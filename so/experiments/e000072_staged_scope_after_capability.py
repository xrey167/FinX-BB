"""E-000072 -- preserve the high-capability memory first, then attach exact scope semantics.

E-000060 failed on both seeds because the CELL/BYPASS/UNKNOWN state machine attenuated reads while the
memory itself was still learning to address facts.  That is an optimisation coupling, not a reason to
make scope part of the knowledge representation.  E-000072 separates the stages:

  1. train the recorded BOS + match-gate + two-channel memory with its high-capability soft read path;
  2. freeze that entire learned memory/routing geometry;
  3. instantiate the E-000060 state-machine read path and copy the trained adapter weights into it;
  4. train ONLY the question-vs-prose scope heads on disjoint training templates/prose;
  5. evaluate fixed hard BYPASS / CELL / terminal-UNKNOWN semantics using the copied learned match_tau.

No held-out metric selects a threshold.  This is a performance-screening experiment only; scope routing
is established prior art and is not part of the CAVI novelty claim.  J-space is not used.
"""
from __future__ import annotations

import argparse, json, os, time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000018_no_key_no_injection as E18
from so.experiments.e000055_relevance_calibration import calibrate_relevance
from so.experiments.e000060_scope_before_routing import ScopeBeforeRoutingAdapter
from so.llm_adapter import AdapterConfig

BARS={
 "train/active_correct":(">=",.95), "heldout/active_correct":(">=",.95),
 "revoke_train_min":(">=",.95), "revoke_heldout_min":(">=",.95), "shred_heldout_min":(">=",.95),
 "heldout/revoked_deleted_object":("<=",.02), "broken1_unknown":(">=",.90), "generic/kl_to_base":("<=",.05),
}


def _metrics(gk,centre,seed):
    m=E17.evaluate_templates(gk,17100+seed,centre,E18.N_TRAIN_TEMPLATES)
    return {k:float(v) for k,v in m.items() if isinstance(v,(int,float,bool))}


def _check(m:Dict[str,float]):
    out={}
    for k,(op,b) in BARS.items():
        v=float(m.get(k,float('nan'))); ok=v>=b if op=='>=' else v<=b
        out[k]={'value':v,'op':op,'bar':b,'pass':bool(ok)}
    return out


def _new_scope_model(cfg):
    old=E8.KnowledgeAdapterLM; E8.KnowledgeAdapterLM=ScopeBeforeRoutingAdapter
    try: return E8.GPT2Knowledge(cfg)
    finally: E8.KnowledgeAdapterLM=old


def run(seed:int,steps:int,cal_steps:int,threads:int,outdir:str):
    if threads: torch.set_num_threads(threads)
    os.environ['SO_BOS']='1'; t0=time.time()
    cfg=AdapterConfig(status_gated=True,match_gate=True,two_channel_null=True)

    # Stage 1: the already strongest capability path, unchanged.
    source=E8.GPT2Knowledge(cfg)
    trained=E18.train_arm(source,seed,steps,generic_share=.25)
    centre=np.asarray(trained['centre']); source_metrics=_metrics(source,centre,seed)
    learned=E8.adapter_state(source.model)

    # Stage 2: same parameters, different execution semantics; no knowledge/routing retraining.
    gk=_new_scope_model(cfg)
    gk.model.load_state_dict(learned,strict=False)
    gk.model.scope_mode='soft'
    transplanted_soft=_metrics(gk,centre,seed)

    # Freeze all but relevance heads. calibrate_relevance does the explicit freezing itself.
    hist=calibrate_relevance(gk,seed,cal_steps)
    calibrated_soft=_metrics(gk,centre,seed)
    gk.model.scope_mode='hard'
    hard=_metrics(gk,centre,seed); checks=_check(hard)

    rec={'experiment':'E-000072','candidate_only':True,'seed':seed,'steps':steps,'cal_steps':cal_steps,
         'source_metrics':source_metrics,'transplanted_soft':transplanted_soft,'calibrated_soft':calibrated_soft,
         'hard_metrics':hard,'criteria':checks,'screening_pass':all(x['pass'] for x in checks.values()),
         'learned_match_tau':[float(x) for x in gk.model.match_tau.detach()],
         'calibration_history':hist,'seconds':time.time()-t0,
         'interpretation':'A pass would show the performance seam is solved by staging an ordinary scope classifier after memory capability; it would not itself add CAVI novelty.',
         'jspace_used':False}
    p=Path(outdir);p.mkdir(parents=True,exist_ok=True);(p/f'e000072-seed{seed}.json').write_text(json.dumps(rec,indent=2),encoding='utf-8')
    print(json.dumps({'seed':seed,'screening_pass':rec['screening_pass'],'source_heldout':source_metrics.get('heldout/active_correct'),
          **{k:round(v['value'],4) for k,v in checks.items()}},indent=2),flush=True)
    return rec

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--steps',type=int,default=1200)
    ap.add_argument('--cal-steps',type=int,default=250); ap.add_argument('--threads',type=int,default=2); ap.add_argument('--results-dir',default='ci-e72')
    a=ap.parse_args(); run(a.seed,a.steps,a.cal_steps,a.threads,a.results_dir)
