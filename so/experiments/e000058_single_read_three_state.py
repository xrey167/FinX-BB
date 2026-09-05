"""E-000058 — interaction test: one intervention site + exact MATCH/UNKNOWN/BYPASS policy.

E-000056 removes accumulated writes; E-000057 makes the decision semantics exact.  Their interaction is
not implied by either ablation, and with one read site there is no ambiguity about an earlier null read
on a one-hop question.  Train the recorded soft full architecture at one read layer, then evaluate the
same weights through E-000057's hard three-state mux.  Diagnostic only; positive cells require held-out
threshold selection, multi-seed replication and attacks.
"""
from __future__ import annotations
import argparse, copy, json, os, time
from pathlib import Path
import numpy as np
import torch
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000018_no_key_no_injection as E18
from so.experiments.e000057_three_state_router import ThreeStateAdapter
from so.llm_adapter import AdapterConfig

BARS={
 "train/active_correct":(">=",.95), "heldout/active_correct":(">=",.95),
 "revoke_train_min":(">=",.95), "revoke_heldout_min":(">=",.95), "shred_heldout_min":(">=",.95),
 "heldout/revoked_deleted_object":("<=",.02), "broken1_unknown":(">=",.90), "generic/kl_to_base":("<=",.05),
}
REL_TAUS=(.30,.50,.70); MATCH_OFFSETS=(-.05,0.0,.05)

def checks(m):
    c={}
    for k,(op,b) in BARS.items():
        v=float(m.get(k,float('nan'))); ok=v>=b if op=='>=' else v<=b
        c[k]={'value':v,'op':op,'bar':b,'pass':bool(ok)}
    return c

def run(layer,seed,steps,threads,outdir):
    if threads: torch.set_num_threads(threads)
    os.environ['SO_BOS']='1'; old=E8.KnowledgeAdapterLM; E8.KnowledgeAdapterLM=ThreeStateAdapter
    try:
        cfg=AdapterConfig(status_gated=True,match_gate=True,two_channel_null=True,read_layers=(layer,))
        gk=E8.GPT2Knowledge(cfg)
    finally: E8.KnowledgeAdapterLM=old
    model=gk.model; model.tri_mode='parent'; t0=time.time()
    out=E18.train_arm(gk,seed,steps,generic_share=.25); centre=np.asarray(out['centre'])
    state=copy.deepcopy(model.state_dict()); rows=[]
    for rt in REL_TAUS:
        for mo in MATCH_OFFSETS:
            model.load_state_dict(state); model.tri_mode='hard'; model.rel_tau=rt; model.match_offset=mo
            m=E17.evaluate_templates(gk,10800+seed+layer*10,centre,E18.N_TRAIN_TEMPLATES)
            mm={k:float(v) for k,v in m.items() if isinstance(v,(int,float,bool))}; c=checks(mm)
            row={'rel_tau':rt,'match_offset':mo,'metrics':mm,'criteria':c,'screening_pass':all(x['pass'] for x in c.values())};rows.append(row)
            print({'layer':layer,'rel_tau':rt,'match_offset':mo,'screening_pass':row['screening_pass'],
                   **{k:round(c[k]['value'],4) for k in BARS}},flush=True)
    rec={'experiment':'E-000058','candidate_only':True,'layer':layer,'seed':seed,'steps':steps,'adapter':cfg.to_dict(),
         'rows':rows,'any_screening_pass':any(r['screening_pass'] for r in rows),'seconds':time.time()-t0}
    p=Path(outdir);p.mkdir(parents=True,exist_ok=True)
    (p/f'e000058-l{layer}-s{seed}.json').write_text(json.dumps(rec,indent=2),encoding='utf-8');return rec

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--layer',type=int,choices=[8,10],required=True);ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--steps',type=int,default=1200);ap.add_argument('--threads',type=int,default=2);ap.add_argument('--results-dir',default='ci-e58')
    a=ap.parse_args();run(a.layer,a.seed,a.steps,a.threads,a.results_dir)
