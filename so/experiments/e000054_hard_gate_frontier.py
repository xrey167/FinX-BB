"""E-000054 — hard decision frontier over the two remaining locality channels.

Train the E-000052 full architecture with soft gates, then evaluate the same weights under
hard inference thresholds for absolute key match and query relevance. This asks whether the
remaining locality/refusal failures are soft-gating leakage rather than representation failure.
Candidate only: any positive cell must be repeated across seeds and attacked.
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import torch
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000018_no_key_no_injection as E18
from so.llm_adapter import AdapterConfig

BARS = {
    "train/active_correct": (">=", .95),
    "heldout/active_correct": (">=", .95),
    "revoke_train_min": (">=", .95),
    "revoke_heldout_min": (">=", .95),
    "shred_heldout_min": (">=", .95),
    "heldout/revoked_deleted_object": ("<=", .02),
    "broken1_unknown": (">=", .90),
    "generic/kl_to_base": ("<=", .05),
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,default=0); ap.add_argument('--steps',type=int,default=1200)
    ap.add_argument('--generic-share',type=float,default=.25)
    ap.add_argument('--hard-match',type=float,default=.5); ap.add_argument('--hard-rel',type=float,default=.5)
    ap.add_argument('--threads',type=int,default=2); ap.add_argument('--results-dir',default='ci-e54')
    a=ap.parse_args(); torch.set_num_threads(a.threads); os.environ['SO_BOS']='1'
    cfg=AdapterConfig(status_gated=True, match_gate=True, two_channel_null=True)
    gk=E8.GPT2Knowledge(cfg); t0=time.time()
    out=E18.train_arm(gk,a.seed,a.steps,generic_share=a.generic_share)
    os.environ['SO_HARD_MATCH_TAU']=str(a.hard_match); os.environ['SO_HARD_REL_TAU']=str(a.hard_rel)
    m=E17.evaluate_templates(gk, 6400+a.seed, out['centre'], E18.N_TRAIN_TEMPLATES)
    metrics={k:float(v) for k,v in m.items() if isinstance(v,(int,float,bool))}
    checks={}
    for k,(op,b) in BARS.items():
        v=metrics.get(k,float('nan')); ok=(v>=b) if op=='>=' else (v<=b)
        checks[k]={'value':v,'op':op,'bar':b,'pass':bool(ok)}
    rec={'experiment':'E-000054','candidate_only':True,'seed':a.seed,'steps':a.steps,
         'generic_share':a.generic_share,'hard_match':a.hard_match,'hard_rel':a.hard_rel,
         'metrics':metrics,'criteria':checks,'screening_pass':all(x['pass'] for x in checks.values()),
         'seconds':time.time()-t0}
    p=Path(a.results_dir); p.mkdir(parents=True,exist_ok=True)
    fn=f"e54-m{a.hard_match}-r{a.hard_rel}-s{a.seed}.json"; (p/fn).write_text(json.dumps(rec,indent=2))
    print(json.dumps({k:rec[k] for k in ('hard_match','hard_rel','seed','screening_pass')},indent=2))
    for k,c in checks.items(): print(k, round(c['value'],4), c['op'], c['bar'], 'PASS' if c['pass'] else 'FAIL')
if __name__=='__main__': main()
