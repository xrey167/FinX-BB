"""E-000056 — reduce the number of intervention sites instead of squeezing the gates harder.

E-000053 shows the two-read-layer BOS adapter can reach the reading/deletion bars but still moves generic
text.  Each read layer independently writes into the frozen residual stream, so generic perturbation can
accumulate even when each gate is small.  This orthogonal ablation trains the same full locality
architecture with exactly ONE read site (layer 8 or 10).  A positive cell must still satisfy the full
joint bars; no relaxed locality bar is used.
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import torch
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000018_no_key_no_injection as E18
from so.llm_adapter import AdapterConfig

BARS={
 "train/active_correct":(">=",.95), "heldout/active_correct":(">=",.95),
 "revoke_train_min":(">=",.95), "revoke_heldout_min":(">=",.95), "shred_heldout_min":(">=",.95),
 "heldout/revoked_deleted_object":("<=",.02), "broken1_unknown":(">=",.90),
 "generic/kl_to_base":("<=",.05),
}

def run(layer:int,seed:int,steps:int,threads:int,outdir:str):
    if threads: torch.set_num_threads(threads)
    os.environ['SO_BOS']='1'
    cfg=AdapterConfig(status_gated=True,match_gate=True,two_channel_null=True,read_layers=(layer,))
    gk=E8.GPT2Knowledge(cfg); t0=time.time()
    out=E18.train_arm(gk,seed,steps,generic_share=.25)
    m=E17.evaluate_templates(gk, 8600+seed+layer*10, out['centre'], E18.N_TRAIN_TEMPLATES)
    metrics={k:float(v) for k,v in m.items() if isinstance(v,(int,float,bool))}
    crit={}
    for k,(op,b) in BARS.items():
        v=metrics.get(k,float('nan')); ok=v>=b if op=='>=' else v<=b
        crit[k]={'value':v,'op':op,'bar':b,'pass':bool(ok)}
    rec={'experiment':'E-000056','candidate_only':True,'read_layer':layer,'seed':seed,'steps':steps,
         'adapter':cfg.to_dict(),'metrics':metrics,'criteria':crit,
         'screening_pass':all(x['pass'] for x in crit.values()),'seconds':time.time()-t0}
    print(json.dumps({'read_layer':layer,'seed':seed,'screening_pass':rec['screening_pass'],
        **{k:round(crit[k]['value'],4) for k in BARS}},indent=2),flush=True)
    p=Path(outdir);p.mkdir(parents=True,exist_ok=True)
    (p/f'e000056-l{layer}-s{seed}.json').write_text(json.dumps(rec,indent=2),encoding='utf-8')
    return rec

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--layer',type=int,choices=[8,10],required=True)
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--steps',type=int,default=1200)
    ap.add_argument('--threads',type=int,default=2);ap.add_argument('--results-dir',default='ci-e56')
    a=ap.parse_args();run(a.layer,a.seed,a.steps,a.threads,a.results_dir)
