"""E-000053 — parallel frontier sweep around E-000052.

Screening only. Sweeps locality pressure and architecture controls in independent jobs.
A candidate is promoted only if the same strict joint bars as E-000052 pass.
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import torch
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000018_no_key_no_injection as E18
from so.llm_adapter import AdapterConfig

CRITERIA = {
    "train/active_correct": (">=", 0.95),
    "heldout/active_correct": (">=", 0.95),
    "revoke_train_min": (">=", 0.95),
    "revoke_heldout_min": (">=", 0.95),
    "shred_heldout_min": (">=", 0.95),
    "heldout/revoked_deleted_object": ("<=", 0.02),
    "broken1_unknown": (">=", 0.90),
    "generic/kl_to_base": ("<=", 0.05),
}

def run(seed:int, steps:int, generic_share:float, variant:str, threads:int, outdir:str):
    if threads: torch.set_num_threads(threads)
    os.environ["SO_BOS"] = "1"
    if variant == "full":
        cfg = AdapterConfig(status_gated=True, match_gate=True, two_channel_null=True)
    elif variant == "match_only":
        cfg = AdapterConfig(status_gated=True, match_gate=True, two_channel_null=False)
    elif variant == "null_only":
        cfg = AdapterConfig(status_gated=True, match_gate=False, two_channel_null=True)
    elif variant == "plain":
        cfg = AdapterConfig(status_gated=True)
    else:
        raise ValueError(variant)
    gk = E8.GPT2Knowledge(cfg)
    t0=time.time()
    trained = E18.train_arm(gk, seed, steps, generic_share=generic_share)
    m = E17.evaluate_templates(gk, 5300 + seed, trained["centre"], E18.N_TRAIN_TEMPLATES)
    m = {k:float(v) for k,v in m.items() if isinstance(v,(int,float,bool))}
    checks={}
    for k,(op,bar) in CRITERIA.items():
        v=m.get(k,float("nan")); ok=(v>=bar) if op==">=" else (v<=bar)
        checks[k]={"value":v,"op":op,"bar":bar,"pass":bool(ok)}
    rec={"experiment":"E-000053","screening_only":True,"seed":seed,"steps":steps,
         "generic_share":generic_share,"variant":variant,"adapter":cfg.to_dict(),"metrics":m,
         "criteria":checks,"screening_pass":all(c["pass"] for c in checks.values()),
         "seconds":time.time()-t0}
    p=Path(outdir); p.mkdir(parents=True,exist_ok=True)
    stem=f"e000053-{variant}-g{generic_share:g}-s{seed}"
    (p/(stem+".json")).write_text(json.dumps(rec,indent=2),encoding="utf-8")
    print(json.dumps({"variant":variant,"generic_share":generic_share,"seed":seed,
        "screening_pass":rec["screening_pass"],
        **{k:round(v["value"],4) for k,v in checks.items()}},indent=2),flush=True)
    return rec

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--steps",type=int,default=1200)
    ap.add_argument("--generic-share",type=float,default=0.25)
    ap.add_argument("--variant",choices=["full","match_only","null_only","plain"],default="full")
    ap.add_argument("--threads",type=int,default=2)
    ap.add_argument("--results-dir",default="ci-e53")
    a=ap.parse_args(); run(a.seed,a.steps,a.generic_share,a.variant,a.threads,a.results_dir)
