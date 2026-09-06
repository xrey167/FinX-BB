"""E-000069 -- enforce incarnation freshness at the actual Bank->adapter consumption boundary.

E-000066 proved an old exported Bank remains replayable after the live store shreds/deletes its pod.
E-000068 proved a live incarnation authority can reject old one-use capabilities. E-000069 composes
those pieces with the REAL frozen-GPT2 KnowledgeAdapterLM API.

No adapter training is needed because the property is authorization, not task accuracy. We choose an
adapter configuration guaranteed to inject a nonzero memory read (marker gate off, one active cell)
and verify that an authorized Bank materially changes GPT-2 logits. Then we invalidate the pod and
attempt both consumed-token replay and an unconsumed pre-invalidation token. Both must be rejected
BEFORE model memory consumption; rejection calls `model(None, ...)`, whose logits must be bit-identical
(up to floating execution determinism) to the explicit no-memory base path.

Registered criteria, every seed:
  active authorized memory changes logits by >1e-6;
  a capability cannot be reused in the same incarnation;
  an unconsumed pre-SHRED capability is invalid after incarnation bump;
  an old capability cannot be rebound to a new request nonce;
  rejected stale memory == explicit no-memory path, max |delta logits| <=1e-7;
  after RESTORE, a fresh new-incarnation capability is accepted and memory changes logits again;
  the pre-SHRED capability remains invalid after RESTORE (ABA resistance);
  after DELETE, old memory remains rejected and no fresh capability can be minted.

Passing closes the interface-level replay hole under the declared trust boundary. It does NOT prove
novelty: commit-time/freshness authorization, capabilities and epochs have extensive systems prior art.

Run: python -m so.experiments.e000069_authorized_injection_boundary --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from so.authorized_memory import AuthorizedSnapshot, consume
from so.data import bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.incarnation import IncarnationAuthority
from so.llm_adapter import AdapterConfig
from so.mvcc import MVCCStore


def nonce(seed:int, tag:str)->bytes:
    return hashlib.sha256(f"e69:{seed}:{tag}".encode()).digest()[:16]


def model_logits(gk:E8.GPT2Knowledge, bank, text:str)->torch.Tensor:
    ids,am,last=E8.encode_texts(gk.tok,[text])
    tensors=None if bank is None else bank.tensors()
    with torch.no_grad():
        _,full,_,_=gk.model(tensors,ids,am,last)
    return full.detach().cpu()


def run(seed:int)->Dict[str,object]:
    torch.manual_seed(seed)
    cfg=AdapterConfig(use_marker_gate=False, status_gated=True, fallback="prior", read_layers=(8,10))
    gk=E8.GPT2Knowledge(cfg)
    centre=np.zeros(cfg.marker_dim,dtype=np.float32)
    s=MVCCStore(marker_dim=cfg.marker_dim,seed=69000+seed,marker_centre=centre)
    # Use IDs that are valid synthetic entities for GPT2Knowledge's candidate vocabulary.
    subject=7; relation=0; obj=11
    root=s.write(subject,relation,obj,provenance="pod")
    bank=bank_from_store(s)
    text=f"{gk.names[subject]}'s relation is"

    auth=IncarnationAuthority(secret=hashlib.sha256(f"e69-secret:{seed}".encode()).digest())
    auth.create(1)
    base=model_logits(gk,None,text)

    n_active=nonce(seed,"active")
    cap_active=auth.issue(1,n_active)
    snap=AuthorizedSnapshot(1,bank,cap_active)
    accepted=consume(auth,snap,n_active)
    active=model_logits(gk,accepted,text)
    active_delta=float((active-base).abs().max())
    consumed_replay=consume(auth,snap,n_active)

    # Mint another token but do NOT consume it; invalidate the pod first.
    n_stale=nonce(seed,"stale-unconsumed")
    cap_stale=auth.issue(1,n_stale)
    stale_snap=AuthorizedSnapshot(1,bank,cap_stale)
    auth.shred(1); s.shred(root)
    rejected_same_nonce=consume(auth,stale_snap,n_stale)
    rejected_new_nonce=consume(auth,stale_snap,nonce(seed,"different"))
    stale_logits=model_logits(gk,rejected_same_nonce,text)
    stale_base_delta=float((stale_logits-base).abs().max())

    # Restore payload in store but to a NEW authority incarnation. Old token must stay dead.
    s.resign(root); auth.restore(1)
    old_after_restore=consume(auth,stale_snap,n_stale)
    fresh_bank=bank_from_store(s)
    n_restore=nonce(seed,"restored")
    fresh_cap=auth.issue(1,n_restore)
    fresh_snap=AuthorizedSnapshot(1,fresh_bank,fresh_cap)
    fresh_accepted=consume(auth,fresh_snap,n_restore)
    restored_logits=model_logits(gk,fresh_accepted,text)
    restored_delta=float((restored_logits-base).abs().max())

    # Delete is terminal.
    n_pre_delete=nonce(seed,"pre-delete")
    pre_delete_cap=auth.issue(1,n_pre_delete)
    pre_delete_snap=AuthorizedSnapshot(1,fresh_bank,pre_delete_cap)
    s.delete(root); auth.delete(1)
    after_delete=consume(auth,pre_delete_snap,n_pre_delete)
    deleted_logits=model_logits(gk,after_delete,text)
    delete_base_delta=float((deleted_logits-base).abs().max())
    mint_after_delete=False
    try: auth.issue(1,nonce(seed,"after-delete"))
    except PermissionError: mint_after_delete=True

    checks={
        "active_memory_changes_logits":active_delta>1e-6,
        "consumed_cap_replay_rejected":consumed_replay is None,
        "unconsumed_pre_shred_rejected":rejected_same_nonce is None,
        "old_cap_wrong_nonce_rejected":rejected_new_nonce is None,
        "stale_rejection_exact_bypass":stale_base_delta<=1e-7,
        "old_cap_not_aba_after_restore":old_after_restore is None,
        "fresh_restore_accepted":fresh_accepted is not None and restored_delta>1e-6,
        "delete_rejects_old_snapshot":after_delete is None and delete_base_delta<=1e-7,
        "delete_denies_new_capability":mint_after_delete,
    }
    return {"seed":seed,"pass":all(checks.values()),"checks":checks,
            "active_vs_base_maxabs":active_delta,"stale_vs_base_maxabs":stale_base_delta,
            "restored_vs_base_maxabs":restored_delta,"deleted_vs_base_maxabs":delete_base_delta}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--seeds",type=int,nargs="*",default=[0,1,2]);ap.add_argument("--threads",type=int,default=2);ap.add_argument("--results-dir",default="so/results")
    a=ap.parse_args();torch.set_num_threads(a.threads);rows:List[Dict[str,object]]=[run(s) for s in a.seeds];all_pass=all(bool(r["pass"]) for r in rows)
    rec={"experiment":"E-000069","all_pass":all_pass,"rows":rows,
         "claim":"interface control only: consume-time live authority makes stale exported Bank fail to exact no-memory path",
         "not_claimed":"HMAC/nonces/epochs/capabilities or commit-time freshness are not new"}
    p=Path(a.results_dir);p.mkdir(parents=True,exist_ok=True);(p/"e000069_authorized_injection_boundary.json").write_text(json.dumps(rec,indent=2),encoding="utf-8")
    print(json.dumps(rec,indent=2))
    if not all_pass: raise SystemExit(2)

if __name__=="__main__":main()
