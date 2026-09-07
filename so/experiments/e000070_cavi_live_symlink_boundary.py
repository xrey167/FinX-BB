"""E-000070 -- CAVI consume-time alias+pod freshness on REAL symlink neural memory.

E-000069 closed stale whole-Bank replay with an all-or-nothing pod capability.  That is necessary but
not sufficient for versioned indirection: a cached alias resolution can become stale while BOTH the
old and new canonical pods remain live.  A simple referent-version check then still accepts the old
pointer.  This experiment is the first direct falsification test of the composed CAVI boundary.

We train the recorded E-000020 frozen-GPT2 link adapter (LINK rows contain only a target address),
export a real symlink Bank, and capture a row manifest.  The neural read receives the serialized Bank
plus a `cell_mask` generated at the LAST boundary before `KnowledgeAdapterLM` consumes it.  Four arms:

  none        serialized Bank is trusted;
  commit      a full mask is computed before mutation and cached (commit/export-time authorization);
  pod_only    live canonical-pod incarnation is rechecked at consumption;
  cavi        BOTH alias binding incarnation and canonical-pod incarnation/reachability are rechecked.

The differentiating attack is ALIAS RELINK: alias A moves from still-live pod P to still-live pod Q.
The old Bank still says A->P. `pod_only` must accept it; CAVI must reject the stale alias row while
leaving bystander pods available.  Root UPDATE/SHRED and ABA controls verify that pod freshness still
works.  A resolve->mutate->inject race verifies that commit-time authorization is not enough.

This is a candidate composition test, NOT a novelty claim.  Any positive seed must be repeated and
compared against systems prior art.  No J-space signal is used for routing or optimization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from so.cavi import CAVIAuthority, RowManifest
from so.data import Bank, bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
from so.llm_adapter import AdapterConfig
from so.mvcc import CellKind
from so.world import UNKNOWN


def _authority_and_manifest(store, bank: Bank) -> Tuple[CAVIAuthority, RowManifest]:
    auth=CAVIAuthority()
    # Canonical FACT cells first; LINK cells bind to those canonical ids.
    for kid,cell in store.cells.items():
        if not cell.versions or cell.status.value == "DELETED":
            continue
        v=cell.version_obj(cell.active_version)
        if v.kind is CellKind.FACT:
            auth.create_pod(int(kid))
    for kid,cell in store.cells.items():
        if not cell.versions or cell.status.value == "DELETED":
            continue
        v=cell.version_obj(cell.active_version)
        if v.kind is CellKind.LINK and v.target is not None:
            auth.create_alias(int(kid),int(v.target))

    pids=[]; pins=[]; aids=[]; ains=[]
    for raw_kid in bank.kid:
        kid=int(raw_kid); cell=store.cells[kid]; v=cell.version_obj(cell.active_version)
        if v.kind is CellKind.LINK and v.target is not None:
            aid=kid; pid=int(v.target); ai=auth.alias_incarnation(aid)
        else:
            aid=-1; pid=kid; ai=0
        pids.append(pid); pins.append(auth.pod_incarnation(pid)); aids.append(aid); ains.append(ai)
    return auth, RowManifest(np.asarray(pids,np.int64),np.asarray(pins,np.int64),
                             np.asarray(aids,np.int64),np.asarray(ains,np.int64))


def _text(gk, key:Tuple[int,int], template:int)->str:
    s,r=key
    return E17.TEMPLATES12[r][template].format(s=gk.names[s])


def _run_logits(gk, bank:Bank|None, text:str, mask:np.ndarray|None=None):
    ids,am,last=E8.encode_texts(gk.tok,[text])
    tensors=None if bank is None else bank.tensors()
    tm=None if mask is None else torch.as_tensor(mask,dtype=torch.bool)
    with torch.no_grad():
        cand,full,routing,_=gk.model(tensors,ids,am,last,cell_mask=tm)
    pred=int(cand.argmax(-1)[0])
    return pred,full.detach().cpu(),None if routing is None else routing.detach().cpu()


def _row(bank:Bank,kid:int)->int:
    hits=np.flatnonzero(bank.kid==kid)
    if len(hits)!=1: raise AssertionError((kid,hits.tolist()))
    return int(hits[0])


def _fresh_read_rate(gk, bank:Bank, keys:Iterable[Tuple[int,int]], world, template:int)->float:
    ok=[]
    for key in keys:
        p,_,_=_run_logits(gk,bank,_text(gk,key,template))
        ok.append(p==int(world.index[key]))
    return float(np.mean(ok)) if ok else 0.0


def run(seed:int,steps:int,n_groups:int,template:int)->Dict[str,object]:
    torch.manual_seed(seed); rng=np.random.default_rng(70000+seed)
    cfg=AdapterConfig(status_gated=True,use_links=True,n_deref=E20.N_DEREF)
    gk=E8.GPT2Knowledge(cfg)
    trained=E20.train_adapter_links(gk,seed,steps,n_groups=max(24,n_groups),verbose=True)
    centre=np.asarray(trained["centre"])

    # Evaluation world is independent of training worlds.
    world,spec=E15.sample_alias_world(rng,180,n_groups,2,gk.n_entities,4,E20.N_TRAIN_TEMPLATES)
    store,kids=E15.load_arm(world,spec,centre,71000+seed,symlink=True)
    bank=bank_from_store(store)
    auth,manifest=_authority_and_manifest(store,bank)
    alias_keys=spec.alias_keys
    fresh_alias_rate=_fresh_read_rate(gk,bank,alias_keys,world,template)

    target,aliases=spec.groups[0]
    other_target,other_aliases=spec.groups[1]
    aid=kids[aliases[0]]; old_pid=kids[target]; new_pid=kids[other_target]
    alias_pos=_row(bank,aid); old_pod_pos=_row(bank,old_pid)
    bystander_key=other_aliases[1]
    bystander_aid=kids[bystander_key]; bystander_pos=_row(bank,bystander_aid)
    txt=_text(gk,aliases[0],template)

    # Baseline neural effect of the real stale alias snapshot.
    stale_pred,stale_logits,_=_run_logits(gk,bank,txt)
    old_truth=int(world.index[target]); new_truth=int(world.index[other_target])

    # COMMIT-TIME baseline caches a valid full mask before the state changes.
    cached_commit_mask=auth.row_mask(manifest,full=True)
    cached_witness=auth.witness(aid)

    # Differentiating attack: alias changes binding, old canonical pod remains current/live.
    store.relink(aid,new_pid); auth.relink_alias(aid,new_pid)
    assert auth.pods[old_pid].live and auth.pods[new_pid].live
    pod_mask=auth.row_mask(manifest,full=False)
    cavi_mask=auth.row_mask(manifest,full=True)
    commit_pred,commit_logits,_=_run_logits(gk,bank,txt,cached_commit_mask)
    pod_pred,pod_logits,_=_run_logits(gk,bank,txt,pod_mask)
    cavi_pred,cavi_logits,_=_run_logits(gk,bank,txt,cavi_mask)
    no_alias_mask=np.ones(bank.size,dtype=bool); no_alias_mask[alias_pos]=False
    explicit_reject_pred,explicit_reject_logits,_=_run_logits(gk,bank,txt,no_alias_mask)

    alias_relink={
        "cached_witness_rejected":not auth.validate_witness(cached_witness),
        "pod_only_witness_accepts":auth.validate_pod_only(cached_witness),
        "commit_alias_row_accepts":bool(cached_commit_mask[alias_pos]),
        "pod_only_alias_row_accepts":bool(pod_mask[alias_pos]),
        "cavi_alias_row_rejects":not bool(cavi_mask[alias_pos]),
        "cavi_preserves_old_pod_bystander":bool(cavi_mask[old_pod_pos]),
        "cavi_preserves_unrelated_alias":bool(cavi_mask[bystander_pos]),
        "cavi_equals_explicit_neural_rejection_maxabs":float((cavi_logits-explicit_reject_logits).abs().max()),
        "pod_only_vs_cavi_maxabs":float((pod_logits-cavi_logits).abs().max()),
        "stale_snapshot_pred":stale_pred,"commit_pred":commit_pred,"pod_only_pred":pod_pred,"cavi_pred":cavi_pred,
        "old_truth":old_truth,"new_truth":new_truth,
    }

    # Root UPDATE attack on a fresh independent world: both pod-only and CAVI must reject root+aliases.
    world2,spec2=E15.sample_alias_world(np.random.default_rng(72000+seed),120,6,2,gk.n_entities,4,E20.N_TRAIN_TEMPLATES)
    s2,k2=E15.load_arm(world2,spec2,centre,73000+seed,symlink=True); b2=bank_from_store(s2); a2,m2=_authority_and_manifest(s2,b2)
    t2,als2=spec2.groups[0]; p2=k2[t2]; al2=k2[als2[0]]; rp2=_row(b2,p2); ra2=_row(b2,al2)
    s2.update(p2,(int(world2.index[t2])+17)%gk.n_entities); a2.update_pod(p2)
    pm2=a2.row_mask(m2,full=False); cm2=a2.row_mask(m2,full=True)
    root_update={"pod_only_rejects_root":not bool(pm2[rp2]),"pod_only_rejects_alias":not bool(pm2[ra2]),
                 "cavi_rejects_root":not bool(cm2[rp2]),"cavi_rejects_alias":not bool(cm2[ra2])}

    # SHRED and ABA controls use live authority as the final consume check.
    world3,spec3=E15.sample_alias_world(np.random.default_rng(74000+seed),120,6,2,gk.n_entities,4,E20.N_TRAIN_TEMPLATES)
    s3,k3=E15.load_arm(world3,spec3,centre,75000+seed,symlink=True); b3=bank_from_store(s3); a3,m3=_authority_and_manifest(s3,b3)
    t3,als3=spec3.groups[0]; p3=k3[t3]; al3=k3[als3[0]]; r3=_row(b3,p3); ar3=_row(b3,al3); w3=a3.witness(al3)
    commit3=a3.row_mask(m3,full=True)
    s3.shred(p3); a3.shred_pod(p3)
    after_shred=a3.row_mask(m3,full=True)
    shred={"cached_commit_would_accept":bool(commit3[r3]) and bool(commit3[ar3]),
           "consume_rejects_root":not bool(after_shred[r3]),"consume_rejects_alias":not bool(after_shred[ar3]),
           "cached_resolver_rejected":not a3.validate_witness(w3)}
    # ABA: same logical pod id is live again at a newer incarnation; old snapshot/witness must remain dead.
    a3.restore_pod(p3)
    aba_mask=a3.row_mask(m3,full=True)
    aba={"old_root_snapshot_rejected":not bool(aba_mask[r3]),"old_alias_snapshot_rejected":not bool(aba_mask[ar3]),
         "old_witness_rejected":not a3.validate_witness(w3),"newer_incarnation":a3.pod_incarnation(p3)>w3.pod_incarnation}

    # Exact explicit BYPASS: memory scope says no memory, so Bank is never passed to the model.
    generic="The ocean was calm under the moon."
    _,bypass1,_=_run_logits(gk,None,generic); _,bypass2,_=_run_logits(gk,None,generic)
    bypass_delta=float((bypass1-bypass2).abs().max())

    checks={
        "fresh_real_symlink_read":fresh_alias_rate>=0.60,  # screening floor, not final performance bar
        "alias_relink_full_rejects":alias_relink["cavi_alias_row_rejects"],
        "alias_relink_pod_only_fails":alias_relink["pod_only_alias_row_accepts"],
        "cached_commit_fails_relink":alias_relink["commit_alias_row_accepts"],
        "cached_resolver_rejected":alias_relink["cached_witness_rejected"],
        "bystanders_preserved":alias_relink["cavi_preserves_old_pod_bystander"] and alias_relink["cavi_preserves_unrelated_alias"],
        "actual_neural_rejection_matches_reference":alias_relink["cavi_equals_explicit_neural_rejection_maxabs"]<=1e-7,
        "rejection_changes_stale_neural_path":alias_relink["pod_only_vs_cavi_maxabs"]>1e-6,
        "root_update_rejected":all(root_update.values()),
        "shred_final_consume_rejects":shred["consume_rejects_root"] and shred["consume_rejects_alias"] and shred["cached_resolver_rejected"],
        "commit_time_not_sufficient":shred["cached_commit_would_accept"],
        "aba_rejected":all(aba.values()),
        "exact_bypass":bypass_delta<=1e-7,
    }
    return {"seed":seed,"steps":steps,"template":template,"screening_pass":all(checks.values()),
            "checks":checks,"fresh_alias_read_rate":fresh_alias_rate,"alias_relink":alias_relink,
            "root_update":root_update,"shred":shred,"aba":aba,"bypass_maxabs":bypass_delta,
            "interpretation":"CAVI is differentiated from a pod-only version check only if alias relink defeats pod_only but not full alias+pod consume validation."}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,nargs="*",default=[0]); ap.add_argument("--steps",type=int,default=1200)
    ap.add_argument("--groups",type=int,default=16); ap.add_argument("--template",type=int,default=9); ap.add_argument("--threads",type=int,default=2)
    ap.add_argument("--results-dir",default="so/results"); a=ap.parse_args(); torch.set_num_threads(a.threads)
    rows:List[Dict[str,object]]=[run(s,a.steps,a.groups,a.template) for s in a.seeds]
    rec={"experiment":"E-000070","candidate_only":True,"all_screening_pass":all(bool(r["screening_pass"]) for r in rows),"rows":rows,
         "not_claimed":"versions, capabilities, HMAC, pointers, commit authorization, or freshness individually; no J-space routing"}
    p=Path(a.results_dir);p.mkdir(parents=True,exist_ok=True);(p/"e000070_cavi_live_symlink_boundary.json").write_text(json.dumps(rec,indent=2),encoding="utf-8")
    print(json.dumps(rec,indent=2))
    if not rec["all_screening_pass"]: raise SystemExit(2)

if __name__=="__main__": main()
