"""E-000071 -- revalidate CAVI authority at the actual neural read hook.

E-000070 supplies a fresh cell_mask immediately before model.forward.  This experiment closes the
remaining caller->read TOCTOU gap: an independent live-authority callback is executed as a pre-hook on
every transformer block where the symlink adapter consumes memory.  A deliberately stale commit mask
is still passed by the caller; alias mutation occurs after that mask was produced.  The read hook must
reject the stale alias while a pod-only read-hook baseline accepts it.

No J-space signal participates in routing/training.  This is a structural screening test, not a novelty
claim.  The differentiating case is alias A->P relinked to Q while P remains live and unchanged.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from so.cavi import CAVIAuthority, NeuralConsumptionGuard, RowManifest
from so.data import Bank, bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments.e000070_cavi_live_symlink_boundary import _authority_and_manifest, _row, _run_logits, _text
from so.llm_adapter import AdapterConfig, transformer_blocks
from so.mvcc import CellKind


def _manifest_from_live(auth:CAVIAuthority, store, bank:Bank)->RowManifest:
    pids=[]; pins=[]; aids=[]; ains=[]
    for raw in bank.kid:
        kid=int(raw); cell=store.cells[kid]; v=cell.version_obj(cell.active_version)
        if v.kind is CellKind.LINK and v.target is not None:
            aid=kid; pid=int(v.target); ai=auth.alias_incarnation(aid)
        else:
            aid=-1; pid=kid; ai=0
        pids.append(pid); pins.append(auth.pod_incarnation(pid)); aids.append(aid); ains.append(ai)
    return RowManifest(np.asarray(pids,np.int64),np.asarray(pins,np.int64),np.asarray(aids,np.int64),np.asarray(ains,np.int64))


def run(seed:int,steps:int,groups:int,template:int)->Dict[str,object]:
    torch.manual_seed(seed); rng=np.random.default_rng(81000+seed)
    gk=E8.GPT2Knowledge(AdapterConfig(status_gated=True,use_links=True,n_deref=E20.N_DEREF))
    tr=E20.train_adapter_links(gk,seed,steps,n_groups=max(groups,24),verbose=True)
    centre=np.asarray(tr["centre"])
    world,spec=E15.sample_alias_world(rng,180,groups,2,gk.n_entities,4,E20.N_TRAIN_TEMPLATES)
    store,kids=E15.load_arm(world,spec,centre,82000+seed,symlink=True); bank=bank_from_store(store)
    auth,manifest=_authority_and_manifest(store,bank)
    target,aliases=spec.groups[0]; target2,_=spec.groups[1]
    aid=kids[aliases[0]]; p1=kids[target]; p2=kids[target2]; apos=_row(bank,aid)
    txt=_text(gk,aliases[0],template); old_truth=int(world.index[target]); new_truth=int(world.index[target2])

    # Establish that this real pointer path is capable before attacking authority.
    fresh_before,base_logits,_=_run_logits(gk,bank,txt)
    cached_commit=auth.row_mask(manifest,full=True)

    # Resolve/authorize then mutate: commit-time/caller mask is stale by construction.
    old_witness=auth.witness(aid); store.relink(aid,p2); auth.relink_alias(aid,p2)
    stale_commit_pred,stale_commit_logits,_=_run_logits(gk,bank,txt,cached_commit)

    # Pod-only revalidation at the actual read hook still accepts A->P because P is unchanged/live.
    with NeuralConsumptionGuard(gk.model,lambda:auth.row_mask(manifest,full=False)):
        pod_hook_pred,pod_hook_logits,_=_run_logits(gk,bank,txt,cached_commit)

    # Full CAVI revalidation at each read hook independently checks alias binding + referent.
    with NeuralConsumptionGuard(gk.model,lambda:auth.row_mask(manifest,full=True)):
        cavi_hook_pred,cavi_hook_logits,_=_run_logits(gk,bank,txt,cached_commit)
    explicit=np.ones(bank.size,dtype=bool); explicit[apos]=False
    explicit_pred,explicit_logits,_=_run_logits(gk,bank,txt,explicit)

    # A newly materialized current-generation Bank retains capability to the NEW referent.
    fresh_bank=bank_from_store(store); fresh_manifest=_manifest_from_live(auth,store,fresh_bank)
    with NeuralConsumptionGuard(gk.model,lambda:auth.row_mask(fresh_manifest,full=True)):
        fresh_after_pred,fresh_after_logits,_=_run_logits(gk,fresh_bank,txt)

    # Stronger race: mutation occurs *inside forward* before the consumption guard pre-hook runs.
    worldr,specr=E15.sample_alias_world(np.random.default_rng(83000+seed),120,6,2,gk.n_entities,4,E20.N_TRAIN_TEMPLATES)
    sr,kr=E15.load_arm(worldr,specr,centre,84000+seed,symlink=True); br=bank_from_store(sr); ar,mr=_authority_and_manifest(sr,br)
    rt,ra=specr.groups[0]; rt2,_=specr.groups[1]; raid=kr[ra[0]]; rp2=kr[rt2]
    rtxt=_text(gk,ra[0],template); rcommit=ar.row_mask(mr,full=True); mutated={"done":False}
    blocks=transformer_blocks(gk.model.lm)
    def mutate_before_guard(module,inputs):
        if not mutated["done"]:
            sr.relink(raid,rp2); ar.relink_alias(raid,rp2); mutated["done"]=True
    # Register mutator BEFORE guard: PyTorch invokes same-kind hooks in registration order.
    h=blocks[gk.model.cfg.read_layers[0]].register_forward_pre_hook(mutate_before_guard)
    try:
        with NeuralConsumptionGuard(gk.model,lambda:ar.row_mask(mr,full=True)):
            race_pred,race_logits,_=_run_logits(gk,br,rtxt,rcommit)
    finally:
        h.remove()
    rpos=_row(br,raid); rexp=np.ones(br.size,dtype=bool); rexp[rpos]=False
    _,race_exp_logits,_=_run_logits(gk,br,rtxt,rexp)

    metrics={
      "fresh_before_correct":fresh_before==old_truth,
      "cached_witness_rejected":not auth.validate_witness(old_witness),
      "pod_only_old_witness_accepts":auth.validate_pod_only(old_witness),
      "cached_commit_alias_accepts":bool(cached_commit[apos]),
      "pod_hook_vs_stale_commit_maxabs":float((pod_hook_logits-stale_commit_logits).abs().max()),
      "cavi_hook_vs_explicit_reject_maxabs":float((cavi_hook_logits-explicit_logits).abs().max()),
      "pod_hook_vs_cavi_hook_maxabs":float((pod_hook_logits-cavi_hook_logits).abs().max()),
      "fresh_after_current_correct":fresh_after_pred==new_truth,
      "fresh_after_vs_stale_maxabs":float((fresh_after_logits-stale_commit_logits).abs().max()),
      "race_mutation_happened":bool(mutated["done"]),
      "race_hook_vs_explicit_reject_maxabs":float((race_logits-race_exp_logits).abs().max()),
      "preds":{"fresh_before":fresh_before,"stale_commit":stale_commit_pred,"pod_hook":pod_hook_pred,
               "cavi_hook":cavi_hook_pred,"explicit_reject":explicit_pred,"fresh_after":fresh_after_pred,"race":race_pred},
    }
    checks={
      "real_symlink_capability_before":metrics["fresh_before_correct"],
      "full_binding_needed":metrics["cached_witness_rejected"] and metrics["pod_only_old_witness_accepts"],
      "commit_time_mask_stale":metrics["cached_commit_alias_accepts"],
      "pod_only_hook_preserves_stale_path":metrics["pod_hook_vs_stale_commit_maxabs"]<=1e-7,
      "cavi_hook_matches_explicit_rejection":metrics["cavi_hook_vs_explicit_reject_maxabs"]<=1e-7,
      "cavi_materially_differs_from_pod_only":metrics["pod_hook_vs_cavi_hook_maxabs"]>1e-6,
      "fresh_current_generation_retains_capability":metrics["fresh_after_current_correct"],
      "in_forward_race_was_exercised":metrics["race_mutation_happened"],
      "read_hook_closes_in_forward_race":metrics["race_hook_vs_explicit_reject_maxabs"]<=1e-7,
    }
    return {"seed":seed,"candidate_only":True,"screening_pass":all(checks.values()),"checks":checks,"metrics":metrics,
            "not_claimed":"freshness, epochs, pointers, capabilities, transactions or commit authorization individually; J-space not used"}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',type=int,nargs='*',default=[0]); ap.add_argument('--steps',type=int,default=1200)
    ap.add_argument('--groups',type=int,default=16); ap.add_argument('--template',type=int,default=9); ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--results-dir',default='so/results'); a=ap.parse_args(); torch.set_num_threads(a.threads)
    rows=[run(s,a.steps,a.groups,a.template) for s in a.seeds]
    out={"experiment":"E-000071","all_screening_pass":all(r['screening_pass'] for r in rows),"rows":rows}
    p=Path(a.results_dir);p.mkdir(parents=True,exist_ok=True);(p/'e000071_cavi_read_hook_race.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2));
    if not out['all_screening_pass']: raise SystemExit(2)
if __name__=='__main__': main()
