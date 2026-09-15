"""E-000073 -- concurrent mutation in the validation->neural-consumption interval.

E-000071 showed that revalidation at a transformer read hook can reject state that changed before the
hook executes.  The stronger systems claim is atomic consume-time authority: another thread must not
be able to relink/revoke the alias after final validation but before the adapter's forward hook consumes
the cached Bank row.

This experiment creates exactly that schedule on a REAL trained symlink GPT-2 adapter.  A coordination
pre-hook is registered after the CAVI guard's pre-hook.  It starts a second thread which attempts to
relink the target alias while the transformer block is between pre-hook validation and the adapter's
memory read.

Arms:
  unlocked -- live mask refreshed at the hook, but no authority lock is held across consumption;
              concurrent relink must complete before the memory read, exposing a stale-path race.
  atomic   -- the same guard holds CAVIAuthority.lock from validation through the adapter memory hook;
              relink must remain blocked until after consumption and then commit normally.

Locks, TOCTOU prevention, MVCC and freshness are established systems concepts and are NOT novelty
claims individually.  The candidate property is their placement at a version-qualified neural-memory
consumption boundary for pointer-only aliases.
"""
from __future__ import annotations

import argparse, json, threading, time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from so.cavi import NeuralConsumptionGuard
from so.data import bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments.e000070_cavi_live_symlink_boundary import _authority_and_manifest, _row, _run_logits, _text
from so.llm_adapter import AdapterConfig, transformer_blocks


def _arm(gk, centre, seed:int, groups:int, template:int, *, atomic:bool)->Dict[str,object]:
    rng=np.random.default_rng(91000 + seed + (1000 if atomic else 0))
    world,spec=E15.sample_alias_world(rng,180,groups,2,gk.n_entities,4,E20.N_TRAIN_TEMPLATES)
    store,kids=E15.load_arm(world,spec,centre,92000+seed+(1000 if atomic else 0),symlink=True)
    bank=bank_from_store(store); auth,manifest=_authority_and_manifest(store,bank)
    target,aliases=spec.groups[0]; target2,_=spec.groups[1]
    aid=kids[aliases[0]]; p2=kids[target2]; apos=_row(bank,aid)
    text=_text(gk,aliases[0],template)
    old_truth=int(world.index[target]); new_truth=int(world.index[target2])

    fresh_pred,_,_=_run_logits(gk,bank,text)
    cached_commit=auth.row_mask(manifest,full=True)

    started=threading.Event(); finished=threading.Event(); state={"error":None,"elapsed":None}
    blocks=transformer_blocks(gk.model.lm); block=blocks[gk.model.cfg.read_layers[0]]

    def mutate():
        t=time.perf_counter(); started.set()
        try:
            store.relink(aid,p2)
            # Store mutation is deliberately outside CAVI authority.  The authorization-critical mutation
            # is this call; in the atomic arm it must block on auth.lock until neural consumption completes.
            auth.relink_alias(aid,p2)
        except Exception as e:  # pragma: no cover - surfaced in result
            state["error"]=repr(e)
        state["elapsed"]=time.perf_counter()-t; finished.set()

    worker={"thread":None,"finished_inside":False}
    def coordinate_after_validation(module,inputs):
        th=threading.Thread(target=mutate,daemon=True); worker["thread"]=th; th.start()
        if not started.wait(1.0): raise RuntimeError("mutator did not start")
        # Give an unlocked authority mutation a deterministic opportunity to commit.  In the atomic arm
        # the second thread is blocked on the authority lock held by NeuralConsumptionGuard.
        worker["finished_inside"]=finished.wait(0.10)

    lock=auth.lock if atomic else None
    with NeuralConsumptionGuard(gk.model,lambda:auth.row_mask(manifest,full=True),lock=lock) as guard:
        # Guard pre-hooks were registered in __init__; registering this hook now puts it after validation.
        h=block.register_forward_pre_hook(coordinate_after_validation)
        try:
            pred,logits,_=_run_logits(gk,bank,text,cached_commit)
        finally:
            h.remove()
    th=worker["thread"]
    if th is not None: th.join(timeout=2.0)
    if th is not None and th.is_alive(): raise RuntimeError("mutator remained blocked after neural read")
    if state["error"] is not None: raise RuntimeError(state["error"])

    # Reference paths from independent masks after the mutation has committed.
    reject=np.ones(bank.size,dtype=bool); reject[apos]=False
    _,reject_logits,_=_run_logits(gk,bank,text,reject)
    stale_pred,stale_logits,_=_run_logits(gk,bank,text,cached_commit)
    return {
      "atomic":atomic,
      "fresh_before_correct":fresh_pred==old_truth,
      "mutation_finished_inside_interval":bool(worker["finished_inside"]),
      "mutation_finished_after_forward":bool(finished.is_set()),
      "pred_during_race":int(pred),"old_truth":old_truth,"new_truth":new_truth,"post_mutation_stale_pred":int(stale_pred),
      "race_vs_stale_maxabs":float((logits-stale_logits).abs().max()),
      "race_vs_reject_maxabs":float((logits-reject_logits).abs().max()),
      "mutation_elapsed_s":float(state["elapsed"] or 0.0),
    }


def run(seed:int,steps:int,groups:int,template:int)->Dict[str,object]:
    torch.manual_seed(seed)
    gk=E8.GPT2Knowledge(AdapterConfig(status_gated=True,use_links=True,n_deref=E20.N_DEREF))
    tr=E20.train_adapter_links(gk,seed,steps,n_groups=max(groups,24),verbose=True)
    centre=np.asarray(tr["centre"])
    unlocked=_arm(gk,centre,seed,groups,template,atomic=False)
    atomic=_arm(gk,centre,seed,groups,template,atomic=True)
    checks={
      "real_symlink_capability":unlocked["fresh_before_correct"] and atomic["fresh_before_correct"],
      "unlocked_exposes_interval":unlocked["mutation_finished_inside_interval"],
      "atomic_blocks_interval_mutation":not atomic["mutation_finished_inside_interval"],
      "atomic_allows_mutation_after_read":atomic["mutation_finished_after_forward"],
      # Atomic read must be materially different from an explicit rejection produced after mutation;
      # it consumed the generation that was current when its atomic validation began.
      "atomic_consumes_validated_generation":atomic["race_vs_reject_maxabs"]>1e-6,
    }
    return {"seed":seed,"screening_pass":all(checks.values()),"checks":checks,"unlocked":unlocked,"atomic":atomic,
            "interpretation":"A positive result establishes that freshness validation alone leaves a neural TOCTOU interval, while locking the version-qualified authority through the actual adapter read closes it."}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',type=int,nargs='*',default=[0]); ap.add_argument('--steps',type=int,default=1200)
    ap.add_argument('--groups',type=int,default=16); ap.add_argument('--template',type=int,default=9); ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--results-dir',default='so/results'); a=ap.parse_args(); torch.set_num_threads(a.threads)
    rows=[run(s,a.steps,a.groups,a.template) for s in a.seeds]
    out={"experiment":"E-000073","candidate_only":True,"all_screening_pass":all(r['screening_pass'] for r in rows),"rows":rows,
         "not_claimed":"locks, TOCTOU, MVCC, freshness, symlinks, external memory, or capabilities individually"}
    p=Path(a.results_dir); p.mkdir(parents=True,exist_ok=True)
    (p/'e000073_concurrent_neural_toctou.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))
    if not out['all_screening_pass']: raise SystemExit(2)

if __name__=='__main__': main()
