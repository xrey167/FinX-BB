"""Experiment E-000032 — the deletion closure of a store, and the certificate that composes with it.

E-000030 produced this programme's first certified deletions: for every value a deleted payload could
hold, the model's computation is bit-identical, so no attack that exists or will be invented can tell
the values apart. That is a statement about a RECORD. A data subject does not ask about a record. They
ask whether the FACT is gone, and this repository already holds the case where the two come apart
completely: `dependency/derivable_recovery_after_revoke_K3 = 1.0` in every seed of E-000019 -- a
certified record deletion under which every derivable fact survives, totally.

THE COMPOSITION THIS EXPERIMENT MEASURES.

    record-level certificate over R  +  R covers the fact closure of the fact
    ------------------------------------------------------------------------
                        fact-level certificate

The second premise is a property of the STORE, not of a checkpoint: `so.closure.fact_closure` computes
it with the mechanical resolver and no model at all, and reports a certified lower bound beside the
greedy answer so "optimal" is verified rather than assumed. The guarantee therefore factorises into
one expensive model-side proof and one cheap store-side search -- and the search is exactly what
canonicalisation makes trivial.

WHY THE SYMLINK IS THE POINT AND NOT AN ORNAMENT. Two stores are built from the SAME world with the
SAME ground truth (E-000015's `sample_alias_world` and `load_arm`), so they present an IDENTICAL
interface: every key resolves to the same object in both. In the canonical arm the group's k access
keys are LINK cells sharing one object; in the duplicated arm each key carries its own copy. At the
record level the two stores are indistinguishable -- per-key closure is one in both -- and their
erasure cost differs by a factor of k. That gap is what a record-level certificate cannot see and what
this experiment puts a number on.

The honest framing, stated here so the record carries it: the gap itself is Codd's deletion anomaly,
and normalization as the remedy is 1971. What is not Codd is that normalization is free in a database,
where a join is exact, and is NOT free in a neural memory, where the reader must LEARN to dereference
and can refuse or fail. E-000025 priced that on a frozen GPT-2: sharing costs 0.0954 and having
trained on links at all costs 0.0688, worst of three seeds over all twelve phrasings. This experiment
adds the other half -- what the price BUYS, in units of certificate.

WHAT COULD FALSIFY IT. Four controls, each able to void the comparison:
  * the two arms must present the same interface, or nothing below is a like-for-like comparison;
  * per-key closure must be one in BOTH arms, or the stores were already distinguishable at the
    record level and the fact level is not doing the work;
  * the model must READ the fact before any deletion, or "the fact is gone afterwards" is vacuous;
  * removing the whole closure must certify in BOTH arms, or the instrument reports "canonical" where
    it should report "enough records removed".

Trains nothing when the E-000015 checkpoints are present.

Run:  python -m so.experiments.e000032_deletion_closure [--seeds 0 1 2] [--n-groups 25]
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from so import ledger
from so.audit import certify_encoding, certify_fact, certify_structural
from so.closure import closure_profile, fact_closure
from so.data import bank_from_store
from so.experiments.e000015_symlink_cells import (EVAL, _q1, encode_slots, load_arm, predict,
                                                  sample_alias_world, train_or_load)
from so.experiments.e000001b_mini_transformer import _sha256, CHECKPOINTS, CKPT_SUFFIX
from so.reference import ReferenceResolver
from so.world import UNKNOWN, World

ARMS = ("canonical", "duplicated")
N_GROUPS = 25            # alias groups certified per seed; each costs one encoding sweep per arm


def group_keys(spec, target_key) -> Tuple[Tuple[int, int], ...]:
    """The group's access keys: the target's own plus its aliases.

    Taken from the SPEC rather than from either store, so the two arms are measured over exactly the
    same key set and the only thing that differs is how the store holds them.
    """
    for t, aliases in spec.groups:
        if t == target_key:
            return tuple([t] + list(aliases))
    raise KeyError(target_key)


def read_keys(model, store, world, keys: Sequence[Tuple[int, int]]) -> np.ndarray:
    return predict(model, bank_from_store(store), world, [_q1(world, k) for k in keys]).answers


def certify_removal(model, store, world, kids_removed: Sequence[int], batch_queries) -> Tuple[Any, Any]:
    """The record-level half, on the bank as it stands after the removal.

    EVICT takes the row out of the bank, so there is no payload left to perturb and the payload sweep
    has nothing to say -- `certify_encoding` over an empty row set certifies with one evaluation, and
    that is vacuous rather than strong. The structural result is what actually carries the claim here:
    autograd is asked whether any output is reachable from the removed payload at all, which is a
    theorem over any domain rather than over a swept one. `certify_fact` refuses the vacuous sweep
    unless that structural result is supplied, so both are returned and both are recorded.
    """
    bank = bank_from_store(store)
    tensors = bank.tensors()
    batch = encode_slots(list(batch_queries), bank, world, model.cfg.max_hops, model.cfg.n_deref)
    record = certify_encoding(model, tensors, [], world.n_entities,
                              interface_keys=("k_f", "v_f", "k_r", "v_r", "active"))
    struct = certify_structural(model, tensors, [], 
                                lambda b: model(b, batch.mode, batch.start, batch.rels, batch.hop_valid),
                                model.cfg.d_model, outputs_of=lambda o: o[0])
    return record, struct


def run_seed(seed: int, n_groups: int, steps: int, verbose: bool = True) -> Dict[str, Any]:
    path = CHECKPOINTS / f"e000015_deref1{CKPT_SUFFIX}_seed{seed}.pt"
    out = train_or_load(seed, steps, n_deref=1)
    model, centre = out["model"], out["centre"]

    rng = np.random.default_rng(seed)
    world, spec = sample_alias_world(rng, EVAL["n_base"], EVAL["n_groups"], EVAL["n_alias_per_group"])
    stores = {"canonical": load_arm(world, spec, centre, seed, symlink=True),
              "duplicated": load_arm(world, spec, centre, seed, symlink=False)}

    m: Dict[str, Any] = {"seed": seed, "n_entities": world.n_entities,
                         "checkpoint_sha256": _sha256(path) if path.exists() else "",
                         "n_alias_per_group": EVAL["n_alias_per_group"],
                         "keys_per_group": EVAL["n_alias_per_group"] + 1}
    t0 = time.time()

    # ---- control 1: the two arms present the same interface, or nothing below compares anything
    views = {a: ReferenceResolver(st).view() for a, (st, _) in stores.items()}
    answers = {a: {k: int(o) for k, (o, _) in v.items()} for a, v in views.items()}
    m["control/interface_identical"] = float(answers["canonical"] == answers["duplicated"])
    m["control/n_keys"] = len(answers["canonical"])

    # ---- control 2: per-key closure is one in BOTH arms -- indistinguishable at the record level
    chosen = [spec.groups[int(i)][0] for i in rng.permutation(len(spec.groups))[:n_groups]]
    all_keys = [k for t in chosen for k in group_keys(spec, t)]
    for arm, (store, _) in stores.items():
        prof = closure_profile(store, all_keys)
        m[f"{arm}/per_key_closure_max"] = float(prof.max)
        m[f"{arm}/per_key_closure_mean"] = float(prof.mean)
        m[f"{arm}/per_key_n"] = int(prof.n)

    # ---- control 3: the model reads the fact before anything is deleted
    truth = np.array([answers["canonical"][k] for k in all_keys])
    for arm, (store, _) in stores.items():
        m[f"{arm}/read_before_deletion"] = float((read_keys(model, store, world, all_keys) == truth).mean())
    m["control/read_before_deletion"] = min(m[f"{a}/read_before_deletion"] for a in ARMS)

    # ---- the closure itself, and the composition
    probe_queries = [_q1(world, k) for k in all_keys[: min(64, len(all_keys))]]
    per_arm: Dict[str, Dict[str, List[float]]] = {a: {} for a in ARMS}
    for arm in ARMS:
        store, kids = stores[arm]
        sizes, optimal, one_valid, all_valid, one_reads, all_reads, one_struct = [], [], [], [], [], [], []
        for t_key in chosen:
            keys = group_keys(spec, t_key)
            obj = answers[arm][t_key]
            fc = fact_closure(store, keys, obj=obj)
            sizes.append(fc.size)
            optimal.append(float(fc.optimal))

            # ARM "one record": remove exactly what a record-level certificate would cover -- the
            # object itself. In a pod that IS the closure; under duplication it is one of k.
            store.evict(kids[t_key])
            rec, st_res = certify_removal(model, store, world, [kids[t_key]], probe_queries)
            cert_one = certify_fact(rec, fc, [kids[t_key]], store_after=store, keys=keys,
                                    structural=st_res,
                                    residual_note="says nothing about what the core knew before the store existed")
            one_valid.append(float(cert_one.valid))
            one_struct.append(float(st_res.certified_structurally))
            one_reads.append(float((read_keys(model, store, world, keys) == obj).mean()))

            # ARM "whole closure": remove every record the store's own semantics needs
            rest = [k for k in fc.records if k != kids[t_key]]
            for kid in rest:
                store.evict(kid)
            rec_all, st_all = certify_removal(model, store, world, list(fc.records), probe_queries)
            cert_all = certify_fact(rec_all, fc, list(fc.records) + [kids[t_key]], store_after=store,
                                    keys=keys, structural=st_all)
            all_valid.append(float(cert_all.valid))
            all_reads.append(float((read_keys(model, store, world, keys) == obj).mean()))

            for kid in list(fc.records) + [kids[t_key]]:
                store.restore(kid)
        per_arm[arm] = dict(sizes=sizes, optimal=optimal, one_valid=one_valid, all_valid=all_valid,
                            one_reads=one_reads, all_reads=all_reads, one_struct=one_struct)
        m[f"{arm}/fact_closure_mean"] = float(np.mean(sizes))
        m[f"{arm}/fact_closure_min"] = float(np.min(sizes))
        m[f"{arm}/fact_closure_max"] = float(np.max(sizes))
        m[f"{arm}/fact_closure_optimal_rate"] = float(np.mean(optimal))
        m[f"{arm}/one_record_fact_certified"] = float(np.mean(one_valid))
        m[f"{arm}/one_record_structurally_certified"] = float(np.mean(one_struct))
        m[f"{arm}/one_record_still_readable"] = float(np.mean(one_reads))
        m[f"{arm}/whole_closure_fact_certified"] = float(np.mean(all_valid))
        m[f"{arm}/whole_closure_still_readable"] = float(np.mean(all_reads))
        if verbose:
            print(f"  seed {seed} {arm:<11} closure {m[f'{arm}/fact_closure_mean']:.2f} "
                  f"(optimal {m[f'{arm}/fact_closure_optimal_rate']:.2f})  one-record certified "
                  f"{m[f'{arm}/one_record_fact_certified']:.2f}  still readable "
                  f"{m[f'{arm}/one_record_still_readable']:.2f}  whole-closure certified "
                  f"{m[f'{arm}/whole_closure_fact_certified']:.2f}  ({time.time() - t0:.0f}s)", flush=True)

    m["gap/certificates_per_record"] = (m["canonical/fact_closure_mean"] and
                                        m["duplicated/fact_closure_mean"] / m["canonical/fact_closure_mean"])
    m["n_groups"] = len(chosen)
    m["seconds"] = time.time() - t0
    return m


KEYS = (["control/interface_identical", "control/read_before_deletion", "control/n_keys",
         "gap/certificates_per_record", "n_groups"] +
        [f"{a}/{k}" for a in ARMS
         for k in ("per_key_closure_max", "per_key_closure_mean", "read_before_deletion",
                   "fact_closure_mean", "fact_closure_min", "fact_closure_max",
                   "fact_closure_optimal_rate", "one_record_fact_certified",
                   "one_record_structurally_certified", "one_record_still_readable",
                   "whole_closure_fact_certified", "whole_closure_still_readable")])

CRITERIA = {
    # controls: any of these failing voids the comparison rather than weakening it
    "control/interface_identical": (">=", 1.0),
    "control/read_before_deletion": (">=", 0.90),
    "canonical/per_key_closure_max": ("<=", 1.0),
    "duplicated/per_key_closure_max": ("<=", 1.0),
    # the claim
    "canonical/fact_closure_max": ("<=", 1.0),
    "duplicated/fact_closure_min": (">=", 3.0),
    "canonical/fact_closure_optimal_rate": (">=", 1.0),
    "duplicated/fact_closure_optimal_rate": (">=", 1.0),
    "canonical/one_record_fact_certified": (">=", 1.0),
    "duplicated/one_record_fact_certified": ("<=", 0.0),
    # the model confirms the verdict rather than the verdict standing alone
    "canonical/one_record_still_readable": ("<=", 0.10),
    "duplicated/one_record_still_readable": (">=", 0.60),
    # and removing the whole closure certifies in BOTH arms: the instrument is not arm-shaped
    "canonical/whole_closure_fact_certified": (">=", 1.0),
    "duplicated/whole_closure_fact_certified": (">=", 1.0),
    "duplicated/whole_closure_still_readable": ("<=", 0.10),
}


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-groups", type=int, default=N_GROUPS)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    per_seed = [run_seed(s, args.n_groups, args.steps) for s in args.seeds]
    numeric = [{k: float(v) for k, v in s.items() if isinstance(v, (bool, int, float))} for s in per_seed]
    keys = [k for k in KEYS if all(k in s for s in numeric)]
    agg = ledger.aggregate(numeric, keys)
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})

    rows = [[arm,
             f"{agg[f'{arm}/per_key_closure_mean']['mean']:.2f}",
             f"{agg[f'{arm}/fact_closure_mean']['mean']:.2f}",
             f"{agg[f'{arm}/fact_closure_optimal_rate']['min']:.2f}",
             f"{agg[f'{arm}/one_record_fact_certified']['min']:.4f}",
             f"{agg[f'{arm}/one_record_still_readable']['min']:.4f}",
             f"{agg[f'{arm}/whole_closure_fact_certified']['min']:.4f}"]
            for arm in ARMS]
    tbl = ledger.table(["store", "closure per KEY", "closure per FACT", "proved optimal",
                        "one record: fact certified", "one record: still readable",
                        "whole closure: fact certified"], rows)

    record = {"experiment": "E-000032",
              "title": "the deletion closure of a store, and the certificate that composes with it",
              "trains_nothing": True, "seeds": args.seeds, "n_groups": args.n_groups,
              "arms": list(ARMS), "per_seed": per_seed, "aggregate": agg, "criteria": check}
    md = [f"# E-000032 — {record['title']}", "",
          f"Seeds {args.seeds}, {args.n_groups} alias groups per seed, the recorded E-000015 one-slot",
          "checkpoints, no training. Both arms are built from the SAME world with the same ground truth,",
          "so they present an identical interface: every key resolves to the same object in both.", "",
          "## The gap a record-level certificate cannot see", "", tbl, "",
          "`closure per KEY` is how many records must go before THAT KEY stops answering. It is one in",
          "both arms, which is the point: at the record level the two stores are indistinguishable.",
          "`closure per FACT` is how many must go before NO key in the group yields the object, and it",
          "is where they separate. `proved optimal` is the fraction where the greedy search MET a",
          "certified lower bound (every live derivation is a must-hit set, so a pairwise-disjoint",
          "subfamily bounds the optimum from below) rather than merely being assumed exact.", "",
          "`one record: fact certified` removes exactly the object -- what a record-level certificate",
          "covers today -- and asks whether that licenses a fact-level statement. `still readable` is",
          "the model's own answer afterwards, so the verdict is confirmed by behaviour and not only by",
          "bookkeeping.", "",
          "## Pre-registered criteria", "", ledger.criteria_table(check), "",
          "## What this is and is not", "",
          "The gap between the two arms is Codd's deletion anomaly and normalization is its 1971 remedy;",
          "this experiment does not claim otherwise. What it adds is that the anomaly decides whether a",
          "DELETION CERTIFICATE for a neural memory means anything, that the store-side half of the",
          "guarantee is computable without the model, and that in a neural memory the normalization is",
          "not free -- E-000025 prices it at 0.0954 for sharing and 0.0688 for link training on a frozen",
          "GPT-2, worst of three seeds across all twelve phrasings.", ""]
    path = ledger.save("e000032_deletion_closure", record, "\n".join(md))
    print(f"\nwritten: {path}")
    print(tbl)
    print(ledger.criteria_table(check))
    return record


if __name__ == "__main__":
    main()
