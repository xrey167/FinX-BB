"""E-000086 — alias fan-out utility/scaling screen.

This is deliberately a store/control-plane engineering measurement, NOT a novelty claim and NOT
a CAVI/LLM capability result. It asks one practical question from the Symlink hypothesis: when k
linguistic access keys share one canonical knowledge object, does one lifecycle operation replace
O(k) duplicated payload mutations while preserving the same visible store semantics?

The comparison uses E-000015's SAME-world symlink and duplicate arms. No neural reader result is
interpreted here, so this experiment cannot satisfy or bypass the >=0.95 three-training-seed gate.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from so.data import bank_from_store
from so.experiments import e000015_symlink_cells as E15
from so.train import make_centre


def _visible(bank, keys):
    return [bank.index_view.get(k, None) for k in keys]


def run_cell(seed: int, fanout: int, groups: int, n_base: int, rounds: int) -> dict[str, Any]:
    # Large entity space guarantees enough unused (subject, relation) keys for high fan-out.
    n_entities = max(2048, n_base + fanout * groups + 256)
    rng = np.random.default_rng(seed * 1009 + fanout)
    world, spec = E15.sample_alias_world(
        rng, n_base=n_base, n_groups=groups, n_alias_per_group=fanout,
        n_entities=n_entities, n_relations=6, n_synonyms=2,
    )
    centre = make_centre(seed, 32)
    symlink_store, symlink_kids = E15.load_arm(world, spec, centre, seed, symlink=True)
    duplicate_store, duplicate_kids = E15.load_arm(world, spec, centre, seed, symlink=False)

    target, aliases = spec.groups[0]
    target_old = world.index[target]
    target_new = int((target_old + 17) % n_entities)

    # Operation-count contract. The duplicate arm must mutate every copied alias to implement
    # the same externally visible lifecycle semantics as the canonical target mutation.
    symlink_update_ops = 1
    duplicate_update_ops = 1 + len(aliases)

    symlink_store.update(symlink_kids[target], target_new)
    symlink_after = bank_from_store(symlink_store)
    symlink_update_correct = (
        symlink_after.index_view.get(target) == target_new
        and all(symlink_after.index_view.get(a) == target_new for a in aliases)
    )

    duplicate_store.update(duplicate_kids[target], target_new)
    duplicate_one_after = bank_from_store(duplicate_store)
    duplicate_one_propagates = all(duplicate_one_after.index_view.get(a) == target_new for a in aliases)
    for a in aliases:
        duplicate_store.update(duplicate_kids[a], target_new)
    duplicate_after = bank_from_store(duplicate_store)
    duplicate_full_correct = (
        duplicate_after.index_view.get(target) == target_new
        and all(duplicate_after.index_view.get(a) == target_new for a in aliases)
    )

    # Rollback same semantic update.
    symlink_store.rollback(symlink_kids[target], 1)
    for k in [target, *aliases]:
        duplicate_store.rollback(duplicate_kids[k], 1)
    symlink_rollback = bank_from_store(symlink_store)
    duplicate_rollback = bank_from_store(duplicate_store)
    rollback_correct = (
        symlink_rollback.index_view.get(target) == target_old
        and all(symlink_rollback.index_view.get(a) == target_old for a in aliases)
        and duplicate_rollback.index_view.get(target) == target_old
        and all(duplicate_rollback.index_view.get(a) == target_old for a in aliases)
    )

    # SHRED semantics: one canonical shred kills all symlink access paths; duplicate semantics
    # require target plus every copied alias to be shredded.
    symlink_store.shred(symlink_kids[target])
    symlink_shred = bank_from_store(symlink_store)
    symlink_shred_correct = (
        target not in symlink_shred.index_view and all(a not in symlink_shred.index_view for a in aliases)
    )
    duplicate_store.shred(duplicate_kids[target])
    duplicate_one_shred = bank_from_store(duplicate_store)
    duplicate_one_shred_propagates = all(a not in duplicate_one_shred.index_view for a in aliases)
    for a in aliases:
        duplicate_store.shred(duplicate_kids[a])
    duplicate_shred = bank_from_store(duplicate_store)
    duplicate_full_shred_correct = (
        target not in duplicate_shred.index_view and all(a not in duplicate_shred.index_view for a in aliases)
    )

    # Fresh stores for paired timing; setup is deliberately excluded and reported separately.
    update_sym_ns, update_dup_ns, shred_sym_ns, shred_dup_ns = [], [], [], []
    for r in range(rounds):
        rrng = np.random.default_rng(seed * 1000003 + fanout * 97 + r)
        w, s = E15.sample_alias_world(
            rrng, n_base=n_base, n_groups=groups, n_alias_per_group=fanout,
            n_entities=n_entities, n_relations=6, n_synonyms=2,
        )
        c = make_centre(seed + r, 32)
        ss, sk = E15.load_arm(w, s, c, seed + r, symlink=True)
        ds, dk = E15.load_arm(w, s, c, seed + r, symlink=False)
        t, aa = s.groups[0]
        nv = int((w.index[t] + 19) % n_entities)

        t0 = time.perf_counter_ns(); ss.update(sk[t], nv); update_sym_ns.append(time.perf_counter_ns() - t0)
        t0 = time.perf_counter_ns()
        ds.update(dk[t], nv)
        for a in aa: ds.update(dk[a], nv)
        update_dup_ns.append(time.perf_counter_ns() - t0)

        # Separate fresh stores so update history cannot affect the shred timing.
        ss2, sk2 = E15.load_arm(w, s, c, seed + r + 10000, symlink=True)
        ds2, dk2 = E15.load_arm(w, s, c, seed + r + 10000, symlink=False)
        t0 = time.perf_counter_ns(); ss2.shred(sk2[t]); shred_sym_ns.append(time.perf_counter_ns() - t0)
        t0 = time.perf_counter_ns()
        ds2.shred(dk2[t])
        for a in aa: ds2.shred(dk2[a])
        shred_dup_ns.append(time.perf_counter_ns() - t0)

    med = lambda x: float(np.median(np.asarray(x, dtype=np.float64)))
    rec = {
        "seed": seed, "fanout": fanout, "groups": groups, "n_base": n_base,
        "aliases_in_measured_group": len(aliases),
        "operation_counts": {
            "symlink_update": symlink_update_ops,
            "duplicate_update_to_same_semantics": duplicate_update_ops,
            "symlink_shred": 1,
            "duplicate_shred_to_same_semantics": 1 + len(aliases),
        },
        "semantic_checks": {
            "single_symlink_update_propagates": symlink_update_correct,
            "single_duplicate_target_update_propagates": duplicate_one_propagates,
            "full_duplicate_update_matches_semantics": duplicate_full_correct,
            "rollback_restores_both_arms": rollback_correct,
            "single_symlink_shred_closes_all_aliases": symlink_shred_correct,
            "single_duplicate_target_shred_closes_all_aliases": duplicate_one_shred_propagates,
            "full_duplicate_shred_matches_semantics": duplicate_full_shred_correct,
        },
        "timing_ns_median": {
            "symlink_update": med(update_sym_ns), "duplicate_update": med(update_dup_ns),
            "symlink_shred": med(shred_sym_ns), "duplicate_shred": med(shred_dup_ns),
        },
        "timing_ratios": {
            "duplicate_over_symlink_update": med(update_dup_ns) / max(med(update_sym_ns), 1.0),
            "duplicate_over_symlink_shred": med(shred_dup_ns) / max(med(shred_sym_ns), 1.0),
        },
        "raw_timing_ns": {
            "symlink_update": update_sym_ns, "duplicate_update": update_dup_ns,
            "symlink_shred": shred_sym_ns, "duplicate_shred": shred_dup_ns,
        },
    }
    required = [
        "single_symlink_update_propagates", "full_duplicate_update_matches_semantics",
        "rollback_restores_both_arms", "single_symlink_shred_closes_all_aliases",
        "full_duplicate_shred_matches_semantics",
    ]
    # The two single-duplicate controls are expected FALSE when fanout>0.
    rec["pass"] = all(rec["semantic_checks"][k] for k in required) and (
        fanout == 0 or (
            not rec["semantic_checks"]["single_duplicate_target_update_propagates"]
            and not rec["semantic_checks"]["single_duplicate_target_shred_closes_all_aliases"]
        )
    )
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--fanouts", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--n-base", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--results-dir", default="so/results")
    a = ap.parse_args()
    rows = [run_cell(s, f, a.groups, a.n_base, a.rounds) for f in a.fanouts for s in a.seeds]
    rec = {
        "experiment": "E-000086",
        "title": "alias fan-out lifecycle utility scaling",
        "result": "utility_screen_pass" if all(r["pass"] for r in rows) else "falsified",
        "rows": rows,
        "breakthrough": False,
        "novelty_claim": False,
        "interpretation": (
            "Canonical symlink storage is a practical fan-out mechanism only: one target lifecycle operation "
            "can replace target+alias duplicated mutations. This screen earns no novelty credit and says nothing "
            "about neural reader capability, stale derived-state closure, J-space audit, or end-to-end latency."
        ),
        "promotion_boundary": (
            "Do not interpret CAVI attacks from this result. Real neural claims still require >=0.95 fresh "
            "real-symlink correctness across >=3 genuine training seeds, held-out/REVOKE/SHRED/leakage/UNKNOWN/" 
            "locality gates, adversarial replay/race closure, J-lens audit and public-backbone replication."
        ),
    }
    out = Path(a.results_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "e000086_alias_fanout_scaling.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    if not all(r["pass"] for r in rows): raise SystemExit(2)

if __name__ == "__main__":
    main()
