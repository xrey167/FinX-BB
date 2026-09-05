"""E-000085 -- strict-capability real-symlink reader against downstream stale-state replay.

E-000084 validates stale serialized Bank rows with the consistency-trained LINK+deref reader.
This experiment pushes the same reader downstream of the Bank boundary.  It wraps the existing
E-000073 post-read hidden-state replay and E-000074 cached-router/resolved-payload replay attacks,
without changing their CAVI semantics.

Each arm trains E-000081's consistency reader, then independently re-measures all four held-out
symlink templates on E-000081's standard fresh evaluation world before attack results are interpreted.
The attack is valid only if every held-out template is >=0.95 in that exact job.

This is an integration/falsification experiment, not a novelty claim.  Versions, dependency tags,
cache validation, capabilities, locks, replay prevention and recomputation are prior-art controls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from so.data import bank_from_store
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments import e000073_cavi_derived_state_replay as E73
from so.experiments import e000074_cavi_cached_router_payload_replay as E74
from so.experiments import e000081_symlink_consistency_reader as E81


def _strict_capability(gk, centre: np.ndarray, seed: int, groups: int) -> Dict[str, Any]:
    rng = np.random.default_rng(70000 + seed)
    world, spec = E15.sample_alias_world(
        rng, 180, groups, 2, gk.n_entities, 4, E20.N_TRAIN_TEMPLATES
    )
    store, _ = E15.load_arm(world, spec, centre, 71000 + seed, symlink=True)
    bank = bank_from_store(store)
    per = {
        str(t): E81._evaluate_template(gk, bank, spec.alias_keys, world, t)
        for t in range(8, 12)
    }
    vals = [float(per[str(t)]["candidate_correct"]) for t in range(8, 12)]
    return {
        "per_heldout_template": per,
        "heldout_mean": float(np.mean(vals)),
        "heldout_min": float(np.min(vals)),
        "template9": float(per["9"]["candidate_correct"]),
        "strict_pass": bool(min(vals) >= 0.95),
    }


def run(mode: str, seed: int, steps: int, groups: int, consistency: float, alt_supervision: float) -> Dict[str, Any]:
    original = E20.train_adapter_links
    cap_box: Dict[str, Any] = {}

    def train_consistent(gk, seed_arg, steps_arg, n_groups=100, verbose=True, **_kwargs):
        out = E81.train_symlink_consistent(
            gk,
            seed_arg,
            steps_arg,
            consistency=consistency,
            alt_supervision=alt_supervision,
            n_groups=max(groups, int(n_groups)),
            verbose=verbose,
        )
        cap_box.clear()
        cap_box.update(_strict_capability(gk, np.asarray(out["centre"]), int(seed_arg), groups))
        return out

    E20.train_adapter_links = train_consistent
    try:
        if mode == "hidden":
            row = E73.run(seed, steps, groups, template=9)
        elif mode == "router_payload":
            row = E74.run(seed, steps, groups)
        else:
            raise ValueError(mode)
    finally:
        E20.train_adapter_links = original

    row = dict(row)
    structural = bool(row.get("screening_pass", row.get("pass", False)))
    strict = bool(cap_box.get("strict_pass", False))
    return {
        "mode": mode,
        "seed": seed,
        "steps": steps,
        "groups": groups,
        "consistency": consistency,
        "alt_supervision": alt_supervision,
        "capability": cap_box,
        "attack": row,
        "structural_pass": structural,
        "strict_capability_pass": strict,
        "strict_interpretable_pass": bool(strict and structural),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("hidden", "router_payload"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--groups", type=int, default=100)
    ap.add_argument("--consistency", type=float, default=0.15)
    ap.add_argument("--alt-supervision", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--results-dir", default="so/results")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rec = {
        "experiment": "E-000085",
        "candidate_only": True,
        "row": run(a.mode, a.seed, a.steps, a.groups, a.consistency, a.alt_supervision),
        "gate": "every held-out real-symlink template >=0.95 in the exact job AND unchanged downstream replay battery",
        "not_claimed": "consistency training, symlinks, dependency/version tags, cache invalidation, capabilities, locks, replay prevention or CAVI composition individually",
    }
    out = Path(a.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"e000085_{a.mode}_s{a.seed}_c{a.consistency:g}_a{a.alt_supervision:g}.json"
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    if not bool(rec["row"]["strict_interpretable_pass"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
