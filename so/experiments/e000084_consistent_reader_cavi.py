"""E-000084 -- strict-capability symlink reader under the existing CAVI battery.

This is an integration experiment, not a novelty claim. It reuses E-000070's
unchanged lifecycle/adversarial logic but swaps in E-000081's consistency-trained
real LINK+deref reader.

Validity correction: attack results are interpretable only when the *same exact
trained reader in that job* scores >=0.95 on every held-out real-symlink template
(8..11) in E-000081's independent evaluation world. Template 9 alone is not a
sufficient capability gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from so.data import bank_from_store
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments import e000070_cavi_live_symlink_boundary as E70
from so.experiments import e000081_symlink_consistency_reader as E81


def _strict_capability(gk, centre: np.ndarray, seed: int, groups: int) -> Dict[str, Any]:
    """Re-measure all held-out templates on E-000081's independent world."""
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
    candidate = [float(per[str(t)]["candidate_correct"]) for t in range(8, 12)]
    full = [float(per[str(t)]["full_vocab_top1_correct"]) for t in range(8, 12)]
    return {
        "per_heldout_template": per,
        "heldout_candidate_mean": float(np.mean(candidate)),
        "heldout_candidate_min": float(np.min(candidate)),
        "heldout_full_vocab_mean": float(np.mean(full)),
        "heldout_full_vocab_min": float(np.min(full)),
        "template9": float(per["9"]["candidate_correct"]),
        "strict_pass": bool(min(candidate) >= 0.95),
    }


def run(seed: int, steps: int, groups: int, consistency: float, alt_supervision: float) -> Dict[str, object]:
    original = E20.train_adapter_links
    trained_box: Dict[str, Any] = {}

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
        trained_box.clear()
        trained_box["gk"] = gk
        trained_box["centre"] = np.asarray(out["centre"])
        return out

    E20.train_adapter_links = train_consistent
    try:
        row = E70.run(seed, steps, groups, template=9)
    finally:
        E20.train_adapter_links = original

    row = dict(row)
    row["consistency"] = consistency
    row["alt_supervision"] = alt_supervision
    if "gk" not in trained_box:
        raise RuntimeError("consistent-reader training did not expose the exact trained reader")
    capability = _strict_capability(
        trained_box["gk"], trained_box["centre"], seed, groups
    )
    structural = bool(row["screening_pass"])
    row["capability"] = capability
    row["strict_capability_ge_095_every_heldout"] = bool(capability["strict_pass"])
    # Retain the historical field for diagnostics, but it no longer authorizes interpretation.
    row["fresh_template9_from_attack_world"] = float(row["fresh_alias_read_rate"])
    row["cavi_structural_pass"] = structural
    row["strict_interpretable_pass"] = bool(capability["strict_pass"] and structural)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--groups", type=int, default=100)
    ap.add_argument("--consistency", type=float, default=0.15)
    ap.add_argument("--alt-supervision", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--results-dir", default="so/results")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rows: List[Dict[str, object]] = [
        run(s, a.steps, a.groups, a.consistency, a.alt_supervision) for s in a.seeds
    ]
    rec = {
        "experiment": "E-000084",
        "candidate_only": True,
        "rows": rows,
        "all_strict_interpretable_pass": all(bool(r["strict_interpretable_pass"]) for r in rows),
        "gate": "every held-out real-symlink template 8..11 >=0.95 in the exact trained-reader job AND unchanged E-000070 structural battery",
        "protocol_correction": "template9-only capability is diagnostic and cannot authorize attack interpretation",
        "not_claimed": "consistency training, symlinks, versions, masks, capabilities, cache invalidation or CAVI composition individually",
    }
    out_dir = Path(a.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"e000084_consistent_reader_cavi_c{a.consistency:g}_a{a.alt_supervision:g}.json"
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    if not rec["all_strict_interpretable_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
