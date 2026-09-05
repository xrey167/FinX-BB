"""E-000084 -- strict-capability symlink reader under the existing CAVI battery.

This is an integration experiment, not a novelty claim.  It reuses E-000070's
unchanged lifecycle/adversarial logic but swaps in E-000081's consistency-trained
real LINK+deref reader.  The original E-000070 screening floor is deliberately
made stricter here: a seed is valid only when its fresh alias read on template 9
is >=0.95 before any CAVI result is interpreted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments import e000070_cavi_live_symlink_boundary as E70
from so.experiments import e000081_symlink_consistency_reader as E81


def run(seed: int, steps: int, groups: int, consistency: float, alt_supervision: float) -> Dict[str, object]:
    original = E20.train_adapter_links

    def train_consistent(gk, seed_arg, steps_arg, n_groups=100, verbose=True, **_kwargs):
        return E81.train_symlink_consistent(
            gk,
            seed_arg,
            steps_arg,
            consistency=consistency,
            alt_supervision=alt_supervision,
            n_groups=max(groups, int(n_groups)),
            verbose=verbose,
        )

    E20.train_adapter_links = train_consistent
    try:
        row = E70.run(seed, steps, groups, template=9)
    finally:
        E20.train_adapter_links = original

    row = dict(row)
    row["consistency"] = consistency
    row["alt_supervision"] = alt_supervision
    strict_capability = float(row["fresh_alias_read_rate"]) >= 0.95
    row["strict_capability_ge_095"] = strict_capability
    row["cavi_structural_pass"] = bool(row["screening_pass"])
    row["strict_interpretable_pass"] = strict_capability and bool(row["screening_pass"])
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
        "gate": "fresh template9 real-symlink correctness >=0.95 AND unchanged E-000070 structural battery",
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
