"""Read-only reassessment of legacy E-000070 records used by E-000076.

No model execution, no historical record mutation, and no novelty verdict.
A legacy screening_pass flag is not evidence that the stronger 0.95 bar passed.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    try:
        v = float(value)
    except (OverflowError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _rate(value: Any) -> float | None:
    v = _number(value)
    return v if v is not None and 0 <= v <= 1 else None


def _nonnegative(value: Any) -> float | None:
    v = _number(value)
    return v if v is not None and v >= 0 else None


def _check(value: float | None, bar: float, *, lower: bool) -> dict[str, Any]:
    if value is None:
        return {"value": None, "bar": bar, "status": "NOT_MEASURED"}
    passed = value >= bar if lower else value <= bar
    return {"value": value, "bar": bar, "status": "PASS" if passed else "FAIL"}


def assess_records(records: Sequence[Mapping[str, Any]], *,
                   expected_seeds: tuple[int, ...] = (0, 1, 2)) -> dict[str, Any]:
    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("expected seeds must be nonempty and unique")
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for record in records:
        if record.get("experiment") != "E-000070":
            raise ValueError("expected original E-000070-format records")
        for row in record.get("rows", []):
            seed = row.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int) or seed in seen:
                raise ValueError("each record must identify one unique integer seed")
            seen.add(seed)
            alias = row.get("alias_relink", {})
            truth = alias.get("old_truth")
            predictions = [alias.get(k) for k in
                           ("stale_snapshot_pred", "commit_pred", "pod_only_pred")]
            valid_ids = all(isinstance(x, int) and not isinstance(x, bool)
                            for x in [truth, *predictions])
            case_status = ("PASS" if all(p == truth for p in predictions) else "FAIL") if valid_ids else "NOT_MEASURED"
            rows.append({
                "seed": seed,
                "historical_screening_pass": row.get("screening_pass"),
                "fresh_alias_capability": _check(_rate(row.get("fresh_alias_read_rate")), .95, lower=True),
                "selected_old_answer_case": {"status": case_status,
                    "scope": "one selected case, not a population recovery rate"},
                "matches_explicit_row_mask_reference": _check(
                    _nonnegative(alias.get("cavi_equals_explicit_neural_rejection_maxabs")), 1e-7, lower=False),
                "learned_scope_bypass": {"status": "NOT_MEASURED",
                    "reason": "legacy bypass compares base(None) with base(None), not the actual scoped path"},
                "fresh_current_answer_after_relink": {"status": "NOT_MEASURED",
                    "reason": "legacy record contains no fresh-current neural answer after relink"},
                "retained_pod_output_locality": {"status": "NOT_MEASURED",
                    "reason": "row-mask preservation does not measure output distributions"},
                "independent_workspace_audit": {"status": "NOT_MEASURED"},
            })
    rows.sort(key=lambda x: x["seed"])
    complete = seen == set(expected_seeds)
    rates = [r["fresh_alias_capability"]["value"] for r in rows]
    capability = complete and bool(rows) and all(r["fresh_alias_capability"]["status"] == "PASS" for r in rows)
    return {"assessment": "strong-symlink-evidence-v1", "expected_seeds": list(expected_seeds),
            "seed_set_complete": complete, "fresh_alias_bar": .95,
            "worst_fresh_alias_rate": min(rates) if rates and all(v is not None for v in rates) else None,
            "capability_gate_pass": capability, "rows": rows,
            "breakthrough_established": False,
            "reason": "this legacy experiment does not contain the full joint capability, scope, locality and audit evidence"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("records", nargs="+", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    report = assess_records([json.loads(p.read_text(encoding="utf-8")) for p in args.records])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["capability_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
