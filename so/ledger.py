"""Result recording for SO experiments.

Every experiment writes one JSON file (machine-readable, complete) and one
Markdown file (human-readable summary) under ``so/results/``.  Each record
carries the evidence level it claims (ledger section 4) and the deletion
level it demonstrates where applicable (ledger section 6), so that a result
can never silently be presented as stronger than it is.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"

EVIDENCE_LEVELS = {
    "E0": "Idea only",
    "E1": "Analytical / conceptual support",
    "E2": "Toy implementation",
    "E3": "Repeated synthetic evidence",
    "E4": "Controlled neural-network evidence",
    "E5": "Transformer evidence",
    "E6": "Real pretrained LLM evidence",
    "E7": "Scalable / externally reproduced evidence",
}

DELETION_LEVELS = {
    "F0": "Access suppression",
    "F1": "Routing removal",
    "F2": "Component removal",
    "F3": "Functional forgetting",
    "F4": "Representational removal",
    "F5": "Reconstruction-resistant deletion",
}


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return _to_jsonable(x.tolist())
    return x


def environment() -> Dict[str, str]:
    info = {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__}
    try:
        import torch  # noqa: WPS433

        info["torch"] = torch.__version__
        info["device"] = "cpu"
        info["threads"] = str(torch.get_num_threads())
    except Exception:  # pragma: no cover
        info["torch"] = "not installed"
    return info


def save(name: str, record: Dict[str, Any], markdown: str) -> Path:
    """Write ``<name>.json`` and ``<name>.md``; ``SO_RESULT_SUFFIX`` (e.g. "-quick") keeps reduced runs apart."""
    import os

    name = name + os.environ.get("SO_RESULT_SUFFIX", "")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record.setdefault("recorded_at", _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    record.setdefault("environment", environment())
    if "evidence_level" in record:
        record["evidence_level_meaning"] = EVIDENCE_LEVELS[record["evidence_level"]]
    if "claim_supported" in record and not record["claim_supported"]:
        record["claim"] = "NOT SUPPORTED BY THE MEASUREMENTS (see criteria): " + record.get("claim", "")
    if record.get("deletion_level"):
        record["deletion_level_meaning"] = DELETION_LEVELS[record["deletion_level"]]
    json_path = RESULTS_DIR / f"{name}.json"
    json_path.write_text(json.dumps(_to_jsonable(record), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (RESULTS_DIR / f"{name}.md").write_text(markdown.rstrip("\n") + "\n", encoding="utf-8")
    return json_path


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(str(h) for h in headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(v) for v in r) + " |")
    return "\n".join(lines)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"
    return str(v)


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def clopper_pearson(successes: int, n: int, alpha: float = 0.05) -> Dict[str, float]:
    """Exact (Clopper–Pearson) two-sided binomial confidence interval for a rate.

    For 0 failures in n trials the upper bound on the failure rate is 1 - lower.
    Uses the beta-quantile identity; computed with a bisection on the regularised
    incomplete beta so that no SciPy dependency is needed.
    """
    successes, n = int(successes), int(n)
    if n <= 0:
        return {"rate": float("nan"), "lower": float("nan"), "upper": float("nan"), "n": 0}

    def beta_cdf(x: float, a: float, b: float) -> float:
        # regularised incomplete beta via continued fraction (Numerical Recipes betacf)
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        import math
        lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)

        def betacf(x: float, a: float, b: float) -> float:
            maxit, eps, fpmin = 300, 3e-14, 1e-300
            qab, qap, qam = a + b, a + 1, a - 1
            c, d = 1.0, 1 - qab * x / qap
            d = 1 / (d if abs(d) > fpmin else fpmin)
            h = d
            for m in range(1, maxit + 1):
                m2 = 2 * m
                aa = m * (b - m) * x / ((qam + m2) * (a + m2))
                d = 1 + aa * d; d = 1 / (d if abs(d) > fpmin else fpmin)
                c = 1 + aa / c; c = c if abs(c) > fpmin else fpmin
                h *= d * c
                aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
                d = 1 + aa * d; d = 1 / (d if abs(d) > fpmin else fpmin)
                c = 1 + aa / c; c = c if abs(c) > fpmin else fpmin
                de = d * c
                h *= de
                if abs(de - 1) < eps:
                    break
            return h

        if x < (a + 1) / (a + b + 2):
            return math.exp(lbeta) * betacf(x, a, b) / a
        return 1 - math.exp(lbeta) * betacf(1 - x, b, a) / b

    def beta_ppf(q: float, a: float, b: float) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if beta_cdf(mid, a, b) < q:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if successes == 0 else beta_ppf(alpha / 2, successes, n - successes + 1)
    upper = 1.0 if successes == n else beta_ppf(1 - alpha / 2, successes + 1, n - successes)
    return {"rate": successes / n, "lower": lower, "upper": upper, "n": n}


def ci_rows(per_seed: List[Dict[str, Any]], keys: Sequence[str], sizes: Dict[str, int],
            lower_is_better: Sequence[str] = ()) -> List[List[Any]]:
    """Rows: metric, mean, worst seed, pooled n, 95% CI lower, 95% CI upper.

    Rates are pooled over seeds with the per-seed sample size ``sizes[metric]`` (exact
    binomial interval, Clopper–Pearson).  "Worst" is the max for lower-is-better metrics
    (leaks, unknown-rates that should be 0) and the min otherwise.
    """
    rows: List[List[Any]] = []
    lib = set(lower_is_better)
    for k in keys:
        vals = [float(s[k]) for s in per_seed if k in s and s[k] == s[k]]
        if not vals:
            continue
        n = sizes.get(k)
        worst = max(vals) if k in lib else min(vals)
        if n:
            successes = int(round(sum(v * n for v in vals)))
            ci = clopper_pearson(successes, n * len(vals))
            rows.append([k, f"{np.mean(vals):.4f}", f"{worst:.4f}", n * len(vals), f"{ci['lower']:.4f}", f"{ci['upper']:.4f}"])
        else:
            rows.append([k, f"{np.mean(vals):.4f}", f"{worst:.4f}", "-", "-", "-"])
    return rows


CI_HEADERS = ["measure", "mean over seeds", "worst seed", "pooled n", "95% CI lower", "95% CI upper"]


def check_criteria(agg: Dict[str, Dict[str, float]], criteria: Dict[str, Tuple[str, float]]) -> Dict[str, Any]:
    """Evaluate pass criteria on the aggregate: ``{"metric": (">=", 0.99)}`` tests the worst seed
    (min for ">=", max for "<=").  Returns per-criterion detail and an overall ``claim_supported``."""
    detail: Dict[str, Dict[str, Any]] = {}
    ok_all = True
    for metric, (op, threshold) in criteria.items():
        if metric not in agg:
            detail[metric] = {"op": op, "threshold": threshold, "observed": None, "pass": False}
            ok_all = False
            continue
        observed = agg[metric]["min"] if op == ">=" else agg[metric]["max"]
        passed = observed >= threshold if op == ">=" else observed <= threshold
        detail[metric] = {"op": op, "threshold": threshold, "observed": observed, "pass": bool(passed)}
        ok_all = ok_all and bool(passed)
    return {"criteria": detail, "claim_supported": ok_all}


def criteria_table(check: Dict[str, Any]) -> str:
    rows = [(m, f"{d['op']} {d['threshold']}", "-" if d["observed"] is None else f"{d['observed']:.4f}",
             "PASS" if d["pass"] else "FAIL") for m, d in check["criteria"].items()]
    return table(["criterion (worst seed)", "required", "observed", "result"], rows)


def aggregate(per_seed: List[Dict[str, Any]], keys: Sequence[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = np.asarray([float(s[k]) for s in per_seed], dtype=float)
        out[k] = {"mean": float(vals.mean()), "min": float(vals.min()), "max": float(vals.max()), "n": int(len(vals))}
    return out
