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
from typing import Any, Dict, Iterable, List, Sequence

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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record.setdefault("recorded_at", _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    record.setdefault("environment", environment())
    if "evidence_level" in record:
        record["evidence_level_meaning"] = EVIDENCE_LEVELS[record["evidence_level"]]
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


def aggregate(per_seed: List[Dict[str, Any]], keys: Sequence[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = np.asarray([float(s[k]) for s in per_seed], dtype=float)
        out[k] = {"mean": float(vals.mean()), "min": float(vals.min()), "max": float(vals.max()), "n": int(len(vals))}
    return out
