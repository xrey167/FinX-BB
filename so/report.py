"""Assemble the session results document from the recorded experiment results.

    python -m so.report            # writes docs/so-results-2026-09-02.md

Everything numeric comes from so/results/*.json; the prose parts are the fixed
boundary statements and the mapping from experiments to the ledger's six
breakthrough properties (section 3).  An experiment whose pre-registered
criteria failed is shown as NOT SUPPORTED — the document cannot overstate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ledger import RESULTS_DIR, EVIDENCE_LEVELS, DELETION_LEVELS, table

DOC = Path(__file__).resolve().parent.parent / "docs" / "so-results-2026-09-02.md"

ORDER = ["e000001a_reference", "e000001b_mini_transformer", "e000002_memorization_control",
         "e000003_retention_generalization", "e000004_reconstruction_attacks", "e000005_causal_interventions",
         "e000006_ablations", "e000007_biomarker", "e000008_gpt2_adapter"]

# ledger §3 properties -> the experiments that bear on them
PROPERTIES = {
    "Selectivity (target disappears)": ["e000001b_mini_transformer", "e000003_retention_generalization", "e000008_gpt2_adapter"],
    "Retention (non-target intact)": ["e000001b_mini_transformer", "e000003_retention_generalization", "e000008_gpt2_adapter"],
    "Generalisation (paraphrases, alternative queries)": ["e000003_retention_generalization", "e000004_reconstruction_attacks", "e000008_gpt2_adapter"],
    "Causal isolation (effect follows from the intended structure)": ["e000005_causal_interventions", "e000006_ablations", "e000007_biomarker"],
    "Reconstruction resistance": ["e000004_reconstruction_attacks", "e000007_biomarker", "e000008_gpt2_adapter"],
    "Scalability (path beyond toy models)": ["e000002_memorization_control", "e000008_gpt2_adapter"],
}

BOUNDARY = """## Boundary of this evidence

Everything below was produced on 4 CPU cores in one session, with no GPU. It is therefore bounded as follows:

- **No LLM-scale evidence.** The largest neural core used is frozen GPT-2 small (124M parameters, E-000008). Nothing here shows editable knowledge inside a large pretrained model, and nothing here shows unlearning of facts that a pretrained model already encodes in its weights.
- **Synthetic worlds.** Facts are `(subject, relation) → object` triples over 256 entities and 4 relations; queries are symbolic (E-000001 … E-000007) or short natural-language templates (E-000008). Real-world knowledge, multi-token entities and free-text questions are not covered.
- **By construction versus learned.** REVOKE removes routing by a hard mask; that is deletion level F1 by construction. The learned results are: answering UNKNOWN instead of using another cell, refusing a payload whose marker is invalid (SHRED), composing hops, and the probe / forced-choice / rank checks after SHRED. Every result table states which is which.
- **Provenance is trained**, not emergent: the routing loss supervises which cell each hop reads. E-000006 (`no_routing_loss`) measures what remains without it.
- **The outstanding C55–C57 real-model / GPU chain of the ledger is still outstanding.** E-000008 is its CPU-feasible analogue on a small model, not its execution.
- **Noise figures are not comparable** with the architecture document's "noise = 0.24 → 68.4%", whose noise definition is not recorded; the sweep here perturbs bank keys and values relative to their RMS.
- **Seeds.** E-000003 … E-000007 evaluate the same five E-000001-B models on fresh worlds; they are not independent replications of training. E-000002, E-000006 and E-000008 train their own models (3 seeds).
"""


def load(name: str) -> Optional[Dict[str, Any]]:
    p = RESULTS_DIR / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def md_of(name: str) -> str:
    p = RESULTS_DIR / f"{name}.md"
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8")
    # shift headings one level down so the experiment sections nest under this document
    return "\n".join(("#" + l if l.startswith("#") else l) for l in text.splitlines())


def status_of(rec: Optional[Dict[str, Any]]) -> str:
    if rec is None:
        return "not run"
    if "claim_supported" in rec:
        return "criteria met" if rec["claim_supported"] else "**criteria NOT met**"
    if "all_pass" in rec:
        return "all tests passed" if rec["all_pass"] else "**not all tests passed**"
    return "recorded"


def main() -> None:
    recs = {n: load(n) for n in ORDER}
    lines: List[str] = [
        "# SO — Session results 2026-09-02",
        "",
        "**What this is:** the record of what the code in `so/` actually demonstrated in this session, assembled "
        "automatically from `so/results/*.json` by `python -m so.report`. Companion documents: "
        "[architecture](so-modular-neural-os.md), [experiment ledger](so-experiment-ledger.md), "
        "[roadmap](so-roadmap-2026-09-02.md).",
        "",
        "**Evidence scale** (ledger section 4): " + ", ".join(f"{k} = {v}" for k, v in EVIDENCE_LEVELS.items()) + ".",
        "",
        "**Deletion levels** (ledger section 6): " + ", ".join(f"{k} = {v}" for k, v in DELETION_LEVELS.items()) + ".",
        "",
        "## Summary",
        "",
        table(["experiment", "title", "evidence level claimed", "deletion level", "status"],
              [(r["experiment"], r["title"], r.get("evidence_level", "-"), r.get("deletion_level") or "-", status_of(r))
               if r else (n, "-", "-", "-", "not run") for n, r in recs.items()]),
        "",
        "## The six breakthrough properties (ledger section 3)",
        "",
        table(["property", "experiments", "status", "highest level"],
              [(prop, ", ".join(recs[e]["experiment"] for e in exps if recs.get(e)),
                "; ".join(f"{recs[e]['experiment']}: {status_of(recs[e])}" for e in exps if recs.get(e)) or "not run",
                max((recs[e].get("evidence_level", "E0") for e in exps if recs.get(e)), default="-"))
               for prop, exps in PROPERTIES.items()]),
        "",
        "Scalability is the property this session can least address: E-000008 shows the mechanism attaches to a "
        "frozen pretrained transformer on CPU; the path to LLM scale is a roadmap item, not a result.",
        "",
        BOUNDARY,
        "",
        "## Reproduction",
        "",
        "```bash",
        "pip install -r so/requirements.txt && pip install transformers safetensors",
        "python -m pytest so/tests -q",
        "python -m so.experiments.run_all          # full chain, then: python -m so.report",
        "```",
        "",
        "Trained models are cached under `so/results/checkpoints/` (not committed); every JSON record carries the "
        "environment, configuration, per-seed numbers, pre-registered criteria and the claim / not-claimed text.",
        "",
        "## Per-experiment records",
        "",
    ]
    for n in ORDER:
        body = md_of(n)
        if body:
            lines += [body, ""]
        else:
            lines += [f"## {n}", "", "_not run in this session_", ""]
    DOC.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
