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
         "e000006_ablations", "e000007_biomarker", "e000008_gpt2_adapter", "e000009_verification_gate",
         "e000010_balanced_gate"]

# ledger §3 properties -> the experiments that bear on them
PROPERTIES = {
    "Selectivity (target disappears)": ["e000001b_mini_transformer", "e000003_retention_generalization", "e000008_gpt2_adapter"],
    "Retention (non-target intact)": ["e000001b_mini_transformer", "e000003_retention_generalization", "e000008_gpt2_adapter"],
    "Generalisation (paraphrases, alternative queries)": ["e000003_retention_generalization", "e000004_reconstruction_attacks", "e000008_gpt2_adapter"],
    "Causal isolation (effect follows from the intended structure)": ["e000005_causal_interventions", "e000006_ablations", "e000007_biomarker"],
    "Reconstruction resistance": ["e000004_reconstruction_attacks", "e000007_biomarker", "e000009_verification_gate", "e000010_balanced_gate", "e000008_gpt2_adapter"],
    "Scalability (path beyond toy models)": ["e000002_memorization_control", "e000008_gpt2_adapter"],
}

# post-hoc interpretation of recorded outcomes (the JSON records are never edited)
NOTES = {
    "e000002_memorization_control": "Only 'fixed_routing' is an empirical control, and it came out clean: with the "
        "layer available, 2000 steps on a fixed world did not copy any fact into the weights (masked-layer accuracy 0%, "
        "leak 0%). This is a bound for that budget, not a guarantee for longer training. The no-layer model memorised "
        "everything and cannot revoke (leak 100%): the copy problem in its purest form.",
    "e000004_reconstruction_attacks": "SHRED column: behaviourally deleted (direct / paraphrase / multi-hop UNKNOWN "
        "100%), but the gate learned without supervision closes to about 9% of the value norm, so the linear probe "
        "(8% worst seed) and forced choice (69% worst seed) recover a residual. Recorded as F3 with a trace; E-000009 "
        "is the response.",
    "e000006_ablations": "Two pre-registered expectations were wrong and are recorded as such. (1) The null cell is NOT "
        "essential: without it, broken paths are still answered UNKNOWN at 100% (the model learns to produce a "
        "low-norm read from non-matching keys). The design claim 'the null cell is what makes broken paths answer "
        "UNKNOWN' is withdrawn. (2) Without the routing loss the model collapsed to answering UNKNOWN for everything "
        "within 2000 steps (identical numbers to 'no_routing'): routing supervision is necessary for the mechanism to "
        "be *learned* at all at this budget, not only for exact provenance. That is an optimisation finding, not a "
        "by-construction one. 'no_marker_gate' and 'no_routing' confirm the information-flow necessities (SHRED 0%, "
        "nothing readable).",
    "e000009_verification_gate": "The verification loss sharpened the gate (signed markers 0.89 -> 0.998, unsigned "
        "mean 0.087 -> 0.065) but a tail of unsigned markers still scores high (max 0.84 soft; under hard gating 3-5% of "
        "shredded payloads pass and answer correctly), so the SHRED residual persists and F4 is still withheld. Cause: "
        "the gate loss is averaged over ~1000 cells of which ~5% are unsigned, so the tail receives almost no gradient. "
        "E-000010 weights the two classes equally.",
    "e000007_biomarker": "The suppression-versus-deletion separation holds in every seed (suppressed: value "
        "contribution 8.3, probe 86%, mean rank 10; shredded: 1.3, 4%, 110). The two failed criteria are the same "
        "SHRED residual as in E-000004, addressed in E-000009.",
}

BOUNDARY = """## Boundary of this evidence

Everything below was produced on 4 CPU cores in one session, with no GPU. It is therefore bounded as follows:

- **No LLM-scale evidence.** The largest neural core used is frozen GPT-2 small (124M parameters, E-000008). Nothing here shows editable knowledge inside a large pretrained model, and nothing here shows unlearning of facts that a pretrained model already encodes in its weights.
- **Synthetic worlds.** Facts are `(subject, relation) → object` triples over 256 entities and 4 relations; queries are symbolic (E-000001 … E-000007) or short natural-language templates (E-000008). Real-world knowledge, multi-token entities and free-text questions are not covered.
- **By construction versus learned.** REVOKE removes routing by a hard mask; that is deletion level F1 by construction. The learned results are: answering UNKNOWN instead of using another cell, refusing a payload whose marker is invalid (SHRED), composing hops, and the probe / forced-choice / rank checks after SHRED. Every result table states which is which.
- **Provenance is trained**, not emergent: the routing loss supervises which cell each hop reads. E-000006 (`no_routing_loss`) measures what remains without it.
- **The outstanding C55–C57 real-model / GPU chain of the ledger is still outstanding.** E-000008 is its CPU-feasible analogue on a small model, not its execution.
- **Noise figures are not comparable** with the architecture document's "noise = 0.24 → 68.4%", whose noise definition is not recorded; the sweep here perturbs bank keys and values relative to their RMS.
- **Seeds.** E-000003 … E-000007 evaluate the same five E-000001-B models on fresh worlds; they are not independent replications of training. E-000002, E-000006 and E-000008 train their own models (3 seeds); E-000009 trains five.
- **The SHRED residual.** E-000004 and E-000007 found that the marker gate learned without explicit supervision closes to about 9% rather than 0 on unsigned payloads, so a linear probe and forced choice recover a residual; their F4 criteria fail and the records say so. E-000009 is the response: a verification loss and a hard verification gate. Whether that closes the residual is a recorded result, not an assumption.
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
            if n in NOTES:
                lines += ["**Interpretation (post hoc, record unchanged):** " + NOTES[n], ""]
        else:
            lines += [f"## {n}", "", "_not run in this session_", ""]
    DOC.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
