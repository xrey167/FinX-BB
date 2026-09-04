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
         "e000010_balanced_gate", "e000011_gpt2_v2", "e000012_status_gated_revoke", "e000013_prior_conflict", "e000014_bank_10k", "e000015_symlink_cells", "e000016_alias_chains", "e000017a_paraphrase_diagnosis", "e000017b_templates8", "e000018_both", "e000018_gate", "e000018_generic", "e000019_fresh_seed_chance", "e000020_symlink_gpt2", "e000021_gate_error_rates", "e000022_two_channel_null"]

# ledger §3 properties -> the experiments that bear on them
PROPERTIES = {
    "Selectivity (target disappears)": ["e000001b_mini_transformer", "e000003_retention_generalization", "e000008_gpt2_adapter", "e000011_gpt2_v2", "e000012_status_gated_revoke", "e000013_prior_conflict"],
    "Retention (non-target intact)": ["e000001b_mini_transformer", "e000003_retention_generalization", "e000008_gpt2_adapter", "e000011_gpt2_v2", "e000012_status_gated_revoke", "e000013_prior_conflict"],
    "Generalisation (paraphrases, alternative queries)": ["e000003_retention_generalization", "e000004_reconstruction_attacks", "e000008_gpt2_adapter", "e000011_gpt2_v2", "e000012_status_gated_revoke", "e000013_prior_conflict"],
    "Causal isolation (effect follows from the intended structure)": ["e000005_causal_interventions", "e000006_ablations", "e000007_biomarker", "e000011_gpt2_v2", "e000012_status_gated_revoke"],
    "Reconstruction resistance": ["e000004_reconstruction_attacks", "e000007_biomarker", "e000009_verification_gate", "e000010_balanced_gate", "e000008_gpt2_adapter", "e000011_gpt2_v2", "e000012_status_gated_revoke", "e000013_prior_conflict", "e000014_bank_10k"],
    "Scalability (path beyond toy models)": ["e000002_memorization_control", "e000008_gpt2_adapter", "e000011_gpt2_v2", "e000012_status_gated_revoke", "e000013_prior_conflict", "e000014_bank_10k"],
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
        "essential: without it, broken paths are still answered UNKNOWN at 100% (how the model does this was not "
        "measured; a plausible mechanism is a diffuse, low-norm read over non-matching keys). The design claim 'the null "
        "cell is what makes broken paths answer UNKNOWN' is withdrawn. (2) Without the routing loss the model collapsed to answering UNKNOWN for everything "
        "within 2000 steps (identical numbers to 'no_routing'): routing supervision is necessary for the mechanism to "
        "be *learned* at all at this budget, not only for exact provenance. That is an optimisation finding, not a "
        "by-construction one. 'no_marker_gate' and 'no_routing' confirm the information-flow necessities (SHRED 0%, "
        "nothing readable).",
    "e000009_verification_gate": "The verification loss sharpened the gate (signed markers 0.89 -> 0.998, unsigned "
        "mean 0.087 -> 0.065) but a tail of unsigned markers still scores high (max 0.84 soft; under hard gating 3-5% of "
        "shredded payloads pass and answer correctly), so the SHRED residual persists and F4 is still withheld. Cause: "
        "the gate loss is averaged over ~1000 cells of which ~5% are unsigned, so the tail receives almost no gradient. "
        "E-000010 weights the two classes equally.",
    "e000010_balanced_gate": "Closes the residual: with signed and unsigned markers weighted equally in the verification "
        "loss (weight 5), every reconstruction attack after SHRED is at or below its pre-registered chance-level "
        "threshold in all five seeds (probe 0.2-0.4% against a chance of 0.39%, forced choice 53-54% against 50%); "
        "this is a tolerance result, not a test of the null that the residual IS chance while the payload remains "
        "physically present and routed to (routing mass 0.998), and no other family degrades. This is the F4-level result "
        "of the session, within the synthetic system. Residual caveat: the soft gate still assigns a high score to a rare "
        "unsigned marker in one seed (max 0.995 among all unsigned cells of that bank); none of the 500 shredded targets leaked.",
    "e000014_bank_10k": "Ten times the bank and ten times the read-out vocabulary at once: every family stays at the "
        "E-000001-B level (direct 100% over 30,000 pooled queries, 3-hop 99.5%, provenance 99.99%), the verified gate keeps "
        "SHRED at F4 on 500 targets with thresholds derived for 2,560 entities, and the same model reads 100% at 1,000, 3,000 "
        "and 10,000 cells with routing mass 0.995 — addressing does not degrade in this range. Residual: 1 in 500 shredded "
        "targets answered (an unsigned marker passing the gate), inside the binomial threshold; the gate's false-accept tail is "
        "the quantity to watch at larger scale.",
    "e000012_status_gated_revoke": "A design result rather than a threshold result. Expressing REVOKE as a status flag "
        "that multiplies the verification gate, instead of removing the cell from routing, raises ' unknown' after "
        "REVOKE from 72.7% to 99.0% in the frozen GPT-2 and improves reading, composition, update and locality at the "
        "same time. The explanation is in E-000011's own numbers: a masked cell releases its routing mass to "
        "neighbouring keys and the model then names another entity. The pre-registered bar is still missed because "
        "deletion does not generalise to held-out paraphrases, which is the open problem of the GPT-2 chain.",
    "e000015_symlink_cells": "The first measurement of the Symlink hypothesis as ledger section 7 states it: sharing "
        "versus duplicating. Both arms hold the identical world and are read by the identical model, so every number in "
        "the contrast is attributable to the storage form alone. One operation on the shared object reaches or deletes "
        "every access path; the same operation in the duplication arm reaches one key and leaves the object recoverable "
        "through the copies by probe (87.3%) and forced choice (1.000). The dereference ablation is the mechanism "
        "control: with the slot disabled, alias reading is 0% and fact reading is 100%. Two results are withheld and "
        "recorded as failures: shredding the alias rather than the payload reaches only 93% on the worst seed, and the "
        "two-slot control does not resolve two-link chains because chains never occur in the training distribution.",
    "e000018_both": "PARTLY WITHDRAWN. The match-gate arm measured nothing because the gate was cancelled by the "
        "RMS-matched injection one line later: a scalar that multiplies a vector and is then divided out by that "
        "vector's own norm cannot act. The conclusion 'all of the improvement is the training and none is the "
        "capacity' therefore does not follow from this record, and the capacity question is reopened in E-000022. "
        "What survives: training on generic text brings injection into unrelated text down by a factor of five and "
        "no arm gets within a factor of twelve of the bar, and both arms that move the number pay for it in reading "
        "and refusal. Original text follows. Read the three arms together rather than one at a time. The match gate alone leaves injection "
        "into unrelated text exactly where it was (3.2681 against a baseline of 3.2741); generic text in training "
        "alone brings it to 0.6035; both together to 0.6736. All of the improvement is the behavioural training and "
        "none is the added capacity, and no arm gets within a factor of twelve of the bar. The reason is that "
        "refusing a question and ignoring prose are routed through one null column and pull in opposite directions, "
        "which is a design fault rather than a tuning failure.",
    "e000021_gate_error_rates": "The number the deletion claim needed and did not have. Across 2.2 million fresh "
        "unsigned markers and eleven checkpoints the gate admits one in about 1,180, with a tight interval and no "
        "false rejects at all. That is the bound on every SHRED result in this programme: behavioural deletion is "
        "complete and the residual sits at chance, but roughly one payload per thousand would pass verification. It "
        "just clears the pre-registered bar of one in a thousand, and it is a limit rather than a guarantee.",
    "e000019_fresh_seed_chance": "The record that turns F4 from a tolerance claim into a chance claim, and does it "
        "outside the seeds that chose the configuration. Forced choice lands on exactly 375 of 750 pooled trials, the "
        "probe on 4 of 750 against a chance of 1 in 256, the true object top-1 on 7 of 750; every exact interval "
        "contains its chance level and stays inside the pre-registered distance. Two objections from the standing "
        "audit are answered in one run. What is not answered: the hard gate still admits an unsigned marker in at "
        "least one seed, and the top-1 interval only just contains chance with a point estimate about two and a half "
        "times the chance rate, so a larger sample could still separate them.",
    "e000017b_templates8": "The remedy run for the fired kill criterion, and it works for the part the criterion is "
        "about: at the prescribed budget of eight trained templates, refusal after REVOKE and SHRED on unseen phrasings "
        "reaches 89.8% (worst seed 86.5%) against 52% at two templates, the conditional figure reaches 99.3%, and the "
        "deleted object returns in exactly 0.0000 of cases. The criterion's own 95% bar is still not met, so it stays "
        "fired, but it is no longer evidence against the deletion mechanism. The run also surfaces a worse problem than "
        "the one it fixed: injection where there is no key degraded rather than improved (generic text 3.27 nats against "
        "a 0.05 bar, above E-000013's 2.27), so more prompt shapes in training mean more shapes that trigger a spurious "
        "read. That is the next thing to fix, because it means the layer perturbs the frozen model on unrelated text.",
    "e000017a_paraphrase_diagnosis": "The record that reframes the programme's one measured failure. Roadmap kill "
        "criterion 5 fired on the unconditional refusal rate, and that stands. Decomposing E-000012's own checkpoints "
        "without training anything shows the cause: conditioned on the model having read the fact at all while the cell "
        "was active, it refuses after REVOKE 96.1% of the time on held-out phrasings and returns the deleted object in "
        "0.15% of those cases. What does not generalise is reading (69.4% against 96.1%), so the defect sits in the "
        "query and routing path. The worst-seed conditional figure is 94.2%, still under the bar, so the remedy run with "
        "the template budget the roadmap prescribes is still owed.",
    "e000016_alias_chains": "The follow-up that turns E-000015's two recorded failures into an explanation. Both were "
        "caused by the training distribution rather than the architecture: with 30% chains in training, two dereference "
        "slots resolve a two-link chain completely, a one-slot model refuses it (100% unknown, 0% answered) instead of "
        "naming another entity, and shredding the pointer rather than the payload rises from 93% to 97% on the worst "
        "seed. The refusal arm is the load-bearing part: it shows the depth the mechanism reaches is set by the number "
        "of slots, and that the model reports the limit instead of hiding it.",
    "e000007_biomarker": "CORRECTION to the prose inside that record: its \"Reading:\" paragraph calls the shredded arm \"F4, learned\" eleven lines below the record's own F3 line. The criteria are evaluated on the worst seed and both F4 criteria fail there (value contribution 1.57 against a bar of 0.10, probe 8% against 5%), so the record is F3. The separation itself holds in every seed (suppressed: value "
        "contribution 8.3, probe 86%, mean rank 10; shredded: 1.3, 4%, 110). The two failed criteria are the same "
        "SHRED residual as in E-000004, addressed in E-000009.",
}

BOUNDARY = """## Boundary of this evidence

Everything below was produced on one 4-core CPU box in one session, with no GPU (experiments ran with 2 or 4 torch threads; each record stores the thread count under `environment`). It is therefore bounded as follows:

- **No LLM-scale evidence.** The largest neural core used is frozen GPT-2 small (124M parameters, E-000008). Nothing here shows editable knowledge inside a large pretrained model, and nothing here shows unlearning of facts that a pretrained model already encodes in its weights.
- **Synthetic worlds.** Facts are `(subject, relation) → object` triples over 256 entities and 4 relations; queries are symbolic (E-000001 … E-000007) or short natural-language templates (E-000008). Real-world knowledge, multi-token entities and free-text questions are not covered.
- **By construction versus learned.** REVOKE removes routing by a hard mask; that is deletion level F1 by construction. The learned results are: answering UNKNOWN instead of using another cell, refusing a payload whose marker is invalid (SHRED), composing hops, and the probe / forced-choice / rank checks after SHRED. Every result table states which is which.
- **Provenance is trained**, not emergent: the routing loss supervises which cell each hop reads. E-000006 (`no_routing_loss`) measures what remains without it.
- **The outstanding C55–C57 real-model / GPU chain of the ledger is still outstanding.** E-000008 is its CPU-feasible analogue on a small model, not its execution.
- **Noise figures are not comparable** with the architecture document's "noise = 0.24 → 68.4%", whose noise definition is not recorded; the sweep here perturbs bank keys and values relative to their RMS.
- **Seeds.** E-000003 … E-000007 evaluate the same five E-000001-B models on fresh worlds; they are not independent replications of training. E-000002, E-000006 and E-000008 train their own models (3 seeds); E-000009 trains five.
- **The SHRED residual.** E-000004 and E-000007 found that the marker gate learned without explicit supervision closes to about 9% rather than 0 on unsigned payloads, so a linear probe and forced choice recover a residual; their F4 criteria fail and the records say so. E-000009 (plain verification loss) narrowed it but left an unsigned tail; E-000010 (class-balanced verification loss) closed it to chance in every seed — a recorded result, not an assumption.
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
        table(["experiment", "title", "evidence level recorded", "deletion level recorded (targeted)", "status", "recorded at (UTC)"],
              [(r["experiment"], r["title"], r.get("evidence_level", "-"),
                (r.get("deletion_level") or "-") + (f" ({r['deletion_level_targeted']})" if r.get("deletion_level_targeted") else ""),
                status_of(r), r.get("recorded_at", "-")[:16].replace("T", " "))
               if r else (n, "-", "-", "-", "not run", "-") for n, r in recs.items()]),
        "",
        "## The six breakthrough properties (ledger section 3)",
        "",
        table(["property", "experiments", "status", "highest level"],
              [(prop, ", ".join(recs[e]["experiment"] for e in exps if recs.get(e)),
                "; ".join(f"{recs[e]['experiment']}: {status_of(recs[e])}" for e in exps if recs.get(e)) or "not run",
                max((recs[e].get("evidence_level", "E0") for e in exps if recs.get(e)), default="-"))
               for prop, exps in PROPERTIES.items()]),
        "",
        ("Scalability is the property this session can least address: E-000008 tests whether the same layer works as an "
         "adapter on a frozen pretrained GPT-2 (" + ("recorded below" if recs.get("e000008_gpt2_adapter") else "not yet recorded") +
         "); the path to LLM scale is a roadmap item, not a result."),
        "",
        "The 'status' column is the status of the whole record's pre-registered criteria; a property can be supported by "
        "an experiment whose record fails on a different criterion (E-000004 and E-000007 fail only their F4 rows while "
        "their behavioural rows pass — see the split claims inside those records).",
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
