"""NOV-005 — run the literature search that produced this programme's novelty verdicts, and check it.

Every load-bearing novelty statement in this repository is a **null**: "Nothing I verified covers
this", "I found no work that demonstrates …", "the combination is unclaimed", "the paired arms in one
reader I could not find". Those nulls came out of a 41-agent literature workflow whose own provenance
note disclaims its external citations, in an environment the drafts asserted had no network — an
assertion that was never tested and was false (§31.57).

So the situation on 2026-09-07 was: the *positives* had been checked (`docs/paper/references.md`
verifies eight clusters against their sources) and the *negatives* had not been checked at all. A
search that has never been shown to find anything cannot support a null. **§31.15, for the fifth
time, now applied to the literature review itself: an instrument that cannot fail is not evidence.**

This module records the audit that closes that. Three things are in it:

  **1. The calibration.** Positive controls — propositions whose prior art is certain and findable,
  phrased as novelty claims — are searched with the same instrument. If the search cannot rediscover
  Codd on the update anomaly, Carlini and Wagner on learned predicates, or the resilience literature
  on minimum contingency sets, then its silence on the live claims means nothing and `run()` returns
  `SEARCH_INVALID`. A negative control — a fabricated construct that cannot exist — must return
  nothing, so that a searcher which manufactures a hit for anything is caught too.

  **2. The verdicts.** Each live claim, with the citation that kills or narrows it, an identifier, and
  a sentence quoted from the source rather than from a summary of it.

  **3. The enforcement.** A claim this audit marks SUPERSEDED or NARROWED must not still be asserted
  flat in the document that asserted it. The paragraph carrying the assertion has to carry the marker
  `NOV-005` too. Without that the correction is a document nobody reads against a claim everybody
  does, which is exactly how §31.62's drift happened. `main()` exits non-zero on a contradiction.

**What this is not.** It is not a systematic review, and a null here is weaker than a null in one:
these are the queries a person ran on one afternoon, recorded so the next person can run better ones.
It reports what the search found, never that nothing exists. Every verdict of SURVIVES should be read
as "not found by these queries", and the queries are in the record so they can be attacked.

Run:  python -m so.experiments.nov005_novelty_claim_audit
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# The marker a corrected paragraph must carry. Short, greppable, and named for the audit that
# produced the correction so the reader can find out why the sentence changed.
MARKER = "NOV-005"

SEARCHED_ON = "2026-09-07"


@dataclass(frozen=True)
class Citation:
    """One source, with what was read and a sentence quoted from it."""

    ident: str          # arXiv id, eprint number, or a stable bibliographic key
    title: str
    authors: str
    date: str
    quote: str          # verbatim from the source, not from a summary of it
    read: str           # FULL TEXT | ABSTRACT | SEARCH METADATA


@dataclass(frozen=True)
class Claim:
    """A novelty null the repository asserted, and what the search did to it."""

    ident: str
    proposition: str        # the null, as the repository states it
    asserted_in: str        # the file that asserts it
    phrase: str             # a distinctive substring of the assertion, for locating it
    record: str             # the so/results record behind the underlying finding
    verdict: str            # SURVIVES | NARROWED | SUPERSEDED
    queries: tuple[str, ...]
    evidence: tuple[Citation, ...] = ()
    what_survives: str = ""  # required when NARROWED: the part the citation does not reach


@dataclass(frozen=True)
class Control:
    """A proposition of known standing. The search must find its prior art, or it is not a search."""

    ident: str
    proposition: str
    query: str
    must_find: str      # a substring that has to appear in the found work's identifier or title
    found: str          # what the search actually returned


_VERDICTS = {"SURVIVES", "NARROWED", "SUPERSEDED"}
_NEEDS_MARKER = {"NARROWED", "SUPERSEDED"}


# ----------------------------------------------------------------------------- the calibration
# Three propositions whose prior art is not in question, phrased the way the novelty statement
# phrases its own claims, and run through the same searches. These passed: each returned the work
# it had to return. That is the only reason a null below is worth anything.

_CONTROLS: tuple[Control, ...] = (
    Control(
        ident="PC-CODD",
        proposition=(
            "That one operation on a shared record reaches every access path, while duplicated "
            "records need one operation each, is unclaimed."
        ),
        query="Codd 1970 relational model update anomaly normalization shared record single operation",
        must_find="Codd",
        found=(
            "E. F. Codd, 'A Relational Model of Data for Large Shared Data Banks', CACM 13(6), 1970. "
            "Normalization beyond 1NF stated by Codd as freeing relations from undesirable insertion, "
            "update and deletion dependencies."
        ),
    ),
    Control(
        ident="PC-CW",
        proposition="That a learned predicate can be made to fail adversarially is unclaimed.",
        query="Carlini Wagner evaluating robustness neural networks detector bypass",
        must_find="1608.04644",
        found="Carlini and Wagner, arXiv:1608.04644, IEEE S&P 2017. Already cluster 6 of references.md.",
    ),
    Control(
        ident="PC-RESILIENCE",
        proposition=(
            "That the minimum number of tuples whose removal falsifies a Boolean query is a named "
            "quantity with a complexity dichotomy is unclaimed."
        ),
        query="resilience minimum contingency set conjunctive query dichotomy database deletion",
        must_find="1907.01129",
        found=(
            "Resilience / minimum contingency set, with the self-join-free dichotomy and the triad "
            "hardness structure; arXiv:1907.01129 and the ILP/LP-relaxation solver line."
        ),
    ),
)

# The other half of the calibration. A searcher that returns something for anything is as useless as
# one that returns nothing for everything, and only the second failure mode is usually tested.
_NEGATIVE_CONTROL = Control(
    ident="NC-FABRICATED",
    proposition=(
        "Deletion certificates for memory stores addressed by prime-indexed Goedel numbering with "
        "tri-symplectic revocation manifolds are unclaimed."
    ),
    query=(
        "deletion certificates for memory stores addressed by prime-indexed Goedel numbering with "
        "tri-symplectic revocation manifolds"
    ),
    must_find="",       # nothing may match
    found=(
        "Topically adjacent noise only — certificate-revocation patents. No work on the construct, "
        "which does not exist. The searcher did not manufacture a hit."
    ),
)


# ----------------------------------------------------------------------------- the verdicts

_CLAIMS: tuple[Claim, ...] = (
    Claim(
        ident="N1-payload-derived-index-channel",
        proposition=(
            "A deletion primitive that gates the value channel to chance while a term of the "
            "computation derived from the same payload still recovers it is uncovered by prior work."
        ),
        asserted_in="docs/so-novelty-2026-09-04.md",
        phrase="Nothing I verified covers this",
        record="so/results/e000028_key_channel.json",
        verdict="NARROWED",
        queries=(
            "unlearning leaves auxiliary index metadata derived from deleted value residual key "
            "recovers content after successful forgetting",
            "recover deleted record through derived index term while value channel gated at chance",
        ),
        evidence=(
            Citation(
                ident="arXiv:2609.04875",
                title="Forgetting Without Restarting: Execution-State Unlearning for Stateful LLM Agents",
                authors="Chao Yao et al.",
                date="2026-09-04",
                quote=(
                    "deployed stacks offer only a forgetting affordance that operates on plaintext at "
                    "a single layer: delete the memory record, edit the Markdown file, drop the "
                    "message from retrieval"
                ),
                read="FULL TEXT",
            ),
        ),
        what_survives=(
            "The general shape — deletion at the plaintext layer leaving payload-derived artifacts "
            "intact — is published, three days before this audit, at agent scale, with behavioural "
            "extraction at 80-100% of episodes against zero string matches. Two things it does not "
            "reach. (i) Its artifacts CARRY the content: summaries and plans paraphrase the target "
            "and KV tensors encode it, so the finding is that a copy survived. E-000028's reverse key "
            "k_rev(LN(o + r)) contains no payload and names none; it discriminates without holding "
            "anything, which is why every value-channel attack is at chance while recovery is exact. "
            "(ii) It measures presence and behaviour, not a candidate-set posterior; PDX-001's "
            "tombstone row is invisible to exactly the binary reading it uses (top-1 0.0000, search "
            "space cut 88x). The transferable rule is no longer ours. The measurement that catches "
            "the case the rule misses still is."
        ),
    ),
    Claim(
        ident="N2-swept-geometry-of-a-learned-gate",
        proposition=(
            "No work demonstrates the specification-versus-learned-boundary gap as a swept geometry "
            "on a deletion mechanism, with the sampled-distribution rate and the predicate-region "
            "rate reported side by side."
        ),
        asserted_in="docs/so-novelty-2026-09-04.md",
        phrase="I found no work that demonstrates the specification-versus-learned-boundary gap",
        record="so/results/e000029_marker_geometry.json",
        verdict="SURVIVES",
        queries=(
            "learned gate declared threshold versus operational decision boundary swept geometry "
            "deletion mechanism false accept rate distribution",
        ),
    ),
    Claim(
        ident="N3-equivalence-margin-floor-heldout",
        proposition=(
            "The combination of a pre-registered equivalence margin, a calibrated attack floor, and "
            "seeds held out from configuration selection is unclaimed."
        ),
        asserted_in="docs/so-novelty-2026-09-04.md",
        phrase="margin-plus-floor-plus-held-out-seeds combination is unclaimed",
        record="so/results/e000019_fresh_seed_chance.json",
        verdict="SUPERSEDED",
        queries=(
            "unlearning evaluation equivalence testing pre-registered margin calibrated attack floor "
            "held-out seeds accept null",
        ),
        evidence=(
            Citation(
                ident="arXiv:2607.19442",
                title=(
                    "Unlearning as Distribution Restoration: A Controlled Counterfactual Study, a "
                    "Validated Selective Screen, and the Limits of Oracle-Free Certification"
                ),
                authors="Sen Yang, Yuen-Hei Yeung",
                date="2026-07-21",
                quote=(
                    "only 28.0% of survivors (CI [22.6%,34.0%]) are TOST-equivalent to never-learned "
                    "at that tolerance"
                ),
                read="FULL TEXT",
            ),
            Citation(
                ident="arXiv:2607.19442#floor",
                title="the same paper's sealed challenge panel, which is the calibrated floor",
                authors="Sen Yang, Yuen-Hei Yeung",
                date="2026-07-21",
                quote="The screen rejects Minj in 45/45 cells and accepts the reference in 44/45",
                read="FULL TEXT",
            ),
        ),
        what_survives="",
    ),
    Claim(
        ident="N5-paired-sharing-and-duplication-arms",
        proposition=(
            "One reader, one ground truth, one calibrated probe, applied to a canonicalised arm and a "
            "duplicated arm after the same lifecycle operation, is unclaimed."
        ),
        asserted_in="docs/so-novelty-2026-09-04.md",
        phrase="The paired arms in one reader I could not find",
        record="so/results/e000032_deletion_closure.json",
        verdict="SURVIVES",
        queries=(
            "duplicated versus shared canonical facts same reader paired arms deletion one operation "
            "reaches all copies experiment language model",
        ),
        what_survives=(
            "The nearest hit pairs states on a different axis: Raeesi and Roed run alias-closure "
            "deletions with retrieval enabled against disabled, not canonicalised against duplicated, "
            "and their canonicalisation proposal is in Future Work and unimplemented (references.md, "
            "cluster 4, read in full). The pairing axis here is still unclaimed."
        ),
    ),
    Claim(
        ident="N6-erasure-disclosure-duality",
        proposition=(
            "That canonicalisation makes erasure a single certifiable operation AND turns every "
            "surviving access path into an oracle naming the deleted key — the closure inverting with "
            "the guarantee — is unclaimed."
        ),
        asserted_in="docs/so-novelty-2026-09-04.md",
        phrase="keeps a dangling reference after deleting its referent has it",
        record="so/results/e000035_deletion_disclosure.json",
        verdict="SURVIVES",
        queries=(
            "canonicalization single canonical record makes deletion cheap but identifiable "
            "duplication hides which record removed tradeoff unlinkability",
            "deletion itself leaks which item was deleted surviving pointers identify removed record",
        ),
        what_survives=(
            "Adjacent work exists and does not reach it. MERIT (arXiv:2607.29173) treats the stale "
            "incoming edges of a deleted node in a proximity graph purely as a cost — 'stale incoming "
            "edges consume search capacity' — and makes no disclosure claim at all. The dedup and "
            "differential-privacy literatures have deletion patterns leaking in aggregate. Neither "
            "states the inversion: that the design decision which minimises the contingency set is "
            "the same one that makes the removed key uniquely nameable, so no configuration is both "
            "cheapest to certify and quietest to observe."
        ),
    ),
    Claim(
        ident="N7-composition-with-a-record-level-certificate",
        proposition=(
            "Composing a store-side closure guarantee with a record-level deletion certificate over a "
            "learned reader is ours."
        ),
        asserted_in="docs/paper/deletion-certificates-draft-2026-09-06.md",
        phrase="composition of a store-side guarantee with a record-level certificate",
        record="so/results/e000030_deletion_certificate.json",
        verdict="SUPERSEDED",
        queries=(
            "Garg Goldwasser Vasudevan Formalizing Data Deletion Right to be Forgotten Eurocrypt 2020",
            "Cohen Smith Swanberg Vasudevan Control Confidentiality Right to be Forgotten CCS 2023",
        ),
        evidence=(
            Citation(
                ident="eprint:2020/254",
                title="Formalizing Data Deletion in the Context of the Right to be Forgotten",
                authors="Sanjam Garg, Shafi Goldwasser, Prashant Nalini Vasudevan",
                date="Eurocrypt 2020",
                quote=(
                    "The data collector M maintains a dataset as a history-independent dictionary "
                    "Dict. […] it updates model to be the output of delete(Dict, model, key)."
                ),
                read="FULL TEXT",
            ),
            Citation(
                ident="eprint:2020/254#thm34",
                title="Theorem 3.4, the composition stated as a bound",
                authors="Garg, Goldwasser and Vasudevan",
                date="Eurocrypt 2020",
                quote=(
                    "The data collector (M, pi, piD) as described in Fig. 5 has 1-representative "
                    "deletion-compliance error at most (1/lambda + poly(lambda)/2^lambda)."
                ),
                read="FULL TEXT",
            ),
            Citation(
                ident="arXiv:2210.07876",
                title="Control, Confidentiality, and the Right to be Forgotten",
                authors="Aloni Cohen, Adam D. Smith, Marika Swanberg, Prashant Nalini Vasudevan",
                date="CCS 2023",
                quote=(
                    "builds a unified formalism for deletion that encompasses previous approaches as "
                    "special cases"
                ),
                read="SEARCH METADATA",
            ),
        ),
        what_survives="",
    ),
    Claim(
        ident="N8-attack-based-standard-is-not-a-guarantee",
        proposition=(
            "That 'no attack recovered it' is not an adequate deletion guarantee is this paper's F1, "
            "argued here."
        ),
        asserted_in="docs/paper/deletion-certificates-draft-2026-09-06.md",
        phrase="F1 says attacking harder is not the fix",
        record="so/results/e000030_deletion_certificate.json",
        verdict="NARROWED",
        queries=(
            "Yang Yeung unlearning distribution restoration oracle-free certification limits 2026",
        ),
        evidence=(
            Citation(
                ident="arXiv:2607.19442#forward-only",
                title="the same paper's adversarial boundary section",
                authors="Sen Yang, Yuen-Hei Yeung",
                date="2026-07-21",
                quote=(
                    "in 12/45 cells the fixed-magnitude penalty lands the forget-answer NLL within "
                    "family tolerance, and the entire forward battery accepts a suppressed model. "
                    "Forward-only certification is not sound"
                ),
                read="FULL TEXT",
            ),
        ),
        what_survives=(
            "The thesis is not ours and is not new: an independent paper reached it two months "
            "earlier, at 45 model-seed cells over five architecture families, and demonstrated it "
            "with a suppression attack that defeats a whole forward battery on a model whose "
            "knowledge is intact. It is corroboration, and it is stronger evidence for F1 than "
            "anything in this repository. What is left to F1 here is the specific mechanism — a "
            "recovery channel through a term that never holds the payload — and the constructive "
            "half, which that paper explicitly declines: it calls its own result 'an empirical "
            "selective test for methods-as-produced, not an adversarially sound certificate'."
        ),
    ),
)


def _paragraphs(text: str) -> list[str]:
    return re.split(r"\n\s*\n", text)


def _assertion_sites(root: Path, claim: Claim) -> tuple[bool, list[str]]:
    """Find the paragraphs asserting a claim, and whether each carries the correction marker."""
    path = root / claim.asserted_in
    if not path.exists():
        return False, []
    text = path.read_text(encoding="utf-8")
    hits = [p for p in _paragraphs(text) if claim.phrase in p]
    return bool(hits), [p for p in hits if MARKER not in p]


def run(root: Path) -> dict:
    for claim in _CLAIMS:
        if claim.verdict not in _VERDICTS:
            raise ValueError(f"{claim.ident}: unknown verdict {claim.verdict!r}")

    # -- the floor. Controls of known standing must have been found, the fabricated one must not.
    controls = []
    floor_met = True
    for c in _CONTROLS:
        ok = c.must_find.lower() in c.found.lower()
        floor_met = floor_met and ok
        controls.append({
            "id": c.ident, "proposition": c.proposition, "query": c.query,
            "must_find": c.must_find, "found": c.found, "prior_art_found": ok,
        })
    neg_ok = _NEGATIVE_CONTROL.found.strip() != "" and "no work" in _NEGATIVE_CONTROL.found.lower()
    floor_met = floor_met and neg_ok

    # -- the verdicts, and whether the asserting document has absorbed them.
    claims, contradictions = [], []
    for claim in _CLAIMS:
        found, unmarked = _assertion_sites(root, claim)
        needs = claim.verdict in _NEEDS_MARKER
        if needs and not found:
            contradictions.append({
                "claim": claim.ident, "kind": "ASSERTION_NOT_FOUND",
                "detail": (
                    f"{claim.asserted_in} no longer contains {claim.phrase!r}. Either the claim was "
                    "silently deleted rather than corrected, or this registry is stale. Both are "
                    "failures: a withdrawn claim has to leave a visible withdrawal."
                ),
            })
        elif needs and unmarked:
            contradictions.append({
                "claim": claim.ident, "kind": "UNCORRECTED_ASSERTION",
                "detail": (
                    f"{claim.asserted_in} still asserts {claim.phrase!r} in {len(unmarked)} "
                    f"paragraph(s) with no {MARKER} marker, but the search verdict is "
                    f"{claim.verdict}."
                ),
            })
        if claim.verdict == "NARROWED" and not claim.what_survives:
            contradictions.append({
                "claim": claim.ident, "kind": "NARROWED_WITHOUT_REMAINDER",
                "detail": "A narrowed claim must say what part of it the citation does not reach.",
            })
        if claim.verdict in _NEEDS_MARKER and not claim.evidence:
            contradictions.append({
                "claim": claim.ident, "kind": "VERDICT_WITHOUT_EVIDENCE",
                "detail": f"{claim.verdict} requires at least one citation with a quoted sentence.",
            })
        claims.append({
            "id": claim.ident,
            "proposition": claim.proposition,
            "asserted_in": claim.asserted_in,
            "record": claim.record,
            "verdict": claim.verdict,
            "queries": list(claim.queries),
            "evidence": [
                {"id": e.ident, "title": e.title, "authors": e.authors, "date": e.date,
                 "quote": e.quote, "read": e.read}
                for e in claim.evidence
            ],
            "what_survives": claim.what_survives,
            "assertion_located": found,
            "uncorrected_paragraphs": len(unmarked),
        })

    by_verdict = {v: sum(1 for c in _CLAIMS if c.verdict == v) for v in sorted(_VERDICTS)}
    return {
        "experiment": "NOV-005",
        "title": "The novelty search, calibrated and then run",
        "searched_on": SEARCHED_ON,
        "claims_examined": len(_CLAIMS),
        "verdict_counts": by_verdict,
        "claims": claims,
        "positive_controls": controls,
        "negative_control": {
            "id": _NEGATIVE_CONTROL.ident, "proposition": _NEGATIVE_CONTROL.proposition,
            "query": _NEGATIVE_CONTROL.query, "found": _NEGATIVE_CONTROL.found,
            "returned_nothing": neg_ok,
        },
        "validity_floor_met": floor_met,
        "contradictions": contradictions,
        "contradiction_count": len(contradictions),
        "decision": (
            "SEARCH_INVALID_CONTROLS_MISSED" if not floor_met
            else "CLAIMS_CONTRADICTED" if contradictions
            else "AUDIT_CLEAN"
        ),
        "how_to_read_this": (
            "The verdicts are only as good as the calibration above them. Three propositions of known "
            "standing were run through the same instrument and it returned their prior art; a "
            "fabricated construct was run through it and it returned nothing. That is what makes a "
            "null here worth more than the nulls it replaces, which were produced by a workflow whose "
            "own citations were disclaimed and whose environment was wrongly believed to have no "
            "network. It is still not a systematic review. SURVIVES means 'these queries did not find "
            "it', and the queries are recorded so the next reader can beat them."
        ),
        "not_claimed": (
            "No cited result has been reproduced. Every verdict rests on reading the source, and a "
            "source read correctly can still be applied wrongly. In particular SUPERSEDED is a claim "
            "about priority, not about quality: N3 and N7 are better done elsewhere, which is the "
            "finding, and nothing here says the work behind them was not worth doing."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("so/results/nov005"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run(args.root)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / "nov005_novelty_claim_audit.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if not result["validity_floor_met"] or result["contradictions"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
