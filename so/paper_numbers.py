"""Bind every number the paper prints to the record it came from, and check it.

The paper's prose has now disagreed with `so/results/` three times, and each was caught by a human
reading a JSON file:

  * E-000028's revoke/delete mean rank was written 128.0; the record says **128.02**.
  * The marker accept rate was written "1.0000 out to 0.70"; the band sweep says **0.9999** at 0.70,
    min 0.9992 across checkpoints.
  * §7's head-to-head table was presented without a seed count; the record is **one seed**.

Three for three is not a run of bad luck, it is the absence of an instrument. Prose drifts from the
record whenever either is edited, and nothing in this repository noticed. This module is the missing
check: a registry binding each printed figure to a JSON path, and a run that fails when they part.

**How a claim can fail.** Three ways, all of them reported rather than skipped:

  ``PATH``      the record path does not resolve — the claim is unanchored, which is worse than
                wrong, because a silent skip would let the registry grow while checking nothing.
  ``ABSENT``    the figure is not present in the paper text as written — usually the paper was
                reworded and the registry not updated, or vice versa.
  ``MISMATCH``  the record's value does not round to the printed figure under the claim's stated
                rule.

**Scope claims** are the other half, and the one that caught §7: a fact about the record's *extent*
(how many seeds, how many cells) that the paper must state in words. A number can be perfectly
accurate and still mislead if the reader is not told it rests on one seed.

**The presence test needed its own calibration.** The first version asked whether the printed string
occurred anywhere in the paper, which is nearly vacuous for a short token: `"0.0"` is inside
`0.0040`, `"12"` is inside `128.02`. Requiring a standalone match immediately turned up two figures
the registry had been passing on a coincidence -- the eleven checkpoints and the twelve templates,
both of which the paper spells in words and never prints as numerals. `appears_as` binds those. And
for a figure round enough to recur across the paper (`1.0000`, `256`) the presence test still does
not discriminate; those are reported weak rather than counted as checks.

**The drawn figures are checked separately**, because they drift on their own and are what a reader
looks at first. `FIGURE_CLAIMS` binds each number printed inside an SVG to its record. What that pass
does *not* claim is the geometry: Figure 2 was once drawn with a y-scale that put 0.2191 at the
height of 0.50, and no string check would have seen it -- a render did.

This registry is **partial by construction** and says so: `coverage()` reports what it binds. A
figure absent from the registry is unchecked, not verified.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

RESULTS = Path("so/results")
PAPER = Path("docs/paper/deletion-certificates-draft-2026-09-06.md")
FIGURES = Path("docs/paper/figures")


# ----------------------------------------------------------------------------- rounding rules
# Each rule says how the record's value becomes the string the paper prints.

def _fmt(value: float, rule: str) -> str:
    if rule == "int":
        return str(int(round(value)))
    if rule == "thousands":
        return f"{int(round(value)):,}"
    if rule.startswith("dp"):
        return f"{value:.{int(rule[2:])}f}"
    if rule.startswith("sig"):
        return f"{value:.{int(rule[3:])}g}"
    if rule.startswith("exp"):
        return f"{value:.{int(rule[3:])}e}"
    raise ValueError(f"unknown rounding rule {rule!r}")


@dataclass(frozen=True)
class Claim:
    label: str
    record: str
    path: str
    printed: str
    rule: str
    reduce: str = ""       # set when the path uses '*' and the paper quotes an aggregate
    appears_as: str = ""   # set when the paper spells the figure some other way ("eleven" for 11)

    @property
    def sought(self) -> str:
        return self.appears_as or self.printed


@dataclass(frozen=True)
class FigureClaim:
    """A number printed inside a figure, bound to the record it came from.

    Figures drift independently of the prose and are what a reader looks at first -- Figure 2 was
    drawn once with a y-scale that put 0.2191 at the height of 0.50, and only a render caught it.
    A number in an SVG is as much a claim as a number in a paragraph.
    """

    figure: str
    label: str
    record: str
    path: str
    printed: str
    rule: str
    reduce: str = ""

    @property
    def sought(self) -> str:
        return self.printed


@dataclass(frozen=True)
class ScopeClaim:
    """A fact about the record's extent that the paper must state in words."""

    label: str
    record: str
    path: str
    expect: object
    must_appear: str


def _resolve(doc: object, path: str):
    """Walk a '/'-separated path.

    These records mix two shapes: genuine nesting (`aggregate` -> `operational_radius` -> `mean`)
    and flat keys that contain slashes (`aggregate` -> `'active/margin_mean'` -> `mean`, and
    `gpt2[0]` -> `'gpt2_soft/shred/residual'`). So at each level try the longest remaining path as
    one key before falling back to a single segment.

    On a list a segment may be a numeric index, `*` to iterate, or `field=value` to select the one
    element whose `field` equals `value`. The selector exists so that a claim about a named policy
    does not silently follow the list's *order*: PDX-001 writes five policies as a list, and a
    positional path would keep resolving, and keep passing, if that order ever changed.
    """
    segs = [s for s in path.split("/") if s != ""]
    cur, i = doc, 0
    while i < len(segs):
        if isinstance(cur, list):
            seg = segs[i]
            if "=" in seg:
                field, _, want = seg.partition("=")
                hits = [it for it in cur if isinstance(it, dict) and str(it.get(field)) == want]
                if len(hits) != 1:
                    return None, f"selector {seg!r} matched {len(hits)} of {len(cur)}, need exactly 1"
                cur, i = hits[0], i + 1
                continue
            if seg == "*":
                out = []
                for item in cur:
                    sub, err = _resolve(item, "/".join(segs[i + 1:]))
                    if err:
                        return None, err
                    out.append(sub)
                return out, None
            if not seg.isdigit():
                return None, f"list needs an index at {seg!r}"
            if int(seg) >= len(cur):
                return None, f"index {seg} out of range ({len(cur)})"
            cur, i = cur[int(seg)], i + 1
            continue
        if not isinstance(cur, dict):
            return None, f"cannot descend into {type(cur).__name__} at {segs[i]!r}"
        for j in range(len(segs), i, -1):
            key = "/".join(segs[i:j])
            if key in cur:
                cur, i = cur[key], j
                break
        else:
            return None, f"missing key {segs[i]!r}"
    return cur, None


def _reduce(values, how: str):
    if how == "mean":
        return sum(values) / len(values)
    if how == "min":
        return min(values)
    if how == "max":
        return max(values)
    raise ValueError(f"unknown reducer {how!r}")


CLAIMS: tuple[Claim, ...] = (
    # E-000028 — the payload-derived index channel
    Claim("E28 active top-1", "e000028_key_channel.json", "aggregate/active/object_top1/mean", "1.0000", "dp4"),
    Claim("E28 shred top-1", "e000028_key_channel.json", "aggregate/shred/object_top1/mean", "1.0000", "dp4"),
    Claim("E28 shred margin", "e000028_key_channel.json", "aggregate/shred/margin_mean/mean", "0.6195", "dp4"),
    Claim("E28 active margin", "e000028_key_channel.json", "aggregate/active/margin_mean/mean", "0.6195", "dp4"),
    Claim("E28 revoke top-1", "e000028_key_channel.json", "aggregate/revoke/object_top1/mean", "0.0040", "dp4"),
    Claim("E28 revoke mean rank", "e000028_key_channel.json", "aggregate/revoke/object_mean_rank/mean", "128.02", "dp2"),
    Claim("E28 revoke margin", "e000028_key_channel.json", "aggregate/revoke/margin_mean/mean", "0.0022", "dp4"),

    Claim("E28 active mean rank", "e000028_key_channel.json", "aggregate/active/object_mean_rank/mean", "0.0", "dp1"),
    Claim("E28 shred mean rank", "e000028_key_channel.json", "aggregate/shred/object_mean_rank/mean", "0.0", "dp1"),
    Claim("E28 chance top-1", "e000028_key_channel.json", "pooled_vs_chance/chance", "0.0039", "dp4"),
    Claim("E28 pooled targets", "e000028_key_channel.json", "pooled_vs_chance/n", "500", "int"),

    # PDX-001 — the same attack over an abstract store, selected by policy name not by position
    Claim("PDX value-gated top-1", "pdx001/pdx001_payload_derived_index_audit.json",
          "policies/policy=value_gated_shred/top1_recovery", "1.0000", "dp4"),
    Claim("PDX value-gated candidates", "pdx001/pdx001_payload_derived_index_audit.json",
          "policies/policy=value_gated_shred/mean_candidates_remaining", "1.00", "dp2"),
    Claim("PDX value-gated posterior", "pdx001/pdx001_payload_derived_index_audit.json",
          "policies/policy=value_gated_shred/mean_posterior_on_true_payload", "1.000000", "dp6"),
    Claim("PDX value-gated cut", "pdx001/pdx001_payload_derived_index_audit.json",
          "policies/policy=value_gated_shred/search_space_reduction_factor", "256", "int"),

    Claim("PDX tombstone top-1", "pdx001/pdx001_payload_derived_index_audit.json",
          "policies/policy=hnsw_tombstone/top1_recovery", "0.0000", "dp4"),
    Claim("PDX tombstone candidates", "pdx001/pdx001_payload_derived_index_audit.json",
          "policies/policy=hnsw_tombstone/mean_candidates_remaining", "2.92", "dp2"),
    Claim("PDX tombstone posterior", "pdx001/pdx001_payload_derived_index_audit.json",
          "policies/policy=hnsw_tombstone/mean_posterior_on_true_payload", "0.391667", "dp6"),
    Claim("PDX tombstone cut", "pdx001/pdx001_payload_derived_index_audit.json",
          "policies/policy=hnsw_tombstone/search_space_reduction_factor", "88", "int"),

    Claim("PDX codebook top-1", "pdx001/pdx001_payload_derived_index_audit.json",
          "policies/policy=codebook_key/top1_recovery", "1.0000", "dp4"),
    Claim("PDX unindexed candidates", "pdx001/pdx001_payload_derived_index_audit.json",
          "policies/policy=revoke_unindex/mean_candidates_remaining", "256.00", "dp2"),
    Claim("PDX unindexed posterior", "pdx001/pdx001_payload_derived_index_audit.json",
          "policies/policy=revoke_unindex/mean_posterior_on_true_payload", "0.003906", "dp6"),
    Claim("PDX gate-all candidates", "pdx001/pdx001_payload_derived_index_audit.json",
          "policies/policy=gate_all_derived/mean_candidates_remaining", "256.00", "dp2"),
    Claim("PDX payload domain", "pdx001/pdx001_payload_derived_index_audit.json",
          "policies/policy=gate_all_derived/payload_domain", "256", "int"),
    Claim("PDX validity floor", "pdx001/pdx001_payload_derived_index_audit.json",
          "validity_control/top1_recovery", "1.0000", "dp4"),

    # E-000029 — the swept geometry
    Claim("E29 declared radius", "e000029_marker_geometry.json", "declared_radius", "0.35", "dp2"),
    Claim("E29 operational radius", "e000029_marker_geometry.json", "aggregate/operational_radius/mean", "0.90", "dp2"),
    Claim("E29 annulus accepted", "e000029_marker_geometry.json", "pooled/annulus/0", "2,199,996", "thousands"),
    Claim("E29 annulus n", "e000029_marker_geometry.json", "pooled_intervals/annulus/n", "2,200,000", "thousands"),
    Claim("E29 checkpoints", "e000029_marker_geometry.json", "n_checkpoints", "11", "int", "", "eleven"),
    Claim("E29 per-band n", "e000029_marker_geometry.json", "n_per_band", "20,000", "thousands"),
    Claim("E29 band 0.60 mean", "e000029_marker_geometry.json", "per_checkpoint/*/band_accept/5", "1.0000", "dp4", "mean"),
    Claim("E29 band 0.70 mean", "e000029_marker_geometry.json", "per_checkpoint/*/band_accept/6", "0.9999", "dp4", "mean"),
    Claim("E29 band 0.70 min", "e000029_marker_geometry.json", "per_checkpoint/*/band_accept/6", "0.9992", "dp4", "min"),
    Claim("E29 band 0.80 mean", "e000029_marker_geometry.json", "per_checkpoint/*/band_accept/7", "0.2191", "dp4", "mean"),
    Claim("E29 band 0.80 min", "e000029_marker_geometry.json", "per_checkpoint/*/band_accept/7", "0.0953", "dp4", "min"),
    Claim("E29 band 0.80 max", "e000029_marker_geometry.json", "per_checkpoint/*/band_accept/7", "0.4014", "dp4", "max"),
    Claim("E29 band 0.90 mean", "e000029_marker_geometry.json", "per_checkpoint/*/band_accept/8", "0.0000", "dp4", "mean"),

    # E-000021 — the published false-accept rate the paper contrasts against
    Claim("E21 false-accept rate", "e000021_gate_error_rates.json", "totals/false_accept_rate", "8.49e-04", "exp2"),
    Claim("E21 n", "e000021_gate_error_rates.json", "totals/ci_false_accept/n", "2,200,000", "thousands"),

    # E-000035 — the deletion oracle
    Claim("E35 canonical disclosed", "e000035_deletion_disclosure.json", "aggregate/canonical/deleted_key_disclosed/mean", "1.0000", "dp4"),
    Claim("E35 canonical unique", "e000035_deletion_disclosure.json", "aggregate/canonical/uniquely_identified/mean", "1.0000", "dp4"),
    Claim("E35 duplicated disclosed", "e000035_deletion_disclosure.json", "aggregate/duplicated/deleted_key_disclosed/mean", "0.0000", "dp4"),
    Claim("E35 key space", "e000035_deletion_disclosure.json", "aggregate/duplicated/candidate_keys_mean/mean", "1,536", "thousands"),

    # E-000019 — the attack validity floor
    Claim("E19 probe floor min", "e000019_fresh_seed_chance.json", "aggregate/verified_hard/probe_calibration_top1/min", "0.893", "dp3"),
    Claim("E19 probe floor max", "e000019_fresh_seed_chance.json", "aggregate/verified_hard/probe_calibration_top1/max", "0.927", "dp3"),
    Claim("E19 forced-choice wins", "e000019_fresh_seed_chance.json", "equivalence/forced_choice_win/successes", "375", "int"),
    Claim("E19 forced-choice n", "e000019_fresh_seed_chance.json", "equivalence/forced_choice_win/n", "750", "int"),
    Claim("E19 probe hits", "e000019_fresh_seed_chance.json", "equivalence/probe_top1/successes", "4", "int"),
    Claim("E19 derivable survival", "e000019_fresh_seed_chance.json",
          "aggregate/verified_hard/dependency/derivable_recovery_after_revoke_K3/min", "1.0", "dp1"),

    # E-000025 — the price of the indirection, worst of three seeds
    Claim("E25 cost of sharing", "e000025_template_rescoring.json", "aggregate/all/cost_of_sharing/max", "0.0954", "dp4"),
    Claim("E25 cost of link training", "e000025_template_rescoring.json", "aggregate/all/cost_of_link_training/max", "0.0688", "dp4"),
    Claim("E25 templates", "e000025_template_rescoring.json", "n_templates", "12", "int", "", "twelve"),

    # E-000032 — the store-side closure, proved rather than sampled
    Claim("E32 canonical closure", "e000032_deletion_closure.json", "aggregate/canonical/fact_closure_mean/mean", "1.00", "dp2"),
    Claim("E32 duplicated closure", "e000032_deletion_closure.json", "aggregate/duplicated/fact_closure_mean/mean", "3.00", "dp2"),

    # E-000035 — the closure inverts, and the false-positive column
    Claim("E35 canonical trace closure", "e000035_deletion_disclosure.json", "aggregate/canonical/trace_closure_mean/mean", "3.00", "dp2"),
    Claim("E35 duplicated trace closure", "e000035_deletion_disclosure.json", "aggregate/duplicated/trace_closure_mean/mean", "1.00", "dp2"),
    Claim("E35 canonical false positives", "e000035_deletion_disclosure.json", "aggregate/canonical/false_positive_keys/mean", "0.00", "dp2"),
    Claim("E35 pods per seed", "e000035_deletion_disclosure.json", "aggregate/n_groups/mean", "100", "int"),
    Claim("E35 blanking closes it", "e000035_deletion_disclosure.json", "aggregate/blanked/channel_closed/mean", "1.0000", "dp4"),

    # E-000024 — rows versus weights
    Claim("E24 cells forced choice", "e000024_weights_vs_cells-seed0.json", "aggregate/cells/after/forced_choice/mean", "0.44", "dp2"),
    Claim("E24 cells perplexity", "e000024_weights_vs_cells-seed0.json", "aggregate/cells/ppl_after/mean", "42.9", "dp1"),
    Claim("E24 cells delete seconds", "e000024_weights_vs_cells-seed0.json", "aggregate/cells/delete_seconds/mean", "0.0008", "dp4"),

    # E-000030 — the certificate
    Claim("E30 swept questions", "e000030_deletion_certificate.json", "per_seed/0/n_queries_swept", "838", "int"),
    Claim("E30 soft-gate residual", "e000030_deletion_certificate.json", "gpt2/0/gpt2_soft/shred/residual", "1.390e-02", "exp3"),
)


_F1 = "fig1-top1-false-negative.svg"
_F2 = "fig2-swept-geometry.svg"
_F3 = "fig3-closure-inversion.svg"
_PDX = "pdx001/pdx001_payload_derived_index_audit.json"

FIGURE_CLAIMS: tuple[FigureClaim, ...] = (
    # Figure 1 — every bar height and every strip reading is a record value
    FigureClaim(_F1, "F1 value-gated posterior", _PDX,
                "policies/policy=value_gated_shred/mean_posterior_on_true_payload", "1.000000", "dp6"),
    FigureClaim(_F1, "F1 tombstone posterior", _PDX,
                "policies/policy=hnsw_tombstone/mean_posterior_on_true_payload", "0.391667", "dp6"),
    FigureClaim(_F1, "F1 certified posterior", _PDX,
                "policies/policy=revoke_unindex/mean_posterior_on_true_payload", "0.003906", "dp6"),
    FigureClaim(_F1, "F1 chance floor", _PDX,
                "policies/policy=hnsw_tombstone/chance_top1", "0.003906", "dp6"),
    FigureClaim(_F1, "F1 top-1 on the leaking policy", _PDX,
                "policies/policy=value_gated_shred/top1_recovery", "1.0000", "dp4"),
    FigureClaim(_F1, "F1 top-1 on the tombstone", _PDX,
                "policies/policy=hnsw_tombstone/top1_recovery", "0.0000", "dp4"),
    FigureClaim(_F1, "F1 candidates left", _PDX,
                "policies/policy=hnsw_tombstone/mean_candidates_remaining", "2.92", "dp2"),
    FigureClaim(_F1, "F1 search-space cut", _PDX,
                "policies/policy=hnsw_tombstone/search_space_reduction_factor", "88", "int"),
    FigureClaim(_F1, "F1 payload domain", _PDX,
                "policies/policy=hnsw_tombstone/payload_domain", "256", "int"),
    FigureClaim(_F1, "F1 validity floor", _PDX, "validity_control/top1_recovery", "1.0000", "dp4"),

    # Figure 2 — the swept geometry: the y-scale bug lived here
    FigureClaim(_F2, "F2 declared radius", "e000029_marker_geometry.json", "declared_radius", "0.35", "dp2"),
    FigureClaim(_F2, "F2 operational radius", "e000029_marker_geometry.json",
                "aggregate/operational_radius/mean", "0.90", "dp2"),
    FigureClaim(_F2, "F2 band 0.60", "e000029_marker_geometry.json",
                "per_checkpoint/*/band_accept/5", "1.0000", "dp4", "mean"),
    FigureClaim(_F2, "F2 band 0.70", "e000029_marker_geometry.json",
                "per_checkpoint/*/band_accept/6", "0.9999", "dp4", "mean"),
    FigureClaim(_F2, "F2 band 0.80", "e000029_marker_geometry.json",
                "per_checkpoint/*/band_accept/7", "0.2191", "dp4", "mean"),
    FigureClaim(_F2, "F2 band 0.80 min", "e000029_marker_geometry.json",
                "per_checkpoint/*/band_accept/7", "0.0953", "dp4", "min"),
    FigureClaim(_F2, "F2 band 0.80 max", "e000029_marker_geometry.json",
                "per_checkpoint/*/band_accept/7", "0.4014", "dp4", "max"),
    FigureClaim(_F2, "F2 band 0.90", "e000029_marker_geometry.json",
                "per_checkpoint/*/band_accept/8", "0.0000", "dp4", "mean"),
    FigureClaim(_F2, "F2 annulus accepted", "e000029_marker_geometry.json",
                "pooled/annulus/0", "2,199,996", "thousands"),
    FigureClaim(_F2, "F2 annulus n", "e000029_marker_geometry.json",
                "pooled_intervals/annulus/n", "2,200,000", "thousands"),
    FigureClaim(_F2, "F2 markers per band", "e000029_marker_geometry.json", "n_per_band", "20,000", "thousands"),
    FigureClaim(_F2, "F2 checkpoints", "e000029_marker_geometry.json", "n_checkpoints", "11", "int"),

    # Figure 3 — the inversion
    FigureClaim(_F3, "F3 canonical fact closure", "e000032_deletion_closure.json",
                "aggregate/canonical/fact_closure_mean/mean", "1.00", "dp2"),
    FigureClaim(_F3, "F3 duplicated fact closure", "e000032_deletion_closure.json",
                "aggregate/duplicated/fact_closure_mean/mean", "3.00", "dp2"),
    FigureClaim(_F3, "F3 canonical trace closure", "e000035_deletion_disclosure.json",
                "aggregate/canonical/trace_closure_mean/mean", "3.00", "dp2"),
    FigureClaim(_F3, "F3 duplicated trace closure", "e000035_deletion_disclosure.json",
                "aggregate/duplicated/trace_closure_mean/mean", "1.00", "dp2"),
    FigureClaim(_F3, "F3 canonical disclosure", "e000035_deletion_disclosure.json",
                "aggregate/canonical/deleted_key_disclosed/mean", "1.0000", "dp4"),
    FigureClaim(_F3, "F3 duplicated disclosure", "e000035_deletion_disclosure.json",
                "aggregate/duplicated/deleted_key_disclosed/mean", "0.0000", "dp4"),
    FigureClaim(_F3, "F3 key space", "e000035_deletion_disclosure.json",
                "aggregate/duplicated/candidate_keys_mean/mean", "1,536", "thousands"),
    FigureClaim(_F3, "F3 pods per seed", "e000035_deletion_disclosure.json",
                "aggregate/n_groups/mean", "100", "int"),
)


SCOPE_CLAIMS: tuple[ScopeClaim, ...] = (
    ScopeClaim(
        "E28's attack pooled five seeds",
        "e000028_key_channel.json", "seeds", [0, 1, 2, 3, 4],
        "Five seeds",
    ),
    ScopeClaim(
        "E32, E35 and E25 are three seeds each",
        "e000035_deletion_disclosure.json", "seeds", [0, 1, 2],
        "Three seeds",
    ),
    ScopeClaim(
        "PDX-001 ran its own validity floor before reporting at-chance readings",
        "pdx001/pdx001_payload_derived_index_audit.json", "attack_validity_floor_met", True,
        "Validity floor",
    ),
    ScopeClaim(
        "PDX-001 runs no published system",
        "pdx001/pdx001_payload_derived_index_audit.json", "instrument_self_consistent", True,
        "No published system is run here",
    ),
    ScopeClaim(
        "E24 is a single seed",
        "e000024_weights_vs_cells-seed0.json", "seeds", [0],
        "one seed",
    ),
    ScopeClaim(
        "E30's frozen-LM arm is a single seed",
        "e000030_deletion_certificate.json", "gpt2/0/seed", 0,
        "one seed",
    ),
    ScopeClaim(
        "E30's frozen-LM arm holds 400 cells, not the synthetic arm's count",
        "e000030_deletion_certificate.json", "gpt2/0/n_cells", 400,
        "400 cells",
    ),
)


def occurrences(printed: str, text: str) -> int:
    """Count the printed figure in the paper as a *standalone* number.

    A plain substring test is far too weak here: `"0.0"` is inside `0.0040`, `"12"` is inside
    `128.02`, and `"4"` is inside almost everything. The match must not be *extended* into a longer
    number on either side -- but a trailing full stop or comma is punctuation, not a digit group, so
    only `.` or `,` followed by a digit disqualifies. Without this, ABSENT means nothing: a claim
    could pass its presence test on a number it has no relation to.
    """
    if not any(ch.isdigit() for ch in printed):
        return len(re.findall(rf"\b{re.escape(printed)}\b", text, flags=re.IGNORECASE))
    return len(re.findall(rf"(?<![\d.,]){re.escape(printed)}(?!\d)(?![.,]\d)", text))


# A figure that appears this often is being matched by coincidence somewhere, so its presence test
# stops discriminating. We report those rather than dropping them: the record comparison still holds.
_PRESENCE_NOISE_FLOOR = 8


def _load(name: str):
    p = RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _figure_text(name: str) -> str | None:
    """The visible text of one figure: its `<text>` runs, its title and its description.

    Read as text rather than parsed, because the numbers we are checking are what a reader sees --
    a value baked into a `d=` path attribute is *geometry*, and the registry deliberately does not
    claim to check geometry. Figure 2's y-scale bug was in a path and no string check would have
    caught it; a render did.
    """
    p = FIGURES / name
    if not p.exists():
        return None
    return " ".join(re.findall(r">([^<>]*)<", p.read_text(encoding="utf-8")))


def _check_value_claim(c, doc, where: str, text: str) -> dict:
    """The shared body: resolve, reduce, render, compare, then look for it in `text`."""
    row: dict = {"label": c.label}
    value, err = _resolve(doc, c.path)
    if err:
        return {**row, "status": "PATH", "detail": f"{c.path}: {err}"}
    if c.reduce:
        if not isinstance(value, list) or not value:
            return {**row, "status": "PATH",
                    "detail": f"{c.path}: reducer {c.reduce!r} needs a non-empty list"}
        value = _reduce(value, c.reduce)
    try:
        rendered = _fmt(float(value), c.rule)
    except (TypeError, ValueError) as exc:
        return {**row, "status": "PATH", "detail": f"{c.path}: {exc}"}
    if rendered != c.printed:
        return {**row, "status": "MISMATCH",
                "detail": f"record {value!r} renders {rendered!r} under {c.rule}, "
                          f"{where} prints {c.printed!r}"}
    hits = occurrences(c.sought, text)
    if hits == 0:
        return {**row, "status": "ABSENT", "occurrences": 0,
                "detail": f"{c.sought!r} matches the record but does not appear in {where}"}
    weak = hits > _PRESENCE_NOISE_FLOOR
    spelled = "" if not getattr(c, "appears_as", "") else f" (spelled {c.appears_as!r})"
    return {**row, "status": "OK", "occurrences": hits, "presence_discriminating": not weak,
            "detail": f"{c.printed}{spelled} = {c.path}"
                      + (f"  [presence test weak: {hits} matches]" if weak else "")}


def check(
    paper_text: str | None = None,
    claims: tuple[Claim, ...] | None = None,
    scope_claims: tuple[ScopeClaim, ...] | None = None,
    figure_claims: tuple[FigureClaim, ...] | None = None,
    figure_texts: dict[str, str] | None = None,
) -> dict:
    """Check the paper, and the figures, against the records.

    Every registry is a parameter so that the validity floor can be exercised: a checker that has
    never been shown to fail is not evidence that the paper agrees with the records, it is only
    evidence that it printed ``0 failing``.
    """
    text = paper_text if paper_text is not None else PAPER.read_text(encoding="utf-8")
    claims = CLAIMS if claims is None else claims
    scope_claims = SCOPE_CLAIMS if scope_claims is None else scope_claims
    figure_claims = FIGURE_CLAIMS if figure_claims is None else figure_claims
    cache: dict[str, object] = {}
    figtext: dict[str, str | None] = dict(figure_texts or {})
    rows = []

    for c in claims:
        doc = cache.setdefault(c.record, _load(c.record))
        if doc is None:
            rows.append({"label": c.label, "status": "PATH", "detail": f"no record {c.record}"})
            continue
        rows.append(_check_value_claim(c, doc, "the paper", text))

    figure_rows = []
    for c in figure_claims:
        doc = cache.setdefault(c.record, _load(c.record))
        if doc is None:
            figure_rows.append({"label": c.label, "status": "PATH", "detail": f"no record {c.record}"})
            continue
        if c.figure not in figtext:
            figtext[c.figure] = _figure_text(c.figure)
        drawn = figtext[c.figure]
        if drawn is None:
            figure_rows.append({"label": c.label, "status": "PATH", "detail": f"no figure {c.figure}"})
            continue
        figure_rows.append({**_check_value_claim(c, doc, c.figure, drawn), "figure": c.figure})

    scope_rows = []
    for s in scope_claims:
        doc = cache.setdefault(s.record, _load(s.record))
        if doc is None:
            scope_rows.append({"label": s.label, "status": "PATH", "detail": f"no record {s.record}"})
            continue
        value, err = _resolve(doc, s.path)
        if err:
            scope_rows.append({"label": s.label, "status": "PATH", "detail": f"{s.path}: {err}"})
            continue
        if value != s.expect:
            scope_rows.append({"label": s.label, "status": "MISMATCH",
                               "detail": f"{s.path} is {value!r}, registry expects {s.expect!r}"})
            continue
        if s.must_appear.lower() not in text.lower():
            scope_rows.append({"label": s.label, "status": "UNDISCLOSED",
                               "detail": f"record confirms it, but the paper never says {s.must_appear!r}"})
            continue
        scope_rows.append({"label": s.label, "status": "OK", "detail": s.must_appear})

    failures = [r for r in rows + figure_rows + scope_rows if r["status"] != "OK"]
    weak = [r["label"] for r in rows + figure_rows if r.get("presence_discriminating") is False]
    return {
        "registered_figures": len(claims),
        "registered_drawn_figures": len(figure_claims),
        "registered_scope_claims": len(scope_claims),
        "figures": rows,
        "drawn": figure_rows,
        "scope": scope_rows,
        "failures": failures,
        "clean": not failures,
        "weak_presence_tests": weak,
        "not_claimed": (
            "Partial by construction. A figure absent from this registry is unchecked, not verified; "
            "`coverage()` reports what is bound. The registry checks that the paper agrees with the "
            "records, never that the records are right. And a round figure that recurs across the "
            "paper (1.0000, 256) has a presence test that no longer discriminates: its record "
            "comparison still holds, but its ABSENT half is reported weak rather than counted as a "
            "check. On the drawn figures it checks the numbers a reader sees, never the geometry "
            "that places them: a bar drawn at the wrong height with the right label passes here, "
            "and only a render catches it."
        ),
    }


def coverage() -> dict:
    return {
        "records_bound": sorted(
            {c.record for c in CLAIMS}
            | {c.record for c in FIGURE_CLAIMS}
            | {s.record for s in SCOPE_CLAIMS}
        ),
        "figures_bound": sorted({c.figure for c in FIGURE_CLAIMS}),
        "figures": len(CLAIMS),
        "drawn_figures": len(FIGURE_CLAIMS),
        "scope_claims": len(SCOPE_CLAIMS),
    }


def main() -> None:
    report = check()
    for r in report["figures"] + report["drawn"] + report["scope"]:
        mark = " ok " if r["status"] == "OK" else r["status"]
        print(f"[{mark:>11}] {r['label']:<52} {r['detail']}")
    print()
    print(f"{report['registered_figures']} figures in the prose + "
          f"{report['registered_drawn_figures']} in the drawn figures + "
          f"{report['registered_scope_claims']} scope claims; "
          f"{len(report['failures'])} failing")
    weak = report["weak_presence_tests"]
    if weak:
        print(f"{len(weak)} figures are round enough to recur; their presence test does not "
              f"discriminate and only the record comparison counts for them.")
    if not report["clean"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
