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
    near: str = ""         # a phrase the figure's line must contain, when several quantities
                           # render identically -- see the note on `100` below

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
class VerdictClaim:
    """A categorical cell — CERTIFIED, yes, no — bound to the record boolean behind it.

    The registry binds numbers, and a table's verdict column carries more weight than any of them:
    a wrong decimal misstates a magnitude, a wrong ``CERTIFIED`` misstates whether the paper's
    central claim holds at all. These are checked against the specific table *row*, not against the
    document, so a verdict cannot pass by appearing somewhere else on the page.
    """

    label: str
    record: str
    path: str
    expect: object      # what the record must hold for the paper's cell to be honest
    row_prefix: str     # identifies one table row, e.g. "| frozen GPT-2, hard gate | SHRED |"
    says: str           # the verdict text that row must carry


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
    Claim("E28 targets per seed", "e000028_key_channel.json", "n_targets",
          "100", "int", "", "", "targets each"),
    # §2's exact intervals: the point estimates alone asked the reader to take the finding on trust
    Claim("E28 active interval lower", "e000028_key_channel.json", "pooled_vs_chance/active/lower", "0.9926", "dp4"),
    Claim("E28 shred interval lower", "e000028_key_channel.json", "pooled_vs_chance/shred/lower", "0.9926", "dp4"),
    Claim("E28 revoke interval lower", "e000028_key_channel.json", "pooled_vs_chance/revoke/lower", "0.0005", "dp4"),
    Claim("E28 revoke interval upper", "e000028_key_channel.json", "pooled_vs_chance/revoke/upper", "0.0144", "dp4"),

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

    # NOV-004 — the sweep's own coverage. This went stale the moment the base branch grew by six
    # experiments, and nothing caught it: the paper's "100" was passing its presence test on
    # E-000035's pod count, a different quantity that renders the same way.
    Claim("NOV004 experiments scanned", "nov004/nov004_instrument_audit.json", "files_scanned",
          "106", "int", "", "", "mechanical sweep"),
    Claim("NOV004 class A confirmed", "nov004/nov004_instrument_audit.json",
          "class_a_confirmed_count", "3", "int", "", "three", "mechanical sweep"),

    # E-000033 — §13(b). Bound because the number that matters is the one that FAILED: the
    # control the experiment set for itself, which is why its two data columns are not reported.
    Claim("E33 control read rate", "e000033_retrieval_closure.json",
          "aggregate/control/read_before_deletion/mean", "0.0467", "dp4"),
    Claim("E33 control worst seed", "e000033_retrieval_closure.json",
          "aggregate/control/read_before_deletion/min", "0.0250", "dp4"),
    Claim("E33 chunks", "e000033_retrieval_closure.json", "per_seed/0/n_chunks", "600", "int"),
    Claim("E33 facts", "e000033_retrieval_closure.json", "n_facts", "150", "int"),

    # E-000021 — the published false-accept rate the paper contrasts against
    Claim("E21 false-accept rate", "e000021_gate_error_rates.json", "totals/false_accept_rate", "8.49e-04", "exp2"),
    Claim("E21 n", "e000021_gate_error_rates.json", "totals/ci_false_accept/n", "2,200,000", "thousands"),

    # E-000035 — the deletion oracle
    Claim("E35 canonical disclosed", "e000035_deletion_disclosure.json", "aggregate/canonical/deleted_key_disclosed/mean", "1.0000", "dp4"),
    Claim("E35 canonical unique", "e000035_deletion_disclosure.json", "aggregate/canonical/uniquely_identified/mean", "1.0000", "dp4"),
    Claim("E35 duplicated disclosed", "e000035_deletion_disclosure.json", "aggregate/duplicated/deleted_key_disclosed/mean", "0.0000", "dp4"),
    Claim("E35 key space", "e000035_deletion_disclosure.json", "aggregate/duplicated/candidate_keys_mean/mean", "1,536", "thousands"),
    Claim("E35 canonical candidate keys", "e000035_deletion_disclosure.json",
          "aggregate/canonical/candidate_keys_mean/mean", "1", "int"),

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
    # E-000032 runs 25 pods per seed, E-000035 runs 100; the paper combined them under one heading
    Claim("E32 pods per seed", "e000032_deletion_closure.json", "n_groups", "25", "int"),

    # E-000029 — the reproduction of the published false-accept rate on its own distribution
    Claim("E29 rejection-sampled accept rate", "e000029_marker_geometry.json",
          "aggregate/rejection_sampled_accept_rate/mean", "8.550e-04", "exp3"),

    # E-000035 — the closure inverts, and the false-positive column
    Claim("E35 canonical trace closure", "e000035_deletion_disclosure.json", "aggregate/canonical/trace_closure_mean/mean", "3.00", "dp2"),
    Claim("E35 duplicated trace closure", "e000035_deletion_disclosure.json", "aggregate/duplicated/trace_closure_mean/mean", "1.00", "dp2"),
    Claim("E35 canonical false positives", "e000035_deletion_disclosure.json", "aggregate/canonical/false_positive_keys/mean", "0.00", "dp2"),
    # pinned with `near`: five different quantities in this paper render as "100"
    Claim("E35 pods per seed", "e000035_deletion_disclosure.json", "aggregate/n_groups/mean",
          "100", "int", "", "", "pod"),
    Claim("E35 blanking closes it", "e000035_deletion_disclosure.json", "aggregate/blanked/channel_closed/mean", "1.0000", "dp4"),

    # E-000024 — rows versus weights. The whole head-to-head table is bound, because leaving it
    # unbound is how "137" (a mean rank read as seconds), "311" and "8.49e+06" got into print.
    Claim("E24 cells answered before", "e000024_weights_vs_cells-seed0.json",
          "aggregate/cells/before/direct_acc/mean", "0.92", "dp2"),
    # both LoRA arms share ONE before-measurement; the table's two 0.96 cells are one number
    Claim("E24 weights answered before", "e000024_weights_vs_cells-seed0.json",
          "aggregate/weights/before/direct_acc/mean", "0.96", "dp2"),
    Claim("E24 cells answers after", "e000024_weights_vs_cells-seed0.json", "aggregate/cells/after/direct_acc/mean", "0.02", "dp2"),
    Claim("E24 ga answers after", "e000024_weights_vs_cells-seed0.json", "aggregate/ga/after/direct_acc/mean", "0.00", "dp2"),
    Claim("E24 relabel answers after", "e000024_weights_vs_cells-seed0.json", "aggregate/relabel/after/direct_acc/mean", "0.02", "dp2"),
    Claim("E24 cells forced choice", "e000024_weights_vs_cells-seed0.json", "aggregate/cells/after/forced_choice/mean", "0.44", "dp2"),
    Claim("E24 ga forced choice", "e000024_weights_vs_cells-seed0.json", "aggregate/ga/after/forced_choice/mean", "0.78", "dp2"),
    Claim("E24 relabel forced choice", "e000024_weights_vs_cells-seed0.json", "aggregate/relabel/after/forced_choice/mean", "1.00", "dp2"),
    Claim("E24 cells perplexity", "e000024_weights_vs_cells-seed0.json", "aggregate/cells/ppl_after/mean", "42.9", "dp1"),
    Claim("E24 ga perplexity", "e000024_weights_vs_cells-seed0.json", "aggregate/ga/ppl_after/mean", "6.19e+09", "exp2"),
    Claim("E24 relabel perplexity", "e000024_weights_vs_cells-seed0.json", "aggregate/relabel/ppl_after/mean", "6.39e+06", "exp2"),
    Claim("E24 cells delete seconds", "e000024_weights_vs_cells-seed0.json", "aggregate/cells/delete_seconds/mean", "0.0008", "dp4"),
    Claim("E24 ga delete seconds", "e000024_weights_vs_cells-seed0.json", "aggregate/ga/delete_seconds/mean", "129", "int"),
    Claim("E24 relabel delete seconds", "e000024_weights_vs_cells-seed0.json", "aggregate/relabel/delete_seconds/mean", "335", "int"),
    # the "parameters changed: 0" cell. The record's nearest quantity is the L2 norm of the weight
    # delta, which is exactly 0.0 -- zero movement, hence zero parameters changed.
    Claim("E24 cells weight delta", "e000024_weights_vs_cells-seed0.json",
          "aggregate/cells/weight_delta_l2/mean", "0", "int"),
    Claim("E24 cells", "e000024_weights_vs_cells-seed0.json", "n_cells", "400", "int"),
    Claim("E24 deletion targets", "e000024_weights_vs_cells-seed0.json", "n_targets", "50", "int"),
    # the prose said "76%" where the record says 0.72 -- the fourth prose/record disagreement,
    # found by reading the paper rather than by this check, because it was not registered
    Claim("E24 relabel relearning recovery", "e000024_weights_vs_cells-seed0.json",
          "aggregate/relabel/relearn/heldout_acc/mean", "0.72", "dp2"),
    Claim("E24 gradient-ascent relearning recovery", "e000024_weights_vs_cells-seed0.json",
          "aggregate/ga/relearn/heldout_acc/mean", "0.48", "dp2"),
    Claim("E24 cells relearning recovery", "e000024_weights_vs_cells-seed0.json",
          "aggregate/cells/relearn/heldout_acc/mean", "0.00", "dp2"),

    # E-000030 — the certificate
    Claim("E30 swept questions", "e000030_deletion_certificate.json", "per_seed/0/n_queries_swept", "838", "int"),
    Claim("E30 targets", "e000030_deletion_certificate.json", "n_targets", "3", "int"),
    Claim("E30 synthetic cells", "e000030_deletion_certificate.json", "per_seed/0/n_cells", "1000", "int"),
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
    FigureClaim(_F3, "F3 unreachability pods per seed", "e000032_deletion_closure.json",
                "n_groups", "25", "int"),
)


_E30 = "e000030_deletion_certificate.json"

VERDICT_CLAIMS: tuple[VerdictClaim, ...] = (
    # §4 — the certificate table. Seven cells, seven record booleans.
    VerdictClaim("E30 synthetic REVOKE not interface-certified", _E30,
                 "per_seed/0/revoke/interface_certified", False,
                 "| synthetic | REVOKE |", "no (certified on 838 swept questions)"),
    VerdictClaim("E30 synthetic REVOKE certified on the swept questions", _E30,
                 "per_seed/0/revoke/outputs_certified", True,
                 "| synthetic | REVOKE |", "certified on 838 swept questions"),
    VerdictClaim("E30 synthetic SHRED not certified", _E30,
                 "per_seed/0/shred/outputs_certified", False,
                 "| synthetic | SHRED |", " no "),
    VerdictClaim("E30 synthetic DELETE is structural", _E30,
                 "per_seed/0/delete/structural", True,
                 "| synthetic | DELETE |", "yes, structurally"),
    VerdictClaim("E30 soft-gate REVOKE certified", _E30,
                 "gpt2/0/gpt2_soft/revoke/interface_certified", True,
                 "| frozen GPT-2, soft gate | REVOKE |", "**CERTIFIED**"),
    VerdictClaim("E30 soft-gate SHRED not certified", _E30,
                 "gpt2/0/gpt2_soft/shred/interface_certified", False,
                 "| frozen GPT-2, soft gate | SHRED |", " no "),
    VerdictClaim("E30 hard-gate REVOKE certified", _E30,
                 "gpt2/0/gpt2_hard/revoke/interface_certified", True,
                 "| frozen GPT-2, hard gate | REVOKE |", "**CERTIFIED**"),
    VerdictClaim("E30 hard-gate SHRED certified", _E30,
                 "gpt2/0/gpt2_hard/shred/interface_certified", True,
                 "| frozen GPT-2, hard gate | SHRED |", "**CERTIFIED**"),
    # the guard that can void a certificate must have held wherever one was issued
    VerdictClaim("E30 hard-gate SHRED mediation guard held", _E30,
                 "gpt2/0/gpt2_hard/shred/mediation_consistent", True,
                 "| frozen GPT-2, hard gate | SHRED |", "**CERTIFIED**"),

    # §3 — PDX-001's certified column, addressed by policy name
    VerdictClaim("PDX value-gated not certified", _PDX,
                 "policies/policy=value_gated_shred/certified", False,
                 "| gated value, ungated derived key |", " no |"),
    VerdictClaim("PDX tombstone not certified", _PDX,
                 "policies/policy=hnsw_tombstone/certified", False,
                 "| tombstoned index node, edges kept |", " no |"),
    VerdictClaim("PDX codebook not certified", _PDX,
                 "policies/policy=codebook_key/certified", False,
                 "| cleared value, codebook key kept |", " no |"),
    VerdictClaim("PDX unindexed certified", _PDX,
                 "policies/policy=revoke_unindex/certified", True,
                 "| row removed from addressable set |", "**yes**"),
    VerdictClaim("PDX gate-all certified", _PDX,
                 "policies/policy=gate_all_derived/certified", True,
                 "| every derived quantity gated |", "**yes**"),
    # the paper says attack and certificate "agree on every policy" -- that is a per-row claim
    VerdictClaim("PDX attack and certificate agree, tombstone", _PDX,
                 "policies/policy=hnsw_tombstone/certificate_matches_attack", True,
                 "| tombstoned index node, edges kept |", "0.391667"),
    VerdictClaim("PDX attack and certificate agree, unindexed", _PDX,
                 "policies/policy=revoke_unindex/certificate_matches_attack", True,
                 "| row removed from addressable set |", "**yes**"),

    # §2 — whether each arm's interval covers chance is the finding, so it is a verdict too
    VerdictClaim("E28 shred's interval excludes chance", "e000028_key_channel.json",
                 "pooled_vs_chance/shred/contains_chance", False,
                 "| **shred** |", "**[0.9926, 1.0000]**"),
    VerdictClaim("E28 active's interval excludes chance", "e000028_key_channel.json",
                 "pooled_vs_chance/active/contains_chance", False,
                 "| active (validity control) |", "[0.9926, 1.0000]"),
    VerdictClaim("E28 revoke's interval contains chance", "e000028_key_channel.json",
                 "pooled_vs_chance/revoke/contains_chance", True,
                 "| revoke / delete |", "[0.0005, 0.0144]"),
)


SCOPE_CLAIMS: tuple[ScopeClaim, ...] = (
    ScopeClaim(
        "E28's attack pooled five seeds",
        "e000028_key_channel.json", "seeds", [0, 1, 2, 3, 4],
        "Five seeds",
    ),
    ScopeClaim(
        "the four-attack battery ran on seeds held out from selection, disjoint from E28's",
        "e000019_fresh_seed_chance.json", "config/seeds", [5, 6, 7],
        "seeds 5–7",
    ),
    ScopeClaim(
        "E32 runs 25 pods per seed, not E35's 100",
        "e000032_deletion_closure.json", "n_groups", 25,
        "25 pods",
    ),
    # bound to the note itself, not to some unrelated field that happens to be stable: if the record's
    # provenance changes, the paper's caveat has to be re-read rather than silently kept
    ScopeClaim(
        "E25's checkpoints cannot all be traced back to E-000020",
        "e000025_template_rescoring.json", "provenance_note",
        "a forced re-run of E-000020 overwrote its seed-0 and seed-1 checkpoints after that record "
        "was written; only seed 2 still matches the SHA-256 recorded there. The SHA of every "
        "checkpoint scored here is in per_seed.",
        "Provenance caveat",
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
    near = getattr(c, "near", "")
    if near:
        # Several distinct quantities can render to the same string. `100` is five different
        # numbers in this paper -- targets per seed, pods per seed, and the count of experiments
        # swept -- and binding one of them let another go stale unnoticed while the check stayed
        # green. A claim that names its context is checked against the lines carrying it.
        # scope to the paragraph, not the line: prose wraps, and "found three" routinely lands on
        # the line after the phrase that identifies which sweep is meant
        scoped = "\n\n".join(b for b in re.split(r"\n\s*\n", text) if near in b)
        if not scoped:
            return {**row, "status": "ABSENT", "occurrences": 0,
                    "detail": f"no paragraph in {where} contains the context {near!r}"}
        text = scoped
    hits = occurrences(c.sought, text)
    if hits == 0:
        return {**row, "status": "ABSENT", "occurrences": 0,
                "detail": f"{c.sought!r} matches the record but does not appear in {where}"
                          + (f" near {near!r}" if near else "")}
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
    verdict_claims: tuple[VerdictClaim, ...] | None = None,
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
    verdict_claims = VERDICT_CLAIMS if verdict_claims is None else verdict_claims
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

    verdict_rows = []
    lines = text.splitlines()
    for v in verdict_claims:
        doc = cache.setdefault(v.record, _load(v.record))
        if doc is None:
            verdict_rows.append({"label": v.label, "status": "PATH", "detail": f"no record {v.record}"})
            continue
        value, err = _resolve(doc, v.path)
        if err:
            verdict_rows.append({"label": v.label, "status": "PATH", "detail": f"{v.path}: {err}"})
            continue
        if value != v.expect:
            verdict_rows.append({"label": v.label, "status": "MISMATCH",
                                 "detail": f"{v.path} is {value!r}, the paper's cell needs {v.expect!r}"})
            continue
        matching = [ln for ln in lines if ln.startswith(v.row_prefix)]
        if len(matching) != 1:
            verdict_rows.append({"label": v.label, "status": "ABSENT",
                                 "detail": f"{len(matching)} rows start {v.row_prefix!r}, need exactly 1"})
            continue
        if v.says not in matching[0]:
            verdict_rows.append({"label": v.label, "status": "CONTRADICTED",
                                 "detail": f"record says {v.expect!r} but the row does not say {v.says!r}: "
                                           f"{matching[0].strip()}"})
            continue
        verdict_rows.append({"label": v.label, "status": "OK",
                             "detail": f"{v.says.strip()} = {v.path} is {v.expect!r}"})

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

    failures = [r for r in rows + figure_rows + verdict_rows + scope_rows if r["status"] != "OK"]
    weak = [r["label"] for r in rows + figure_rows if r.get("presence_discriminating") is False]
    return {
        "registered_figures": len(claims),
        "registered_drawn_figures": len(figure_claims),
        "registered_verdicts": len(verdict_claims),
        "registered_scope_claims": len(scope_claims),
        "figures": rows,
        "drawn": figure_rows,
        "verdicts": verdict_rows,
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


# --------------------------------------------------------------------------- coverage
# Tokens that look numeric but are not measurements. Each carries the reason it is excluded, so
# the exclusion list can be argued with rather than trusted. Anything NOT here and NOT registered
# is reported as unbound.
_NOT_A_MEASUREMENT: tuple[tuple[str, str], ...] = (
    (r"E-\d{6}",              "experiment identifier"),
    (r"§\s?\d+(?:\.\d+)*(?:\s?[–-]\s?\d+)?", "section reference or range"),
    (r"(?m)^#{1,4} \d+(?:\.\d+)?\.", "markdown heading number"),
    (r"Part [IVX]+, \s*[–-]?\s*\d+", "section range in the roadmap table"),
    (r"Revision \d\b|revision\b",  "draft revision number"),
    (r"\bseeds \d\s?[–-]\s?\d\b",  "seed range, carried by a scope claim"),
    (r"\bseed-\d\b|\bseed \d\b", "seed name"),
    (r"\(Figure \d\)",        "inline figure reference"),
    (r"\d{4}-\d{2}-\d{2}",    "ISO date — must precede the bare-year rule"),
    (r"\b(?:19|20)\d{2}\b",   "calendar year"),
    (r"(?m)^\d+\. ",          "numbered list item in §11"),
    (r"\*Figure \d+ —",       "figure caption number"),
    (r"\b1 in 256\b",         "chance stated in words; the figure itself is bound as 0.0039"),
    (r"\b\d+ (?:figures printed in this text|more printed inside|\*\*verdicts\*\*|verdicts,"
     r"|\*\*scope claims\*\*|\*\*scope\s+claims\*\*|figures in the prose|in the drawn figures|scope\s+claims)",
                              "the registry's own size: checked by check_self_description, not bound to a record"),
    (r"PDX-\d+|NOV-\d+",      "experiment identifier"),
    (r"\bF[123]\b",           "sub-question label"),
    (r"GPT-2|top-[15]|rank-\d", "term of art containing a digit"),
    (r"\b124M\b|\b7B\b",      "model size named in prose, not read from a record"),
    (r"\brev \d\b",           "revision number"),
    (r"figures/fig\d[^)]*|\bfig\d\b", "figure filename"),
    (r"\b41-agent\b",         "provenance of the literature workflow, not a measurement"),
    (r"\b2,359,296\b|\b2,370,692\b", "LoRA and adapter parameter counts: configuration, not in E-000024"),
    (r"\b127\.5\b",           "analytic chance rank, 255/2, not a recorded value"),
    (r"\b0\.55\b",            "derived in the text: 0.90 minus 0.35"),
    # the paper's convention: an asserted figure is plain or bold, a figure being *talked about*
    # (an example, or an error being reported) is in backticks. Only the former is a claim.
    (r"`\d[\d,.e+-]*`",       "a numeral in backticks: quoted or illustrative, not asserted"),
    (r"\b76\b|\b128\.0\b|\b137\.2\b", "quoted in §9 as an error that was corrected"),
    (r"\b45 of 95\b",         "the coverage instrument's own first-run reading, recorded in ledger §31.51"),
    (r"\b0\.50\b",            "the stated chance level of a forced choice, by construction"),
    (r"\b95%",                "training stopping criterion, not a measured outcome"),
    (r"\b0\.6\b|\b0\.7\b|\b2\.0\b|\b0\.60\b|\b0\.70\b|\b0\.80\b|\b0\.90\b",
                              "shell radii: the sweep's x-axis, bound by the accept rates at them"),
    (r"\b1000\b",             "cell count of E-000030's synthetic arm, stated in prose"),
    (r"\bn = 1\b|\bk−1\b|\bk-1\b", "algebra in prose"),
)


MAKEFILE = Path("Makefile")

# An experiment the paper cites without a record is honest only if the paper says it has not been
# run. Each entry pairs the id with the words that must appear for the citation to stand.
# Empty, and deliberately kept rather than deleted: E-000033 lived here until it was run. The
# machinery matters more than the entries — an id declared unrun that acquires a record is a
# contradiction the reference pass reports, which is how §13(b) got corrected.
_CITED_BUT_UNRUN: dict[str, str] = {}


def check_references(paper_text: str | None = None) -> list[dict]:
    """The fifth kind of claim: that the paper's pointers point at something.

    Numbers, extent, coverage and verdicts are all checked. None of them looks at whether `make
    certify` is a target, whether a cited E-id has a record, whether a §-reference resolves, or
    whether a quoted repository path exists. §0 promises every number is "reproducible by the `make`
    target named beside it" -- a target that does not exist makes that promise false in a way no
    amount of correct arithmetic would reveal.
    """
    text = paper_text if paper_text is not None else PAPER.read_text(encoding="utf-8")
    rows: list[dict] = []

    # (a) every backticked `make X` names a real target
    makefile = MAKEFILE.read_text(encoding="utf-8") if MAKEFILE.exists() else ""
    targets = set(re.findall(r"^([a-z][a-z0-9_-]*):", makefile, flags=re.M))
    cited: set[str] = set()
    for m in re.finditer(r"`make ([a-z0-9_ -]+)`", text):
        cited.update(m.group(1).split())
    for t in sorted(cited):
        rows.append({"kind": "make target", "name": t,
                     "status": "OK" if t in targets else "MISSING",
                     "detail": "in the Makefile" if t in targets else "no such Makefile target"})

    # (b) every cited experiment has a record, or is declared unrun in the text
    records = {p.name for p in RESULTS.glob("**/*.json")}
    for eid in sorted(set(re.findall(r"E-\d{6}", text))):
        stem = "e" + eid.split("-")[1]
        has_record = any(r.startswith(stem) for r in records)
        if has_record and eid in _CITED_BUT_UNRUN:
            # the direction this originally missed: the id was declared unrun, and then somebody ran
            # it. Record and paper now contradict each other, and the "record present" branch would
            # have reported OK while the paper still said "has never been run".
            rows.append({"kind": "experiment", "name": eid, "status": "CONTRADICTED",
                         "detail": "declared unrun in the paper, but a record now exists"})
        elif has_record:
            rows.append({"kind": "experiment", "name": eid, "status": "OK", "detail": "record present"})
        elif eid in _CITED_BUT_UNRUN and _CITED_BUT_UNRUN[eid] in text:
            rows.append({"kind": "experiment", "name": eid, "status": "OK",
                         "detail": f"no record, and the paper says so ({_CITED_BUT_UNRUN[eid]!r})"})
        elif eid in _CITED_BUT_UNRUN:
            rows.append({"kind": "experiment", "name": eid, "status": "UNDISCLOSED",
                         "detail": "cited with no record, and the paper no longer says it is unrun"})
        else:
            rows.append({"kind": "experiment", "name": eid, "status": "MISSING",
                         "detail": "cited, but no record and not declared unrun"})

    # (c) every §N points at a section that exists (§31.x refers to the ledger, not this paper)
    sections = set(re.findall(r"^#{1,3} (\d+)\.", text, flags=re.M))
    for ref in sorted(set(re.findall(r"§(\d+)(?!\.\d)", text)), key=int):
        ok = ref in sections
        rows.append({"kind": "section", "name": f"§{ref}",
                     "status": "OK" if ok else "MISSING",
                     "detail": "section exists" if ok else "no such section in this paper"})

    # (d) every quoted repository path, and every embedded figure, exists on disk
    for path in sorted(set(re.findall(r"`((?:so|docs)/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*/?)`", text))):
        ok = Path(path).exists()
        rows.append({"kind": "path", "name": path, "status": "OK" if ok else "MISSING",
                     "detail": "exists" if ok else "no such file or directory"})
    for fig in sorted(set(re.findall(r"\(figures/([^)]+\.svg)\)", text))):
        ok = (FIGURES / fig).exists()
        rows.append({"kind": "figure", "name": fig, "status": "OK" if ok else "MISSING",
                     "detail": "exists" if ok else "embedded but not on disk"})

    return rows


EXPERIMENTS = Path("so/experiments")

# Records whose value is a count over the repository itself, and therefore goes stale when the
# repository changes rather than when anyone edits anything. Each pairs a record field with the
# live quantity it should equal.
_REPO_DERIVED: tuple[tuple[str, str, str], ...] = (
    ("nov004/nov004_instrument_audit.json", "files_scanned", "recorded experiment files"),
)


def check_record_freshness() -> list[dict]:
    """Is the *record* still true of the repository it counted?

    The sixth kind of staleness was a paper claim invalidated by the repository growing. Binding it
    to NOV-004's record fixed the paper, and left the same hole one level down: if the repository
    grows again and nobody re-runs the sweep, the record and the paper agree with each other and
    disagree with the world. Everything upstream stays green.

    So this compares the record's own count against a live count made by the same rule the sweep
    uses. A registry that only ever compares two documents to each other cannot notice this.
    """
    pattern = re.compile(r"^e\d{6}[a-z]?_.*\.py$")
    live = sorted(p.name for p in EXPERIMENTS.glob("*.py") if pattern.match(p.name))
    rows = []
    for record, field, what in _REPO_DERIVED:
        doc = _load(record)
        if doc is None:
            rows.append({"record": record, "status": "PATH", "detail": f"no record {record}"})
            continue
        value, err = _resolve(doc, field)
        if err:
            rows.append({"record": record, "status": "PATH", "detail": f"{field}: {err}"})
            continue
        if value != len(live):
            rows.append({"record": record, "status": "STALE",
                         "detail": f"{field} says {value}, the repository now holds {len(live)} "
                                   f"{what} — re-run the sweep"})
            continue
        rows.append({"record": record, "status": "OK",
                     "detail": f"{field} = {value} {what}, matching the repository"})
    return rows


def check_self_description(paper_text: str | None = None,
                           claims: tuple[Claim, ...] | None = None,
                           figure_claims: tuple[FigureClaim, ...] | None = None,
                           scope_claims: tuple[ScopeClaim, ...] | None = None,
                           verdict_claims: tuple[VerdictClaim, ...] | None = None) -> list[dict]:
    """§9 and the Reproduction note state how big this registry is. Check that they are right.

    These three numbers are the one set the registry cannot bind to a record, because their source
    is the registry itself. They have already drifted twice (33 -> 65 -> 70) and were corrected by
    hand both times, which is the same failure this module exists to stop. So they get checked
    against `len()` instead.
    """
    text = paper_text if paper_text is not None else PAPER.read_text(encoding="utf-8")
    sizes = {
        "prose figures": len(CLAIMS if claims is None else claims),
        "drawn figures": len(FIGURE_CLAIMS if figure_claims is None else figure_claims),
        "verdicts": len(VERDICT_CLAIMS if verdict_claims is None else verdict_claims),
        "scope claims": len(SCOPE_CLAIMS if scope_claims is None else scope_claims),
    }
    # both places the paper states them, in the order prose / drawn / verdicts / scope
    patterns = {
        "§9": r"binding (\d+) figures printed in this text\s+and (\d+) more printed inside the three drawn figures.*?plus (\d+) \*\*verdicts\*\*.*?and (\d+) \*\*scope\s+claims\*\*",
        "Reproduction": r"`make papernums` \((\d+) figures in the prose, (\d+) in the drawn figures, (\d+) verdicts, (\d+) scope\s+claims",
    }
    rows = []
    for where, pattern in patterns.items():
        m = re.search(pattern, text, flags=re.DOTALL)
        if not m:
            rows.append({"where": where, "status": "ABSENT",
                         "detail": "the paper no longer states the registry's size here"})
            continue
        stated = [int(g) for g in m.groups()]
        actual = list(sizes.values())
        if stated != actual:
            rows.append({"where": where, "status": "MISMATCH",
                         "detail": f"states {stated}, registry holds {actual}"})
            continue
        rows.append({"where": where, "status": "OK",
                     "detail": " / ".join(f"{v} {k}" for k, v in sizes.items())})
    return rows


def unregistered(paper_text: str | None = None,
                 claims: tuple[Claim, ...] | None = None) -> dict:
    """Every number in the paper that no claim covers.

    `coverage()` says what the registry binds. It cannot say what the registry *misses*, and the
    difference is where four errors lived through a green run. This is the other half: strip the
    tokens that are not measurements (each with its reason), subtract the registered figures, and
    report the remainder. A number here is not necessarily wrong -- it is unchecked, which is the
    condition that let ``137`` (a mean rank printed as seconds) reach the draft.
    """
    text = paper_text if paper_text is not None else PAPER.read_text(encoding="utf-8")
    claims = CLAIMS if claims is None else claims
    stripped = text
    for pattern, _reason in _NOT_A_MEASUREMENT:
        stripped = re.sub(pattern, " ", stripped)
    # a trailing comma or full stop is punctuation, not part of the number
    token = re.compile(r"(?<![\w.])\d(?:[\d,]*\d)?(?:\.\d+)?(?:e[-+]?\d+)?(?![\w])")

    # Coverage is computed per paragraph, because a *pinned* claim only checks the paragraph it
    # names. Treating it as covering every occurrence of its figure is the same concealment it was
    # introduced to stop, one level up: `100` would read as bound everywhere on the strength of a
    # claim that only ever looks at the pods-per-seed sentence.
    unpinned = {c.sought for c in claims if not c.near} | {c.printed for c in claims if not c.near}
    pinned = [c for c in claims if c.near]

    counts: dict[str, int] = {}
    where: dict[str, str] = {}
    unbound_counts: dict[str, int] = {}
    for block in re.split(r"\n\s*\n", stripped):
        here = set(unpinned)
        for c in pinned:
            if c.near in block:
                here.add(c.sought)
                here.add(c.printed)
        for m in token.finditer(block):
            tok = m.group(0)
            counts[tok] = counts.get(tok, 0) + 1
            where.setdefault(tok, block[max(0, m.start() - 40):m.end() + 25].replace("\n", " "))
            if tok not in here:
                unbound_counts[tok] = unbound_counts.get(tok, 0) + 1
    unbound = unbound_counts
    return {
        "distinct_numeric_tokens": len(counts),
        "unbound": dict(sorted(unbound.items(), key=lambda kv: (-kv[1], kv[0]))),
        "unbound_count": len(unbound),
        "first_context": {tok: where[tok] for tok in unbound},
        "excluded_with_reason": [{"pattern": p, "reason": r} for p, r in _NOT_A_MEASUREMENT],
        "not_claimed": (
            "A token counted as bound because its printed form matches a registered figure may in "
            "fact be a different quantity that renders the same way. This measures coverage, not "
            "correctness."
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
    for r in report["figures"] + report["drawn"] + report["verdicts"] + report["scope"]:
        mark = " ok " if r["status"] == "OK" else r["status"]
        print(f"[{mark:>11}] {r['label']:<52} {r['detail']}")

    self_rows = check_self_description()
    for r in self_rows:
        mark = " ok " if r["status"] == "OK" else r["status"]
        print(f"[{mark:>11}] {('self-description: ' + r['where']):<52} {r['detail']}")

    fresh_rows = check_record_freshness()
    for r in fresh_rows:
        mark = " ok " if r["status"] == "OK" else r["status"]
        print(f"[{mark:>11}] {('freshness: ' + r['record']):<52} {r['detail']}")
    stale = [r for r in fresh_rows if r["status"] != "OK"]

    ref_rows = check_references()
    for r in ref_rows:
        if r["status"] != "OK":
            print(f"[{r['status']:>11}] {(r['kind'] + ' ' + r['name']):<52} {r['detail']}")
    bad_refs = [r for r in ref_rows if r["status"] != "OK"]
    print(f"[{(' ok ' if not bad_refs else 'REFS'):>11}] {'references':<52} {len(ref_rows)} checked, {len(bad_refs)} failing")

    cov = unregistered()
    print()
    print(f"coverage: {cov['distinct_numeric_tokens']} distinct numeric tokens in the paper, "
          f"{cov['unbound_count']} bound to no claim")
    for tok, n in cov["unbound"].items():
        print(f"  {n:>3}x  {tok:<12} {cov['first_context'][tok].strip()[:78]}")
    print()
    print(f"{report['registered_figures']} figures in the prose + "
          f"{report['registered_drawn_figures']} in the drawn figures + "
          f"{report['registered_verdicts']} verdicts + "
          f"{report['registered_scope_claims']} scope claims; "
          f"{len(report['failures'])} failing")
    weak = report["weak_presence_tests"]
    if weak:
        print(f"{len(weak)} figures are round enough to recur; their presence test does not "
              f"discriminate and only the record comparison counts for them.")
    failed_self = [r for r in self_rows if r["status"] != "OK"]
    if not report["clean"] or failed_self or cov["unbound_count"] or bad_refs or stale:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
