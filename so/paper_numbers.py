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
    reduce: str = ""   # set when the path uses '*' and the paper quotes an aggregate


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
    one key before falling back to a single segment. A numeric segment indexes a list, and `*`
    iterates one.
    """
    segs = [s for s in path.split("/") if s != ""]
    cur, i = doc, 0
    while i < len(segs):
        if isinstance(cur, list):
            seg = segs[i]
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

    # E-000029 — the swept geometry
    Claim("E29 declared radius", "e000029_marker_geometry.json", "declared_radius", "0.35", "dp2"),
    Claim("E29 operational radius", "e000029_marker_geometry.json", "aggregate/operational_radius/mean", "0.90", "dp2"),
    Claim("E29 annulus accepted", "e000029_marker_geometry.json", "pooled/annulus/0", "2,199,996", "thousands"),
    Claim("E29 annulus n", "e000029_marker_geometry.json", "pooled_intervals/annulus/n", "2,200,000", "thousands"),
    Claim("E29 checkpoints", "e000029_marker_geometry.json", "n_checkpoints", "11", "int"),
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

    # E-000024 — rows versus weights
    Claim("E24 cells forced choice", "e000024_weights_vs_cells-seed0.json", "aggregate/cells/after/forced_choice/mean", "0.44", "dp2"),
    Claim("E24 cells perplexity", "e000024_weights_vs_cells-seed0.json", "aggregate/cells/ppl_after/mean", "42.9", "dp1"),
    Claim("E24 cells delete seconds", "e000024_weights_vs_cells-seed0.json", "aggregate/cells/delete_seconds/mean", "0.0008", "dp4"),

    # E-000030 — the certificate
    Claim("E30 swept questions", "e000030_deletion_certificate.json", "per_seed/0/n_queries_swept", "838", "int"),
    Claim("E30 soft-gate residual", "e000030_deletion_certificate.json", "gpt2/0/gpt2_soft/shred/residual", "1.390e-02", "exp3"),
)


SCOPE_CLAIMS: tuple[ScopeClaim, ...] = (
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


def _load(name: str):
    p = RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def check(
    paper_text: str | None = None,
    claims: tuple[Claim, ...] | None = None,
    scope_claims: tuple[ScopeClaim, ...] | None = None,
) -> dict:
    """Check the paper against the records.

    `claims` and `scope_claims` default to the registries above. They are parameters so that the
    validity floor can be exercised: a checker that has never been shown to fail is not evidence
    that the paper agrees with the records, it is only evidence that it printed ``0 failing``.
    """
    text = paper_text if paper_text is not None else PAPER.read_text(encoding="utf-8")
    claims = CLAIMS if claims is None else claims
    scope_claims = SCOPE_CLAIMS if scope_claims is None else scope_claims
    cache: dict[str, object] = {}
    rows = []

    for c in claims:
        doc = cache.setdefault(c.record, _load(c.record))
        if doc is None:
            rows.append({"label": c.label, "status": "PATH", "detail": f"no record {c.record}"})
            continue
        value, err = _resolve(doc, c.path)
        if err:
            rows.append({"label": c.label, "status": "PATH", "detail": f"{c.path}: {err}"})
            continue
        if c.reduce:
            if not isinstance(value, list) or not value:
                rows.append({"label": c.label, "status": "PATH",
                             "detail": f"{c.path}: reducer {c.reduce!r} needs a non-empty list"})
                continue
            value = _reduce(value, c.reduce)
        try:
            rendered = _fmt(float(value), c.rule)
        except (TypeError, ValueError) as exc:
            rows.append({"label": c.label, "status": "PATH", "detail": f"{c.path}: {exc}"})
            continue
        if rendered != c.printed:
            rows.append({"label": c.label, "status": "MISMATCH",
                         "detail": f"record {value!r} renders {rendered!r} under {c.rule}, paper prints {c.printed!r}"})
            continue
        if c.printed not in text:
            rows.append({"label": c.label, "status": "ABSENT",
                         "detail": f"{c.printed!r} matches the record but does not appear in the paper"})
            continue
        rows.append({"label": c.label, "status": "OK", "detail": f"{c.printed} = {c.path}"})

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

    failures = [r for r in rows + scope_rows if r["status"] != "OK"]
    return {
        "registered_figures": len(claims),
        "registered_scope_claims": len(scope_claims),
        "figures": rows,
        "scope": scope_rows,
        "failures": failures,
        "clean": not failures,
        "not_claimed": (
            "Partial by construction. A figure absent from this registry is unchecked, not verified; "
            "`coverage()` reports what is bound. The registry checks that the paper agrees with the "
            "records, never that the records are right."
        ),
    }


def coverage() -> dict:
    return {
        "records_bound": sorted({c.record for c in CLAIMS} | {s.record for s in SCOPE_CLAIMS}),
        "figures": len(CLAIMS),
        "scope_claims": len(SCOPE_CLAIMS),
    }


def main() -> None:
    report = check()
    for r in report["figures"] + report["scope"]:
        mark = " ok " if r["status"] == "OK" else r["status"]
        print(f"[{mark:>11}] {r['label']:<52} {r['detail']}")
    print()
    print(f"{report['registered_figures']} figures + {report['registered_scope_claims']} scope claims; "
          f"{len(report['failures'])} failing")
    if not report["clean"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
