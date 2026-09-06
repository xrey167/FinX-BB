"""The audit window: where an accessibility audit of an external memory may be read at all.

An accessibility audit asks whether some item is still reachable in a model's activations after the
store has been told to delete it. E-000063 composes one such audit with a canonical pod SHRED, and its
three-seed record (ledger 31.57) returns the headline verdict -- *no workspace trace after deletion* --
while its own pre-registered validity row, *can this instrument see a LIVE pod at all*, fails on every
seed. WSC-001 measured why, and the reason is not the lens, the basis, the probe or the dimension: the
audit was read at a block the pod's content had not reached.

That failure has a shape, and this module is that shape written down. A site is admissible for an
accessibility audit only if BOTH of the following hold there:

    ARRIVAL       the content is here. Formally, the state at this site MOVES when the item is
                  removed from the store. This is a mediator, not a readout: it separates "the audit
                  is blind" from "the audit is correctly built and pointed at the wrong block", and
                  without it a null result is unfalsifiable. Reported relative to the largest movement
                  over the candidate sites, because the absolute scale of a residual stream is not
                  comparable across depths.

    DISTINCTNESS  the audit is a different instrument from reading the output. The J-lens vector of a
                  token converges onto that token's unembedding row as the site approaches the last
                  block -- there is no computation left for the Jacobian to be about -- and at cosine
                  1.000 an audit "in J-space" is the logit lens under another name. E-000042 already
                  recorded what the logit lens is worth here: removing every one of eight top logit-lens
                  directions did not stop GPT-2 answering a single fact.

The AUDIT WINDOW is the set of sites where both hold. It can be empty, and when it is, no siting of
that audit on that model certifies anything -- which is a statement about the memory system's design,
because ARRIVAL is decided by where the memory writes and DISTINCTNESS by how much depth is left
afterwards. Both are set by the same number: the read layer.

WHAT IS CALIBRATED AND WHAT IS PREDICTED, stated here because a bar chosen after seeing the number it
judges is not a bar. ``ARRIVAL_FLOOR`` and ``DEGENERACY_CEILING`` were fixed in
``docs/novelty/sit001-preregister.md`` with the recorded adapter's numbers ALREADY KNOWN (arrival
0.032 at its first read site, cosine 0.783 / 1.000 at its candidate sites), so the empty window on
``read_layers=(8, 10)`` is a re-statement of two existing measurements and is reported as a
consolidation, never as a finding. What those bars had not seen, and what SIT-001 tests, is any number
from an adapter whose writes land earlier.

This module holds no model and runs no forward pass: it takes measurements and returns a verdict, so
that the verdict is auditable separately from the harness that produced the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = ["ARRIVAL_FLOOR", "DEGENERACY_CEILING", "SiteReading", "lens_alignment", "arrival_ratios",
           "audit_window", "window_sites", "certify"]

# Pre-registered in docs/novelty/sit001-preregister.md. See the calibration disclosure above.
ARRIVAL_FLOOR = 0.50        # the site must carry at least half of the largest movement seen anywhere
DEGENERACY_CEILING = 0.90   # mean |cos(v_u, W_U[u])| above this and the lens IS the logit lens


@dataclass
class SiteReading:
    """One candidate site, judged. ``reasons`` is empty exactly when the site is in the window."""

    site: str
    arrival: float
    arrival_ratio: float
    lens_cos: float
    in_window: bool
    reasons: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["reasons"] = list(self.reasons)
        return d


def lens_alignment(vectors, w_rows) -> float:
    """Mean ``|cos|`` between each token's lens vector and that token's own unembedding row.

    Both arguments are (n_tokens, d) and are matched row by row: row i of ``vectors`` must be the lens
    vector of the same token as row i of ``w_rows``, or this measures nothing. Neither is assumed
    normalised. Returned as a mean over tokens because the question is whether the FAMILY has
    collapsed onto the vocabulary basis, not whether one token has.
    """
    import torch
    v = torch.as_tensor(vectors).float()
    w = torch.as_tensor(w_rows).float()
    if v.shape != w.shape:
        raise ValueError(f"lens vectors {tuple(v.shape)} and unembedding rows {tuple(w.shape)} must match "
                         "row for row: row i of each has to belong to the same token")
    v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    w = w / w.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return float((v * w).sum(-1).abs().mean())


def arrival_ratios(mediators: Mapping[str, float]) -> Dict[str, float]:
    """Each site's mediator as a fraction of the largest one measured.

    The absolute movement of a residual stream is not comparable across depths -- GPT-2 small's state
    norm grows by more than an order of magnitude from block 0 to block 11 -- so the floor is relative
    and the denominator is the strongest arrival observed anywhere in the same sweep. If every site is
    at zero the ratios are all zero rather than undefined, because a store operation that moves nothing
    anywhere means the counterfactual did not fire and NO site is admissible.
    """
    if not mediators:
        return {}
    vals = [float(v) for v in mediators.values()]
    if any(v < 0 for v in vals):
        raise ValueError("a mediator is a magnitude and cannot be negative")
    top = max(vals)
    if top <= 0.0:
        return {k: 0.0 for k in mediators}
    return {k: float(v) / top for k, v in mediators.items()}


def audit_window(mediators: Mapping[str, float], lens_cos: Mapping[str, float],
                 arrival_floor: float = ARRIVAL_FLOOR,
                 degeneracy_ceiling: float = DEGENERACY_CEILING,
                 order: Optional[Sequence[str]] = None) -> List[SiteReading]:
    """Judge every candidate site. Returns one ``SiteReading`` per site, in ``order`` if given.

    ``mediators`` and ``lens_cos`` must cover the same sites: a site with a mediator and no lens
    measurement cannot be judged on distinctness, and admitting it by default is exactly the mistake
    this module exists to prevent, so it raises instead.
    """
    if set(mediators) != set(lens_cos):
        missing = sorted(set(mediators) ^ set(lens_cos))
        raise ValueError(f"every candidate site needs both a mediator and a lens alignment; unmatched: {missing}")
    ratios = arrival_ratios(mediators)
    names = list(order) if order is not None else list(mediators)
    if set(names) != set(mediators):
        raise ValueError("`order` must list exactly the sites measured")
    out: List[SiteReading] = []
    for s in names:
        r, c = ratios[s], float(lens_cos[s])
        reasons: List[str] = []
        if r < arrival_floor:
            reasons.append(f"arrival {r:.3f} < {arrival_floor:.2f}: the content has not reached this site, so "
                           "a null readout here is not evidence of absence")
        if c > degeneracy_ceiling:
            reasons.append(f"lens |cos| to the unembedding {c:.3f} > {degeneracy_ceiling:.2f}: the audit at this "
                           "site is the logit lens, not an independent instrument")
        out.append(SiteReading(s, float(mediators[s]), r, c, not reasons, tuple(reasons)))
    return out


def window_sites(readings: Sequence[SiteReading]) -> List[str]:
    return [r.site for r in readings if r.in_window]


def certify(readings: Sequence[SiteReading], site: str, verdict: Mapping[str, object]) -> Dict[str, object]:
    """Admit or refuse an audit verdict, given where it was read.

    This is the guard E-000063 did not have. A verdict read outside the window is returned marked
    ``admissible: False`` together with the reasons, and callers are expected to report it as VACUOUS
    rather than as a negative result -- the audit did not fail to find a trace, it was never in a
    position to find one. A verdict read inside the window is returned unchanged; this function does
    not second-guess the audit, it only decides whether the audit was in a position to speak.
    """
    by_site = {r.site: r for r in readings}
    if site not in by_site:
        raise ValueError(f"site {site!r} was not among the candidate sites judged: {sorted(by_site)}")
    r = by_site[site]
    return {"site": site, "admissible": bool(r.in_window), "reasons": list(r.reasons),
            "window": window_sites(readings), "siting": r.to_dict(), "verdict": dict(verdict),
            "status": "ADMITTED" if r.in_window else "VACUOUS"}
