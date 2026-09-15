"""The reduction screen, with the three defects NOV-001 found closed.

Every experiment from E-000102 to E-000108 wrote its own copy of the kill predicate, and each copy
had the same shape:

    kill = ... and candidate_generic_mismatch == 0 and work_mismatch == 0
               and total_candidate_mult == total_generic_mult

NOV-001 ran mechanisms of known standing through it and found three failures, each with a witness,
and NOV-002 then showed that the third of them did not produce the two kills it was most likely to
have produced. The verdicts stand; the predicate still needs replacing, because an instrument that
can pass a strict regression will eventually pass one.

This module is that replacement. It is deliberately **not** wired into any recorded experiment —
E-000102 … E-000108 keep the predicate that produced their records, because rewriting an instrument
under a recorded result is how a ledger stops meaning anything. New work should import from here.

What changes, and why:

**1. Improvement, not equality.** The old predicate asked whether two cost figures *differ*. A
candidate spending strictly more arithmetic for the same states therefore escaped the kill
(NOV-001's M5, 32 multiplies against 16, better at nothing, `PROMOTE`). A screen should ask whether
the candidate is *better*, so `PROMOTE` here requires a strict improvement on at least one
coordinate.

**2. Every coordinate the mechanism moves, not one.** The old predicate read multiplies alone, so a
mechanism that tied on arithmetic and moved strictly less memory was refused (NOV-001's M1, 36 words
against 60, `KILL`). Declare the coordinates the comparison is entitled to read; anything left
undeclared is recorded as unread rather than silently scored.

**3. The representation is charged to whoever builds it.** The old convention handed the generic
baseline the candidate's own representation, so any mechanism whose advantage *is* its representation
tied by construction (NOV-001's M4, verdict flipped by the accounting alone). Here
`representation_multiplies` folds into the total.

And one guard the old predicate had no way to express:

**4. A comparison whose two sides share a provenance label is refused as vacuous.** In
`e000105` lines 176-178 and `e000104` lines 224-225 both work figures are the same pure function of
the same argument, so neither `work_mismatch` nor the totals term could take any value but
zero/equal. Requiring each side to name where its number came from makes that unrepresentable:
identical labels raise. This is ledger §31.15 — *an instrument that cannot fail is not evidence* —
enforced at the type level rather than remembered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "COORDINATES",
    "CostVector",
    "Measurement",
    "VacuousComparison",
    "Verdict",
    "screen",
]

COORDINATES = ("multiplies", "slow_memory_words", "sequential_rounds", "representation_multiplies")


class VacuousComparison(RuntimeError):
    """Both arms sourced their cost from the same place, so the comparison cannot fail."""


@dataclass(frozen=True)
class CostVector:
    multiplies: int = 0
    slow_memory_words: int = 0
    sequential_rounds: int = 0
    representation_multiplies: int = 0

    def as_dict(self) -> dict[str, int]:
        return {c: getattr(self, c) for c in COORDINATES}

    def charged(self) -> "CostVector":
        """Fold the representation build into arithmetic — defect 3."""
        return CostVector(
            multiplies=self.multiplies + self.representation_multiplies,
            slow_memory_words=self.slow_memory_words,
            sequential_rounds=self.sequential_rounds,
            representation_multiplies=0,
        )


@dataclass(frozen=True)
class Measurement:
    """One arm of a comparison: what it cost, and **where that number came from**.

    `label` is not decoration. Two arms carrying the same label are the same measurement written
    twice, which is the E-000104 / E-000105 defect, and `screen` refuses it.
    """

    label: str
    cost: CostVector
    declared_coordinates: tuple[str, ...] = COORDINATES

    def __post_init__(self) -> None:
        unknown = set(self.declared_coordinates) - set(COORDINATES)
        if unknown:
            raise ValueError(f"unknown cost coordinates: {sorted(unknown)}")
        if not self.label.strip():
            raise ValueError("a measurement must name its provenance")


@dataclass
class Verdict:
    verdict: str
    states_equal: bool
    improved: dict[str, list[int]] = field(default_factory=dict)
    regressed: dict[str, list[int]] = field(default_factory=dict)
    unread_coordinates: tuple[str, ...] = ()
    candidate_label: str = ""
    generic_label: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "states_equal": self.states_equal,
            "improved_coordinates": self.improved,
            "regressed_coordinates": self.regressed,
            "unread_coordinates": list(self.unread_coordinates),
            "candidate_label": self.candidate_label,
            "generic_label": self.generic_label,
        }


def screen(candidate: Measurement, generic: Measurement, states_equal: bool) -> Verdict:
    """Screen a candidate mechanism against a generic baseline.

    `states_equal` is the caller's exact-state comparison and is not second-guessed here; it is the
    one thing the old predicate got right. Verdicts:

      ``NOT_EXACT``   the arms do not compute the same thing, so cost is not yet the question
      ``PROMOTE``     identical states and a strict improvement on at least one declared coordinate
      ``KILL``        identical states and no improvement anywhere — a tie or a regression

    Raises `VacuousComparison` when both arms name the same provenance.
    """
    if candidate.label == generic.label:
        raise VacuousComparison(
            f"both arms report cost from {candidate.label!r}; a comparison of a measurement with "
            "itself cannot fail (ledger §31.15). Measure the baseline independently."
        )

    read = tuple(c for c in COORDINATES
                 if c in candidate.declared_coordinates and c in generic.declared_coordinates)
    unread = tuple(c for c in COORDINATES if c not in read)

    if not states_equal:
        return Verdict("NOT_EXACT", False, unread_coordinates=unread,
                       candidate_label=candidate.label, generic_label=generic.label)

    c, g = candidate.cost.charged().as_dict(), generic.cost.charged().as_dict()
    improved = {k: [c[k], g[k]] for k in read if c[k] < g[k]}
    regressed = {k: [c[k], g[k]] for k in read if c[k] > g[k]}

    return Verdict(
        "PROMOTE" if improved else "KILL",
        True,
        improved=improved,
        regressed=regressed,
        unread_coordinates=unread,
        candidate_label=candidate.label,
        generic_label=generic.label,
    )
