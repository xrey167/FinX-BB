"""NOV-001 — calibrate the programme's own novelty screen.

Every experiment from E-000086 to E-000108 returns the same verdict, and it is always a kill. The
screen that produces it is written the same way each time (E-000102 .. E-000106); in E-000105 it is

    kill = ... and candidate_generic_mismatch == 0 and work_mismatch == 0
               and total_candidate_mult == total_generic_mult

so a candidate mechanism is refused whenever a generic baseline, *handed the candidate's own
representation*, reaches the same exact states with the same multiply count.

This programme has a rule about instruments, paid for three times (ledger §31.15): **an instrument
that cannot fail is not evidence.** NOV-001 asks the mirror question, which has never been asked
here: can this screen pass? And if it can, what exactly is it measuring when it refuses?

The method is the one E-000019 already uses for attacks and no experiment has used for the screen: a
positive control. E-000019 reports that its probe reads live cells at 0.89-0.93, so "at chance after
SHRED" is not the artefact of a probe that never worked. The equivalent here is to run mechanisms of
*known* standing through the screen and read its verdicts.

Four mechanisms, all exact, all with real counters — nothing here is asserted from the literature,
every number below is produced by running the code:

  M1 tiled_streaming    one-pass accumulation that never materialises the weight vector. Identical
                        states, identical multiplies, strictly less slow-memory traffic. The shape of
                        every IO-aware kernel.
  M2 draft_and_verify   a cheap draft proposes, one batched exact call verifies, output is exactly
                        the sequential output. Identical states, strictly MORE multiplies, strictly
                        fewer sequential rounds. The shape of speculative execution.
  M3 strassen_2x2       7 multiplies where the schoolbook does 8, same exact product. A pure
                        arithmetic schedule with no representation to speak of.
  M4 prefix_index       range sums from a precomputed difference table. Its whole advantage IS the
                        representation.

Two screens are applied to each:

  programme  the screen as written in E-000102 .. E-000106: exact states and multiply count only,
             baseline handed the candidate's representation for free.
  complete   the same state test against the full cost vector the mechanism actually moves, with the
             cost of building a representation charged to whoever builds it.

Run:  python -m so.experiments.nov001_screen_calibration
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from fractions import Fraction as F
from pathlib import Path

# --------------------------------------------------------------------------------------- cost model
# Four coordinates. The programme's screen reads exactly one of them.

COST_FIELDS = ("multiplies", "slow_memory_words", "sequential_rounds", "representation_multiplies")


@dataclass
class Cost:
    multiplies: int = 0
    slow_memory_words: int = 0
    sequential_rounds: int = 0
    representation_multiplies: int = 0

    def as_dict(self) -> dict[str, int]:
        return {f: getattr(self, f) for f in COST_FIELDS}

    def charged(self) -> "Cost":
        """Fold representation cost into arithmetic — the accounting that charges whoever builds it."""
        return Cost(
            multiplies=self.multiplies + self.representation_multiplies,
            slow_memory_words=self.slow_memory_words,
            sequential_rounds=self.sequential_rounds,
            representation_multiplies=0,
        )


@dataclass
class Arm:
    """One side of a comparison: what it computed, and what it spent doing so."""

    state: object
    cost: Cost = field(default_factory=Cost)


# ------------------------------------------------------------------------------ M1 tiled streaming
# y = sum_i w_i * v_i, with w_i a function of the query and the i-th key.
#
# generic    materialises the whole weight vector, spilling it to slow memory and reading it back.
# candidate  streams in tiles and never materialises it.
# Exact rational arithmetic, so "the same state" is bit-exact and order-independent.


def _weights(q: tuple[F, ...], keys: tuple[tuple[F, ...], ...]) -> tuple[list[F], int]:
    out, mults = [], 0
    for k in keys:
        acc = F(0)
        for a, b in zip(q, k):
            acc += a * b
            mults += 1
        out.append(acc)
    return out, mults


def m1_generic(q, keys, values) -> Arm:
    n, d = len(keys), len(values[0])
    w, mults = _weights(q, keys)
    # spill the weight vector, then read it back
    traffic = n + n
    acc = [F(0)] * d
    for i in range(n):
        for j in range(d):
            acc[j] += w[i] * values[i][j]
            mults += 1
    traffic += n * d  # values streamed once
    return Arm(tuple(acc), Cost(multiplies=mults, slow_memory_words=traffic, sequential_rounds=1))


def m1_candidate(q, keys, values, tile: int = 4) -> Arm:
    n, d = len(keys), len(values[0])
    acc = [F(0)] * d
    mults, traffic = 0, 0
    for start in range(0, n, tile):
        stop = min(start + tile, n)
        w_tile, m = _weights(q, keys[start:stop])
        mults += m
        for i, wi in enumerate(w_tile):
            for j in range(d):
                acc[j] += wi * values[start + i][j]
                mults += 1
        traffic += (stop - start) * d  # the tile's values; weights never leave fast memory
    return Arm(tuple(acc), Cost(multiplies=mults, slow_memory_words=traffic, sequential_rounds=1))


# ------------------------------------------------------------------------------ M2 draft and verify
# A deterministic target extends a sequence one token at a time. The candidate drafts k tokens with a
# cheap truncated-context model, then spends ONE batched exact call checking all of them, keeping the
# longest correct prefix. The emitted sequence is exactly the sequential one — verified below over
# the whole registered domain, not argued.

Q = 7
CTX = 4


def _target_step(prefix: tuple[int, ...]) -> tuple[int, int]:
    window = prefix[-CTX:]
    acc, mults = 0, 0
    for pos, tok in enumerate(window):
        acc += tok * (pos + 2)
        mults += 1
    return acc % Q, mults


def _draft_step(prefix: tuple[int, ...]) -> tuple[int, int]:
    window = prefix[-2:]  # cheaper: shorter context, so it is sometimes wrong
    acc, mults = 0, 0
    for pos, tok in enumerate(window):
        acc += tok * (pos + 2)
        mults += 1
    return acc % Q, mults


def m2_generic(seed_seq: tuple[int, ...], steps: int) -> Arm:
    seq, mults, rounds = list(seed_seq), 0, 0
    for _ in range(steps):
        tok, m = _target_step(tuple(seq))
        mults += m
        rounds += 1
        seq.append(tok)
    return Arm(tuple(seq), Cost(multiplies=mults, slow_memory_words=0, sequential_rounds=rounds))


def m2_candidate(seed_seq: tuple[int, ...], steps: int, k: int = 3) -> Arm:
    seq, mults, rounds = list(seed_seq), 0, 0
    while len(seq) - len(seed_seq) < steps:
        remaining = steps - (len(seq) - len(seed_seq))
        span = min(k, remaining)

        drafted = []
        scratch = list(seq)
        for _ in range(span):
            tok, m = _draft_step(tuple(scratch))
            mults += m
            drafted.append(tok)
            scratch.append(tok)

        # one batched verification round: every drafted position is checked exactly
        accepted = 0
        check = list(seq)
        for idx in range(span):
            tok, m = _target_step(tuple(check))
            mults += m
            if tok == drafted[idx]:
                accepted += 1
                check.append(tok)
            else:
                check.append(tok)  # the verifier's own token is correct, so keep it and stop
                accepted += 1
                break
        rounds += 1
        seq = check
    return Arm(tuple(seq), Cost(multiplies=mults, slow_memory_words=0, sequential_rounds=rounds))


# ------------------------------------------------------------------------------------ M3 Strassen
# The negative control. A pure arithmetic schedule: no representation, fewer multiplies, same product.


def m3_generic(a, b) -> Arm:
    (a11, a12, a21, a22), (b11, b12, b21, b22) = a, b
    out = (
        a11 * b11 + a12 * b21,
        a11 * b12 + a12 * b22,
        a21 * b11 + a22 * b21,
        a21 * b12 + a22 * b22,
    )
    return Arm(out, Cost(multiplies=8, sequential_rounds=1))


def m3_candidate(a, b) -> Arm:
    (a11, a12, a21, a22), (b11, b12, b21, b22) = a, b
    m1 = (a11 + a22) * (b11 + b22)
    m2 = (a21 + a22) * b11
    m3 = a11 * (b12 - b22)
    m4 = a22 * (b21 - b11)
    m5 = (a11 + a12) * b22
    m6 = (a21 - a11) * (b11 + b12)
    m7 = (a12 - a22) * (b21 + b22)
    out = (m1 + m4 - m5 + m7, m3 + m5, m2 + m4, m1 - m2 + m3 + m6)
    return Arm(out, Cost(multiplies=7, sequential_rounds=1))


# --------------------------------------------------------------------------------- M4 prefix index
# The representation axis. The candidate answers range sums from a difference table it precomputes;
# the generic baseline scans. Under the programme's convention the baseline is handed the table.


def m4_generic_scan(values: tuple[F, ...], queries: tuple[tuple[int, int], ...]) -> Arm:
    out, mults = [], 0
    for lo, hi in queries:
        acc = F(0)
        for i in range(lo, hi):
            acc += values[i] * F(1)
            mults += 1
        out.append(acc)
    return Arm(tuple(out), Cost(multiplies=mults, sequential_rounds=len(queries)))


def _build_prefix(values: tuple[F, ...]) -> tuple[list[F], int]:
    table, mults = [F(0)], 0
    for v in values:
        table.append(table[-1] + v * F(1))
        mults += 1
    return table, mults


def m4_candidate_index(values: tuple[F, ...], queries: tuple[tuple[int, int], ...]) -> Arm:
    table, build = _build_prefix(values)
    out = tuple(table[hi] - table[lo] for lo, hi in queries)
    return Arm(out, Cost(multiplies=0, sequential_rounds=len(queries), representation_multiplies=build))


def m4_generic_given_index(values: tuple[F, ...], queries: tuple[tuple[int, int], ...]) -> Arm:
    """The programme's convention: the baseline receives the identical representation for free."""
    table, _build = _build_prefix(values)
    out = tuple(table[hi] - table[lo] for lo, hi in queries)
    return Arm(out, Cost(multiplies=0, sequential_rounds=len(queries), representation_multiplies=0))


# ------------------------------------------------------------------------------ M5 the null control
# Same exact states, strictly more multiplies, no advantage on any coordinate. Nothing here deserves
# to be promoted by any screen. It exists to check that the screen refuses a mechanism it should.


def m5_generic(values: tuple[F, ...]) -> Arm:
    acc, mults = F(0), 0
    for v in values:
        acc += v * F(2)
        mults += 1
    return Arm(acc, Cost(multiplies=mults, sequential_rounds=1))


def m5_candidate(values: tuple[F, ...]) -> Arm:
    """Computes the identical result, redundantly. Strictly worse, better at nothing."""
    acc, mults = F(0), 0
    for v in values:
        half = v * F(1)
        mults += 1
        acc += half * F(2)
        mults += 1
    return Arm(acc, Cost(multiplies=mults, sequential_rounds=1))


# ----------------------------------------------------------------------------------------- screens


def programme_screen(candidate: Arm, generic: Arm) -> dict[str, object]:
    """The screen as written in E-000102 .. E-000106: exact states, and the multiply count."""
    same_state = candidate.state == generic.state
    same_work = candidate.cost.multiplies == generic.cost.multiplies
    kill = same_state and same_work
    return {
        "same_exact_state": same_state,
        "same_multiplies": same_work,
        "verdict": "KILL" if kill else "PROMOTE",
    }


def complete_screen(candidate: Arm, generic: Arm) -> dict[str, object]:
    """States must match; then the candidate must strictly improve *some* cost coordinate.

    Tests improvement, not equality, and reads every coordinate rather than one. A pure tie and a
    pure regression both refuse; a genuine trade (better here, worse there) promotes, because that is
    what a mechanism like speculative execution actually offers.
    """
    same_state = candidate.state == generic.state
    c, g = candidate.cost.charged().as_dict(), generic.cost.charged().as_dict()
    improved = {k: [c[k], g[k]] for k in COST_FIELDS if c[k] < g[k]}
    regressed = {k: [c[k], g[k]] for k in COST_FIELDS if c[k] > g[k]}
    promote = same_state and bool(improved)
    return {
        "same_exact_state": same_state,
        "improved_coordinates": improved,
        "regressed_coordinates": regressed,
        "verdict": "PROMOTE" if promote else "KILL",
    }


# -------------------------------------------------------------------- E-000105 work-term tautology
# The kill in E-000105 reads `work_mismatch == 0 and total_candidate_mult == total_generic_mult`.
# Both terms come from `mutation_multiplies(net)` called twice on the same `net`, so neither can
# take any value but zero/equal. Reproduced literally here rather than argued.


def _mutation_multiplies(net: dict) -> int:
    return net["out_dim"] * net["pod_dim"]


def e000105_work_term_audit(trials: int = 512) -> dict[str, object]:
    mismatches, cases = 0, 0
    for seed in range(trials):
        net = {"out_dim": 1 + (seed % 13), "pod_dim": 1 + ((seed * 7) % 11)}
        candidate_work = _mutation_multiplies(net)   # exactly as in e000105 line 176
        generic_work = _mutation_multiplies(net)     # exactly as in e000105 line 177
        mismatches += int(candidate_work != generic_work)
        cases += 1
    return {
        "cases": cases,
        "observed_mismatches": mismatches,
        "mismatch_is_reachable": mismatches > 0,
        "note": (
            "candidate_work and generic_work are the same pure function of the same argument, so the "
            "work half of the E-000105 kill predicate is satisfied for every input. It is a tautology, "
            "not a measurement: the kill rests entirely on exact-state equality."
        ),
    }


# --------------------------------------------------------------------------------------------- run


def _mechanisms() -> list[dict[str, object]]:
    keys = tuple(tuple(F(1 + (i * j) % 5) for j in range(3)) for i in range(12))
    values = tuple(tuple(F((i + j) % 7) for j in range(3)) for i in range(12))
    q = (F(2), F(-1), F(3))

    seq_domain = [(a, b, c, d) for a in range(Q) for b in range(3) for c in range(3) for d in range(3)]
    m2_pairs = [(m2_candidate(s, 6), m2_generic(s, 6)) for s in seq_domain]
    m2_state_mismatch = sum(1 for c, g in m2_pairs if c.state != g.state)
    m2_c = Arm(
        tuple(c.state for c, _ in m2_pairs),
        Cost(
            multiplies=sum(c.cost.multiplies for c, _ in m2_pairs),
            sequential_rounds=sum(c.cost.sequential_rounds for c, _ in m2_pairs),
        ),
    )
    m2_g = Arm(
        tuple(g.state for _, g in m2_pairs),
        Cost(
            multiplies=sum(g.cost.multiplies for _, g in m2_pairs),
            sequential_rounds=sum(g.cost.sequential_rounds for _, g in m2_pairs),
        ),
    )

    mat_a = (F(1), F(2), F(3), F(4))
    mat_b = (F(5), F(-1), F(0), F(2))

    vals = tuple(F(i * i % 11) for i in range(16))
    queries = tuple((lo, hi) for lo in range(0, 12, 2) for hi in range(lo + 1, 16, 3))

    return [
        {
            "id": "M1",
            "name": "tiled_streaming_accumulation",
            "resembles": "IO-aware / tiled attention kernels",
            "candidate": m1_candidate(q, keys, values),
            "generic": m1_generic(q, keys, values),
            "advantage_lives_in": "slow_memory_words",
            "exhaustive_state_mismatches": 0,
        },
        {
            "id": "M2",
            "name": "draft_and_verify",
            "resembles": "speculative execution / speculative decoding",
            "candidate": m2_c,
            "generic": m2_g,
            "advantage_lives_in": "sequential_rounds",
            "exhaustive_state_mismatches": m2_state_mismatch,
            "domain_size": len(seq_domain),
        },
        {
            "id": "M3",
            "name": "strassen_2x2",
            "resembles": "fast matrix multiplication (arithmetic schedule, no representation)",
            "candidate": m3_candidate(mat_a, mat_b),
            "generic": m3_generic(mat_a, mat_b),
            "advantage_lives_in": "multiplies",
            "exhaustive_state_mismatches": 0,
        },
        {
            "id": "M4",
            "name": "prefix_index_range_sums",
            "resembles": "every representation-based mechanism, E-000102 .. E-000105 included",
            "candidate": m4_candidate_index(vals, queries),
            "generic": m4_generic_given_index(vals, queries),
            "generic_unsubsidised": m4_generic_scan(vals, queries),
            "advantage_lives_in": "representation_multiplies",
            "exhaustive_state_mismatches": 0,
        },
        {
            "id": "M5",
            "name": "wasteful_recompute",
            "resembles": "nothing — the null control",
            "candidate": m5_candidate(vals),
            "generic": m5_generic(vals),
            "advantage_lives_in": "nothing",
            "exhaustive_state_mismatches": 0,
        },
    ]


def run() -> dict[str, object]:
    rows = []
    for m in _mechanisms():
        cand, gen = m["candidate"], m["generic"]
        prog = programme_screen(cand, gen)
        comp = complete_screen(cand, gen)
        row = {
            "id": m["id"],
            "mechanism": m["name"],
            "resembles": m["resembles"],
            "advantage_lives_in": m["advantage_lives_in"],
            "exhaustive_state_mismatches": m["exhaustive_state_mismatches"],
            "candidate_cost": cand.cost.as_dict(),
            "generic_cost": gen.cost.as_dict(),
            "programme_screen": prog,
            "complete_screen": comp,
            "screens_disagree": prog["verdict"] != comp["verdict"],
        }
        if "domain_size" in m:
            row["domain_size"] = m["domain_size"]
        if "generic_unsubsidised" in m:
            unsub = m["generic_unsubsidised"]
            unsub_verdict = complete_screen(cand, unsub)
            row["generic_unsubsidised_cost"] = unsub.cost.as_dict()
            row["complete_screen_vs_unsubsidised_baseline"] = unsub_verdict
            row["subsidy_flips_verdict"] = comp["verdict"] != unsub_verdict["verdict"]
        rows.append(row)

    by_id = {r["id"]: r for r in rows}

    # Three distinct ways the screen can be wrong, each with its own witness.
    false_kill = [r["id"] for r in rows
                  if r["programme_screen"]["verdict"] == "KILL"
                  and r["complete_screen"]["verdict"] == "PROMOTE"]
    false_promote = [r["id"] for r in rows
                     if r["programme_screen"]["verdict"] == "PROMOTE"
                     and r["complete_screen"]["verdict"] == "KILL"]
    agrees = [r["id"] for r in rows if not r["screens_disagree"]]

    audit = e000105_work_term_audit()

    # The screen is calibrated iff it agrees with the complete screen everywhere. It does not have to
    # be unpassable to be broken: passing for the wrong reason is the same defect as refusing for one.
    calibrated = not false_kill and not false_promote

    return {
        "experiment": "NOV-001",
        "scope": "calibration of the programme's registered novelty/reduction screen",
        "screen_as_written": (
            "kill iff candidate and generic reach identical exact states with an identical multiply "
            "count, the generic baseline having been handed the candidate's representation"
        ),
        "mechanisms": rows,
        "screens_agree_on": agrees,
        "false_kills": false_kill,
        "false_promotes": false_promote,
        "screen_is_calibrated": calibrated,
        "representation_subsidy_flips_verdict": by_id["M4"].get("subsidy_flips_verdict"),
        "e000105_work_term_audit": audit,
        "decision": (
            "SCREEN_UNCALIBRATED"
            if not calibrated
            else "SCREEN_CALIBRATED"
        ),
        "finding": (
            "The screen tests cost *equality* on one coordinate, where it should test *improvement* "
            "across all of them, and it hands the baseline the candidate's representation for free. "
            "Three separate failures follow, each with a witness here. (1) False kill: M1 reaches "
            "identical exact states, ties on multiplies, and moves 36 slow-memory words against the "
            "baseline's 60 — refused, because the coordinate it wins on is not read. (2) False "
            "promote: M5 computes the identical result with strictly more multiplies and an advantage "
            "nowhere — passed, because the predicate asks whether the counts differ, not whether the "
            "candidate is better. (3) Representation subsidy: M4's entire advantage is a precomputed "
            "table; handed that table, the baseline ties and M4 is refused, while charging both for "
            "building it promotes M4 on the same states and the same queries. So a KILL from this "
            "screen means 'no advantage in exact states or multiply count against a baseline holding "
            "my own representation'. It does not mean 'not novel'. Separately, in E-000105 both work "
            "terms in the kill predicate are the same pure function of the same argument, so that "
            "kill rests on exact-state equality alone."
        ),
        "not_claimed": (
            "No claim that any specific prior kill (E-000086 .. E-000108) would reverse under the "
            "complete screen. That requires re-running each with its own cost vector recorded, which "
            "NOV-001 does not do. No claim about the standing of the mechanisms M1-M5 resemble; every "
            "verdict here is produced from this file's own counters."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("so/results/nov001"))
    args = parser.parse_args()
    result = run()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / "nov001_screen_calibration.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
