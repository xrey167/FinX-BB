from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

STAGE = "R319-GENERATIONAL-CAUSAL-REWIND"
TRIALS = int(os.environ.get("SO_R319_TRIALS", "60000"))
PODS = 256
TOKENS = 96
REPORT_PATH = Path(os.environ.get("SO_R319_REPORT", "r319_report.json"))
SEED = 3190914


@dataclass
class Cell:
    generation: int
    value: int


@dataclass
class Trace:
    tokens: list[int]
    state_before: list[int]
    first_read_pos: dict[tuple[int, int], int]
    final_deps: frozenset[tuple[int, int]]


def init_world(rng: random.Random) -> list[Cell]:
    return [Cell(1, rng.randrange(65536)) for _ in range(PODS)]


def step_token(state: int, t: int, reads: list[int]) -> tuple[int, int]:
    # Deterministic autoregressive transducer. After a mutable read, all later
    # output is causally downstream through `state`, exactly like a decode suffix.
    x = (state * 1103515245 + 12345 + 7919 * (t + 1)) & 0xFFFFFFFF
    for v in reads:
        x = ((x ^ (v * 2654435761)) * 2246822519 + 3266489917) & 0xFFFFFFFF
    token = (x ^ (x >> 16)) & 0x3FFF
    new_state = ((state << 7) ^ x ^ token * 40503) & 0xFFFFFFFF
    return token, new_state


def read_schedule(t: int, state: int, mutable_start: int, plan_seed: int) -> list[int]:
    if t < mutable_start:
        return []
    # Sparse late reads. Control flow after the first mutable point is allowed to
    # depend on prior generated state, so a changed value can alter later reads.
    h = (plan_seed * 1315423911 + t * 2654435761 + state) & 0xFFFFFFFF
    reads = []
    if (h & 7) < 3:
        reads.append((h >> 8) % PODS)
    if ((h >> 3) & 15) == 0:
        reads.append((h >> 17) % PODS)
    return list(dict.fromkeys(reads))


def generate(world: list[Cell], mutable_start: int, plan_seed: int, *, prefix_tokens=None, start_pos=0, start_state=None, prefix_first_reads=None, prefix_deps=None) -> Trace:
    tokens = list(prefix_tokens or [])
    if start_state is None:
        state = (plan_seed * 2654435761 + 17) & 0xFFFFFFFF
    else:
        state = start_state
    state_before = [0] * TOKENS
    # Only positions >= start_pos are consumed on partial replay. Prefix states
    # are irrelevant except for checkpoint extraction in an original full trace.
    first_read = dict(prefix_first_reads or {})
    deps = set(prefix_deps or ())

    for t in range(start_pos, TOKENS):
        state_before[t] = state
        pids = read_schedule(t, state, mutable_start, plan_seed)
        values = []
        for pid in pids:
            c = world[pid]
            dep = (pid, c.generation)
            deps.add(dep)
            first_read.setdefault(dep, t)
            values.append(c.value)
        tok, state = step_token(state, t, values)
        if t < len(tokens):
            tokens[t] = tok
        else:
            tokens.append(tok)
    return Trace(tokens, state_before, first_read, frozenset(deps))


def regenerate_from_checkpoint(old: Trace, world: list[Cell], mutable_start: int, plan_seed: int, rewind: int) -> Trace:
    if rewind >= TOKENS:
        return old
    if rewind == 0:
        return generate(world, mutable_start, plan_seed)

    prefix = old.tokens[:rewind]
    start_state = old.state_before[rewind]
    # Preserve only read dependencies whose first use is before the rewind point.
    prefix_first = {dep: pos for dep, pos in old.first_read_pos.items() if pos < rewind}
    prefix_deps = set(prefix_first)
    tail = generate(
        world,
        mutable_start,
        plan_seed,
        prefix_tokens=prefix,
        start_pos=rewind,
        start_state=start_state,
        prefix_first_reads=prefix_first,
        prefix_deps=prefix_deps,
    )
    # `generate` does not know state_before for earlier prefix positions; copy it
    # only for audit/debug parity.
    tail.state_before[:rewind] = old.state_before[:rewind]
    return tail


def one_trial(rng: random.Random, mutable_start: int):
    world = init_world(rng)
    plan_seed = rng.randrange(1, 1 << 30)
    old = generate(world, mutable_start, plan_seed)

    changed_old_deps = []
    update_count = rng.randint(1, 3)
    same_value_updates = 0
    for _ in range(update_count):
        pid = rng.randrange(PODS)
        c = world[pid]
        old_dep = (pid, c.generation)
        same = rng.random() < 0.45
        old_value = c.value
        c.generation += 1
        if not same:
            c.value = rng.randrange(65536)
        else:
            same_value_updates += 1
            c.value = old_value
        changed_old_deps.append(old_dep)

    affected_positions = [old.first_read_pos[d] for d in changed_old_deps if d in old.first_read_pos]
    rewind = min(affected_positions) if affected_positions else TOKENS

    partial = regenerate_from_checkpoint(old, world, mutable_start, plan_seed, rewind)
    full = generate(world, mutable_start, plan_seed)

    # Bytes alone are insufficient: same-value generation rewrites can preserve
    # tokens while requiring a refreshed lifetime/read-set.
    output_exact = partial.tokens == full.tokens
    lifetime_exact = partial.final_deps == full.final_deps
    prefix_preserved = rewind
    invalidated_fraction = (TOKENS - rewind) / TOKENS
    return {
        "output_exact": output_exact,
        "lifetime_exact": lifetime_exact,
        "rewind": rewind,
        "prefix_preserved": prefix_preserved,
        "invalidated_fraction": invalidated_fraction,
        "had_affected_read": bool(affected_positions),
        "same_value_updates": same_value_updates,
        "same_value_changed_dep_was_read": any(d in old.first_read_pos for d in changed_old_deps) and same_value_updates > 0,
        "old_and_full_tokens_equal": old.tokens == full.tokens,
        "old_and_full_lifetimes_equal": old.final_deps == full.final_deps,
    }


def run_policy(name: str, mutable_start: int, seed_offset: int):
    rng = random.Random(SEED + seed_offset)
    exact = 0
    lifetime_exact = 0
    affected = 0
    same_value_read_witnesses = 0
    same_value_byte_equal_but_lifetime_changed = 0
    rewinds = []
    invalidated = []
    t0 = time.perf_counter()
    for _ in range(TRIALS):
        r = one_trial(rng, mutable_start)
        exact += int(r["output_exact"])
        lifetime_exact += int(r["lifetime_exact"])
        affected += int(r["had_affected_read"])
        rewinds.append(r["rewind"])
        invalidated.append(r["invalidated_fraction"])
        if r["same_value_changed_dep_was_read"]:
            same_value_read_witnesses += 1
            if r["old_and_full_tokens_equal"] and not r["old_and_full_lifetimes_equal"]:
                same_value_byte_equal_but_lifetime_changed += 1
    elapsed = time.perf_counter() - t0
    affected_invalidated = [x for x, rw in zip(invalidated, rewinds) if rw < TOKENS]
    return {
        "name": name,
        "mutable_start_token": mutable_start,
        "trials": TRIALS,
        "partial_rewind_output_exact_rate": exact / TRIALS,
        "partial_rewind_lifetime_exact_rate": lifetime_exact / TRIALS,
        "affected_update_trials": affected,
        "same_value_read_witnesses": same_value_read_witnesses,
        "same_value_byte_equal_but_lifetime_changed_witnesses": same_value_byte_equal_but_lifetime_changed,
        "mean_recomputed_fraction_all_updates": statistics.mean(invalidated),
        "mean_recomputed_fraction_affected_updates": statistics.mean(affected_invalidated) if affected_invalidated else 0.0,
        "median_rewind_token": statistics.median(rewinds),
        "p90_rewind_token": sorted(rewinds)[int(0.90 * (len(rewinds) - 1))],
        "elapsed_seconds": elapsed,
    }


def main():
    early = run_policy("early_binding", 4, 0)
    late = run_policy("late_binding", 67, 1000003)
    report = {
        "stage": STAGE,
        "architecture_candidate": "Generational Causal Rewind Decoding (GCRD)",
        "mechanism": (
            "Each mutable generation records its earliest causal entry point into an autoregressive decode. When a generation changes, "
            "the runtime rewinds to the earliest affected neural checkpoint rather than invalidating the whole answer. The safe prefix is "
            "reused; the suffix is regenerated against current capabilities and receives fresh generation lifetimes."
        ),
        "proof_obligation_tested": (
            "Partial replay must be byte-identical and generation-read-set-identical to full current-world recomputation, including "
            "same-value ABA rewrites where output bytes can remain unchanged while lifetime must change."
        ),
        "early_binding": early,
        "late_binding": late,
        "late_vs_early_recompute_fraction_reduction_affected": (
            early["mean_recomputed_fraction_affected_updates"] - late["mean_recomputed_fraction_affected_updates"]
        ),
        "full_answer_recompute_fraction": 1.0,
        "dod_status": "NOT_DOD; causal-rewind mechanism gate",
        "claim_boundary": (
            "Checkpointing, incremental recomputation, MVCC and dependency invalidation are established. R319 tests their generation-lifetime "
            "composition at an autoregressive neural decode boundary; standalone novelty is not claimed."
        ),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
