"""E-000097: exact associative-transport reduction witness.

This is a scoped baseline/falsification assay, not a neural invention.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, List, Sequence, Tuple, TypeVar

T = TypeVar("T")


@dataclass
class SegmentTree(Generic[T]):
    values: List[T]
    identity: T
    compose: Callable[[T, T], T]

    def __post_init__(self) -> None:
        n = 1
        while n < len(self.values):
            n *= 2
        self.n = n
        self.tree = [self.identity for _ in range(2 * n)]
        for i, value in enumerate(self.values):
            self.tree[n + i] = value
        for i in range(n - 1, 0, -1):
            self.tree[i] = self.compose(self.tree[2 * i], self.tree[2 * i + 1])

    @property
    def root(self) -> T:
        return self.tree[1]

    def update(self, index: int, value: T) -> int:
        pos = self.n + index
        self.tree[pos] = value
        compositions = 0
        pos //= 2
        while pos:
            self.tree[pos] = self.compose(self.tree[2 * pos], self.tree[2 * pos + 1])
            compositions += 1
            pos //= 2
        return compositions


def affine_apply(s: Tuple[int, int], x: int, p: int) -> int:
    a, b = s
    return (a * x + b) % p


def affine_compose(left: Tuple[int, int], right: Tuple[int, int], p: int) -> Tuple[int, int]:
    """Return right ∘ left."""
    a1, b1 = left
    a2, b2 = right
    return ((a2 * a1) % p, (a2 * b1 + b2) % p)


def lookup_apply(s: Tuple[int, ...], x: int) -> int:
    return s[x]


def lookup_compose(left: Tuple[int, ...], right: Tuple[int, ...]) -> Tuple[int, ...]:
    """Return right ∘ left."""
    return tuple(right[left[x]] for x in range(len(left)))


def dense_affine(seq: Sequence[Tuple[int, int]], x: int, p: int) -> int:
    for s in seq:
        x = affine_apply(s, x, p)
    return x


def dense_lookup(seq: Sequence[Tuple[int, ...]], x: int) -> int:
    for s in seq:
        x = lookup_apply(s, x)
    return x


def compose_all(seq: Sequence[T], identity: T, compose: Callable[[T, T], T]) -> T:
    out = identity
    for item in seq:
        out = compose(out, item)
    return out


def run(seed_count: int, mutations_per_case: int) -> dict:
    p = 65537
    state_count = 31
    lengths = [8, 31, 64, 127]
    initial_affine = [0, 1, 2, 17, 101, 4096, 65536]
    initial_lookup = list(range(state_count))

    output_cases = 0
    output_mismatches = 0
    root_cases = 0
    root_mismatches = 0
    update_compositions = 0
    dense_transition_evals = 0
    traces = 0
    affine_summary_words = 0
    lookup_summary_words = 0

    for seed in range(seed_count):
        rng = random.Random(970000 + seed)
        for length in lengths:
            # Affine family: compact exact summary.
            aff = [(rng.randrange(1, p), rng.randrange(p)) for _ in range(length)]
            tree = SegmentTree(
                aff.copy(),
                (1, 0),
                lambda l, r: affine_compose(l, r, p),
            )
            affine_summary_words += 2 * (2 * tree.n)

            for _ in range(mutations_per_case):
                idx = rng.randrange(length)
                new = (rng.randrange(1, p), rng.randrange(p))
                aff[idx] = new
                update_compositions += tree.update(idx, new)
                traces += 1

                gold_root = compose_all(aff, (1, 0), lambda l, r: affine_compose(l, r, p))
                root_cases += 1
                if tree.root != gold_root:
                    root_mismatches += 1

                for x in initial_affine:
                    gold = dense_affine(aff, x, p)
                    got = affine_apply(tree.root, x, p)
                    dense_transition_evals += length
                    output_cases += 1
                    if gold != got:
                        output_mismatches += 1

            # Nonlinear finite-state family: arbitrary maps, exact but large summary.
            ident = tuple(range(state_count))
            maps = [tuple(rng.randrange(state_count) for _ in range(state_count)) for _ in range(length)]
            ltree = SegmentTree(maps.copy(), ident, lookup_compose)
            lookup_summary_words += state_count * (2 * ltree.n)

            for _ in range(mutations_per_case):
                idx = rng.randrange(length)
                new = tuple(rng.randrange(state_count) for _ in range(state_count))
                maps[idx] = new
                update_compositions += ltree.update(idx, new)
                traces += 1

                gold_root = compose_all(maps, ident, lookup_compose)
                root_cases += 1
                if ltree.root != gold_root:
                    root_mismatches += 1

                for x in initial_lookup:
                    gold = dense_lookup(maps, x)
                    got = lookup_apply(ltree.root, x)
                    dense_transition_evals += length
                    output_cases += 1
                    if gold != got:
                        output_mismatches += 1

    result = {
        "experiment": "E-000097",
        "scope": "associative exact-transition summary / generic dynamic segment-tree reduction",
        "seed_count": seed_count,
        "lengths": lengths,
        "mutations_per_family_length_seed": mutations_per_case,
        "mutation_traces": traces,
        "root_summary_cases": root_cases,
        "root_summary_mismatches": root_mismatches,
        "output_cases": output_cases,
        "output_mismatches": output_mismatches,
        "generic_tree_update_compositions": update_compositions,
        "dense_replay_transition_evaluations_for_checked_outputs": dense_transition_evals,
        "affine_tree_summary_words_accumulated_across_cases": affine_summary_words,
        "nonlinear_lookup_tree_summary_words_accumulated_across_cases": lookup_summary_words,
        "kill_screen_pass": root_mismatches == 0 and output_mismatches == 0,
        "decision": "KILL_ASSOCIATIVE_RECOMPOSITION_ALONE_AS_NOVELTY_SEAM" if root_mismatches == 0 and output_mismatches == 0 else "INVESTIGATE_ASSAY_FAILURE",
        "boundary": "A genuinely new compact exact neural-specific summary representation may escape; the tree/associativity itself does not.",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-count", type=int, default=32)
    parser.add_argument("--mutations", type=int, default=24)
    parser.add_argument("--results-dir", type=Path, default=Path("results/e000097"))
    args = parser.parse_args()
    result = run(args.seed_count, args.mutations)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "e000097-summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["kill_screen_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
