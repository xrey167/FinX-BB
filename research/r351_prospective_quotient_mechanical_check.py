from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
from pathlib import Path

import numpy as np

STAGE = "R351-PROSPECTIVE-QUOTIENT-MECHANICAL-CHECK"
REPORT_PATH = Path(os.environ.get("SO_R351_REPORT", "ci-r351/report.json"))
SEED = 3510914
RANDOM_LINEAR = int(os.environ.get("SO_R351_RANDOM_LINEAR", "250000"))
RANDOM_NONLINEAR = int(os.environ.get("SO_R351_RANDOM_NONLINEAR", "100000"))


def gf2_mm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a @ b) & 1


def bits(n: int, width: int) -> np.ndarray:
    return np.array([(n >> i) & 1 for i in range(width)], dtype=np.uint8)


def linear_trial(rng: random.Random):
    # y = Tq xor Uw; cache k=Aq xor B w_old; serve y'=Ck xor Dw.
    qd, wd, kd, yd = 2, 2, 3, 2
    rand = lambda r, c: np.array([[rng.randrange(2) for _ in range(c)] for _ in range(r)], dtype=np.uint8)
    A, B, C, D, T, U = rand(kd, qd), rand(kd, wd), rand(yd, kd), rand(yd, wd), rand(yd, qd), rand(yd, wd)
    algebra_exact = (
        np.array_equal(gf2_mm(C, A), T)
        and np.count_nonzero(gf2_mm(C, B)) == 0
        and np.array_equal(D, U)
    )
    exhaustive_exact = True
    for qn, oldn, wn in itertools.product(range(1 << qd), range(1 << wd), range(1 << wd)):
        q, old, w = bits(qn, qd), bits(oldn, wd), bits(wn, wd)
        k = (A @ q ^ B @ old) & 1
        got = (C @ k ^ D @ w) & 1
        want = (T @ q ^ U @ w) & 1
        if not np.array_equal(got, want):
            exhaustive_exact = False
            break
    return algebra_exact, exhaustive_exact, int(np.count_nonzero(gf2_mm(C, B)))


def nonlinear_trial(rng: random.Random):
    # Tiny arbitrary finite functions. Q={0,1}; W={0,1,2}; cache has four states.
    Q, W, C, Y = range(2), range(3), range(4), range(2)
    # Random target semantics F(q,w).
    F = {(q,w): rng.randrange(2) for q in Q for w in W}
    # Random cache builder and serve function.
    K = {(q,old): rng.randrange(4) for q in Q for old in W}
    S = {(c,w): rng.randrange(2) for c in C for w in W}
    uff = all(S[K[q,old], w] == F[q,w] for q in Q for old in W for w in W)
    quotient_holds = True
    if uff:
        for q in Q:
            for u in W:
                for v in W:
                    if any(S[K[q,u], w] != S[K[q,v], w] for w in W):
                        quotient_holds = False
    return uff, quotient_holds


def explicit_constructive_suite():
    # Exhaustively enumerate every cache builder for a fixed nontrivial F, then
    # construct the only S rows necessary to satisfy UFF when possible. This avoids
    # relying on rare random exact protocols in the nonlinear finite check.
    Q, W, C = range(2), range(2), range(3)
    F = {(0,0):0, (0,1):1, (1,0):1, (1,1):0}
    builders = protocols = quotient_violations = physically_world_dependent_but_redundant = 0
    keys = [(q,o) for q in Q for o in W]
    for vals in itertools.product(C, repeat=len(keys)):
        builders += 1
        K = dict(zip(keys, vals))
        # A serve row for cache state c is forced to equal F(q,.) for every q that
        # can map to c. If two queries with different target behavior share c, UFF
        # is impossible. Otherwise assign the forced row; unused c is arbitrary.
        forced = {}
        possible = True
        for q in Q:
            signature = tuple(F[q,w] for w in W)
            for old in W:
                c = K[q,old]
                if c in forced and forced[c] != signature:
                    possible = False
                forced[c] = signature
        if not possible:
            continue
        protocols += 1
        S = {(c,w): (forced[c][w] if c in forced else 0) for c in C for w in W}
        holds = True
        for q in Q:
            cs = [K[q,old] for old in W]
            if len(set(cs)) > 1:
                physically_world_dependent_but_redundant += 1
            for u in W:
                for v in W:
                    if any(S[K[q,u],w] != S[K[q,v],w] for w in W):
                        holds = False
        quotient_violations += int(not holds)
    return {
        "cache_builders_enumerated": builders,
        "uff_protocols_constructible": protocols,
        "uff_protocols_with_physical_old_world_cache_variation": physically_world_dependent_but_redundant,
        "quotient_theorem_violations": quotient_violations,
    }


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    linear_algebra_exact = linear_exhaustive_exact = linear_disagreements = 0
    exact_with_observable_old_world = 0
    for _ in range(RANDOM_LINEAR):
        ae, ee, cb_rank_proxy = linear_trial(rng)
        linear_algebra_exact += int(ae)
        linear_exhaustive_exact += int(ee)
        linear_disagreements += int(ae != ee)
        exact_with_observable_old_world += int(ee and cb_rank_proxy > 0)

    nonlinear_uff = nonlinear_violations = 0
    for _ in range(RANDOM_NONLINEAR):
        uff, qh = nonlinear_trial(rng)
        nonlinear_uff += int(uff)
        nonlinear_violations += int(uff and not qh)

    constructive = explicit_constructive_suite()
    report = {
        "stage": STAGE,
        "random_linear_protocols": RANDOM_LINEAR,
        "linear_exact_by_algebra": linear_algebra_exact,
        "linear_exact_by_exhaustive_truth_table": linear_exhaustive_exact,
        "linear_algebra_vs_truth_table_disagreements": linear_disagreements,
        "linear_exact_protocols_with_decoder_observable_old_world_subspace": exact_with_observable_old_world,
        "random_nonlinear_protocols": RANDOM_NONLINEAR,
        "random_nonlinear_uff_protocols_found": nonlinear_uff,
        "random_nonlinear_quotient_violations": nonlinear_violations,
        "constructive_exhaustive_suite": constructive,
        "contract_pass": (
            linear_disagreements == 0
            and exact_with_observable_old_world == 0
            and nonlinear_violations == 0
            and constructive["quotient_theorem_violations"] == 0
            and constructive["uff_protocols_with_physical_old_world_cache_variation"] > 0
        ),
        "interpretation": (
            "The linear check mechanically confirms that exact write-maintenance-free future freshness is equivalent to CA=T, CB=0, D=U in the "
            "sampled GF(2) systems: any old-world cache component observable through C breaks exactness. The nonlinear finite check confirms the "
            "more general quotient statement: physically different historical-world cache states may exist, but every such difference must be "
            "observationally redundant for all current worlds."
        ),
        "claim_boundary": (
            "The mathematical statement is elementary and is not a novelty claim. Its research value here is as a normal-form criterion for "
            "future-bindable neural caches and a falsification rule for Prospective Neural State implementations."
        ),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["contract_pass"] else 2

if __name__ == "__main__": raise SystemExit(main())
