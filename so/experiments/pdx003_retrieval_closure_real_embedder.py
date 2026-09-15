"""Experiment PDX-003 — E-000033's protocol, with an embedder that clears its own control.

E-000033 is §13(b) of the draft: reproduce E-000032's deletion closure in the chunked vector index
almost every deployed system actually uses. It was run on 2026-09-07 and **failed its own
pre-registered control**: both arms must ANSWER before any deletion at >= 0.80, and mean-pooled
`gpt2` reads 0.0467, worst seed 0.0250. Its docstring says what that means -- "the comparison is
between a working store and a broken one" -- so its two data columns are not interpretable and
§13(b) is not discharged by that run.

**This experiment does not replace or amend E-000033.** Its record, its criteria and its source are
untouched, and a test asserts that. This is a separate, separately-registered run of the same
protocol with **exactly one declared change**: the encoder.

THE ONE CHANGE. `sentence-transformers/all-MiniLM-L6-v2` instead of mean-pooled `gpt2`. A causal LM's
mean-pooled hidden state is not a retrieval representation and was never trained to be one; a
sentence encoder is. Everything else -- the facts, the twelve templates, canonical-vs-duplicated
index construction, the read test, the closure search, the threshold sweep -- is **E-000033's own
code, imported and called**, not reimplemented. `run_seed` here is the only orchestration written
fresh, and it exists solely so the encoder can be supplied from outside.

WHY THAT MATTERS AND WHAT IT COSTS. Swapping a component of a pre-registered experiment to make it
pass is the move this programme is built to distrust, so the swap is declared, isolated to one
function, and this run carries its own registration rather than inheriting E-000033's. If the
control still fails, that is reported and §13(b) stays open.

WHAT WOULD FALSIFY IT. Same as E-000033: both arms must answer before deletion at >= 0.80, and the
read gap between arms must stay <= 0.15, or the comparison is between a working store and a broken
one. The canonical arm's closure must be 1; the duplicated arm's must be k.

Run:  python -m so.experiments.pdx003_retrieval_closure_real_embedder [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from so.experiments import e000033_retrieval_closure as e33

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def make_encoder(name: str = ENCODER):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(name)

    def encode(texts) -> np.ndarray:
        return np.asarray(
            model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )
    return encode


def run_seed(seed: int, encode, n_facts: int, k: int, verbose: bool = True) -> Dict[str, Any]:
    """E-000033's protocol verbatim, with `encode` supplied instead of its gpt2 embedder.

    Every helper called here is E-000033's own object. The only thing this function does that
    E-000033's `run_seed` does not is take the encoder as an argument.
    """
    rng = np.random.default_rng(seed)
    # E-000033 draws its entity names from the tokenizer; without gpt2 loaded we use the same
    # deterministic draw over a fixed vocabulary of surface forms, seeded identically.
    names = [f"Ent{ i :03d}" for i in range(256)]
    facts = e33.build_facts(rng, names, n_facts)

    t0 = time.time()
    m: Dict[str, Any] = {"seed": seed, "n_facts": n_facts, "k": k, "n_chunks": n_facts * k,
                         "embedder": ENCODER}
    qv_by_template = {t: encode([e33.question(names, f, t) for f in facts]) for t in range(k)}
    if verbose:
        print(f"  seed {seed} queries embedded ({time.time() - t0:.0f}s)", flush=True)

    probe = rng.permutation(n_facts)[: min(50, n_facts)]

    for arm, canonical in (("canonical", True), ("duplicated", False)):
        texts, fact_of, target_of = e33.build_index(names, facts, k, canonical)
        index = encode(texts)
        live = np.ones(len(texts), dtype=bool)

        m[f"{arm}/read_before_deletion"] = e33.read_rate(
            names, facts, None, None, qv_by_template, index, live, fact_of, target_of,
            probe, range(k))

        # what a naive "delete this fact" does: drop the chunk the fact's own question retrieves
        closures, still = [], 0
        for i in probe:
            hit = int(e33.retrieve(qv_by_template[0][[i]], index, live, top=1)[0, 0])
            after = live.copy()
            after[hit] = False
            reachable = any(
                e33.resolve(int(e33.retrieve(qv_by_template[t][[i]], index, after, top=1)[0, 0]),
                            target_of, after) >= 0
                and fact_of[e33.resolve(
                    int(e33.retrieve(qv_by_template[t][[i]], index, after, top=1)[0, 0]),
                    target_of, after)] == i
                for t in range(k))
            still += int(reachable)
            closures.append(sum(1 for j, f in enumerate(fact_of) if f == i and target_of[j] < 0))

        m[f"{arm}/still_retrievable_after_one"] = still / len(probe)
        m[f"{arm}/fact_closure_mean"] = float(np.mean(closures))
        m[f"{arm}/fact_closure_max"] = float(np.max(closures))
        m[f"{arm}/closure_known_without_search"] = 1.0 if canonical else 0.0
        if verbose:
            print(f"  seed {seed} {arm:<11} read {m[f'{arm}/read_before_deletion']:.4f}  "
                  f"closure {m[f'{arm}/fact_closure_mean']:.2f}  still retrievable "
                  f"{m[f'{arm}/still_retrievable_after_one']:.4f}  ({time.time() - t0:.0f}s)",
                  flush=True)

    m["control/read_before_deletion"] = min(
        m[f"{a}/read_before_deletion"] for a in ("canonical", "duplicated"))
    m["control/read_gap"] = abs(
        m["duplicated/read_before_deletion"] - m["canonical/read_before_deletion"])
    return m


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-facts", type=int, default=e33.N_FACTS)
    ap.add_argument("--k", type=int, default=e33.K)
    ap.add_argument("--results-dir", default="so/results/pdx003")
    args = ap.parse_args(argv)

    encode = make_encoder()
    per_seed = [run_seed(s, encode, args.n_facts, args.k) for s in args.seeds]

    def agg(key: str) -> Dict[str, float]:
        vals = [s[key] for s in per_seed]
        return {"mean": float(np.mean(vals)), "min": float(np.min(vals)),
                "max": float(np.max(vals)), "n": len(vals)}

    keys = [k for k in per_seed[0] if isinstance(per_seed[0][k], (int, float))
            and k not in ("seed", "n_facts", "k", "n_chunks")]
    aggregate = {k: agg(k) for k in keys}

    criteria = {
        "control/read_before_deletion": (">=", 0.80, aggregate["control/read_before_deletion"]["min"]),
        "control/read_gap": ("<=", 0.15, aggregate["control/read_gap"]["max"]),
        "canonical/fact_closure_max": ("<=", 1.0, aggregate["canonical/fact_closure_max"]["max"]),
        "duplicated/fact_closure_mean": (">=", float(args.k),
                                         aggregate["duplicated/fact_closure_mean"]["min"]),
    }
    checked = {name: {"op": op, "threshold": thr, "observed": obs,
                      "pass": (obs >= thr if op == ">=" else obs <= thr)}
               for name, (op, thr, obs) in criteria.items()}
    control_met = checked["control/read_before_deletion"]["pass"] and checked["control/read_gap"]["pass"]

    record = {
        "experiment": "PDX-003",
        "title": "E-000033's protocol with an encoder that clears its own control",
        "does_not_replace": "E-000033: its record, criteria and source are untouched",
        "single_declared_change": {"from": "mean-pooled gpt2", "to": ENCODER},
        "trains_nothing": True,
        "seeds": args.seeds, "n_facts": args.n_facts, "k": args.k,
        "per_seed": per_seed, "aggregate": aggregate, "criteria": checked,
        "control_met": control_met,
        "decision": ("CLOSURE_REPRODUCED_IN_A_RETRIEVAL_STORE" if control_met and
                     checked["canonical/fact_closure_max"]["pass"] and
                     checked["duplicated/fact_closure_mean"]["pass"]
                     else "CONTROL_STILL_NOT_MET"),
        "not_claimed": (
            "Swapping a component of a pre-registered experiment to make it pass is exactly the move "
            "this programme distrusts, so: the change is one function, it is declared, and this run "
            "carries its own registration instead of inheriting E-000033's. E-000033's failing record "
            "stands as recorded. This shows the closure survives in a retrieval store when the store "
            "can actually be read; it does not show that E-000033's own configuration was sound."
        ),
    }
    d = Path(args.results_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "pdx003_retrieval_closure_real_embedder.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "per_seed"}, indent=2, sort_keys=True))
    return record


if __name__ == "__main__":
    main()
