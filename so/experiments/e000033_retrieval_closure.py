"""Experiment E-000033 — the same closure in a chunked retrieval store, where practitioners meet it.

E-000032 measures the deletion closure inside this repository's own store and model. The obvious
objection is that both are ours: the pod is one shape a knowledge store can take, and a result about
it is a result about our design. This experiment moves the same measurement into the arrangement
almost every deployed system actually uses -- a corpus of text chunks, embedded and retrieved by
cosine similarity -- and changes nothing else.

THE TWO ARRANGEMENTS. The same 150 facts, the same twelve surface forms from E-000017, the same
frozen GPT-2 as the embedder, indexed two ways:

  duplicated   each fact is written out in k different phrasings, each its own chunk carrying both the
               question form and the answer. This is what overlapping chunking, re-ingested documents
               and paraphrased sources produce, and it is the default.
  canonical    each fact is written out ONCE. The other k-1 chunks carry the question form and a
               POINTER -- "... record #17" -- and no answer at all. Retrieval that lands on a pointer
               dereferences it. This is the symlink, in a substrate that has no symlinks.

The addressing text is identical in both arms. What differs is only whether the answer is repeated.

WHAT IS MEASURED, AND WHY THE SECOND HALF MATTERS MORE THAN THE FIRST.

That deleting one chunk of k leaves the fact retrievable is not news; it is Codd's deletion anomaly in
a vector index, and any practitioner would predict it. The half that is not obvious is what it costs
to FIND the closure, because a duplicated store's defenders will say: search for the fact and delete
every chunk that matches. So the experiment asks that search to perform, and reports the trade-off it
cannot escape:

  recall at precision 1.0     how much of the closure a similarity search recovers before it starts
                              deleting chunks about OTHER facts
  precision at recall 1.0     how many innocent chunks it must destroy to be complete

In the canonical arm there is nothing to search for: the store holds the pointers, so the closure is a
lookup and its recall is 1.0 by construction with no false positives. That is the difference the
symlink buys, stated in a currency a retrieval engineer already uses.

WHAT COULD FALSIFY IT. Both arms must ANSWER before any deletion, or the comparison is between a
working store and a broken one; the pod pays for its dereference and that price is measured here, in
this substrate, rather than assumed from E-000025. And the search must be given its best shot: the
threshold is swept and the best achievable point is reported, not a threshold chosen to lose.

Trains nothing. Downloads GPT-2 only (already cached by the other experiments).

Run:  python -m so.experiments.e000033_retrieval_closure [--seeds 0 1 2] [--n-facts 150] [--k 4]
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from so import ledger
from so.experiments.e000008_gpt2_adapter import select_entities
from so.experiments.e000017_paraphrase_gap import TEMPLATES12

N_FACTS, K, N_RELATIONS = 150, 4, 4
MODEL = "gpt2"


# ------------------------------------------------------------------------------------ the corpus
def build_facts(rng: np.random.Generator, names: List[str], n_facts: int) -> List[Tuple[int, int, int]]:
    """``(subject, relation, object)`` with distinct subjects, so one key names one fact."""
    subs = rng.choice(len(names), size=n_facts, replace=False)
    return [(int(s), int(rng.integers(N_RELATIONS)), int(rng.integers(len(names)))) for s in subs]


def statement(names, fact, template: int) -> str:
    s, r, o = fact
    return f"{TEMPLATES12[r][template].format(s=names[s])} {names[o]}."


def pointer(names, fact, template: int, record_id: int) -> str:
    """The same addressing text, and a reference where the answer would be. No payload."""
    s, r, _ = fact
    return f"{TEMPLATES12[r][template].format(s=names[s])} record #{record_id}."


def question(names, fact, template: int) -> str:
    s, r, _ = fact
    return TEMPLATES12[r][template].format(s=names[s])


def build_index(names, facts, k: int, canonical: bool) -> Tuple[List[str], List[int], List[int]]:
    """``(texts, fact_of_chunk, target_of_chunk)``; ``target`` is -1 for a chunk that holds an answer.

    Both arms produce exactly k chunks per fact with exactly the same addressing text; the canonical
    arm replaces k-1 of the ANSWERS with a pointer at the one chunk that keeps it.
    """
    texts: List[str] = []
    fact_of: List[int] = []
    target_of: List[int] = []
    for i, f in enumerate(facts):
        record = len(texts)
        texts.append(statement(names, f, 0)); fact_of.append(i); target_of.append(-1)
        for t in range(1, k):
            if canonical:
                texts.append(pointer(names, f, t, record)); target_of.append(record)
            else:
                texts.append(statement(names, f, t)); target_of.append(-1)
            fact_of.append(i)
    return texts, fact_of, target_of


# ------------------------------------------------------------------------------------ the embedder
@torch.no_grad()
def embed(texts: Sequence[str], tok, model, batch: int = 64) -> np.ndarray:
    """Mean-pooled last hidden state over the real tokens, L2-normalised. A standard cheap encoder."""
    out: List[np.ndarray] = []
    for i in range(0, len(texts), batch):
        chunk = list(texts[i: i + batch])
        enc = tok(chunk, return_tensors="pt", padding=True)
        h = model(**enc, output_hidden_states=True).hidden_states[-1]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        out.append(torch.nn.functional.normalize(pooled, dim=-1).numpy())
    return np.concatenate(out).astype(np.float32)


def retrieve(qv: np.ndarray, index: np.ndarray, live: np.ndarray, top: int = 5) -> np.ndarray:
    """Top-``top`` LIVE chunks per query, most similar first. Deletion is a liveness bit, as in a store."""
    sim = qv @ index.T
    sim[:, ~live] = -np.inf
    return np.argsort(-sim, axis=1)[:, :top]


def resolve(hit: int, target_of: Sequence[int], live: np.ndarray, depth: int = 2) -> int:
    """Follow a pointer chunk to the chunk that holds the answer; -1 if it dangles."""
    cur = int(hit)
    for _ in range(depth + 1):
        if cur < 0 or not live[cur]:
            return -1
        if target_of[cur] < 0:
            return cur
        cur = int(target_of[cur])
    return -1


# --------------------------------------------------------------------------------- the measurement
def read_rate(names, facts, tok, model, qv_by_template, index, live, fact_of, target_of,
              which: Sequence[int], templates: Sequence[int]) -> float:
    """Does the store still answer, for these facts, across these phrasings?

    An answer counts only if the chunk finally reached is the one that states THIS fact -- following a
    pointer where there is one -- so a dangling pointer is a miss and never a leak.
    """
    hits = 0
    total = 0
    for t in templates:
        top = retrieve(qv_by_template[t][which], index, live, top=1)
        for row, i in enumerate(which):
            got = resolve(int(top[row, 0]), target_of, live)
            hits += int(got >= 0 and fact_of[got] == i and target_of[got] < 0)
            total += 1
    return hits / max(total, 1)


def search_closure(index: np.ndarray, live: np.ndarray, deleted_vec: np.ndarray,
                   truth: Sequence[int], thresholds: np.ndarray) -> Tuple[float, float]:
    """Give the content search its best shot, and report the trade-off it cannot escape.

    Returns ``(best recall while precision is still 1.0, precision when recall first reaches 1.0)``.
    """
    sim = (index @ deleted_vec).astype(np.float64)
    sim[~live] = -np.inf
    want = set(int(x) for x in truth)
    best_recall_at_p1 = 0.0
    precision_at_r1 = 0.0
    for th in thresholds:
        found = set(int(i) for i in np.nonzero(sim >= th)[0])
        if not found:
            continue
        tp = len(found & want)
        precision = tp / len(found)
        recall = tp / max(len(want), 1)
        if precision >= 1.0:
            best_recall_at_p1 = max(best_recall_at_p1, recall)
        if recall >= 1.0:
            precision_at_r1 = max(precision_at_r1, precision)
    return best_recall_at_p1, precision_at_r1


def run_seed(seed: int, n_facts: int, k: int, verbose: bool = True) -> Dict[str, Any]:
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModel.from_pretrained(MODEL).eval()

    rng = np.random.default_rng(seed)
    names = [tok.convert_ids_to_tokens(i)[1:] for i in select_entities(tok, 256)]
    facts = build_facts(rng, names, n_facts)
    t0 = time.time()

    m: Dict[str, Any] = {"seed": seed, "n_facts": n_facts, "k": k, "n_chunks": n_facts * k,
                         "embedder": MODEL}
    qv_by_template = {t: embed([question(names, f, t) for f in facts], tok, model) for t in range(k)}
    if verbose:
        print(f"  seed {seed} queries embedded ({time.time() - t0:.0f}s)", flush=True)

    probe = rng.permutation(n_facts)[: min(50, n_facts)]
    thresholds = np.linspace(0.0, 1.0, 201)
    for arm, canonical in (("canonical", True), ("duplicated", False)):
        texts, fact_of, target_of = build_index(names, facts, k, canonical)
        index = embed(texts, tok, model)
        live = np.ones(len(texts), dtype=bool)
        m[f"{arm}/read_before_deletion"] = read_rate(names, facts, tok, model, qv_by_template, index,
                                                     live, fact_of, target_of, probe, range(k))

        # what a naive "delete this fact" does: remove the chunk the fact's own question retrieves
        still, closures, recalls, precisions, exact = [], [], [], [], []
        for i in probe:
            rows = [c for c in range(len(texts)) if fact_of[c] == i]
            answer_rows = [c for c in rows if target_of[c] < 0]
            # the fact closure: every chunk that must go before no phrasing yields the answer. In the
            # canonical arm it is the single record; in the duplicated arm it is every copy.
            closure = answer_rows
            closures.append(len(closure))

            top1 = int(retrieve(qv_by_template[0][[i]], index, live, top=1)[0, 0])
            deleted = resolve(top1, target_of, live)
            if deleted < 0:
                deleted = closure[0]
            live[deleted] = False
            still.append(read_rate(names, facts, tok, model, qv_by_template, index, live, fact_of,
                                   target_of, [i], range(k)))
            r, p = search_closure(index, live, index[deleted], [c for c in closure if c != deleted],
                                  thresholds)
            recalls.append(r); precisions.append(p)
            # the canonical store does not search: it holds the pointers, so the closure is a lookup
            exact.append(float(len(closure) == 1) if canonical else 0.0)
            live[deleted] = True

        m[f"{arm}/fact_closure_mean"] = float(np.mean(closures))
        m[f"{arm}/fact_closure_max"] = float(np.max(closures))
        m[f"{arm}/still_retrievable_after_one"] = float(np.mean(still))
        m[f"{arm}/search_recall_at_precision_1"] = float(np.mean(recalls))
        m[f"{arm}/search_precision_at_recall_1"] = float(np.mean(precisions))
        m[f"{arm}/closure_known_without_search"] = float(np.mean(exact))
        if verbose:
            print(f"  seed {seed} {arm:<11} read {m[f'{arm}/read_before_deletion']:.4f}  closure "
                  f"{m[f'{arm}/fact_closure_mean']:.2f}  still retrievable "
                  f"{m[f'{arm}/still_retrievable_after_one']:.4f}  search recall@P1 "
                  f"{m[f'{arm}/search_recall_at_precision_1']:.4f}  precision@R1 "
                  f"{m[f'{arm}/search_precision_at_recall_1']:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    m["control/read_gap"] = m["duplicated/read_before_deletion"] - m["canonical/read_before_deletion"]
    m["control/read_before_deletion"] = min(m[f"{a}/read_before_deletion"] for a in ("canonical", "duplicated"))
    m["seconds"] = time.time() - t0
    return m


KEYS = (["control/read_gap", "control/read_before_deletion", "n_chunks"] +
        [f"{a}/{x}" for a in ("canonical", "duplicated")
         for x in ("read_before_deletion", "fact_closure_mean", "fact_closure_max",
                   "still_retrievable_after_one", "search_recall_at_precision_1",
                   "search_precision_at_recall_1", "closure_known_without_search")])

CRITERIA = {
    # controls: both arms must work before anything is deleted, and comparably
    "control/read_before_deletion": (">=", 0.80),
    "control/read_gap": ("<=", 0.15),
    # the anomaly, in a vector index
    "canonical/fact_closure_max": ("<=", 1.0),
    "duplicated/fact_closure_mean": (">=", 4.0),
    "canonical/still_retrievable_after_one": ("<=", 0.10),
    "duplicated/still_retrievable_after_one": (">=", 0.50),
    # the half that is not obvious: the search cannot be both complete and clean
    "duplicated/search_recall_at_precision_1": ("<=", 0.90),
    "duplicated/search_precision_at_recall_1": ("<=", 0.90),
    "canonical/closure_known_without_search": (">=", 1.0),
}


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-facts", type=int, default=N_FACTS)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    per_seed = [run_seed(s, args.n_facts, args.k) for s in args.seeds]
    numeric = [{k: float(v) for k, v in s.items() if isinstance(v, (bool, int, float))} for s in per_seed]
    keys = [k for k in KEYS if all(k in s for s in numeric)]
    agg = ledger.aggregate(numeric, keys)
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})

    rows = [[arm,
             f"{agg[f'{arm}/read_before_deletion']['mean']:.4f}",
             f"{agg[f'{arm}/fact_closure_mean']['mean']:.2f}",
             f"{agg[f'{arm}/still_retrievable_after_one']['mean']:.4f}",
             f"{agg[f'{arm}/search_recall_at_precision_1']['mean']:.4f}",
             f"{agg[f'{arm}/search_precision_at_recall_1']['mean']:.4f}",
             f"{agg[f'{arm}/closure_known_without_search']['min']:.4f}"]
            for arm in ("canonical", "duplicated")]
    tbl = ledger.table(["index", "answers before deletion", "chunks holding the fact",
                        "still retrievable after one deletion", "search recall at precision 1.0",
                        "search precision at recall 1.0", "closure known without searching"], rows)

    record = {"experiment": "E-000033",
              "title": "the deletion closure in a chunked retrieval store",
              "trains_nothing": True, "seeds": args.seeds, "n_facts": args.n_facts, "k": args.k,
              "embedder": MODEL, "per_seed": per_seed, "aggregate": agg, "criteria": check}
    md = [f"# E-000033 — {record['title']}", "",
          f"Seeds {args.seeds}, {args.n_facts} facts, {args.k} chunks per fact, frozen {MODEL} as the",
          "embedder, cosine retrieval, no training. Both indexes carry the same addressing text for",
          "every chunk; they differ only in whether the ANSWER is repeated or replaced by a pointer.", "",
          "## Erasure in a vector index", "", tbl, "",
          "`still retrievable after one deletion` removes the chunk the fact's own question retrieves —",
          "what a naive erasure does — and then asks all four phrasings again. The two search columns",
          "give the content-based remedy its best shot: the similarity threshold is swept and the best",
          "achievable point reported. They are the half that is not obvious. Deleting one of k copies",
          "leaving the fact readable is Codd's deletion anomaly and any practitioner would predict it;",
          "that the search for the remaining copies cannot be both complete and clean is the reason",
          "canonicalisation is not merely tidier.", "",
          "## Pre-registered criteria", "", ledger.criteria_table(check), ""]
    path = ledger.save("e000033_retrieval_closure", record, "\n".join(md))
    print(f"\nwritten: {path}")
    print(tbl)
    print(ledger.criteria_table(check))
    return record


if __name__ == "__main__":
    main()
