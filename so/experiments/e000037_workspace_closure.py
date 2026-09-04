"""Experiment E-000037 — a carrier is a pod for the readers that go through it, and a duplicate for
the rest.

E-000032 measures the STORE closure: how many records must go before no query yields a fact. A
canonical pod makes it one. That is the guarantee this programme has been building, and it has a hole
in it that its own records keep showing. E-000017 fired the roadmap's kill criterion because deletion
does not reach phrasings nobody trained on. E-000026 records one UPDATE reaching every alias at
0.8850 against a 0.90 bar. E-000025 records alias resolution at 0.9250 on one held-out phrasing and
0.3078 on the worst. A store-side guarantee cannot explain any of that, because the store did exactly
what it promised.

THE SECOND TERM. Ask the same question of the model instead of the store: how many DIRECTIONS in its
representation must go before no reader produces the object. Call it the workspace closure
(``so/workspace.py``). A store-side pod does not imply a workspace closure of one -- sharing a record
makes one write reach every key, and says nothing about whether one direction carries the fact for
every reader.

WHY THE READERS ARE SPLIT, AND WHY THAT IS THE POINT. Anthropic's workspace paper (Gurnee, Sofroniew
et al., "Verbalizable Representations Form a Global Workspace in Language Models", Transformer
Circuits, 2026) shows a single J-lens vector behaving exactly like a pod object -- swap it and many
downstream predicates follow -- and also shows where it stops: swapping the vector for a passage's
language flips explicit report and the flexible-inference predicates and leaves continuation and
anomaly detection unmoved, while the concept still appears in the lens readouts of all four tasks. So
the carrier is shared by SOME readers and not others. In pod vocabulary that is a pod for some access
paths and a duplicate for the rest, and the closure over a workload that MIXES reader types is
exactly the number that says how partial it is.

This experiment measures that number on a public model, on facts the model demonstrably holds.

  produce    eight direct question phrasings, answered by restricted top-1 over the candidate
             capitals -- E-000013's protocol, which records true_capital_restricted_top1 = 0.96 here.
  continue   four narrative frames in which the capital is never asked for, only continued into.
             THIS IS THE PAPER'S OWN DISCONFIRMING READER: continuation is one of the two classes its
             concept swap left unmoved, so if the carrier is reader-incomplete, this is where it
             shows.
  reverse    the association read backwards, capital to country, over the candidate countries. A
             different computation on the same fact.

A judgement reader was tried first -- yes to the true city and no to a distractor -- and DROPPED
rather than shipped: GPT-2 small scores 0.031 on it, so it holds no fact for an ablation to remove
and every number computed from it would have been noise. The three readers above were each measured
before being written in: produce 0.914, continue 0.938 to 1.000, reverse 0.562 to 0.875.

The prediction that would make the claim, and the one that would break it. If the fact has one carrier
for every reader, the closure over produce alone equals the closure over all three together, and
ablating what silences produce silences continuation too. If the carriers are partly separate, the
union costs more and continuation survives -- which is what the paper reports for its own case, on a
different model and a different lens. A null result here says the workspace is a COMPLETE pod for
these readers, which would be a cleaner world and a shorter paper.

COLLATERAL IS REPORTED WITH EVERY CLOSURE. A closure of one is worthless if the direction removed was
carrying every capital in the model, and the pair is the finding.

Trains nothing. Downloads GPT-2 only.

Run:  python -m so.experiments.e000037_workspace_closure [--layer 7] [--n-facts 16]
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from so import ledger
from so.workspace import project_out, workspace_closure

MODEL = "gpt2"
LAYER = 7

PAIRS = [("France", " Paris"), ("Japan", " Tokyo"), ("Italy", " Rome"), ("Germany", " Berlin"),
         ("Russia", " Moscow"), ("China", " Beijing"), ("Spain", " Madrid"), ("Egypt", " Cairo"),
         ("Canada", " Ottawa"), ("Greece", " Athens"), ("Cuba", " Havana"), ("Iran", " Tehran"),
         ("Poland", " Warsaw"), ("Austria", " Vienna"), ("Norway", " Oslo"), ("Kenya", " Nairobi")]

PRODUCE = ["The capital of {s} is", "{s}'s capital city is", "Q: What is the capital of {s}? A:",
           "The city that serves as the capital of {s} is", "People say the capital of {s} is",
           "In {s}, the capital is", "The seat of government of {s} is located in",
           "Everyone knows that the capital of {s} is"]
# never asks for the capital, only continues into it -- the reader class the workspace paper's own
# concept swap left unmoved, measured here at 0.938 to 1.000 before anything is ablated
CONTINUE = ["I landed in the capital of {s}, which is", "She moved to the capital of {s},",
            "Walking through the capital of {s}, called", "The flight to the capital of {s} lands in"]
# the same association read backwards, over the candidate countries: a different computation
REVERSE = ["The country whose capital is{c} is", "{c} is the capital of", "{c} is the capital city of"]


class Probe:
    """The model, its hook, and the two readers. Kept in one object so the hook state is never global."""

    def __init__(self, layer: int, threads: int = 0):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if threads:
            torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.tok.pad_token = self.tok.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(MODEL).eval()
        from so.llm_adapter import transformer_blocks
        self.blocks = transformer_blocks(self.lm)
        self.dirs: Optional[torch.Tensor] = None
        for l in range(layer, len(self.blocks)):
            self.blocks[l].register_forward_hook(self._hook)
        self.layer = layer
        self.caps = [self.tok.encode(o)[0] for _, o in PAIRS]
        self.countries = [self.tok.encode(" " + s)[0] for s, _ in PAIRS]

    def _hook(self, module, inputs, output):
        if self.dirs is None or self.dirs.numel() == 0:
            return None
        h = output[0] if isinstance(output, tuple) else output
        h2 = project_out(h, self.dirs)
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2

    def _enc(self, prompts: Sequence[str]):
        e = self.tok(list(prompts), return_tensors="pt", padding=True)
        return e, e["attention_mask"].sum(1) - 1

    @torch.no_grad()
    def _logits(self, prompts: Sequence[str]) -> torch.Tensor:
        e, last = self._enc(prompts)
        out = self.lm(**e)
        return out.logits[torch.arange(len(prompts)), last]

    def _restricted(self, prompts: Sequence[str], cand: Sequence[int]) -> List[int]:
        lg = self._logits(prompts)[:, torch.as_tensor(list(cand))]
        return [cand[i] for i in lg.argmax(-1).tolist()]

    def produce(self, subj: str) -> List[int]:
        """Restricted top-1 over the candidate capitals: E-000013's readout."""
        return self._restricted([t.format(s=subj) for t in PRODUCE], self.caps)

    def continue_(self, subj: str) -> List[int]:
        """The capital is never asked for, only continued into. The paper's disconfirming reader."""
        return self._restricted([t.format(s=subj) for t in CONTINUE], self.caps)

    def reverse(self, subj: str, cap: str) -> List[int]:
        """Capital to country. Returned as the OBJECT token when it lands on the right country, so
        every reader hands the closure the same interface."""
        obj_id = self.tok.encode(cap)[0]
        want = self.tok.encode(" " + subj)[0]
        got = self._restricted([t.format(c=cap) for t in REVERSE], self.countries)
        return [obj_id if g == want else -1 for g in got]

    @torch.no_grad()
    def residual(self, prompts: Sequence[str]) -> torch.Tensor:
        e, last = self._enc(prompts)
        hs = self.lm(**e, output_hidden_states=True).hidden_states[self.layer]
        return hs[torch.arange(len(prompts)), last]


def carriers_of(p: Probe, subj: str, all_res: Dict[str, torch.Tensor], n_dirs: int) -> List[torch.Tensor]:
    """The fact-specific component, and the directions its spread across phrasings occupies.

    The first carrier is what every phrasing SHARES about this fact -- the mean of its own residuals
    minus the mean of the other facts' at the same phrasings, so the phrasing itself cancels. The rest
    are the principal components of the spread, which are the directions a per-phrasing carrier would
    live in. Ordering them this way is deliberate: the shared direction is offered first, so a closure
    of one means the fact really is carried in one place and not that the search got lucky.
    """
    others = torch.stack([all_res[s] for s, _ in PAIRS if s != subj]).mean(0)
    spec = all_res[subj] - others
    mean = spec.mean(0)
    centred = spec - mean[None]
    u, sv, _ = torch.linalg.svd(centred, full_matrices=False)
    basis = [mean] + [centred.t() @ u[:, i] for i in range(min(n_dirs - 1, centred.shape[0] - 1))]
    return [b / b.norm().clamp(min=1e-8) for b in basis]


def run(layer: int, n_facts: int, n_dirs: int, threads: int, verbose: bool = True) -> Dict[str, Any]:
    p = Probe(layer, threads)
    pairs = PAIRS[:n_facts]
    t0 = time.time()

    # attack validity first, per reader: a fact a reader does not hold cannot be evidence that an
    # ablation removed anything. A judgement reader was dropped at this step, at 0.031.
    p.dirs = None
    prod, cont, rev = {}, {}, {}
    for subj, obj in pairs:
        obj_id = p.tok.encode(obj)[0]
        prod[subj] = sum(g == obj_id for g in p.produce(subj)) / len(PRODUCE)
        cont[subj] = sum(g == obj_id for g in p.continue_(subj)) / len(CONTINUE)
        rev[subj] = sum(g == obj_id for g in p.reverse(subj, obj)) / len(REVERSE)
    m: Dict[str, Any] = {"layer": layer, "n_facts": len(pairs), "n_dirs": n_dirs,
                         "n_produce": len(PRODUCE), "n_continue": len(CONTINUE),
                         "n_reverse": len(REVERSE), "model": MODEL,
                         "control/produce_before": float(np.mean(list(prod.values()))),
                         "control/continue_before": float(np.mean(list(cont.values()))),
                         "control/reverse_before": float(np.mean(list(rev.values())))}
    if verbose:
        print(f"  attack validity: produce {m['control/produce_before']:.4f}  continue "
              f"{m['control/continue_before']:.4f}  reverse {m['control/reverse_before']:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    all_res = {s: p.residual([t.format(s=s) for t in PRODUCE]) for s, _ in pairs}
    strong = [(s, o) for s, o in pairs if prod[s] >= 0.75 and cont[s] >= 0.75]
    m["n_measured"] = len(strong)
    if verbose:
        print(f"  {len(strong)} of {len(pairs)} facts held by BOTH produce and continue; measuring those",
              flush=True)

    rows: List[Dict[str, Any]] = []
    for subj, obj in strong:
        obj_id = p.tok.encode(obj)[0]
        basis = carriers_of(p, subj, all_res, n_dirs)
        bys = [(s, o) for s, o in strong if s != subj][:8]

        def set_dirs(idx):
            p.dirs = torch.stack([basis[j] for j in idx]) if idx else torch.zeros(0, basis[0].shape[0])

        def coll(idx):
            set_dirs(list(idx))
            return float(np.mean([p.produce(s)[0] == p.tok.encode(o)[0] for s, o in bys]))

        def ans_produce(idx):
            set_dirs(list(idx)); return p.produce(subj)

        def ans_all(idx):
            set_dirs(list(idx))
            return list(p.produce(subj)) + list(p.continue_(subj)) + list(p.reverse(subj, obj))

        cand = list(range(len(basis)))
        wp = workspace_closure(ans_produce, cand, obj_id, len(PRODUCE), max_dirs=len(basis),
                               workload="produce: %d phrasings" % len(PRODUCE),
                               lens=f"fact-specific PCA at layer {layer}", collateral_with=coll)
        wu = workspace_closure(ans_all, cand, obj_id, len(PRODUCE) + len(CONTINUE) + len(REVERSE),
                               max_dirs=len(basis), workload="produce, continue and reverse together",
                               lens=f"fact-specific PCA at layer {layer}", collateral_with=coll)

        # THE REPLICATION NUMBER. Ablate exactly what silences the produce reader, then ask the other
        # two. The workspace paper reports continuation unmoved by the swap that flips report; if the
        # carrier here is reader-incomplete the same way, continuation survives.
        set_dirs(list(wp.directions))
        surv_c = float(np.mean([g == obj_id for g in p.continue_(subj)])) if wp.size else float("nan")
        surv_r = float(np.mean([g == obj_id for g in p.reverse(subj, obj)])) if wp.size else float("nan")
        p.dirs = None
        rows.append({"subject": subj, "produce": wp.size, "union": wu.size,
                     "produce_exhausted": wp.exhausted, "union_exhausted": wu.exhausted,
                     "continue_survives": surv_c, "reverse_survives": surv_r,
                     "collateral_after": wu.collateral, "collateral_before": wu.collateral_before,
                     "produce_lower": wp.lower_bound, "union_lower": wu.lower_bound})
        if verbose:
            print(f"  {subj:8s} produce {wp.size:2d}  union {wu.size:2d}  after the produce ablation: "
                  f"continue {surv_c:.2f} reverse {surv_r:.2f}  collateral "
                  f"{wu.collateral_before:.2f} -> {wu.collateral:.2f}  ({time.time() - t0:.0f}s)",
                  flush=True)

    def agg(key):
        vals = [r[key] for r in rows if not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    m["produce/closure_mean"] = agg("produce")
    m["union/closure_mean"] = agg("union")
    m["union/minus_produce"] = m["union/closure_mean"] - m["produce/closure_mean"]
    m["produce/exhausted_rate"] = agg("produce_exhausted")
    m["union/exhausted_rate"] = agg("union_exhausted")
    m["continue/survives_produce_ablation"] = agg("continue_survives")
    m["reverse/survives_produce_ablation"] = agg("reverse_survives")
    m["collateral_before"] = agg("collateral_before")
    m["collateral_after"] = agg("collateral_after")
    m["collateral_cost"] = m["collateral_before"] - m["collateral_after"]
    m["per_fact"] = rows
    m["seconds"] = time.time() - t0
    return m


KEYS = ["control/produce_before", "control/continue_before", "control/reverse_before", "n_measured",
        "produce/closure_mean", "union/closure_mean", "union/minus_produce",
        "continue/survives_produce_ablation", "reverse/survives_produce_ablation",
        "produce/exhausted_rate", "union/exhausted_rate", "collateral_before", "collateral_after",
        "collateral_cost"]

CRITERIA = {
    # the floors: a fact a reader does not hold cannot be evidence that an ablation removed anything
    "control/produce_before": (">=", 0.80),
    "control/continue_before": (">=", 0.80),
    "control/reverse_before": (">=", 0.40),
    "n_measured": (">=", 6.0),
    # the search has to terminate, or the numbers are budget rather than closure
    "union/exhausted_rate": ("<=", 0.25),
    # THE CLAIM, and the direction that breaks it. If the union costs no more than produce alone and
    # continuation does not survive, the carrier is a COMPLETE pod for these readers and the
    # partialness claim is wrong. Both are pre-registered so either can fail.
    "union/minus_produce": (">=", 0.5),
    "continue/survives_produce_ablation": (">=", 0.30),
    # and the closure must not be bought by destroying everything else
    "collateral_cost": ("<=", 0.50),
}


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--n-facts", type=int, default=len(PAIRS))
    ap.add_argument("--n-dirs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    args = ap.parse_args(argv)

    m = run(args.layer, args.n_facts, args.n_dirs, args.threads)
    numeric = [{k: float(v) for k, v in m.items() if isinstance(v, (bool, int, float))}]
    keys = [k for k in KEYS if k in numeric[0]]
    aggd = ledger.aggregate(numeric, keys)
    check = ledger.check_criteria(aggd, {k: v for k, v in CRITERIA.items() if k in aggd})

    tbl = ledger.table(["reader workload", "held before ablation", "directions to remove the fact",
                        "search exhausted"],
                       [["produce (8 phrasings)", f"{m['control/produce_before']:.4f}",
                         f"{m['produce/closure_mean']:.2f}", f"{m['produce/exhausted_rate']:.2f}"],
                        ["continue (4 narrative frames)", f"{m['control/continue_before']:.4f}", "-", "-"],
                        ["reverse (3 backward frames)", f"{m['control/reverse_before']:.4f}", "-", "-"],
                        ["all three together", "-", f"{m['union/closure_mean']:.2f}",
                         f"{m['union/exhausted_rate']:.2f}"]])
    rep = ledger.table(["after ablating exactly what silences PRODUCE", "still answers"],
                       [["continue (the paper's disconfirming reader)",
                         f"{m['continue/survives_produce_ablation']:.4f}"],
                        ["reverse", f"{m['reverse/survives_produce_ablation']:.4f}"]])

    record = {"experiment": "E-000037",
              "title": "a carrier is a pod for the readers that go through it",
              "trains_nothing": True, "model": MODEL, "layer": args.layer,
              "measures": m, "aggregate": aggd, "criteria": check}
    md = [f"# E-000037 — {record['title']}", "",
          f"Frozen {MODEL}, layer {args.layer}, {m['n_measured']} facts the model demonstrably holds",
          f"(produce {m['control/produce_before']:.4f}, continue {m['control/continue_before']:.4f},",
          f"reverse {m['control/reverse_before']:.4f} before any ablation; a judgement reader was tried",
          "and DROPPED at 0.031 rather than shipped). No training. Carriers are the fact-specific",
          "component of the residual and the",
          "principal directions of its spread across phrasings; ablation projects the residual out of",
          "their span at every layer from the read site on.", "",
          "## How many directions carry the fact, by who is reading", "", tbl, "",
          "## The replication number", "", rep, "",
          "The workspace paper reports that swapping the one J-lens vector for a passage's language",
          "flips explicit report and all three flexible-inference predicates and leaves CONTINUATION",
          "and anomaly detection unmoved, while the concept still appears in the lens readouts of all",
          "four tasks. This is that finding asked of a different model, a different lens and a",
          "different concept class: ablate exactly the directions that silence the direct question,",
          "then ask the narrative frames.", "",
          f"Collateral on facts nobody asked to delete: {m['collateral_before']:.4f} before the",
          f"ablation, {m['collateral_after']:.4f} after. A closure without that number can always be",
          "made to look good by removing more.", "",
          "## Pre-registered criteria", "", ledger.criteria_table(check), "",
          "## What this is and is not", "",
          "It is not a measurement of J-space. The lens here is a fact-specific PCA, not the Jacobian",
          "lens, and the ablation is the paper's intervention applied to a different basis. What is",
          "borrowed is the question -- does one direction carry a concept for every reader -- and the",
          "answer the paper gives for its own case: a single J-lens vector flips explicit report and",
          "flexible inference and leaves continuation and anomaly detection unmoved. Counting the",
          "directions is also not new: LEACE proves the minimum for erasing a concept against LINEAR",
          "predictors is rank of the cross-covariance. What is measured here is the closure over the",
          "model's own generative readout across a DECLARED workload of readers, reported with its",
          "collateral -- which is the form the store-side certificate of E-000032 can compose with.", ""]
    path = ledger.save("e000037_workspace_closure", record, "\n".join(md))
    print(f"\nwritten: {path}")
    print(tbl)
    print(ledger.criteria_table(check))
    return record


if __name__ == "__main__":
    main()
