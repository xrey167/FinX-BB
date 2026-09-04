"""Experiment E-000036 — a carrier is a pod for the readers that go through it, and a duplicate for
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

  produce   eight phrasings, answered by restricted top-1 over the candidate capitals -- E-000013's
            protocol, which records true_capital_restricted_top1 = 0.96 for this model.
  verify    the same fact asked as a judgement: is this city the capital of that country. A reader
            that reaches the fact must say yes to the true city AND no to a distractor, so a model
            that has learned to say yes cannot pass.

The prediction that would make the claim, and the one that would break it. If the fact has one
carrier for every reader, the closure over produce alone equals the closure over produce and verify
together. If the carriers are partly separate, the union costs more -- and the size of the difference
is the partialness. A null result here (the union costs the same) says the workspace is a complete pod
for these readers, which would be a cleaner world and a shorter paper.

COLLATERAL IS REPORTED WITH EVERY CLOSURE. A closure of one is worthless if the direction removed was
carrying every capital in the model, and the pair is the finding.

Trains nothing. Downloads GPT-2 only.

Run:  python -m so.experiments.e000036_workspace_closure [--layer 7] [--n-facts 16]
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
VERIFY = ["Q: Is{c} the capital of {s}? A:", "True or false:{c} is the capital of {s}. Answer:"]


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
        self.yes = self.tok.encode(" Yes")[0]
        self.no = self.tok.encode(" No")[0]

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

    def produce(self, subj: str) -> List[int]:
        """Restricted top-1 over the candidate capitals: E-000013's readout."""
        lg = self._logits([t.format(s=subj) for t in PRODUCE])[:, torch.as_tensor(self.caps)]
        return [self.caps[i] for i in lg.argmax(-1).tolist()]

    def verify(self, subj: str, true_cap: str, distractor: str) -> List[int]:
        """Yes to the true city AND no to a distractor. A model that always says yes fails.

        Returned as the object token when the reader reaches the fact and -1 when it does not, so the
        closure sees the same interface as the produce reader.
        """
        obj_id = self.tok.encode(true_cap)[0]
        prompts = ([t.format(s=subj, c=true_cap) for t in VERIFY]
                   + [t.format(s=subj, c=distractor) for t in VERIFY])
        lg = self._logits(prompts)
        yes = (lg[:, self.yes] > lg[:, self.no]).tolist()
        n = len(VERIFY)
        return [obj_id if (yes[i] and not yes[n + i]) else -1 for i in range(n)]

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

    # attack validity first: the model must hold these facts, per reader, before anything is ablated
    p.dirs = None
    prod_ok, ver_ok = {}, {}
    for i, (subj, obj) in enumerate(pairs):
        obj_id = p.tok.encode(obj)[0]
        dis = pairs[(i + 1) % len(pairs)][1]
        prod_ok[subj] = sum(g == obj_id for g in p.produce(subj)) / len(PRODUCE)
        ver_ok[subj] = sum(g == obj_id for g in p.verify(subj, obj, dis)) / len(VERIFY)
    m: Dict[str, Any] = {"layer": layer, "n_facts": len(pairs), "n_dirs": n_dirs,
                         "n_produce": len(PRODUCE), "n_verify": len(VERIFY), "model": MODEL,
                         "control/produce_before": float(np.mean(list(prod_ok.values()))),
                         "control/verify_before": float(np.mean(list(ver_ok.values())))}
    if verbose:
        print(f"  attack validity: produce {m['control/produce_before']:.4f}  "
              f"verify {m['control/verify_before']:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    all_res = {s: p.residual([t.format(s=s) for t in PRODUCE]) for s, _ in pairs}
    # only facts BOTH readers hold can separate the two workloads; anything else is uninterpretable
    strong = [(s, o) for i, (s, o) in enumerate(pairs)
              if prod_ok[s] >= 0.75 and ver_ok[s] >= 0.5]
    m["n_measured"] = len(strong)
    if verbose:
        print(f"  {len(strong)} of {len(pairs)} facts held by BOTH readers; measuring those", flush=True)

    rows: List[Dict[str, Any]] = []
    for i, (subj, obj) in enumerate(strong):
        obj_id = p.tok.encode(obj)[0]
        dis = strong[(i + 1) % len(strong)][1]
        basis = carriers_of(p, subj, all_res, n_dirs)
        bys = [(s, o) for s, o in strong if s != subj][:8]

        def set_dirs(idx):
            p.dirs = torch.stack([basis[j] for j in idx]) if idx else torch.zeros(0, basis[0].shape[0])

        def coll(idx):
            set_dirs(list(idx))
            return float(np.mean([p.produce(s)[0] == p.tok.encode(o)[0] for s, o in bys]))

        def ans_produce(idx):
            set_dirs(list(idx)); return p.produce(subj)

        def ans_verify(idx):
            set_dirs(list(idx)); return p.verify(subj, obj, dis)

        def ans_both(idx):
            set_dirs(list(idx)); return list(p.produce(subj)) + list(p.verify(subj, obj, dis))

        cand = list(range(len(basis)))
        wp = workspace_closure(ans_produce, cand, obj_id, len(PRODUCE), max_dirs=len(basis),
                               workload="produce: %d phrasings" % len(PRODUCE),
                               lens=f"fact-specific PCA at layer {layer}", collateral_with=coll)
        wv = workspace_closure(ans_verify, cand, obj_id, len(VERIFY), max_dirs=len(basis),
                               workload="verify: %d judgements" % len(VERIFY),
                               lens=f"fact-specific PCA at layer {layer}", collateral_with=coll)
        wb = workspace_closure(ans_both, cand, obj_id, len(PRODUCE) + len(VERIFY), max_dirs=len(basis),
                               workload="produce and verify together",
                               lens=f"fact-specific PCA at layer {layer}", collateral_with=coll)
        p.dirs = None
        rows.append({"subject": subj, "produce": wp.size, "verify": wv.size, "both": wb.size,
                     "produce_exhausted": wp.exhausted, "verify_exhausted": wv.exhausted,
                     "both_exhausted": wb.exhausted, "produce_optimal": wp.optimal,
                     "collateral_after": wb.collateral, "collateral_before": wb.collateral_before,
                     "produce_lower": wp.lower_bound, "both_lower": wb.lower_bound})
        if verbose:
            print(f"  {subj:8s} produce {wp.size:2d}  verify {wv.size:2d}  union {wb.size:2d}  "
                  f"collateral {wb.collateral_before:.2f} -> {wb.collateral:.2f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    def agg(key):
        vals = [r[key] for r in rows]
        return float(np.mean(vals)) if vals else float("nan")

    m["produce/closure_mean"] = agg("produce")
    m["verify/closure_mean"] = agg("verify")
    m["union/closure_mean"] = agg("both")
    m["union/minus_produce"] = m["union/closure_mean"] - m["produce/closure_mean"]
    m["produce/exhausted_rate"] = agg("produce_exhausted")
    m["union/exhausted_rate"] = agg("both_exhausted")
    m["collateral_before"] = agg("collateral_before")
    m["collateral_after"] = agg("collateral_after")
    m["collateral_cost"] = m["collateral_before"] - m["collateral_after"]
    m["per_fact"] = rows
    m["seconds"] = time.time() - t0
    return m


KEYS = ["control/produce_before", "control/verify_before", "n_measured", "produce/closure_mean",
        "verify/closure_mean", "union/closure_mean", "union/minus_produce", "produce/exhausted_rate",
        "union/exhausted_rate", "collateral_before", "collateral_after", "collateral_cost"]

CRITERIA = {
    # the floors: a fact neither reader holds cannot be evidence that an ablation removed anything
    "control/produce_before": (">=", 0.80),
    "control/verify_before": (">=", 0.50),
    "n_measured": (">=", 6.0),
    # the search has to terminate, or the numbers are budget rather than closure
    "union/exhausted_rate": ("<=", 0.25),
    # the claim, and the direction that would break it: if the union costs no more than produce alone,
    # the carrier is a COMPLETE pod for these readers and the partialness claim is wrong
    "union/minus_produce": (">=", 0.5),
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

    tbl = ledger.table(["reader workload", "directions to remove the fact", "search exhausted"],
                       [["produce (8 phrasings)", f"{m['produce/closure_mean']:.2f}",
                         f"{m['produce/exhausted_rate']:.2f}"],
                        ["verify (2 judgements)", f"{m['verify/closure_mean']:.2f}", "-"],
                        ["both together", f"{m['union/closure_mean']:.2f}",
                         f"{m['union/exhausted_rate']:.2f}"]])

    record = {"experiment": "E-000036",
              "title": "a carrier is a pod for the readers that go through it",
              "trains_nothing": True, "model": MODEL, "layer": args.layer,
              "measures": m, "aggregate": aggd, "criteria": check}
    md = [f"# E-000036 — {record['title']}", "",
          f"Frozen {MODEL}, layer {args.layer}, {m['n_measured']} facts the model demonstrably holds",
          f"(produce {m['control/produce_before']:.4f}, verify {m['control/verify_before']:.4f} before",
          "any ablation). No training. Carriers are the fact-specific component of the residual and the",
          "principal directions of its spread across phrasings; ablation projects the residual out of",
          "their span at every layer from the read site on.", "",
          "## How many directions carry the fact, by who is reading", "", tbl, "",
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
    path = ledger.save("e000036_workspace_closure", record, "\n".join(md))
    print(f"\nwritten: {path}")
    print(tbl)
    print(ledger.criteria_table(check))
    return record


if __name__ == "__main__":
    main()
