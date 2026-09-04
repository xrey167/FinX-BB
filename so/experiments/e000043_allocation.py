"""Experiment E-000043 — is the failure to delete cleanly a capacity limit or an allocation failure?

THE ARGUMENT, and why it needs a measurement rather than an assertion. A clean deletion of fact i is
an orthogonal projection removing a minimal subspace ``A_i`` such that no access path of fact i still
yields the object and every access path of every other fact still does. Minimality makes every
direction of ``A_i`` load-bearing for one of fact i's paths; zero collateral requires the projection to
fix every other fact's readout subspace ``V_j``. Hence ``A_i`` must be orthogonal to ``V_j`` for all
j != i, mutually orthogonal subspaces satisfy ``sum_i dim A_i <= d``, and so

    n <= d / s    -- CLEAN-DELETION CAPACITY IS LINEAR IN THE DIMENSION,

while representation capacity is not: Johnson-Lindenstrauss gives exponentially many almost-orthogonal
directions in d dimensions and superposition is the observation that models use them. Superposition
buys representation capacity and does not buy deletion capacity.

THAT IS A STATEMENT ABOUT WHAT IS POSSIBLE, AND SAYS NOTHING ABOUT WHAT THIS MODEL DID. The difference
is the whole experiment, and two numbers separate them (``so/capacity.py``):

    pressure   = sum_i dim A_i / d
    efficiency = rank(union A_i) / sum_i dim A_i

High pressure means the bound is binding and no objective fixes it without more dimensions. LOW
pressure with LOW efficiency means the model had room to give each fact a private subspace and did not
-- an allocation failure, which is a training objective.

AND A REFINEMENT THE THEOREM ACTUALLY NEEDS, measured separately because it may be where the truth is.
The orthogonality the argument requires is between ``A_i`` and ``V_j``, not between ``A_i`` and
``A_j``. Deletion subspaces can be mutually independent while each still intrudes on other facts'
READOUT subspaces, and that would produce exactly the collateral E-000040 measured with efficiency
near one. Both quantities are reported; if they disagree, the second is the one the mechanism runs
through and the first is the one that would have been quoted.

WHAT WOULD KILL THE READING. ``pressure`` above 0.5 makes the allocation reading unavailable -- the
bound would be close enough to binding that overlap is partly forced -- and it is pre-registered as a
criterion that can fail. ``efficiency`` at 0.95 or above with collateral still present would say the
deletion subspaces are already well allocated and the mechanism is elsewhere, which is why
``overlap_AV`` is measured in the same run rather than after seeing the first number.

Trains nothing.

Run:  python -m so.experiments.e000043_allocation [--layer 7]
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from so import ledger
from so.capacity import allocation, capacity_bound, subspace_overlap
from so.workspace import project_out

MODEL = "gpt2"
LAYER = 7

PAIRS = [("France", " Paris"), ("Japan", " Tokyo"), ("Italy", " Rome"), ("Germany", " Berlin"),
         ("Russia", " Moscow"), ("China", " Beijing"), ("Spain", " Madrid"), ("Egypt", " Cairo"),
         ("Canada", " Ottawa"), ("Greece", " Athens"), ("Cuba", " Havana"), ("Iran", " Tehran"),
         ("Poland", " Warsaw"), ("Norway", " Oslo"), ("Peru", " Lima"), ("Chile", " Santiago"),
         ("Sweden", " Stockholm")]

TEMPLATES = ["The capital of {s} is", "{s}'s capital city is", "Q: What is the capital of {s}? A:",
             "The city that serves as the capital of {s} is", "People say the capital of {s} is",
             "In {s}, the capital is", "The seat of government of {s} is located in",
             "Everyone knows that the capital of {s} is"]


class Probe:
    def __init__(self, layer: int, threads: int = 0):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if threads:
            torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.tok.pad_token = self.tok.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(MODEL).eval()
        from so.llm_adapter import transformer_blocks
        self.blocks = transformer_blocks(self.lm)
        self.layer = layer
        self.dirs: Optional[torch.Tensor] = None
        for l in range(layer, len(self.blocks)):
            self.blocks[l].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self.dirs is None or self.dirs.numel() == 0:
            return None
        h = output[0] if isinstance(output, tuple) else output
        h = project_out(h, self.dirs)
        return ((h,) + tuple(output[1:])) if isinstance(output, tuple) else h

    def _enc(self, prompts: Sequence[str]):
        e = self.tok(list(prompts), return_tensors="pt", padding=True)
        return e, e["attention_mask"].sum(1) - 1

    @torch.no_grad()
    def restricted(self, prompts: Sequence[str], cand: Sequence[int]) -> List[int]:
        e, last = self._enc(prompts)
        lg = self.lm(**e).logits[torch.arange(len(prompts)), last][:, torch.as_tensor(list(cand))]
        return [cand[i] for i in lg.argmax(-1).tolist()]

    @torch.no_grad()
    def residual(self, prompts: Sequence[str]) -> torch.Tensor:
        d, self.dirs = self.dirs, None
        e, last = self._enc(prompts)
        hs = self.lm(**e, output_hidden_states=True).hidden_states[self.layer]
        self.dirs = d
        return hs[torch.arange(len(prompts)), last]


def fact_basis(res_self: torch.Tensor, res_others: torch.Tensor) -> torch.Tensor:
    """E-000037's basis: the shared fact direction first, then the phrasing spread's PCs.

    This is ``V_j`` -- the subspace the fact's readout is built out of -- and the minimal deletion
    subspace ``A_j`` is searched for inside it.
    """
    spec = res_self - res_others
    centred = spec - spec.mean(0, keepdim=True)
    u, _, _ = torch.linalg.svd(centred, full_matrices=False)
    rows = [spec.mean(0)] + [centred.t() @ u[:, i] for i in range(centred.shape[0] - 1)]
    return torch.stack([r / r.norm().clamp(min=1e-8) for r in rows])


def run(layer: int, threads: int, verbose: bool = True) -> Dict[str, Any]:
    p = Probe(layer, threads)
    tok = p.tok
    pairs = [(s, o) for s, o in PAIRS if len(tok.encode(o)) == 1]
    caps = [tok.encode(o)[0] for _, o in pairs]
    obj_of = {s: tok.encode(o)[0] for s, o in pairs}
    prompts_of = {s: [t.format(s=s) for t in TEMPLATES] for s, _ in pairs}
    t0 = time.time()

    p.dirs = None
    held_rate = {}
    for s, _ in pairs:
        held_rate[s] = float(np.mean([g == obj_of[s] for g in p.restricted(prompts_of[s], caps)]))
    held = [s for s, _ in pairs if held_rate[s] >= 0.75]
    d = int(p.lm.config.n_embd)
    if verbose:
        print(f"  {len(held)}/{len(pairs)} facts answered at >= 0.75 "
              f"(mean {np.mean([held_rate[s] for s in held]):.4f}), d={d}  ({time.time()-t0:.0f}s)",
              flush=True)

    res = {s: p.residual(prompts_of[s]) for s, _ in pairs}
    V: Dict[str, torch.Tensor] = {}
    for s, _ in pairs:
        others = torch.stack([res[x] for x, _ in pairs if x != s]).mean(0)
        V[s] = fact_basis(res[s], others)

    rows: List[Dict[str, Any]] = []
    A: Dict[str, torch.Tensor] = {}
    for s in held:
        obj_id, prompts, basis = obj_of[s], prompts_of[s], V[s]
        chosen: List[int] = []
        live = list(range(len(prompts)))
        for i in range(basis.shape[0]):
            if not live:
                break
            chosen.append(i)
            p.dirs = basis[torch.as_tensor(chosen)]
            got = p.restricted(prompts, caps)
            live = [q for q in live if got[q] == obj_id]
        p.dirs = basis[torch.as_tensor(chosen)] if chosen else None
        after = float(np.mean([g == obj_id for g in p.restricted(prompts, caps)]))
        bys = [b for b in held if b != s]
        coll = float(np.mean([g == obj_of[b] for g, b in
                              zip(p.restricted([TEMPLATES[0].format(s=b) for b in bys], caps), bys)]))
        p.dirs = None
        silenced = bool(after <= 0.25)
        if silenced:
            A[s] = basis[torch.as_tensor(chosen)]
        rows.append({"subject": s, "dim_A": len(chosen), "silenced": float(silenced),
                     "answer_before": held_rate[s], "answer_after": after, "collateral": coll})
        if verbose:
            print(f"  {s:<10} dim(A) {len(chosen)} | answer {held_rate[s]:.2f} -> {after:.2f} | "
                  f"collateral {coll:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    subs = [A[s] for s in held if s in A]
    alloc = allocation(subs, d)

    # THE REFINEMENT: the orthogonality the theorem needs is A_i against V_j, not A_i against A_j.
    names = [s for s in held if s in A]
    av = [subspace_overlap(A[a], V[b]) for a in names for b in held if a != b]
    aa = [subspace_overlap(A[a], A[b]) for a in names for b in names if a != b]

    m: Dict[str, Any] = {
        "layer": layer, "d": d, "n_held": len(held), "n_silenced": len(subs),
        "silenced_rate": float(np.mean([r["silenced"] for r in rows])),
        "answer_before": float(np.mean([r["answer_before"] for r in rows])),
        "answer_after": float(np.mean([r["answer_after"] for r in rows])),
        "collateral": float(np.mean([r["collateral"] for r in rows])),
        "dim_A_mean": float(np.mean([r["dim_A"] for r in rows])),
        "pressure": alloc.pressure, "efficiency": alloc.efficiency, "headroom": alloc.headroom,
        "union_rank": float(alloc.union_rank), "sum_dims": float(sum(alloc.dims)),
        "overlap_AA_max": alloc.max_overlap, "overlap_AA_mean": alloc.mean_overlap,
        "overlap_AV_max": float(np.max(av)) if av else float("nan"),
        "overlap_AV_mean": float(np.mean(av)) if av else float("nan"),
        "capacity_bound": capacity_bound(d, alloc.pressure * d / max(len(subs), 1)) if subs else 0.0,
        "verdict": alloc.verdict(), "per_fact": rows, "seconds": time.time() - t0,
    }
    if verbose:
        print("\n  " + alloc.verdict(), flush=True)
        print(f"  overlap A_i vs A_j: mean {m['overlap_AA_mean']:.4f} max {m['overlap_AA_max']:.4f}  |  "
              f"A_i vs V_j (what the theorem needs): mean {m['overlap_AV_mean']:.4f} "
              f"max {m['overlap_AV_max']:.4f}", flush=True)
    return m


KEYS = ["d", "n_held", "n_silenced", "silenced_rate", "answer_before", "answer_after", "collateral",
        "dim_A_mean", "pressure", "efficiency", "headroom", "union_rank", "sum_dims",
        "overlap_AA_max", "overlap_AA_mean", "overlap_AV_max", "overlap_AV_mean", "capacity_bound"]

CRITERIA = {
    # attack validity, and a deletion to measure
    "answer_before": (">=", 0.75),
    "answer_after": ("<=", 0.25),
    # THE READING DEPENDS ON THIS AND IT CAN FAIL: if the deletion subspaces demand more than half the
    # dimension budget, the bound is close enough to binding that overlap is partly forced, and
    # "allocation, not capacity" is not available as a conclusion.
    "pressure": ("<=", 0.50),
    # THE CLAIM: with budget to spare, the subspaces still are not independent. Failing this would say
    # the deletion subspaces ARE well allocated and the collateral comes from somewhere else -- which
    # overlap_AV is measured in the same run to catch.
    "efficiency": ("<=", 0.95),
}

DECISION_RULE = (
    "pressure <= 0.5 and efficiency <= 0.95 -> ALLOCATION, not capacity: the model had room for private "
    "subspaces and did not take it, so a training objective is the remedy. pressure > 0.8 -> the bound "
    "is binding and no objective fixes it without more dimensions. efficiency > 0.95 with collateral "
    "still present -> the deletion subspaces are independent and the mechanism runs through A_i against "
    "V_j instead, which is measured in the same run. Fixed before the numbers were seen.")


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    args = ap.parse_args(argv)

    m = run(args.layer, args.threads)
    numeric = {k: float(v) for k, v in m.items() if isinstance(v, (bool, int, float))}
    agg = ledger.aggregate([numeric], [k for k in KEYS if k in numeric])
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})

    record = {"experiment": "E-000043",
              "title": "clean-deletion capacity, and whether GPT-2 is against the bound or below it",
              "evidence_level": "E5", "trains_nothing": True, "model": MODEL, "layer": args.layer,
              "decision_rule": DECISION_RULE, "result": m, "aggregate": agg, "criteria": check}
    md = [f"# E-000043 — {record['title']}", "",
          f"Frozen {MODEL}, d = {m['d']}, no training. For each fact the minimal subspace whose removal",
          "silences every phrasing is found, and the set of those subspaces is read two ways.", "",
          "## The dimension budget", "",
          ledger.table(["measure", "value"],
                       [["facts the model answers", f"{m['n_held']}"],
                        ["facts a subspace of their own basis silences", f"{m['n_silenced']}"],
                        ["directions per deletion, mean", f"{m['dim_A_mean']:.2f}"],
                        ["directions demanded in total", f"{m['sum_dims']:.0f} of {m['d']}"],
                        ["**pressure** (demand / d)", f"**{m['pressure']:.4f}**"],
                        ["headroom left unused", f"{m['headroom']:.4f}"],
                        ["**efficiency** (rank of the union / total)", f"**{m['efficiency']:.4f}**"]]), "",
          "## What the deletion costs, and where the overlap actually is", "",
          ledger.table(["measure", "value"],
                       [["the model answers, before", f"{m['answer_before']:.4f}"],
                        ["the model answers, after its own deletion", f"{m['answer_after']:.4f}"],
                        ["bystander facts under the same ablation", f"{m['collateral']:.4f}"],
                        ["overlap of A_i with A_j, mean", f"{m['overlap_AA_mean']:.4f}"],
                        ["overlap of A_i with A_j, max", f"{m['overlap_AA_max']:.4f}"],
                        ["**overlap of A_i with V_j** (what the theorem needs), mean",
                         f"**{m['overlap_AV_mean']:.4f}**"],
                        ["overlap of A_i with V_j, max", f"{m['overlap_AV_max']:.4f}"]]), "",
          "The last two rows are the refinement. The orthogonality the argument requires is between a",
          "fact's DELETION subspace and every other fact's READOUT subspace, not between two deletion",
          "subspaces. They can be mutually independent while each still intrudes on what other facts",
          "read from, and that produces collateral with efficiency near one.", "",
          "## Verdict", "", m["verdict"], "", "## The rule, fixed before the numbers", "",
          DECISION_RULE, "", "## Pre-registered criteria", "", ledger.criteria_table(check), ""]
    path = ledger.save("e000043_allocation", record, "\n".join(md))
    print("\n".join(md[1:]))
    print(f"\nwritten: {path}")
    return record


if __name__ == "__main__":
    main()
