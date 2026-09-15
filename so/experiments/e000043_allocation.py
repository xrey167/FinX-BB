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
    orthogonality = sigma_min of the stacked orthonormal bases, 1.0 iff mutually orthogonal

High pressure means the bound is binding and no objective fixes it without more dimensions. LOW
pressure with LOW orthogonality means the model had room to give each fact a private subspace and did not
-- an allocation failure, which is a training objective.

AND A REFINEMENT THE THEOREM ACTUALLY NEEDS, measured separately because it may be where the truth is.
The orthogonality the argument requires is between ``A_i`` and ``V_j``, not between ``A_i`` and
``A_j``. Deletion subspaces can be mutually independent while each still intrudes on other facts'
READOUT subspaces, and that would produce exactly the collateral E-000040 measured with orthogonality
near one. Both quantities are reported; if they disagree, the second is the one the mechanism runs
through and the first is the one that would have been quoted.

WHAT WOULD KILL THE READING. ``pressure`` above 0.5 makes the allocation reading unavailable -- the
bound would be close enough to binding that overlap is partly forced -- and it is pre-registered as a
criterion that can fail. ``orthogonality`` at 0.95 or above with collateral still present would say the
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

    # ------------------------------------------------------------------ the null, and which null
    # TWO NULLS, and the first version of this experiment used the wrong one.
    #
    # RANDOM STATES through the identical code path overlap at 0.1448 rather than 0, so a raw overlap
    # quoted against zero overstates the effect. That much was right. But random states carry NO
    # TEMPLATE STRUCTURE, and every fact here is asked with the same eight templates, so the rows of
    # the basis beyond the first are phrasing directions that the DESIGN shares out to every fact.
    # Against a null with no phrasing structure at all, any experiment of this shape reports a large
    # excess and would do so whatever the model had learned.
    #
    # The design-matched null is a PERMUTATION: within each template, shuffle which fact's state sits
    # where. Both marginals survive -- the template effect exactly, the fact effect as a multiset --
    # and only the fact x template interaction is destroyed, which is the part a claim about the MODEL
    # has to be about. Measured, it runs the other way: the permuted null overlaps MORE than the real
    # states (0.7895 against 0.5898 at six directions), so the real interaction makes the subspaces
    # more distinct rather than less. Both nulls are reported; the permutation one is primary.
    g = torch.Generator().manual_seed(0)
    rnd = {s: torch.randn(len(TEMPLATES), d, generator=g) for s in held}
    Vr: Dict[str, torch.Tensor] = {}
    for s in held:
        others = torch.stack([rnd[x] for x in held if x != s]).mean(0)
        Vr[s] = fact_basis(rnd[s], others)

    Vp_runs: List[Dict[str, torch.Tensor]] = []
    for pseed in (0, 1, 2):
        pr = np.random.default_rng(pseed)
        perm = {s: torch.zeros_like(res[s]) for s in held}
        for t in range(len(TEMPLATES)):
            order = pr.permutation(len(held))
            for i, s in enumerate(held):
                perm[s][t] = res[held[order[i]]][t]
        vp = {}
        for s in held:
            others = torch.stack([perm[x] for x in held if x != s]).mean(0)
            vp[s] = fact_basis(perm[s], others)
        Vp_runs.append(vp)
    Vn = Vp_runs[0]

    def mean_overlap(mats: Dict[str, torch.Tensor], names: Sequence[str]) -> float:
        pairs = [subspace_overlap(mats[a], mats[b])
                 for i, a in enumerate(names) for b in names[i + 1:]]
        return float(np.mean(pairs)) if pairs else float("nan")

    # ------------------------------------------------------------------ content against addressing
    # Row 0 of the basis is what every phrasing of this fact SHARES -- its content direction. Rows 1..
    # are the phrasing spread, which is how the fact is ADDRESSED. Splitting them is the whole point:
    # a fact can have a private content direction and still be undeletable because its addressing
    # machinery is shared with every other fact that is asked the same way.
    content = {s: V[s][:1] for s in held}
    address = {s: V[s][1:] for s in held}
    content_n = {s: Vn[s][:1] for s in held}
    address_n = {s: Vn[s][1:] for s in held}

    def null_mean(sel) -> Tuple[float, float]:
        perm = float(np.mean([mean_overlap({s: v[s][sel] for s in held}, held) for v in Vp_runs]))
        rand = mean_overlap({s: Vr[s][sel] for s in held}, held)
        return perm, rand

    ov_full = mean_overlap(V, held); ov_full_n, ov_full_r = null_mean(slice(None))
    ov_cont = mean_overlap(content, held); ov_cont_n, ov_cont_r = null_mean(slice(0, 1))
    ov_addr = mean_overlap(address, held); ov_addr_n, ov_addr_r = null_mean(slice(1, None))

    subs = [A[s] for s in held if s in A]
    names_A = [s for s in held if s in A]
    null_A = float(np.mean([mean_overlap({s: v[s][:A[s].shape[0]] for s in names_A}, names_A)
                            for v in Vp_runs]))
    null_A_rand = mean_overlap({s: Vr[s][:A[s].shape[0]] for s in names_A}, names_A)
    alloc = allocation(subs, d, null_overlap=null_A)

    # THE REFINEMENT: the orthogonality the theorem needs is A_i against V_j, not A_i against A_j.
    names = names_A
    av = [subspace_overlap(A[a], V[b]) for a in names for b in held if a != b]
    aa = [subspace_overlap(A[a], A[b]) for a in names for b in names if a != b]

    m: Dict[str, Any] = {
        "layer": layer, "d": d, "n_held": len(held), "n_silenced": len(subs),
        "silenced_rate": float(np.mean([r["silenced"] for r in rows])),
        "answer_before": float(np.mean([r["answer_before"] for r in rows])),
        "answer_after_all": float(np.mean([r["answer_after"] for r in rows])),
        "answer_after": float(np.mean([r["answer_after"] for r in rows if r["silenced"]]))
        if any(r["silenced"] for r in rows) else float("nan"),
        "collateral": float(np.mean([r["collateral"] for r in rows])),
        "dim_A_mean": float(np.mean([r["dim_A"] for r in rows])),
        "pressure": alloc.pressure, "orthogonality": alloc.orthogonality,
        "rank_efficiency": alloc.rank_efficiency, "headroom": alloc.headroom,
        "union_rank": float(alloc.union_rank), "sum_dims": float(sum(alloc.dims)),
        "overlap_AA_max": alloc.max_overlap, "overlap_AA_mean": alloc.mean_overlap,
        "overlap_AV_max": float(np.max(av)) if av else float("nan"),
        "overlap_AV_mean": float(np.mean(av)) if av else float("nan"),
        "null_overlap": alloc.null_overlap, "excess_overlap": alloc.excess,
        "null_overlap_random": null_A_rand, "excess_overlap_random": alloc.mean_overlap - null_A_rand,
        "null_full_random": ov_full_r, "null_content_random": ov_cont_r,
        "null_address_random": ov_addr_r,
        "overlap_full": ov_full, "null_full": ov_full_n, "excess_full": ov_full - ov_full_n,
        "overlap_content": ov_cont, "null_content": ov_cont_n, "excess_content": ov_cont - ov_cont_n,
        "overlap_address": ov_addr, "null_address": ov_addr_n, "excess_address": ov_addr - ov_addr_n,
        "address_over_content": (ov_addr - ov_addr_n) - (ov_cont - ov_cont_n),
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
        "answer_after_all", "dim_A_mean", "pressure", "orthogonality", "rank_efficiency",
        "headroom", "union_rank", "sum_dims",
        "overlap_AA_max", "overlap_AA_mean", "overlap_AV_max", "overlap_AV_mean", "capacity_bound",
        "null_overlap", "excess_overlap", "null_overlap_random", "excess_overlap_random",
        "overlap_full", "null_full", "excess_full", "null_full_random",
        "null_content_random", "null_address_random",
        "overlap_content", "null_content", "excess_content",
        "overlap_address", "null_address", "excess_address", "address_over_content"]

CRITERIA = {
    # attack validity, and a deletion to measure
    "answer_before": (">=", 0.75),
    # over the facts a subspace of their own basis actually silences; the rate at which that fails
    # is its own criterion rather than being averaged into this one
    "answer_after": ("<=", 0.25),
    "silenced_rate": (">=", 0.50),
    # THE READING DEPENDS ON THIS AND IT CAN FAIL: if the deletion subspaces demand more than half the
    # dimension budget, the bound is close enough to binding that overlap is partly forced, and
    # "allocation, not capacity" is not available as a conclusion.
    "pressure": ("<=", 0.50),
    # THE CLAIM: with budget to spare, the subspaces still are not independent. Failing this would say
    # the deletion subspaces ARE well allocated and the collateral comes from somewhere else -- which
    # overlap_AV is measured in the same run to catch.
    # the deletion subspaces must overlap MORE than a matched null, or there is no effect to explain
    "excess_overlap": (">=", 0.10),
    # THESE TWO WERE REGISTERED AGAINST A RANDOM-STATE NULL AND ARE KEPT AS REGISTERED so the record
    # shows them failing rather than being quietly rewritten. Against the design-matched permutation
    # null both reverse: excess_overlap is -0.1306 and address_over_content is -0.2170. The reading
    # they encoded -- that the model allocates badly and that the sharing is in the addressing --
    # is withdrawn. What replaced it is in the report above.
    "address_over_content": (">=", 0.10),
}

DECISION_RULE = (
    "pressure <= 0.5 and orthogonality <= 0.95 -> ALLOCATION, not capacity: the model had room for private "
    "subspaces and did not take it, so a training objective is the remedy. pressure > 0.8 -> the bound "
    "is binding and no objective fixes it without more dimensions. orthogonality > 0.95 with collateral "
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
                        ["mean pairwise overlap of the deletion subspaces",
                         f"{m['overlap_AA_mean']:.4f}"],
                        ["the same on a MATCHED NULL (random states, identical construction)",
                         f"{m['null_overlap']:.4f}"],
                        ["**excess over the null**", f"**{m['excess_overlap']:+.4f}**"],
                        ["sigma_min of the stacked bases (a dependency check, not a summary)",
                         f"{m['orthogonality']:.4f}"]]), "",
          "## Where the sharing is: content or addressing", "",
          "Row 0 of a fact's basis is what all its phrasings share -- its CONTENT direction. Rows 1 and",
          "up are the phrasing spread, which is how the fact is ADDRESSED. Both are compared against",
          "the same matched null.", "",
          ledger.table(["subspace", "overlap", "matched null", "excess"],
                       [["content direction only", f"{m['overlap_content']:.4f}",
                         f"{m['null_content']:.4f}", f"{m['excess_content']:+.4f}"],
                        ["addressing rows only", f"{m['overlap_address']:.4f}",
                         f"{m['null_address']:.4f}", f"{m['excess_address']:+.4f}"],
                        ["the whole basis", f"{m['overlap_full']:.4f}",
                         f"{m['null_full']:.4f}", f"{m['excess_full']:+.4f}"],
                        ["**addressing minus content**", "", "",
                         f"**{m['address_over_content']:+.4f}**"]]), "",
          "READ THE SIGNS. Against the design-matched null the addressing rows overlap LESS than",
          "chance, not more: permuting the fact x template interaction RAISES their overlap. So the",
          "sharing in the addressing is a property of the design -- every fact asked with the same",
          "templates -- and the model's own structure makes those subspaces more distinct rather than",
          "less. An earlier version of this experiment used a random-state null, which carries no",
          "template structure at all, and reported the opposite sign on both rows.", "",
          "What survives is the structural point, and it is stronger for not being a training defect:",
          "a fact's deletion subspace necessarily CONTAINS addressing directions, and addressing is",
          "shared across facts because facts are asked in the same ways. In a store the address and",
          "the object are separate records, so deleting the object leaves the addressing untouched. In",
          "a representation they cannot be pulled apart by allocation, because the sharing is in the",
          "task and not in the model.", "",
          "## What the deletion costs, and where the overlap actually is", "",
          ledger.table(["measure", "value"],
                       [["the model answers, before", f"{m['answer_before']:.4f}"],
                        ["facts a subspace of their own basis silences", f"{m['silenced_rate']:.4f}"],
                        ["answers after its own deletion, over those", f"{m['answer_after']:.4f}"],
                        ["the same over every fact, silenced or not", f"{m['answer_after_all']:.4f}"],
                        ["bystander facts under the same ablation", f"{m['collateral']:.4f}"],
                        ["overlap of A_i with A_j, mean", f"{m['overlap_AA_mean']:.4f}"],
                        ["overlap of A_i with A_j, max", f"{m['overlap_AA_max']:.4f}"],
                        ["**overlap of A_i with V_j** (what the theorem needs), mean",
                         f"**{m['overlap_AV_mean']:.4f}**"],
                        ["overlap of A_i with V_j, max", f"{m['overlap_AV_max']:.4f}"]]), "",
          "The last two rows are the refinement. The orthogonality the argument requires is between a",
          "fact's DELETION subspace and every other fact's READOUT subspace, not between two deletion",
          "subspaces. They can be mutually independent while each still intrudes on what other facts",
          "read from, and that produces collateral with orthogonality near one.", "",
          "", "TWO INSTRUMENT FAULTS THIS EXPERIMENT FOUND IN ITSELF, both recorded because each "
          "changed a published number. (1) It first measured `rank(union)/total`, which is LINEAR "
          "INDEPENDENCE, and reported 1.0000 on twelve subspaces whose pairwise principal cosines "
          "were 0.5566 and 0.8559 -- a direct sum, nowhere near orthogonal. (2) It then compared "
          "overlap against a RANDOM-STATE null, which carries no template structure, and reported "
          "+0.4118 where the design-matched permutation null gives -0.1306. The verdict reversed.",
          "",
          "## Verdict", "", m["verdict"], "", "## The rule, fixed before the numbers", "",
          DECISION_RULE, "", "## Pre-registered criteria", "", ledger.criteria_table(check), ""]
    path = ledger.save("e000043_allocation", record, "\n".join(md))
    print("\n".join(md[1:]))
    print(f"\nwritten: {path}")
    return record


if __name__ == "__main__":
    main()
