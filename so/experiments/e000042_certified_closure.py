"""Experiment E-000042 — a certified lower bound for deletion in a representation.

THE GAP THIS CLOSES, IN MY OWN WORDS. ``so/workspace.py`` says:

    "A model hands out no such trace. There is no set of directions the computation 'used' that a
     solution must intersect, so the disjointness argument does not transfer, and computing it anyway
     would be a bound that certifies nothing."

That is why a store's deletion closure comes with a proof on the low side -- every live derivation is
a must-hit set, pairwise-disjoint derivations bound the optimum from below, and E-000032 reports
``proved optimal`` at 1.00 in every arm -- while a representation's closure has been a greedy number
with nothing under it.

**The J-lens decomposition is the trace.** Anthropic's workspace work writes a residual as a sparse
nonnegative combination of lens directions. If the state a phrasing computes on is a nonnegative
combination over a support S, then any set of directions whose removal stops that phrasing yielding
the object has to touch S. Disjoint supports across phrasings then bound the closure from below by
exactly the store's argument -- and sparsity is what makes the must-hit property CHECKABLE, because
the complement of an eight-atom pool has 256 subsets where the complement of a 768-dimensional space
has 2**768.

HOW THIS IS KEPT FROM BEING THE SIXTH INSTRUMENT THAT CANNOT FAIL. The ablation table is enumerated
EXHAUSTIVELY: every one of the 2**|pool| subsets is run, and every quantity below is then read off
that one table with no further model calls. That buys three things a greedy search cannot have.

  * The TRUE optimum U*, by enumeration rather than by greedy. So the bound is not compared against
    another estimate; it is compared against the answer.
  * SOUNDNESS as a measurement. ``bound_sound`` is 1.0 only if the certified lower bound is <= U* for
    every fact. One violation anywhere falsifies the claim outright, and the criterion is written so
    that it can.
  * A CONTROL THAT CAN KILL IT. A random support of the same size is put through the identical
    exhaustive must-hit test. If random supports pass at the same rate, the pool is such that
    everything is a must-hit set, the J-lens is doing no work, and the claim is about counting rather
    than about the decomposition.

And because a bound of one is the correct answer for a fact stored as a pod -- every phrasing running
through one shared direction -- a POSITIVE CONTROL is run on a synthetic table with disjoint supports
by construction, where the bound must report the number of them. Without it, a bound that returned 1
unconditionally would look like a finding.

WHAT THE SEEDS VARY, stated because in this experiment it is nearly nothing. The model is frozen and
the table is exhaustive, so U*, the supports, the certificates and the bound are all deterministic.
The seeds resample the RANDOM-SUPPORT CONTROL, which is the only stochastic quantity here.

Trains nothing.

Run:  python -m so.experiments.e000042_certified_closure [--layer 7] [--pool 8] [--max-facts 8]
"""

from __future__ import annotations

import argparse
import os
import time
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from so import ledger
from so.support import (certified_closure, certify_must_hit, disjoint_lower_bound, nonneg_pursuit)
from so.workspace import carrier_candidates, lens_logits, project_out

MODEL = "gpt2"
LAYER = 7
POOL = 8

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
    """E-000037's ablation: project the directions out at every layer from the read layer up."""

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
        self.w_out = self.lm.get_output_embeddings().weight
        self.ln = self.lm.transformer.ln_f

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
        d = self.dirs
        self.dirs = None
        e, last = self._enc(prompts)
        hs = self.lm(**e, output_hidden_states=True).hidden_states[self.layer]
        self.dirs = d
        return hs[torch.arange(len(prompts)), last]


def _subsets(pool: Sequence[int]) -> List[Tuple[int, ...]]:
    """Every subset of the pool, smallest first, so the first silencing one found is a minimum."""
    out: List[Tuple[int, ...]] = []
    for size in range(len(pool) + 1):
        out.extend(combinations(pool, size))
    return out


def positive_control(n_groups: int = 3, per_group: int = 2) -> Dict[str, Any]:
    """The bound must be able to report more than one, or a pod-shaped model would flatter it.

    A synthetic table with ``n_groups`` pairwise-disjoint supports, each of which really is must-hit by
    construction. If the machinery cannot recover ``n_groups`` here, every 1 it reports on the model is
    uninterpretable.
    """
    supports = [tuple(range(g * per_group, (g + 1) * per_group)) for g in range(n_groups)]
    pool = list(range(n_groups * per_group))
    certs = [certify_must_hit(lambda d, s=s: bool(set(d) & set(s)), s, pool) for s in supports]
    bound = disjoint_lower_bound(certs)
    return {"expected": n_groups, "bound": bound.lower_bound, "certified": float(bound.certified),
            "ok": float(bound.lower_bound == n_groups and bound.certified)}


def run(layer: int, pool_size: int, max_facts: int, seeds: Sequence[int], threads: int,
        verbose: bool = True) -> Dict[str, Any]:
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
        got = p.restricted(prompts_of[s], caps)
        held_rate[s] = float(np.mean([g == obj_of[s] for g in got]))
    held = [s for s, _ in pairs if held_rate[s] >= 0.75]
    targets = held[:max_facts] if max_facts else held
    if verbose:
        print(f"  {len(held)}/{len(pairs)} facts answered at >= 0.75 "
              f"(mean {np.mean([held_rate[s] for s in held]):.4f}); measuring {len(targets)}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    rows: List[Dict[str, Any]] = []
    for s in targets:
        obj_id = obj_of[s]
        prompts = prompts_of[s]
        bys = [b for b in held if b != s]
        bys_prompts = [TEMPLATES[0].format(s=b) for b in bys]

        res = p.residual(prompts)                                  # (T, d) at the read layer
        ids = carrier_candidates(res.mean(0), p.w_out, obj_id, n=pool_size, ln=p.ln)
        atoms = p.w_out[torch.as_tensor(ids)].detach().to(torch.float32)
        atoms = atoms / atoms.norm(dim=1, keepdim=True)

        # ---------------------------------------------------------------- the exhaustive table
        subsets = _subsets(ids)
        table: Dict[Tuple[int, ...], np.ndarray] = {}
        coll: Dict[Tuple[int, ...], float] = {}
        for d in subsets:
            p.dirs = (atoms[[ids.index(x) for x in d]] if d else None)
            got = p.restricted(list(prompts) + bys_prompts, caps)
            table[d] = np.array([g == obj_id for g in got[:len(prompts)]])
            coll[d] = float(np.mean([g == obj_of[b] for g, b in zip(got[len(prompts):], bys)]))
        p.dirs = None

        answering = [q for q in range(len(prompts)) if table[()][q]]
        if len(answering) < 2:
            rows.append({"subject": s, "excluded": "fewer than two phrasings answer with nothing removed"})
            if verbose:
                print(f"  {s:<10} EXCLUDED: {len(answering)} phrasing(s) answer  ", flush=True)
            continue

        # ---------------------------------------------------------------- the TRUE optimum
        star = None
        for d in subsets:                                          # smallest first
            if not any(table[d][q] for q in answering):
                star = d
                break
        if star is None:
            rows.append({"subject": s, "excluded": "no subset of the pool silences every phrasing"})
            if verbose:
                print(f"  {s:<10} EXCLUDED: the pool cannot silence it  ", flush=True)
            continue

        # ---------------------------------------------------------------- greedy, for comparison
        greedy: List[int] = []
        live = list(answering)
        for atom in ids:
            if not live:
                break
            greedy.append(atom)
            live = [q for q in live if table[tuple(sorted(greedy, key=ids.index))][q]]
        greedy_key = tuple(sorted(greedy, key=ids.index))

        # ---------------------------------------------------------------- supports and certificates
        def silences_for(q: int):
            def f(d: Sequence[int]) -> bool:
                return not table[tuple(sorted((int(x) for x in d), key=ids.index))][q]
            return f

        # THE SUPPORT IS CAPPED AT HALF THE POOL, and the reason is the vacuity hole. A pursuit run to
        # a tight tolerance on a small pool takes every atom; the complement is then empty, there is no
        # disjoint ablation to try, and the must-hit property holds by having nothing to test against.
        # Capping keeps the complement non-empty, which makes the test HARDER -- a smaller set is a
        # stronger claim and easier to refute -- and the pursuit takes the largest coefficients first,
        # so what is kept is the dominant part of the state rather than an arbitrary truncation.
        n_atoms = max(1, pool_size // 2)
        supports, certs, resid = [], [], []
        for q in answering:
            sup = nonneg_pursuit(res[q], atoms, ids=ids, n_atoms=n_atoms, tol=0.05)
            supports.append(sup)
            resid.append(sup.residual_fraction)
            certs.append(certify_must_hit(silences_for(q), sup.directions, ids))
        bound = disjoint_lower_bound(certs)
        cc = certified_closure(obj_id, len(star), bound, len(answering),
                               workload=f"{len(answering)} phrasings")

        # ---------------------------------------------------------------- the control that can kill it
        # TWO CONTROLS, and the second is the one that can actually kill the claim.
        #
        # The first is a random support of the same size. If it is must-hit as often as the J-lens
        # support, this pool is one where nearly everything is must-hit and the bound is about
        # counting rather than about the decomposition.
        #
        # The second is sharper, and is the honest test of what the decomposition CONTRIBUTES. The
        # object's own lens direction is atom 0 of the pool by construction, and a support containing
        # it is must-hit for a nearly trivial reason: every ablation disjoint from that support leaves
        # the direction that reads the answer untouched. So the second control draws a random support
        # of the same size that ALSO contains atom 0. If that passes as often as the J-lens support,
        # the decomposition has added nothing beyond "include the object direction", and this
        # experiment should say so rather than report a tautology wearing a certificate.
        obj_atom = ids[0]
        rest_idx = list(range(1, len(ids)))
        rand_hold, rand_bound, rand_obj_hold = [], [], []
        for seed in seeds:
            rng = np.random.default_rng(seed * 1000 + obj_id)
            rcerts, ocerts = [], []
            for sup, q in zip(supports, answering):
                k = max(1, min(sup.size, len(ids) - 1))
                pick = tuple(sorted((int(ids[i]) for i in rng.choice(len(ids), size=k, replace=False)),
                                    key=ids.index))
                rcerts.append(certify_must_hit(silences_for(q), pick, ids))
                extra = rng.choice(len(rest_idx), size=min(k - 1, len(rest_idx) - 1), replace=False)
                opick = tuple(sorted([obj_atom] + [int(ids[rest_idx[i]]) for i in extra],
                                     key=ids.index))
                ocerts.append(certify_must_hit(silences_for(q), opick, ids))
            rand_hold.append(float(np.mean([c.holds and not c.vacuous for c in rcerts])))
            rand_obj_hold.append(float(np.mean([c.holds and not c.vacuous for c in ocerts])))
            rand_bound.append(float(disjoint_lower_bound(rcerts).lower_bound))

        row = {
            "subject": s, "excluded": None, "pool": ids, "n_answering": len(answering),
            "answer_before": held_rate[s],
            "optimum": len(star), "optimum_set": list(star), "greedy": len(greedy_key),
            "collateral_at_optimum": coll[star], "collateral_before": coll[()],
            "support_size": float(np.mean([x.size for x in supports])),
            "support_residual": float(np.mean(resid)),
            "musthit_rate": float(np.mean([c.holds for c in certs])),
            "musthit_exhaustive": float(np.mean([c.exhaustive for c in certs])),
            "musthit_vacuous": float(np.mean([c.vacuous for c in certs])),
            "musthit_subsets_tested": float(np.mean([c.subsets_tested for c in certs])),
            "lower_bound": bound.lower_bound, "bound_certified": float(bound.certified),
            "shared_atoms": len(bound.shared_atoms), "core_atoms": list(bound.shared_atoms),
            "support_atoms": [list(x.directions) for x in supports],
            "bound_sound": float(bound.lower_bound <= len(star)),
            "tightness": float(bound.lower_bound / max(len(star), 1)),
            "greedy_excess": float(len(greedy_key) - len(star)),
            "musthit_rate_random": float(np.mean(rand_hold)),
            "musthit_rate_random_with_object": float(np.mean(rand_obj_hold)),
            "object_atom_in_support": float(np.mean([obj_atom in x.directions for x in supports])),
            "lower_bound_random": float(np.mean(rand_bound)),
            "summary": cc.summary(),
        }
        rows.append(row)
        if verbose:
            print(f"  {s:<10} {cc.summary()}  | greedy {len(greedy_key)}, true optimum {len(star)}, "
                  f"support {row['support_size']:.1f} atoms explaining "
                  f"{100 * (1 - row['support_residual']):.0f}% | must-hit {row['musthit_rate']:.2f} "
                  f"vs random {row['musthit_rate_random']:.2f} | collateral "
                  f"{row['collateral_at_optimum']:.2f} from {row['collateral_before']:.2f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    good = [r for r in rows if r.get("excluded") is None]
    m: Dict[str, Any] = {"layer": layer, "pool_size": pool_size, "n_held": len(held),
                         "n_attempted": len(rows), "n_measured": len(good),
                         "positive_control": positive_control()}
    m["control_bound_can_exceed_one"] = m["positive_control"]["ok"]
    if len(good) < 3:
        m["void"] = "fewer than three facts the pool can silence; there is nothing to bound"
        m["per_fact"] = rows
        return m
    for k in ("answer_before", "optimum", "greedy", "greedy_excess", "collateral_at_optimum",
              "collateral_before", "support_size", "support_residual", "musthit_rate",
              "musthit_exhaustive", "musthit_vacuous", "musthit_subsets_tested",
              "lower_bound", "bound_certified", "shared_atoms", "bound_sound",
              "tightness", "musthit_rate_random", "musthit_rate_random_with_object",
              "object_atom_in_support", "lower_bound_random"):
        m[k] = float(np.mean([r[k] for r in good]))
    m["bound_sound_min"] = float(np.min([r["bound_sound"] for r in good]))
    m["musthit_advantage"] = m["musthit_rate"] - m["musthit_rate_random"]
    m["musthit_advantage_over_object"] = m["musthit_rate"] - m["musthit_rate_random_with_object"]

    # ------------------------------------------------------------------ the pod, both ways round
    # WITHIN a fact, the core is the atoms every access path runs through: non-empty means the fact is
    # stored as one object with several keys, which is a symlink detected in activation space, and it
    # is why the closure is small. ACROSS facts, those same atoms turning up in another fact's core is
    # the privacy failure, and it is why the collateral is not small. A design wants the first and not
    # the second; a frozen model was not asked and these two numbers say what it did anyway.
    cores = {r["subject"]: set(r["core_atoms"]) for r in good}
    m["pod_rate"] = float(np.mean([len(c) > 0 for c in cores.values()]))
    m["core_size"] = float(np.mean([len(c) for c in cores.values()]))
    shares = [len(cores[a] & cores[b]) / len(cores[a])
              for a in cores for b in cores if a != b and cores[a]]
    m["cross_fact_core_overlap"] = float(np.mean(shares)) if shares else float("nan")
    allsup = [set(x) for r in good for x in r["support_atoms"]]
    pairs = [len(x & y) / max(len(x | y), 1) for i, x in enumerate(allsup) for y in allsup[i + 1:]]
    m["support_jaccard_all"] = float(np.mean(pairs)) if pairs else float("nan")
    m["per_fact"] = rows
    m["seconds"] = time.time() - t0
    return m


KEYS = ["n_held", "n_attempted", "n_measured", "control_bound_can_exceed_one", "answer_before",
        "optimum", "greedy", "greedy_excess", "collateral_at_optimum", "collateral_before",
        "support_size", "support_residual", "musthit_rate", "musthit_exhaustive", "musthit_vacuous",
        "musthit_subsets_tested", "lower_bound",
        "bound_certified", "shared_atoms", "bound_sound", "bound_sound_min", "tightness",
        "musthit_rate_random", "musthit_rate_random_with_object", "object_atom_in_support",
        "lower_bound_random", "musthit_advantage", "musthit_advantage_over_object",
        "pod_rate", "core_size", "cross_fact_core_overlap", "support_jaccard_all"]

CRITERIA = {
    # attack validity: there must be a fact to delete
    "answer_before": (">=", 0.75),
    # the instrument must be able to report a bound above one, or a 1 everywhere means nothing
    "control_bound_can_exceed_one": (">=", 1.0),
    # THE CLAIM'S SOUNDNESS, checked against the true optimum found by enumeration. One violation
    # anywhere falsifies it, which is why the worst fact is the criterion and not the mean.
    "bound_sound_min": (">=", 1.0),
    # the certificate must be exhaustive, not truncated by budget, and must have had something to
    # test: a support filling the pool passes by having no disjoint ablation to try
    "musthit_exhaustive": (">=", 1.0),
    "musthit_vacuous": ("<=", 0.0),
    # the J-lens support must pass the must-hit test
    "musthit_rate": (">=", 0.70),
    # AND THE CONTROL THAT CAN KILL IT: a random support of the same size must NOT, or the pool is
    # one where everything is must-hit and the decomposition is doing no work
    "musthit_rate_random": ("<=", 0.40),
    "musthit_advantage": (">=", 0.30),
    # and the sharper one: the support must beat a random set that ALSO contains the object's own
    # direction, or the certificate's content is "include the object direction" and nothing else
    "musthit_advantage_over_object": (">=", 0.15),
}


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--pool", type=int, default=POOL)
    ap.add_argument("--max-facts", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    args = ap.parse_args(argv)

    m = run(args.layer, args.pool, args.max_facts, args.seeds, args.threads)
    numeric = {k: float(v) for k, v in m.items() if isinstance(v, (bool, int, float))}
    agg = ledger.aggregate([numeric], [k for k in KEYS if k in numeric])
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})

    record: Dict[str, Any] = {
        "experiment": "E-000042",
        "title": "a certified lower bound for deletion in a representation, from the J-lens support",
        "evidence_level": "E5", "trains_nothing": True, "model": MODEL, "layer": args.layer,
        "pool_size": args.pool, "seeds": args.seeds,
        "seeds_vary": "the random-support control only; the ablation table is exhaustive and the model "
                      "is frozen, so the optimum, the supports, the certificates and the bound are "
                      "deterministic",
        "result": m, "aggregate": agg, "criteria": check}

    md = [f"# E-000042 — {record['title']}", "",
          f"Frozen {MODEL}, layer {args.layer}, no training. For each fact, EVERY subset of an "
          f"{args.pool}-direction pool",
          "is ablated and the answers recorded, so the true optimum is found by enumeration and the",
          "bound is compared against the answer rather than against another estimate.", ""]
    if "void" in m:
        md += ["## VOID", "", f"No claim is made: {m['void']}", ""]
        record["void"] = m["void"]
    else:
        md += ["## The interval, against the optimum found by enumeration", "",
               ledger.table(["measure", "mean over facts"],
                            [["the model answers the fact, before anything", f"{m['answer_before']:.4f}"],
                             ["TRUE optimum, by exhaustive enumeration", f"{m['optimum']:.2f}"],
                             ["greedy upper bound", f"{m['greedy']:.2f}"],
                             ["greedy's excess over the optimum", f"{m['greedy_excess']:+.2f}"],
                             ["certified lower bound from disjoint J-lens supports",
                              f"{m['lower_bound']:.2f}"],
                             ["bound / optimum (tightness)", f"{m['tightness']:.4f}"],
                             ["**bound <= optimum, worst fact**", f"**{m['bound_sound_min']:.4f}**"]]), "",
               "## Is the support really a must-hit set", "",
               ledger.table(["measure", "mean over facts"],
                            [["atoms in the support", f"{m['support_size']:.2f}"],
                             ["share of the state the support explains",
                              f"{1.0 - m['support_residual']:.4f}"],
                             ["disjoint ablations actually tried per phrasing",
                              f"{m['musthit_subsets_tested']:.1f}"],
                             ["certificates that had nothing to test (vacuous)",
                              f"{m['musthit_vacuous']:.4f}"],
                             ["support passes the exhaustive must-hit test", f"{m['musthit_rate']:.4f}"],
                             ["a RANDOM support of the same size does (control 1)",
                              f"{m['musthit_rate_random']:.4f}"],
                             ["a random support that ALSO holds the object direction (control 2)",
                              f"{m['musthit_rate_random_with_object']:.4f}"],
                             ["the J-lens support holds the object direction",
                              f"{m['object_atom_in_support']:.4f}"],
                             ["advantage over control 1", f"{m['musthit_advantage']:+.4f}"],
                             ["**advantage over control 2**",
                              f"**{m['musthit_advantage_over_object']:+.4f}**"],
                             ["atoms every phrasing runs through (the pod core)",
                              f"{m['shared_atoms']:.2f}"]]), "",
               "Control 2 is the one that can really kill the claim. The object's own lens direction is",
               "atom 0 of the pool by construction, and any support containing it is must-hit for a",
               "nearly trivial reason: every ablation disjoint from that support leaves the direction",
               "that reads the answer untouched. So control 2 draws a random support of the same size",
               "that ALSO contains atom 0. If it passes as often as the J-lens support, the",
               "decomposition has added nothing beyond 'include the object direction', and what is",
               "reported here is a tautology wearing a certificate. The bolded row is that difference.",
               "", "Control 1 is the weaker form of the same question: if any random set of the same",
               "size is must-hit just as often, this pool is one where everything is, and the bound is",
               "about counting.", "",
               "## The pod, both ways round", "",
               "WITHIN a fact, the core is the set of atoms every access path runs through. Non-empty",
               "means the fact is stored as one object with several keys -- a symlink detected in",
               "activation space -- and it is why the closure is small. ACROSS facts, those same atoms",
               "turning up in another fact's core is the privacy failure, and it is why the collateral",
               "is not small. A design wants the first and not the second; a frozen model was not asked",
               "and these two numbers say what it did anyway.", "",
               ledger.table(["measure", "mean over facts"],
                            [["facts whose access paths share a core (stored as a pod)",
                              f"{m['pod_rate']:.4f}"],
                             ["atoms in that core", f"{m['core_size']:.2f}"],
                             ["share of a fact's core that is ANOTHER fact's core too",
                              f"{m['cross_fact_core_overlap']:.4f}"],
                             ["Jaccard overlap of supports across all facts and phrasings",
                              f"{m['support_jaccard_all']:.4f}"]]), "",
               "## What the deletion costs bystanders", "",
               ledger.table(["measure", "mean over facts"],
                            [["bystander facts with nothing removed", f"{m['collateral_before']:.4f}"],
                             ["bystander facts at the optimum", f"{m['collateral_at_optimum']:.4f}"]]), "",
               "## The positive control", "",
               f"A synthetic table with {m['positive_control']['expected']} pairwise-disjoint supports "
               f"by construction: the bound reports {m['positive_control']['bound']}. Without this, a "
               "bound that returned 1 unconditionally would read as a finding about the model.", ""]
    md += ["## Pre-registered criteria", "", ledger.criteria_table(check), ""]
    path = ledger.save("e000042_certified_closure", record, "\n".join(md))
    print("\n".join(md[1:]))
    print(f"\nwritten: {path}")
    return record


if __name__ == "__main__":
    main()
