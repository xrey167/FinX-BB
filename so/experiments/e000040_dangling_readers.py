"""Experiment E-000040 — the traceless price in a representation: a deletion that announces itself.

E-000041 established the law in the STORE. For a fact reachable through k access paths, U -- the
records that must go for the fact to become unreachable -- falls from k to 1 as the store is
canonicalised, but T -- the records that must go for it to become unreachable AND leave no surviving
row pointing at anything removed -- is k in every cell of the spectrum. Canonicalisation is not a
reduction in the cost of erasure; it buys the cheap-but-visible regime and gives nothing in the
traceless one. E-000035 measured the disclosure that makes T > U: a pod's surviving aliases name the
deleted key at 1.0000, uniquely, against 0.0000 for a duplicated store.

This asks whether the same gap exists in a REPRESENTATION, which is the half the store cannot answer.

WHY IT IS THE SAME QUESTION. Anthropic's workspace paper (Gurnee, Sofroniew et al., Transformer
Circuits 2026) shows a J-lens direction behaving as a shared carrier: swap one and many downstream
predicates follow. That is a pod in activation space -- one write, many reads. The paper reports the
residue directly, in the case where its own intervention does NOT take effect: swapping the vector for
a passage's language leaves continuation and anomaly detection unmoved, **while the concept still
appears in the lens readouts of all four tasks**. A residue where the behaviour has changed is what a
dangling pointer looks like from the inside.

THE DELETION IS THE CLOSURE, NOT A CARRIER. A first version of this experiment removed one hand-built
fact direction at the answer position and measured what was left. It measured nothing: the answer rate
was 0.88 before and 0.88 after, so there was no deletion whose trace could be looked for. E-000037's
number is the honest one -- a fact GPT-2 holds in its own weights occupies 4 to 8 directions across
eight phrasings -- so the deletion here is that fact's **workspace closure**, computed per fact by the
same greedy procedure, with the same mandatory collateral check on facts nobody asked to delete.

WHY THE NEVER-HELD ARM IS BUILT FROM BYSTANDER STATES. The first version split a pool of countries into
facts the model holds and facts it does not, and went VOID on the measurement: of the eighteen capitals
in the pool that are single tokens, seventeen are answered at 0.9338 and none is weak. There is no weak
arm to be had at this pool, and widening it buys one only by comparing real countries against obscure
ones -- a difference an adversary could exploit with no deletion having happened. So the never-held
reference is the SAME subspace read on OTHER subjects' states: how much of a state that does not hold
this fact nevertheless lies in this fact's closure. Same span, same layer, same position, same token
type; the only difference is whether the fact is there.

THE OBSERVABLE, AND THE INTERVENTION THAT MAKES IT CONTINUOUS. Deleting by projection sets the state's
component in the closure span to exactly zero, and no natural state has a zero component in any span.
So the intervention here is not a projection but a SCALE: h -> h - (1 - a) P h, applied at every layer
from the read layer up and at every position, exactly where E-000037's ablation applies. a = 0 is the
projection, a = 1 is the identity, and a is the size of the hole an adversary measures.

  a_hide     the component a state that never held the fact has in this span, as a fraction of the
             component the state that holds it has: the smallest hole that looks natural
  a_answer   the largest hole at which the model still does not answer: the deletion's own ceiling

If a_answer < a_hide, unreachable and traceless are not jointly achievable by scaling the fact's own
component, and the erasure announces itself -- the representation-space form of T > U. If a_answer is
at or above a_hide, they are jointly achievable and the store's law does not carry over. Either way the
gap is the measurement, and it is pre-registered as such.

FOUR CONTROLS, ANY OF WHICH KILLS THE RESULT.
  answer/before      the model must answer the fact, or there was nothing to delete
  answer/deleted     the closure must actually silence it, or there is no deletion to leave a trace
  collateral         bystander facts must survive the same ablation, or what was removed is not this
                     fact but the model's ability to answer at all
  hole_detectable    the deleted component must fall below EVERY bystander's component in the same
                     span, per fact -- an adversary that cannot separate them has nothing to read

Trains nothing.

Run:  python -m so.experiments.e000040_dangling_readers [--layer 7] [--seeds 0 1 2] [--max-facts N]
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from so import ledger
from so.workspace import workspace_closure

MODEL = "gpt2"
LAYER = 7

POOL = [("France", " Paris"), ("Japan", " Tokyo"), ("Italy", " Rome"), ("Germany", " Berlin"),
        ("Russia", " Moscow"), ("China", " Beijing"), ("Spain", " Madrid"), ("Egypt", " Cairo"),
        ("Canada", " Ottawa"), ("Greece", " Athens"), ("Cuba", " Havana"), ("Iran", " Tehran"),
        ("Poland", " Warsaw"), ("Austria", " Vienna"), ("Norway", " Oslo"), ("Kenya", " Nairobi"),
        ("Latvia", " Riga"), ("Estonia", " Tallinn"), ("Nepal", " Kathmandu"), ("Malta", " Valletta"),
        ("Laos", " Vientiane"), ("Bhutan", " Thimphu"), ("Fiji", " Suva"), ("Chad", " Ndjamena"),
        ("Eritrea", " Asmara"), ("Suriname", " Paramaribo"), ("Slovenia", " Ljubljana"),
        ("Myanmar", " Naypyidaw"), ("Ghana", " Accra"), ("Peru", " Lima"), ("Chile", " Santiago"),
        ("Sweden", " Stockholm")]

TEMPLATES = ["The capital of {s} is", "{s}'s capital city is", "Q: What is the capital of {s}? A:",
             "The city that serves as the capital of {s} is", "People say the capital of {s} is",
             "In {s}, the capital is", "The seat of government of {s} is located in",
             "Everyone knows that the capital of {s} is"]

ALPHAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0]


class Probe:
    """E-000037's ablation, made continuous.

    The hook scales the state's component in ``span`` by ``alpha`` at every layer from the read layer
    up and at every position -- the same reach as E-000037's projection, which ``alpha = 0`` reproduces
    exactly. Persistent, so a later layer cannot write the component back and hand the readout
    something the rest of the model never saw.
    """

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
        self.q: Optional[torch.Tensor] = None      # (d, k) orthonormal basis of the span
        self.alpha = 1.0
        for l in range(layer, len(self.blocks)):
            self.blocks[l].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self.q is None or self.q.numel() == 0 or self.alpha == 1.0:
            return None
        h = output[0] if isinstance(output, tuple) else output
        h = h - (1.0 - self.alpha) * ((h @ self.q) @ self.q.t())
        return ((h,) + tuple(output[1:])) if isinstance(output, tuple) else h

    def set_span(self, dirs: Optional[torch.Tensor], alpha: float = 0.0) -> None:
        """``dirs`` is (k, d) and need not be orthonormal; it is orthonormalised here."""
        if dirs is None or dirs.numel() == 0:
            self.q, self.alpha = None, 1.0
            return
        self.q, _ = torch.linalg.qr(dirs.t().to(torch.float32))
        self.alpha = float(alpha)

    def _enc(self, prompts: Sequence[str]):
        e = self.tok(list(prompts), return_tensors="pt", padding=True)
        return e, e["attention_mask"].sum(1) - 1

    @torch.no_grad()
    def restricted(self, prompts: Sequence[str], cand: Sequence[int]) -> List[int]:
        """E-000013's readout: argmax over the candidate capitals, not over the whole vocabulary."""
        e, last = self._enc(prompts)
        lg = self.lm(**e).logits[torch.arange(len(prompts)), last][:, torch.as_tensor(list(cand))]
        return [cand[i] for i in lg.argmax(-1).tolist()]

    @torch.no_grad()
    def residual(self, prompts: Sequence[str]) -> torch.Tensor:
        """The unablated state at the read layer, answer position."""
        q, a = self.q, self.alpha
        self.q, self.alpha = None, 1.0
        e, last = self._enc(prompts)
        hs = self.lm(**e, output_hidden_states=True).hidden_states[self.layer]
        self.q, self.alpha = q, a
        return hs[torch.arange(len(prompts)), last]


def fact_basis(res_self: torch.Tensor, res_others: torch.Tensor) -> List[torch.Tensor]:
    """E-000037's carrier basis: the shared fact direction first, then the phrasing spread's PCs.

    ``res_self - res_others`` is the part of the state that is about this fact rather than about the
    phrasing or the position. Its mean over phrasings is what every phrasing shares; the PCs of what is
    left are the ways the phrasings differ. The mean goes first because it is the direction a caller is
    trying to remove, and a search that had to discover it would be measuring the ranking.
    """
    spec = res_self - res_others
    centred = spec - spec.mean(0, keepdim=True)
    u, _, _ = torch.linalg.svd(centred, full_matrices=False)
    basis = [spec.mean(0)] + [centred.t() @ u[:, i] for i in range(centred.shape[0] - 1)]
    return [b / b.norm().clamp(min=1e-8) for b in basis]


def component(h: torch.Tensor, q: torch.Tensor) -> float:
    """Fraction of the state's norm that lies in the span. The hole an adversary measures."""
    return float(((h @ q).norm(dim=-1) / h.norm(dim=-1).clamp(min=1e-8)).mean())


def run(layer: int, seed: int, threads: int, max_facts: int, verbose: bool = True) -> Dict[str, Any]:
    p = Probe(layer, threads)
    tok = p.tok
    pool = [(s, o) for s, o in POOL if len(tok.encode(o)) == 1]
    caps = [tok.encode(o)[0] for _, o in pool]
    obj_of = {s: tok.encode(o)[0] for s, o in pool}
    prompts_of = {s: [t.format(s=s) for t in TEMPLATES] for s, _ in pool}
    t0 = time.time()

    # attack validity: which facts the model answers, measured, never assumed
    p.set_span(None)
    held_rate = {}
    for s, _ in pool:
        got = p.restricted(prompts_of[s], caps)
        held_rate[s] = float(np.mean([g == obj_of[s] for g in got]))
    held = [s for s, _ in pool if held_rate[s] >= 0.75]
    if verbose:
        print(f"  pool {len(pool)} single-token capitals: {len(held)} answered at >= 0.75 "
              f"(mean {np.mean([held_rate[s] for s in held]):.4f})  ({time.time() - t0:.0f}s)", flush=True)
    if len(held) < 4:
        return {"seed": seed, "layer": layer, "n_pool": len(pool), "n_held": len(held),
                "void": "the model answers too few of the pool for a deletion to be measured"}

    res = {s: p.residual(prompts_of[s]) for s, _ in pool}
    rng = np.random.default_rng(seed)
    order = list(held)
    rng.shuffle(order)
    targets = order[:max_facts] if max_facts else order

    rows: List[Dict[str, Any]] = []
    for s in targets:
        obj_id = obj_of[s]
        others = torch.stack([res[x] for x, _ in pool if x != s]).mean(0)
        basis = fact_basis(res[s], others)
        prompts = prompts_of[s]
        bys = [b for b in held if b != s][:6]

        def ans(idx: Sequence[int], _b=basis, _pr=prompts) -> List[int]:
            p.set_span(torch.stack([_b[i] for i in idx]) if len(idx) else None, 0.0)
            return p.restricted(_pr, caps)

        def coll(idx: Sequence[int], _b=basis, _by=bys) -> float:
            p.set_span(torch.stack([_b[i] for i in idx]) if len(idx) else None, 0.0)
            got = p.restricted([TEMPLATES[0].format(s=b) for b in _by], caps)
            return float(np.mean([g == obj_of[b] for g, b in zip(got, _by)]))

        wc = workspace_closure(ans, list(range(len(basis))), obj_id, len(prompts),
                               max_dirs=len(basis), workload=f"{len(TEMPLATES)} phrasings",
                               lens=f"fact-specific PCA at layer {layer}",
                               collateral_with=coll, bound=False)
        p.set_span(None)
        if not wc.directions or wc.exhausted:
            rows.append({"subject": s, "closure": len(wc.directions), "exhausted": True,
                         "answer/before": held_rate[s]})
            if verbose:
                print(f"  {s:<10} NO CLOSURE within {len(basis)} directions -- excluded", flush=True)
            continue

        span = torch.stack([basis[i] for i in wc.directions])
        q, _ = torch.linalg.qr(span.t().to(torch.float32))
        q1 = basis[0].reshape(-1, 1)            # the fact's own shared direction, the deletion's target

        # THE HOLE, read two ways. The span as a whole is not fact-specific -- its later directions are
        # the phrasing spread, which every capital question has -- so a bystander state has nearly as
        # much of it as the holding state does, and a ratio built on it would be near 1 whatever the
        # deletion did. The primary statistic is therefore the component along the fact's OWN shared
        # direction, which is the part the deletion targets and the part bystanders demonstrably do not
        # share. The span ratio is reported beside it, because a defender must match EVERY statistic an
        # adversary might read with the one scale it gets to choose.
        c_held = component(res[s], q1)
        c_bys = [component(res[b], q1) for b in held if b != s]
        alpha_hide = float(np.mean(c_bys) / max(c_held, 1e-8))
        cs_held = component(res[s], q)
        cs_bys = [component(res[b], q) for b in held if b != s]
        alpha_hide_span = float(np.mean(cs_bys) / max(cs_held, 1e-8))

        # the sweep: the largest hole at which the model still does not answer
        sweep = []
        for a in ALPHAS:
            p.set_span(span, a)
            got = p.restricted(prompts, caps)
            sweep.append(float(np.mean([g == obj_id for g in got])))
        p.set_span(None)
        alpha_answer = 0.0
        for a, acc in zip(ALPHAS, sweep):
            if acc <= 0.25:
                alpha_answer = a
            else:
                break

        row = {
            "subject": s, "closure": len(wc.directions), "exhausted": False,
            "answer/before": held_rate[s], "answer/deleted": sweep[0],
            "collateral": float(wc.collateral), "collateral_before": float(wc.collateral_before),
            # ADMISSION, declared before any number was seen: the traceless question is only about a
            # deletion OF THIS FACT, so a closure that takes the bystanders down with it does not
            # enter the measurement. The rate at which that happens is itself reported.
            "specific": float(wc.collateral >= 0.60 * max(wc.collateral_before, 1e-8)),
            "component/held": c_held, "component/bystander_mean": float(np.mean(c_bys)),
            "component/bystander_min": float(np.min(c_bys)),
            "component_span/held": cs_held, "component_span/bystander_mean": float(np.mean(cs_bys)),
            "hole_detectable": float(0.0 < np.min(c_bys)),   # the projection leaves exactly 0
            "alpha_hide": alpha_hide, "alpha_hide_span": alpha_hide_span, "alpha_answer": alpha_answer,
            "traceless_gap": alpha_hide - alpha_answer,
            "traceless_gap_span": alpha_hide_span - alpha_answer,
            "one_alpha_hides_both": float(abs(alpha_hide - alpha_hide_span) < 0.05),
            "sweep": dict(zip([f"{a:g}" for a in ALPHAS], sweep)),
        }
        rows.append(row)
        if verbose:
            print(f"  {s:<10} closure {len(wc.directions)} dirs | answer {row['answer/before']:.2f} -> "
                  f"{row['answer/deleted']:.2f}, collateral {row['collateral']:.2f} from "
                  f"{row['collateral_before']:.2f}"
                  f"{'' if row['specific'] else '  [NOT SPECIFIC -- not admitted]'} | hole to hide "
                  f"{alpha_hide:.3f} (span {alpha_hide_span:.3f}), hole the deletion can afford "
                  f"{alpha_answer:.2f} -> gap {row['traceless_gap']:+.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    good = [r for r in rows if not r["exhausted"]]
    spec = [r for r in good if r["specific"]]
    m: Dict[str, Any] = {"seed": seed, "layer": layer, "n_pool": len(pool), "n_held": len(held),
                         "n_measured": len(good), "n_no_closure": len(rows) - len(good),
                         "n_specific": len(spec),
                         "specific_rate": float(len(spec) / max(len(good), 1)),
                         "collateral_all_facts": float(np.mean([r["collateral"] for r in good]))
                         if good else float("nan")}
    if len(spec) < 4:
        m["void"] = ("too few facts admitted a deletion specific to themselves: a closure that takes "
                     "the bystanders down with it is not a deletion of one fact")
        m["per_fact"] = rows
        return m
    for k in ("closure", "answer/before", "answer/deleted", "collateral", "collateral_before",
              "component/held", "component/bystander_mean", "component_span/held",
              "component_span/bystander_mean", "hole_detectable", "alpha_hide", "alpha_hide_span",
              "alpha_answer", "traceless_gap", "traceless_gap_span", "one_alpha_hides_both"):
        m[k] = float(np.mean([r[k] for r in spec]))
    m["traceless_impossible"] = float(np.mean([r["traceless_gap"] > 0 for r in spec]))
    m["mean_sweep"] = {f"{a:g}": float(np.mean([r["sweep"][f"{a:g}"] for r in spec])) for a in ALPHAS}
    m["per_fact"] = rows
    m["seconds"] = time.time() - t0
    return m


KEYS = ["n_held", "n_measured", "n_no_closure", "n_specific", "specific_rate",
        "collateral_all_facts", "closure", "answer/before", "answer/deleted", "collateral",
        "collateral_before", "component/held", "component/bystander_mean", "component_span/held",
        "component_span/bystander_mean", "hole_detectable", "alpha_hide", "alpha_hide_span",
        "alpha_answer", "traceless_gap", "traceless_gap_span", "traceless_impossible",
        "one_alpha_hides_both"]

# The criteria are VALIDITY criteria, and the sign of the gap is deliberately not among them. A
# criterion on a quantity whose direction is not predicted in advance is either trivial (set the
# threshold where the answer already is) or arbitrary, and this ledger has closed four instruments that
# certified by not testing. What is pre-registered instead is the DECISION RULE below: both signs are
# named, with what each would mean, before the measurement.
CRITERIA = {
    # attack validity: there must be a fact, and the closure must silence it
    "answer/before": (">=", 0.75),
    "answer/deleted": ("<=", 0.25),
    # the deletion must be of THIS fact -- enforced per fact as an admission rule, so what is checked
    # here is that the admitted set is not a handful cherry-picked out of a mostly-failing pool
    "collateral": (">=", 0.60),
    "specific_rate": (">=", 0.20),
    # the adversary must have something to read: the projection leaves exactly zero and no natural
    # state does. If this ever fails there is no channel and the rest of the experiment is void.
    "hole_detectable": (">=", 0.95),
}

DECISION_RULE = (
    "traceless_gap = a_hide - a_answer, over facts admitted by the collateral rule. POSITIVE: the "
    "smallest hole that looks natural is bigger than the largest hole the deletion can afford, so "
    "unreachable and traceless are not jointly achievable by scaling the fact's component and the "
    "store's law (E-000041, T > U) carries into the representation. NEGATIVE: there is a scale that "
    "both silences the fact and leaves a natural-sized component, so the representation admits a "
    "traceless erasure the store does not, and the reason would be that the closure span has "
    "dimensions the fact does not need. Both readings, and this rule, are fixed before the run.")


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--max-facts", type=int, default=0, help="0 = every fact the model answers")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    args = ap.parse_args(argv)

    per_seed = [run(args.layer, s, args.threads, args.max_facts) for s in args.seeds]
    void = next((s["void"] for s in per_seed if "void" in s), None)
    numeric = [{k: float(v) for k, v in s.items() if isinstance(v, (bool, int, float))} for s in per_seed]
    keys = [k for k in KEYS if all(k in s for s in numeric)]
    agg = ledger.aggregate(numeric, keys)
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})

    record: Dict[str, Any] = {
        "experiment": "E-000040",
        "title": "the traceless price in a representation: a deletion that announces itself",
        "evidence_level": "E5", "trains_nothing": True, "model": MODEL, "layer": args.layer,
        "seeds": args.seeds, "alphas": ALPHAS, "decision_rule": DECISION_RULE,
        "admission_rule": "a fact enters the measurement only if its closure leaves bystander facts at "
                          "0.60 of what they scored with nothing removed",
        "per_seed": per_seed, "aggregate": agg, "criteria": check}
    md = [f"# E-000040 — {record['title']}", "",
          f"Frozen {MODEL}, layer {args.layer}, no training. The deletion is each fact's workspace",
          "closure (E-000037), and the ablation is that closure made continuous: the state's component",
          "in the closure span is scaled by `a` at every layer from the read layer up and at every",
          "position. `a = 0` is E-000037's projection exactly; `a = 1` is the identity; `a` is the size",
          "of the hole an adversary measures.", ""]
    if void:
        md += ["## VOID", "", f"No claim is made: {void}", ""]
        record["void"] = void
    else:
        md += ["## The deletion, and what it costs bystanders", "",
               ledger.table(["measure", "mean over seeds", "worst seed"],
                            [["directions in the closure", f"{agg['closure']['mean']:.2f}",
                              f"{agg['closure']['max']:.2f}"],
                             ["the model answers, before", f"{agg['answer/before']['mean']:.4f}",
                              f"{agg['answer/before']['min']:.4f}"],
                             ["the model answers, closure removed", f"{agg['answer/deleted']['mean']:.4f}",
                              f"{agg['answer/deleted']['max']:.4f}"],
                             ["bystander facts under the same ablation", f"{agg['collateral']['mean']:.4f}",
                              f"{agg['collateral']['min']:.4f}"],
                             ["bystander facts with nothing removed",
                              f"{agg['collateral_before']['mean']:.4f}",
                              f"{agg['collateral_before']['min']:.4f}"]]), "",
               "## Which facts admit a deletion of their own", "",
               "The traceless question is only about a deletion OF ONE FACT, so a closure that takes",
               "the bystanders down with it does not enter the measurement. The admission rule was",
               "fixed in advance and the rate at which facts fail it is a finding, not a filter:", "",
               ledger.table(["measure", "mean over seeds", "worst seed"],
                            [["facts with a closure whose collateral survives",
                              f"{agg['specific_rate']['mean']:.4f}", f"{agg['specific_rate']['min']:.4f}"],
                             ["bystander accuracy over ALL facts with a closure",
                              f"{agg['collateral_all_facts']['mean']:.4f}",
                              f"{agg['collateral_all_facts']['min']:.4f}"],
                             ["bystander accuracy over the admitted facts",
                              f"{agg['collateral']['mean']:.4f}", f"{agg['collateral']['min']:.4f}"]]), "",
               "## The hole, and the price of hiding it", "",
               "The span as a whole is not fact-specific -- its later directions are the phrasing",
               "spread, which every capital question has -- so the primary statistic is the component",
               "along the fact's own shared direction, the part the deletion targets. The span ratio",
               "is beside it because a defender gets ONE scale and an adversary may read either.", "",
               ledger.table(["measure", "mean over seeds", "worst seed"],
                            [["component of the holding state, fact direction",
                              f"{agg['component/held']['mean']:.4f}", f"{agg['component/held']['min']:.4f}"],
                             ["component of a state that never held it, same direction",
                              f"{agg['component/bystander_mean']['mean']:.4f}",
                              f"{agg['component/bystander_mean']['min']:.4f}"],
                             ["smallest hole that looks natural (a_hide)",
                              f"{agg['alpha_hide']['mean']:.4f}", f"{agg['alpha_hide']['max']:.4f}"],
                             ["the same on the whole closure span",
                              f"{agg['alpha_hide_span']['mean']:.4f}",
                              f"{agg['alpha_hide_span']['max']:.4f}"],
                             ["largest hole the deletion can afford (a_answer)",
                              f"{agg['alpha_answer']['mean']:.4f}", f"{agg['alpha_answer']['min']:.4f}"],
                             ["the gap, fact direction",
                              f"{agg['traceless_gap']['mean']:+.4f}",
                              f"{agg['traceless_gap']['min']:+.4f}"],
                             ["the gap, whole span",
                              f"{agg['traceless_gap_span']['mean']:+.4f}",
                              f"{agg['traceless_gap_span']['min']:+.4f}"],
                             ["facts where the gap is positive",
                              f"{agg['traceless_impossible']['mean']:.4f}",
                              f"{agg['traceless_impossible']['min']:.4f}"],
                             ["facts where ONE scale hides both statistics at once",
                              f"{agg['one_alpha_hides_both']['mean']:.4f}",
                              f"{agg['one_alpha_hides_both']['min']:.4f}"]]), "",
               "## The rule, fixed before the run", "", DECISION_RULE, "",
               "The last row is the sharper form of the same question. The defender gets one scale `a`.",
               "If the two statistics call for different scales, no single `a` hides both and the",
               "deletion is visible at every setting -- to an adversary holding one snapshot and a",
               "hypothesis about which fact was deleted, which is the setting of Chen et al. (CCS 2021)",
               "and Gao et al. (arXiv:2202.03460) minus the two model versions they both require.", ""]
    md += ["## Pre-registered criteria", "", ledger.criteria_table(check), ""]
    path = ledger.save("e000040_dangling_readers", record, "\n".join(md))
    print("\n".join(md[1:]))
    print(f"\nwritten: {path}")
    return record


if __name__ == "__main__":
    main()
