"""Experiment E-000040 — does an ablated fact leave a pointer behind, the way a deleted pod does?

E-000035 measured the store half of a structural claim: **canonicalisation converts redundancy into
references, and references outlive referents.** A pod makes a fact unreachable in one record deletion
instead of k, and every surviving alias then carries the removed cell's key, so an adversary reading
only the store names the deleted key at 1.0000, uniquely, against 0.0000 for a duplicated store. Cheap
erasure and silent erasure were not jointly achievable there.

This asks the same question of a representation, which is the half the store cannot answer.

WHY IT IS THE SAME QUESTION. Anthropic's workspace paper (Gurnee, Sofroniew et al., Transformer
Circuits 2026) shows a J-lens vector behaving as a shared carrier: swap one and many downstream
predicates follow. That is a pod in activation space -- one write, many reads -- and the readers are
the aliases. Remove the carrier and the readers are still there. The paper reports the residue
directly, in the case where its own intervention does NOT take effect: swapping the vector for a
passage's language leaves continuation and anomaly detection unmoved, **while the concept still
appears in the lens readouts of all four tasks**. A lens residue where the behaviour has changed is
what a dangling pointer looks like from the inside.

THE ADVERSARY, AND WHY THE COMPARISON IS NOT TRIVIAL. One activation snapshot. No before-and-after
model, which is what separates this from deletion inference (Chen et al., CCS 2021, and Gao et al.,
arXiv:2202.03460, both of which need the discrepancy between two model versions). The adversary must
distinguish

  ABLATED   a fact the model HELD, with its carrier projected out, so it no longer answers
  WEAK      a fact the model does NOT hold, with the SAME construction and the SAME projection applied

Both fail to answer. If the adversary can still tell them apart, the removal left something behind.
Applying the projection to only one side would measure nothing but the projection, so it is applied to
both, and a random projection of matched norm is run as the control that can kill the result.

WHICH FACTS ARE WEAK IS MEASURED, NOT ASSUMED. GPT-2 small holds all sixteen capitals of E-000037 at
0.914, so a weak arm needs obscurer ones; the pool below is scored first and split by what the model
actually does, which is also the attack-validity floor.

Trains nothing.

Run:  python -m so.experiments.e000040_dangling_readers [--layer 7] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from so import ledger
from so.workspace import lens_logits, project_out

MODEL = "gpt2"
LAYER = 7

# a wide pool: which of these the model holds is measured below, never assumed
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
        self.last: Optional[torch.Tensor] = None
        for l in range(layer, len(self.blocks)):
            self.blocks[l].register_forward_hook(self._hook)
        self.w_out = self.lm.get_output_embeddings().weight
        self.ln = self.lm.transformer.ln_f

    def _hook(self, module, inputs, output):
        if self.dirs is None or self.dirs.numel() == 0:
            return None
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        ar = torch.arange(h.shape[0])
        h[ar, self.last] = project_out(h[ar, self.last], self.dirs)   # answer position only
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    def _enc(self, prompts):
        e = self.tok(list(prompts), return_tensors="pt", padding=True)
        return e, e["attention_mask"].sum(1) - 1

    @torch.no_grad()
    def top1(self, prompts: Sequence[str], cand: Sequence[int]) -> List[int]:
        e, last = self._enc(prompts)
        self.last = last
        lg = self.lm(**e).logits[torch.arange(len(prompts)), last][:, torch.as_tensor(list(cand))]
        self.last = None
        return [cand[i] for i in lg.argmax(-1).tolist()]

    @torch.no_grad()
    def residual(self, prompts: Sequence[str]) -> torch.Tensor:
        e, last = self._enc(prompts)
        self.last = last
        hs = self.lm(**e, output_hidden_states=True).hidden_states[self.layer]
        out = hs[torch.arange(len(prompts)), last]
        self.last = None
        return out

    @torch.no_grad()
    def lens_rank(self, prompts: Sequence[str], obj_id: int) -> Tuple[float, float]:
        """The object's rank and score in the LENS readout at the read layer, after any ablation.

        This is the residue the workspace paper reports where its own intervention does not change
        behaviour: the concept still appears in the lens readouts. If a removed fact leaves one and a
        never-held fact does not, the removal left a pointer.
        """
        e, last = self._enc(prompts)
        self.last = last
        hs = self.lm(**e, output_hidden_states=True).hidden_states[self.layer]
        h = hs[torch.arange(len(prompts)), last]
        if self.dirs is not None and self.dirs.numel():
            h = project_out(h, self.dirs)
        self.last = None
        lg = lens_logits(h, self.w_out, self.ln)
        rank = (lg > lg[:, obj_id: obj_id + 1]).sum(-1).float()
        z = (lg[:, obj_id] - lg.mean(-1)) / lg.std(-1).clamp(min=1e-6)
        return float(rank.mean()), float(z.mean())


def carrier_of(res_self: torch.Tensor, res_others: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
    """The fact-specific direction, built identically whether or not the model holds the fact.

    Symmetry is the point: a construction that could only be applied to facts the model knows would
    make the two conditions differ by their construction rather than by the deletion.
    """
    spec = res_self - res_others
    spec = spec - (spec @ glob)[:, None] * glob[None]
    v = spec.mean(0)
    return v / v.norm().clamp(min=1e-8)


def run(layer: int, seed: int, threads: int, verbose: bool = True) -> Dict[str, Any]:
    p = Probe(layer, threads)
    tok = p.tok
    pool = [(s, o) for s, o in POOL if len(tok.encode(o)) == 1]
    caps = [tok.encode(o)[0] for _, o in pool]
    t0 = time.time()

    # attack validity, and the split: measured, never assumed
    p.dirs = None
    held = {}
    for s, o in pool:
        got = p.top1([t.format(s=s) for t in TEMPLATES], caps)
        held[s] = sum(g == tok.encode(o)[0] for g in got) / len(TEMPLATES)
    strong = [(s, o) for s, o in pool if held[s] >= 0.75]
    weak = [(s, o) for s, o in pool if held[s] <= 0.25]
    m: Dict[str, Any] = {"seed": seed, "layer": layer, "n_pool": len(pool),
                         "n_strong": len(strong), "n_weak": len(weak),
                         "control/strong_before": float(np.mean([held[s] for s, _ in strong])) if strong else float("nan"),
                         "control/weak_before": float(np.mean([held[s] for s, _ in weak])) if weak else float("nan")}
    if verbose:
        print(f"  pool {len(pool)} single-token capitals: {len(strong)} strong "
              f"({m['control/strong_before']:.4f}), {len(weak)} weak ({m['control/weak_before']:.4f})"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    if len(strong) < 4 or len(weak) < 4:
        m["void"] = "not enough facts on one side of the split to compare"
        return m

    res = {s: p.residual([t.format(s=s) for t in TEMPLATES]) for s, _ in pool}
    stack = torch.stack([res[s] for s, _ in pool])
    glob = stack.reshape(-1, stack.shape[-1]).mean(0)
    glob = glob / glob.norm()

    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for group, items in (("strong", strong), ("weak", weak)):
        for s, o in items:
            obj_id = tok.encode(o)[0]
            others = torch.stack([res[x] for x, _ in pool if x != s]).mean(0)
            v = carrier_of(res[s], others, glob)
            prompts = [t.format(s=s) for t in TEMPLATES]
            # the real ablation
            p.dirs = v[None]
            after = sum(g == obj_id for g in p.top1(prompts, caps)) / len(TEMPLATES)
            rank, z = p.lens_rank(prompts, obj_id)
            # the control: a random direction of the same norm, so "a projection happened" is matched
            r = torch.as_tensor(rng.normal(size=v.shape[0]), dtype=v.dtype)
            r = r / r.norm()
            p.dirs = r[None]
            rank_r, z_r = p.lens_rank(prompts, obj_id)
            p.dirs = None
            rows.append({"subject": s, "group": group, "held": held[s], "after": after,
                         "lens_rank": rank, "lens_z": z, "rand_rank": rank_r, "rand_z": z_r})
            if verbose:
                print(f"  {group:<6} {s:<10} held {held[s]:.2f} -> after ablation {after:.2f}  "
                      f"lens rank {rank:7.1f} z {z:6.2f}  | random-dir control rank {rank_r:7.1f} "
                      f"z {z_r:6.2f}  ({time.time() - t0:.0f}s)", flush=True)

    def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
        """P(a random positive scores above a random negative). 0.5 is no separation."""
        if not pos or not neg:
            return float("nan")
        wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
        return float(wins / (len(pos) * len(neg)))

    sz = [r["lens_z"] for r in rows if r["group"] == "strong"]
    wz = [r["lens_z"] for r in rows if r["group"] == "weak"]
    szr = [r["rand_z"] for r in rows if r["group"] == "strong"]
    wzr = [r["rand_z"] for r in rows if r["group"] == "weak"]
    m["strong/after_ablation"] = float(np.mean([r["after"] for r in rows if r["group"] == "strong"]))
    m["weak/after_ablation"] = float(np.mean([r["after"] for r in rows if r["group"] == "weak"]))
    m["strong/lens_z"] = float(np.mean(sz))
    m["weak/lens_z"] = float(np.mean(wz))
    m["disclosure_auc"] = auc(sz, wz)
    m["random_direction_auc"] = auc(szr, wzr)
    m["auc_gap"] = m["disclosure_auc"] - m["random_direction_auc"]
    m["per_fact"] = rows
    m["seconds"] = time.time() - t0
    return m


KEYS = ["n_strong", "n_weak", "control/strong_before", "control/weak_before", "strong/after_ablation",
        "weak/after_ablation", "strong/lens_z", "weak/lens_z", "disclosure_auc",
        "random_direction_auc", "auc_gap"]

CRITERIA = {
    # attack validity: the split must be real, and the ablation must actually remove the answer
    "control/strong_before": (">=", 0.75),
    "control/weak_before": ("<=", 0.25),
    "strong/after_ablation": ("<=", 0.30),
    # THE CLAIM: after the same treatment, the fact that WAS there is still distinguishable
    "disclosure_auc": (">=", 0.70),
    # and the control that can kill it: a random direction of matched norm must NOT separate them,
    # or the adversary is detecting the projection rather than anything that survived a deletion
    "random_direction_auc": ("<=", 0.65),
}


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    args = ap.parse_args(argv)

    per_seed = [run(args.layer, s, args.threads) for s in args.seeds]
    if any("void" in s for s in per_seed):
        print("VOID:", per_seed[0].get("void"))
    numeric = [{k: float(v) for k, v in s.items() if isinstance(v, (bool, int, float))} for s in per_seed]
    keys = [k for k in KEYS if all(k in s for s in numeric)]
    agg = ledger.aggregate(numeric, keys)
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})

    tbl = ledger.table(["condition", "held before", "answers after the ablation", "lens z of the object"],
                       [["strong (had it, carrier removed)", f"{agg['control/strong_before']['mean']:.4f}",
                         f"{agg['strong/after_ablation']['mean']:.4f}", f"{agg['strong/lens_z']['mean']:.3f}"],
                        ["weak (never held it, same treatment)", f"{agg['control/weak_before']['mean']:.4f}",
                         f"{agg['weak/after_ablation']['mean']:.4f}", f"{agg['weak/lens_z']['mean']:.3f}"]])
    tbl2 = ledger.table(["what the adversary projects out", "AUC separating removed from never-held"],
                        [["the fact's own carrier", f"{agg['disclosure_auc']['mean']:.4f}"],
                         ["a random direction of matched norm (control)",
                          f"{agg['random_direction_auc']['mean']:.4f}"]])

    record = {"experiment": "E-000040",
              "title": "does an ablated fact leave a pointer behind, the way a deleted pod does",
              "trains_nothing": True, "model": MODEL, "layer": args.layer, "seeds": args.seeds,
              "per_seed": per_seed, "aggregate": agg, "criteria": check}
    md = [f"# E-000040 — {record['title']}", "",
          f"Frozen {MODEL}, layer {args.layer}, no training. The adversary sees ONE activation snapshot:",
          "no before-and-after model, which is what separates this from deletion inference (Chen et al.,",
          "CCS 2021; Gao et al., arXiv:2202.03460), both of which need the discrepancy between two model",
          "versions.", "",
          "## The two conditions, treated identically", "", tbl, "",
          "## Can the adversary tell them apart", "", tbl2, "",
          "The second row is the control that can kill the result. Both conditions have a direction",
          "projected out; if a RANDOM direction of matched norm separates them just as well, the",
          "adversary is detecting the projection and not anything that survived a deletion.", "",
          "## Pre-registered criteria", "", ledger.criteria_table(check), ""]
    path = ledger.save("e000040_dangling_readers", record, "\n".join(md))
    print(f"\nwritten: {path}")
    print(tbl); print(tbl2); print(ledger.criteria_table(check))
    return record


if __name__ == "__main__":
    main()
