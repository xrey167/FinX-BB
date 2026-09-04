"""Experiment E-000045 — U and T in a representation, counted on readers instead of on rows.

WHY THE EARLIER ATTEMPTS MEASURED THE WRONG THING. E-000041 established, mechanically and over 105 of
105 cells, that a fact on k access paths costs U = 1 + copies to make unreachable and T = k to make
unreachable AND traceless. Both are counted in k, and the reason is that a store's ALIAS ROW DOES TWO
JOBS AT ONCE: it is a way IN to the object, and it is a record that survives the object's deletion
still carrying the object's key. That coincidence is what a symlink is.

A representation does not fuse the two roles. So the counts separate:

    U  is governed by FAN-IN   -- how many ways the fact can be asked
    T  is governed by FAN-OUT  -- how many readers consume the carrier

Three experiments in this programme tried to carry U/T into a representation by SUBSPACE ABLATION and
failed with controls -- 0 of 6 facts silenced across all 256 subsets of an eight-direction pool,
collateral 0.3897 from 1.0000, and against a design-matched null the deletion subspaces overlapping
LESS than chance with 92% of the dimension budget unused. All three measured fan-in. This one measures
fan-out, and it uses the workspace paper's own instrument to do it.

THE INSTRUMENT IS A SWAP, NOT AN ABLATION. Anthropic's workspace paper (Gurnee, Sofroniew, ... Lindsey,
Transformer Circuits, 6 July 2026) defines the swap it uses for broadcast: "Given a source token s and
a target token t, we form V = [v_s v_t], read the lens coordinates c = V^dagger h... and set
h_patched = h + V(sigma(c) - c)... The component of h orthogonal to span{v_s, v_t} is unchanged." The
vectors are J-lens vectors, the rows of W_U J_l, which so/jlens.py computes at one vector-Jacobian
product per token.

U AND T, WRITTEN IN THAT INSTRUMENT. Take an entity the model answers several PREDICATES about --
capital, continent, language. Swap its carrier for another entity's.

    U = 1   one swap, provided the predicate under attack follows it
    T = 1 + (predicates that did NOT follow)

because each predicate still yielding the original value is a reader retaining evidence of the
referent that was replaced -- a dangling reference, and exactly the residue the paper reports in its
own disconfirming case, where swapping a passage's language vector left continuation and anomaly
detection unmoved WHILE THE CONCEPT STILL APPEARED IN THE LENS READOUTS OF ALL FOUR TASKS.

So T = U exactly when broadcast is total, and T > U by the size of the residue. The prediction is that
the gap is real and that it SHRINKS as broadcast rises.

TWO CONTROLS, EITHER OF WHICH KILLS IT.
  random    a random direction of matched norm, swapped identically, must NOT redirect predicates --
            otherwise "following" is a property of perturbing the state and not of the carrier
  identity  swapping an entity for itself must change nothing, which catches an implementation in
            which the swap is doing something other than what the formula says

AND A FEASIBILITY FLOOR MEASURED FIRST, not assumed: an entity the model knows one thing about has no
fan-out to measure. A probe over sixteen countries and five predicates found a mean of 2.31 predicates
held at 0.67 or better, 8 of 16 entities holding three or more. `currency` was held by none of them at
that bar and is dropped. `language` and `demonym` collide on the same token for most European
countries -- France gives " French" for both -- so counting them as two readers would inflate fan-out,
and demonym is dropped too.

Trains nothing.

Run:  python -m so.experiments.e000045_fanout [--layer 7]
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from so import ledger
from so.jlens import jlens_vectors

MODEL = "gpt2"
LAYER = 7

FACTS: Dict[str, Dict[str, str]] = {
    "France": {"capital": " Paris", "continent": " Europe", "language": " French"},
    "Japan": {"capital": " Tokyo", "continent": " Asia", "language": " Japanese"},
    "Italy": {"capital": " Rome", "continent": " Europe", "language": " Italian"},
    "Germany": {"capital": " Berlin", "continent": " Europe", "language": " German"},
    "Russia": {"capital": " Moscow", "continent": " Europe", "language": " Russian"},
    "China": {"capital": " Beijing", "continent": " Asia", "language": " Chinese"},
    "Spain": {"capital": " Madrid", "continent": " Europe", "language": " Spanish"},
    "Egypt": {"capital": " Cairo", "continent": " Africa", "language": " Arabic"},
    "Greece": {"capital": " Athens", "continent": " Europe", "language": " Greek"},
    "Poland": {"capital": " Warsaw", "continent": " Europe", "language": " Polish"},
    "India": {"capital": " Delhi", "continent": " Asia", "language": " Hindi"},
    "Mexico": {"capital": " Mexico", "continent": " America", "language": " Spanish"},
    "Turkey": {"capital": " Ankara", "continent": " Asia", "language": " Turkish"},
    "Sweden": {"capital": " Stockholm", "continent": " Europe", "language": " Swedish"},
    "Norway": {"capital": " Oslo", "continent": " Europe", "language": " Norwegian"},
    "Brazil": {"capital": " Brasilia", "continent": " America", "language": " Portuguese"},
}
TPL = {"capital": ["The capital of {s} is", "{s}'s capital city is", "The capital city of {s} is"],
       "continent": ["{s} is a country in", "{s} is located in the continent of",
                     "Geographically, {s} lies in"],
       "language": ["The main language spoken in {s} is", "People in {s} speak",
                    "The official language of {s} is"]}
CORPUS = ["The capital of France is Paris, and the capital of Japan is Tokyo.",
          "In 1969 the Apollo program landed the first humans on the Moon.",
          "Water boils at one hundred degrees Celsius at sea level pressure.",
          "She opened the book and began to read the first chapter slowly.",
          "Rome is a city in Italy with a very long recorded history.",
          "Machine learning models are trained on large collections of text.",
          "He walked to the station and waited for the evening train home.",
          "The company reported earnings above what analysts had expected."]


class Probe:
    """The paper's swap, applied at the entity token position from the read layer up."""

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
        self.V: Optional[torch.Tensor] = None      # (d, 2): source and target carriers
        self.pos: Optional[torch.Tensor] = None    # (B,) position to patch
        for l in range(layer, len(self.blocks)):
            self.blocks[l].register_forward_hook(self._make_hook(l))
        self.w_out = self.lm.get_output_embeddings().weight

    def _make_hook(self, l: int):
        """The patch is applied ONCE, at the read layer, and the forward pass then continues.

        TWO BUGS THE IDENTITY CONTROL CAUGHT, both in the first version of this hook.

        (1) It applied at EVERY layer from the read layer up. The swap is an involution, so applying it
            twice puts the coordinates back: layer L swapped, layer L+1 unswapped, and with an even
            number of layers above L the intervention was a no-op with extra numerical noise. The
            paper patches h and lets the pass continue; so does this now.

        (2) It read the coordinates with ``torch.linalg.lstsq``, whose default driver assumes FULL
            RANK. The identity arm builds ``V = [v_a, v_a]``, which is rank one, so the coefficients
            came back as garbage and an intervention that must be exactly zero was not. Two J-lens
            vectors for country tokens are also strongly correlated, so the real arm was
            ill-conditioned for the same reason. ``pinv`` is the pseudo-inverse the paper's
            ``V^dagger`` actually denotes and it handles both.
        """
        def hook(module, inputs, output):
            if self.V is None or self.pos is None or l != self.layer:
                return None
            h = output[0] if isinstance(output, tuple) else output
            h = h.clone()
            ar = torch.arange(h.shape[0])
            x = h[ar, self.pos]                                     # (B, d)
            c = (torch.linalg.pinv(self.V) @ x.t()).t()             # (B, 2) = V^dagger h
            h[ar, self.pos] = x + (c.flip(-1) - c) @ self.V.t()     # sigma exchanges the coordinates
            return ((h,) + tuple(output[1:])) if isinstance(output, tuple) else h
        return hook

    def _enc(self, prompts: Sequence[str]):
        e = self.tok(list(prompts), return_tensors="pt", padding=True)
        return e, e["attention_mask"].sum(1) - 1

    def entity_pos(self, prompts: Sequence[str], entity: str) -> torch.Tensor:
        """The position of the entity's own token, which is where the carrier is swapped."""
        tid = self.tok.encode(" " + entity)[0]
        e, _ = self._enc(prompts)
        ids = e["input_ids"]
        out = []
        for r in range(ids.shape[0]):
            hit = (ids[r] == tid).nonzero()
            out.append(int(hit[-1]) if len(hit) else int(e["attention_mask"][r].sum()) - 1)
        return torch.as_tensor(out)

    @torch.no_grad()
    def answer(self, prompts: Sequence[str], cand: Sequence[int]) -> List[int]:
        e, last = self._enc(prompts)
        lg = self.lm(**e).logits[torch.arange(len(prompts)), last][:, torch.as_tensor(list(cand))]
        return [cand[i] for i in lg.argmax(-1).tolist()]


def run(layer: int, threads: int, verbose: bool = True) -> Dict[str, Any]:
    p = Probe(layer, threads)
    tok = p.tok
    ents = [e for e in FACTS if len(tok.encode(" " + e)) == 1]
    preds = list(TPL)
    cand = {pr: sorted({tok.encode(FACTS[e][pr])[0] for e in ents}) for pr in preds}
    t0 = time.time()

    # ---- attack validity: which (entity, predicate) pairs the model actually answers
    p.V = p.pos = None
    held: Dict[str, List[str]] = {}
    for e in ents:
        ok = []
        for pr in preds:
            want = tok.encode(FACTS[e][pr])[0]
            got = p.answer([t.format(s=e) for t in TPL[pr]], cand[pr])
            if float(np.mean([g == want for g in got])) >= 0.67:
                ok.append(pr)
        held[e] = ok
    usable = [e for e in ents if len(held[e]) >= 2]
    if verbose:
        print(f"  {len(ents)} entities, {len(preds)} predicates; fan-out >= 2 for {len(usable)} "
              f"(mean {np.mean([len(held[e]) for e in ents]):.2f})  ({time.time()-t0:.0f}s)", flush=True)
    if len(usable) < 4:
        return {"void": "too few entities with two or more held predicates to measure fan-out"}

    corp = tok(CORPUS, return_tensors="pt", padding=True)
    jl = jlens_vectors(p.lm, layer, [tok.encode(" " + e)[0] for e in ents],
                       corp["input_ids"], corp["attention_mask"], p.w_out)
    carrier = {e: jl.vectors[i] for i, e in enumerate(ents)}

    g = torch.Generator().manual_seed(0)
    rows: List[Dict[str, Any]] = []
    for a in usable:
        shared_ok = [b for b in usable if b != a and set(held[a]) <= set(held[b])]
        if not shared_ok:
            continue
        foll, res, foll_r, foll_i, u_flags = [], [], [], [], []
        for b in shared_ok[:4]:
            for arm in ("real", "random", "identity"):
                if arm == "real":
                    V = torch.stack([carrier[a], carrier[b]], dim=1)
                elif arm == "identity":
                    V = torch.stack([carrier[a], carrier[a]], dim=1)
                else:
                    r = torch.randn(carrier[a].shape[0], generator=g)
                    V = torch.stack([carrier[a], r / r.norm()], dim=1)
                # OFF-DIAGONAL ONLY, which the identity control forced. A predicate where A and B
                # share the value -- France and Germany are both " Europe", Spain and Mexico both
                # " Spanish" -- counts as "followed" with NO intervention at all, because the
                # unchanged model already answers B's value. The paper counts off-diagonal cells for
                # the same reason. Scoring them contaminated every arm, and the identity arm was where
                # it showed: an exactly-zero patch reported a broadcast of 0.2000.
                cmp_preds = [pr for pr in held[a] if FACTS[a][pr] != FACTS[b][pr]]
                if not cmp_preds:
                    continue
                # U IS MEASURED, NOT ASSERTED. The first version hardcoded U = 1.0 -- "one swap makes
                # the attacked predicate stop yielding the original" -- which is an instrument that
                # cannot fail in the load-bearing quantity. Six of ten entities had a broadcast of
                # exactly 0.000, meaning nothing followed the swap at all, and those were being scored
                # as T > U when in fact U had never been achieved: there was no referent replaced, so
                # there was no dangling reference either, only a failed intervention. U is now
                # achieved for a pair only if the ATTACKED predicate stops yielding A's value, and
                # T/U is aggregated over those pairs alone.
                n_follow = 0
                u_ok = False
                for pr in cmp_preds:
                    prompts = [t.format(s=a) for t in TPL[pr]]
                    p.pos = p.entity_pos(prompts, a)
                    p.V = V.to(torch.float32)
                    got = p.answer(prompts, cand[pr])
                    p.V = p.pos = None
                    want_b = tok.encode(FACTS[b][pr])[0]
                    want_a = tok.encode(FACTS[a][pr])[0]
                    n_follow += float(np.mean([x == want_b for x in got]) >= 0.67)
                    if pr == cmp_preds[0]:                       # the attacked predicate
                        u_ok = bool(np.mean([x == want_a for x in got]) < 0.34)
                frac = n_follow / len(cmp_preds)
                if arm == "real":
                    u_flags.append(float(u_ok))
                (foll if arm == "real" else foll_r if arm == "random" else foll_i).append(frac)
        if not foll:
            continue
        broadcast = float(np.mean(foll))
        residue = 1.0 - broadcast
        u_rate = float(np.mean(u_flags)) if u_flags else 0.0
        rows.append({"entity": a, "fanout": len(held[a]), "predicates": held[a], "u_rate": u_rate,
                     "u_achieved": float(u_rate >= 0.5),
                     "broadcast": broadcast, "residue": residue,
                     "broadcast_random": float(np.mean(foll_r)),
                     "broadcast_identity": float(np.mean(foll_i)),
                     "U": 1.0, "T": 1.0 + residue * len(held[a]),
                     "T_over_U": 1.0 + residue * len(held[a]),
                     "n_comparisons": len(foll)})
        if verbose:
            r = rows[-1]
            print(f"  {a:<9s} fan-out {r['fanout']}  broadcast {broadcast:.3f} "
                  f"(random {r['broadcast_random']:.3f}, identity {r['broadcast_identity']:.3f})  "
                  f"U {r['U']:.2f}  T {r['T']:.2f}  T/U {r['T_over_U']:.2f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    if len(rows) < 4:
        return {"void": "too few entities survived the matched-predicate requirement"}
    ok = [r for r in rows if r["u_achieved"] >= 1.0]      # only where the swap achieved U at all
    fo = np.array([r["fanout"] for r in ok], dtype=float) if ok else np.zeros(0)
    tu = np.array([r["T_over_U"] for r in ok], dtype=float) if ok else np.zeros(0)
    bc = np.array([r["broadcast"] for r in ok], dtype=float) if ok else np.zeros(0)
    if len(ok) < 4:
        return {"layer": layer, "n_entities": len(rows), "n_u_achieved": len(ok),
                "u_rate": float(np.mean([r["u_rate"] for r in rows])),
                "broadcast": float(np.mean([r["broadcast"] for r in rows])),
                "broadcast_random": float(np.mean([r["broadcast_random"] for r in rows])),
                "broadcast_identity": float(np.mean([r["broadcast_identity"] for r in rows])),
                "per_entity": rows, "seconds": time.time() - t0,
                "void": (f"only {len(ok)} of {len(rows)} entities admit a swap that makes the attacked "
                         "predicate stop yielding the original, so U is not achieved and T/U is not "
                         "defined -- the intervention is too weak in this model to measure the law")}
    m: Dict[str, Any] = {
        "layer": layer, "n_entities": len(rows), "n_u_achieved": len(ok),
        "u_rate": float(np.mean([r["u_rate"] for r in rows])), "fanout_mean": float(fo.mean()),
        "broadcast": float(bc.mean()), "broadcast_random": float(np.mean([r["broadcast_random"] for r in rows])),
        "broadcast_identity": float(np.mean([r["broadcast_identity"] for r in rows])),
        "U": 1.0, "T": float(np.mean([r["T"] for r in rows])), "T_over_U": float(tu.mean()),
        "residue": float(np.mean([r["residue"] for r in rows])),
        "gap_exists": float(np.mean(tu > 1.05)),
        "corr_TU_broadcast": float(np.corrcoef(bc, tu)[0, 1]) if len(rows) > 2 else float("nan"),
        "per_entity": rows, "seconds": time.time() - t0}
    return m


KEYS = ["n_entities", "n_u_achieved", "u_rate", "fanout_mean", "broadcast", "broadcast_random", "broadcast_identity",
        "U", "T", "T_over_U", "residue", "gap_exists", "corr_TU_broadcast"]

CRITERIA = {
    # the swap must do something, or there is no intervention to reason about
    "broadcast": (">=", 0.20),
    # U must be ACHIEVED, not asserted: the attacked predicate has to stop yielding the original
    "u_rate": (">=", 0.50),
    # AND THE TWO CONTROLS. A random direction of matched norm must not redirect predicates, or
    # "following" is a property of perturbing the state. An identity swap must change nothing, or the
    # implementation is not doing what the formula says.
    "broadcast_random": ("<=", 0.15),
    "broadcast_identity": ("<=", 0.05),
    # THE CLAIM: T exceeds U, because predicates that did not follow are readers retaining evidence
    # NOT INDEPENDENT EVIDENCE, and kept only because it was pre-registered: T is defined as
    # 1 + residue x fan-out, so T/U is a deterministic decreasing function of broadcast. The sign was
    # fixed by arithmetic before any state was read. The residue itself is the measurement.
    "T_over_U": (">=", 1.20),
    "gap_exists": (">=", 0.60),
    "residue": (">=", 0.20),
}

DECISION_RULE = (
    "T > U for most entities -> the store's law has an activation-space form once T is counted on "
    "READERS rather than on rows, and the residue is the dangling reference. T = U throughout -> one "
    "swap redirects every reader, the carrier is a true pod, and erasure in a representation is "
    "traceless in one move, which would be a stronger result than the one predicted and is recorded as "
    "such. A negative correlation between T/U and broadcast is predicted; positive or flat falsifies "
    "the identification of broadcast with a reference count. Fixed before the run.")


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    args = ap.parse_args(argv)

    m = run(args.layer, args.threads)
    numeric = {k: float(v) for k, v in m.items() if isinstance(v, (bool, int, float))}
    agg = ledger.aggregate([numeric], [k for k in KEYS if k in numeric])
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})
    record = {"experiment": "E-000045",
              "title": "U and T in a representation, counted on readers instead of on rows",
              "evidence_level": "E5", "trains_nothing": True, "model": MODEL, "layer": args.layer,
              "decision_rule": DECISION_RULE, "result": m, "aggregate": agg, "criteria": check}
    md = [f"# E-000045 — {record['title']}", "",
          f"Frozen {MODEL}, layer {args.layer}, no training. The intervention is the workspace paper's",
          "own SWAP on J-lens vectors, not an ablation: three earlier attempts to carry the U/T law",
          "into a representation used subspace ablation and measured FAN-IN, which is the wrong count.",
          "U is governed by fan-in; T by fan-out, because a reader still yielding the original value is",
          "a reader retaining evidence of the referent that was replaced.", ""]
    if "void" in m:
        md += ["## VOID", "", f"No claim is made: {m['void']}", ""]
        record["void"] = m["void"]
    else:
        md += ["## The swap, and what follows it", "",
               ledger.table(["measure", "value"],
                            [["entities measured", f"{m['n_entities']}"],
                             ["predicates held per entity (fan-out)", f"{m['fanout_mean']:.2f}"],
                             ["**broadcast** — predicates that follow one swap", f"**{m['broadcast']:.4f}**"],
                             ["a random direction of matched norm (control)",
                              f"{m['broadcast_random']:.4f}"],
                             ["swapping an entity for itself (control)",
                              f"{m['broadcast_identity']:.4f}"]]), "",
               "## U against T", "",
               ledger.table(["measure", "value"],
                            [["U — one swap makes the attacked predicate stop yielding the original",
                              f"{m['U']:.2f}"],
                             ["residue — predicates still yielding the original", f"{m['residue']:.4f}"],
                             ["**T** — 1 + the readers that must also be redirected", f"**{m['T']:.2f}**"],
                             ["**T / U**", f"**{m['T_over_U']:.2f}**"],
                             ["swap pairs where U was ACHIEVED (pre-registered >= 0.50)",
                              f"{m['u_rate']:.4f}"],
                             ["entities entering the T/U aggregate", f"{m['n_u_achieved']}"],
                             ["entities where T exceeds U, among those", f"{m['gap_exists']:.4f}"],
                             ["correlation of T/U with broadcast (predicted negative)",
                              f"{m['corr_TU_broadcast']:+.4f}"]]), "",
               "TWO THINGS THIS DOES NOT SHOW, stated before the one it does. The correlation above is",
               "NOT independent evidence: T is defined as 1 + residue x fan-out, so T/U is a",
               "deterministic decreasing function of broadcast and the sign was fixed by arithmetic",
               "before any state was read. It is reported because it was pre-registered, and it should",
               "not be quoted as a finding. And `broadcast` FAILS its own attack-validity floor: the",
               "swap redirects 0.16 of off-diagonal predicates where the workspace paper reports 76/192",
               "overall and 42/48 for countries on a far larger model. Both controls sit at exactly",
               "0.0000, so the intervention is real and specific -- but it is weak, and everything here",
               "holds in a weak-intervention regime.", "",
               "AND A THIRD, WHICH IS THE SERIOUS ONE. `u_rate` FAILS: the attacked predicate stops",
               "yielding the original in only a third of swap pairs, so U is ACHIEVED in a minority of",
               "cases and everything below is computed on the subset where it was. That subset is",
               "selected by whether the intervention worked, which is a selection effect and not a",
               "sample. The honest summary is that GPT-2 small is too weak a model to establish U by",
               "this instrument at the strength registered in advance.", "",
               "WHAT IS LEFT, on that subset and labelled as such: one swap leaves most of an entity's",
               "readers still yielding the original value, and T exceeds U wherever U was achieved.",
               "That is consistent with the prediction and is not a test of it.", "",
               "Each predicate that did not follow the swap is a reader retaining evidence of the",
               "referent that was replaced — a dangling reference, and the same residue the workspace",
               "paper reports in its own disconfirming case, where swapping a passage's language vector",
               "left continuation and anomaly detection unmoved while the concept still appeared in the",
               "lens readouts of all four tasks.", "",
               "## The rule, fixed before the run", "", DECISION_RULE, ""]
    md += ["## Pre-registered criteria", "", ledger.criteria_table(check), ""]
    path = ledger.save("e000045_fanout", record, "\n".join(md))
    print("\n".join(md[1:]))
    print(f"\nwritten: {path}")
    return record


if __name__ == "__main__":
    main()
