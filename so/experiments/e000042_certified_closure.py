"""Experiment E-000042 — a certified lower bound for deletion in a representation.

THE GAP THIS ADDRESSES, IN MY OWN WORDS. ``so/workspace.py`` says:

    "A model hands out no such trace. There is no set of directions the computation 'used' that a
     solution must intersect, so the disjointness argument does not transfer, and computing it anyway
     would be a bound that certifies nothing."

That is why a store's deletion closure comes with a proof on the low side -- every live derivation is
a must-hit set, pairwise-disjoint derivations bound the optimum from below, and E-000032 reports
``proved optimal`` at 1.00 in every arm -- while a representation's closure has been a greedy number
with nothing under it.

The claim under test is that a J-lens sparse nonnegative decomposition is that trace, and that
sparsity makes the must-hit property CHECKABLE rather than merely arguable.

WHAT THE FIRST RUN TAUGHT, RECORDED BECAUSE IT CHANGED THE DESIGN TWICE.

  1. It used rows of ``W_U`` -- the logit lens, the J = I special case -- and went VOID: eight
     directions removed, not one fact silenced. The paper's J-lens vectors are the rows of
     ``W_U J_l``, which ``so/jlens.py`` now computes at one vector-Jacobian product per token.
  2. A follow-up check appeared to vindicate the change and did not. Removing the eight logit-lens
     rows of the eight CANDIDATE CAPITALS took the answer to 0.00 -- but those rows are the readout of
     the candidate set, so removing them makes the restricted argmax noise. That silences nothing; it
     blinds the readout. The paper guards against exactly this: its global ablation "does not ablate
     any tokens that appear in the top-10 tokens of a clean forward pass". The guard is carried over
     here, and it is why the pool is chosen the way it is.

AND THE CONTROL THAT SEPARATES A DELETION FROM A BLINDING. An ablation that moves the argmax has not
necessarily removed anything: a parallel investigation on this repository's own checkpoint found a
closure of 1.00 at collateral 0.0044 whose ablated states STILL yielded the object to a freshly
fitted linear probe at 0.9300 held out. So every deletion reported here is put to a refitted probe.
A closure whose states a probe walks straight through is reported as a readout-path removal and not
as a deletion, and ``probe_after`` is a pre-registered criterion rather than a diagnostic.

HOW IT IS KEPT FROM BEING AN INSTRUMENT THAT CANNOT FAIL. The ablation table is enumerated
EXHAUSTIVELY over the pool, and every quantity is read off that one table. That buys the TRUE optimum
by enumeration rather than by greedy, so the bound is compared against the answer; SOUNDNESS as a
measurement, since ``bound_sound_min`` fails on one violation anywhere; and two controls -- a random
support of the same size, and a random support that also contains the object's own direction, because
a support containing it is must-hit for a nearly trivial reason. A synthetic positive control proves
the bound can report more than one before any 1 it returns is read as a finding.

SCOPE, stated because the number invites over-reading. The count is relative to this pool and this
dictionary. arXiv:2608.10566 shows erasure counts are not affine-invariant -- "both quantities can
change under an information-preserving invertible reparameterization" -- so a support size is a
property of the J-lens dictionary, not of the model. What the certificate licenses is a statement
about ablations drawn from this pool, exactly as ``certify_store_absence`` licenses one about payloads
inside the domain it sweeps.

Trains nothing.

Run:  python -m so.experiments.e000042_certified_closure [--layer 7] [--pool 8] [--max-facts 6]
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
from so.jlens import jlens_vectors
from so.support import certified_closure, certify_must_hit, disjoint_lower_bound, nonneg_pursuit
from so.workspace import project_out

MODEL = "gpt2"
LAYER = 7
POOL = 8
NOMINATE = 40          # tokens the logit lens nominates; J-lens vectors are computed for these
CLEAN_GUARD = 10       # the paper's guard: never ablate a token in the clean top-k output

PAIRS = [("France", " Paris"), ("Japan", " Tokyo"), ("Italy", " Rome"), ("Germany", " Berlin"),
         ("Russia", " Moscow"), ("China", " Beijing"), ("Spain", " Madrid"), ("Egypt", " Cairo"),
         ("Canada", " Ottawa"), ("Greece", " Athens"), ("Cuba", " Havana"), ("Iran", " Tehran"),
         ("Poland", " Warsaw"), ("Norway", " Oslo"), ("Peru", " Lima"), ("Chile", " Santiago"),
         ("Sweden", " Stockholm")]

TEMPLATES = ["The capital of {s} is", "{s}'s capital city is", "Q: What is the capital of {s}? A:",
             "The city that serves as the capital of {s} is", "People say the capital of {s} is",
             "In {s}, the capital is", "The seat of government of {s} is located in",
             "Everyone knows that the capital of {s} is"]

CORPUS = ["The capital of France is Paris, and the capital of Japan is Tokyo.",
          "In 1969 the Apollo program landed the first humans on the Moon.",
          "Water boils at one hundred degrees Celsius at sea level pressure.",
          "She opened the book and began to read the first chapter slowly.",
          "The company reported earnings above what analysts had expected.",
          "Rome is a city in Italy with a very long recorded history.",
          "Machine learning models are trained on large collections of text.",
          "He walked to the station and waited for the evening train home."]


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
    def clean_top(self, prompts: Sequence[str], k: int) -> List[int]:
        """The tokens a clean forward pass already produces -- the paper's ablation never touches these."""
        d, self.dirs = self.dirs, None
        e, last = self._enc(prompts)
        lg = self.lm(**e).logits[torch.arange(len(prompts)), last]
        self.dirs = d
        return sorted({int(t) for t in lg.topk(k, dim=-1).indices.reshape(-1).tolist()})

    @torch.no_grad()
    def state(self, prompts: Sequence[str]) -> torch.Tensor:
        """The residual at the read layer, answer position, under whatever ablation is set."""
        e, last = self._enc(prompts)
        hs = self.lm(**e, output_hidden_states=True).hidden_states[self.layer]
        h = hs[torch.arange(len(prompts)), last]
        if self.dirs is not None and self.dirs.numel():
            h = project_out(h, self.dirs)
        return h


def linear_probe(x: torch.Tensor, y: torch.Tensor, groups: torch.Tensor, n_class: int,
                 steps: int = 400, lr: float = 0.5) -> float:
    """Leave-one-group-out accuracy of a linear readout: can the object still be read off the state?

    This is what separates a DELETION from a BLINDING. An ablation that moves the model's argmax may
    have left the fact entirely intact and merely removed the path the output head was using; a probe
    refitted on the ablated states finds it immediately if so. Groups are phrasings, so the probe is
    never scored on a phrasing it was fitted on.
    """
    x = (x - x.mean(0)) / x.std(0).clamp(min=1e-6)
    accs = []
    for g in groups.unique():
        tr, te = groups != g, groups == g
        w = torch.zeros(x.shape[1], n_class, requires_grad=True)
        b = torch.zeros(n_class, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(x[tr] @ w + b, y[tr]) + 1e-3 * w.pow(2).sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            accs.append(float(((x[te] @ w + b).argmax(-1) == y[te]).float().mean()))
    return float(np.mean(accs))


def _subsets(pool: Sequence[int]) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []
    for size in range(len(pool) + 1):
        out.extend(combinations(pool, size))
    return out


def positive_control(n_groups: int = 3, per_group: int = 2) -> Dict[str, Any]:
    """The bound must be able to report more than one, or a pod-shaped model would flatter it."""
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
    corpus = tok(CORPUS, return_tensors="pt", padding=True)
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

    # states for the probe, one row per (fact, phrasing), collected under each ablation of interest
    probe_y = torch.tensor([held.index(s) for s in held for _ in TEMPLATES])
    probe_g = torch.tensor([t for _ in held for t in range(len(TEMPLATES))])

    def probe_under(dirs) -> float:
        d, p.dirs = p.dirs, dirs
        x = torch.cat([p.state(prompts_of[s]) for s in held])
        p.dirs = d
        return linear_probe(x, probe_y, probe_g, len(held))

    rows: List[Dict[str, Any]] = []
    for s in targets:
        obj_id = obj_of[s]
        prompts = prompts_of[s]
        bys = [b for b in held if b != s]
        bys_prompts = [TEMPLATES[0].format(s=b) for b in bys]

        p.dirs = None
        h = p.state(prompts)
        guard = set(p.clean_top(prompts, CLEAN_GUARD))
        lens0 = (h.mean(0) @ p.w_out.t())
        nominated = [int(t) for t in lens0.topk(NOMINATE).indices.tolist() if int(t) not in guard]
        jl = jlens_vectors(p.lm, layer, nominated[:NOMINATE], corpus["input_ids"],
                           corpus["attention_mask"], p.w_out)
        score = (h.mean(0) @ jl.vectors.t())
        order = [int(i) for i in score.argsort(descending=True).tolist()][:pool_size]
        ids = [int(jl.token_ids[i]) for i in order]
        atoms = jl.vectors[torch.as_tensor(order)].detach()

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
        star = None
        for d in subsets:
            if answering and not any(table[d][q] for q in answering):
                star = d
                break

        base = {"subject": s, "pool": ids, "n_answering": len(answering),
                "answer_before": held_rate[s], "guard_tokens": len(guard),
                "silenceable": float(star is not None)}
        if star is None:
            rows.append({**base, "excluded": "no subset of the J-lens pool silences every phrasing"})
            if verbose:
                print(f"  {s:<10} the pool CANNOT silence it: every one of {len(subsets)} subsets "
                      f"leaves at least one phrasing answering  ({time.time() - t0:.0f}s)", flush=True)
            continue

        greedy: List[int] = []
        live = list(answering)
        for atom in ids:
            if not live:
                break
            greedy.append(atom)
            live = [q for q in live if table[tuple(sorted(greedy, key=ids.index))][q]]
        greedy_key = tuple(sorted(greedy, key=ids.index))

        def silences_for(q: int):
            def f(d: Sequence[int]) -> bool:
                return not table[tuple(sorted((int(x) for x in d), key=ids.index))][q]
            return f

        n_atoms = max(1, pool_size // 2)
        supports, certs, resid = [], [], []
        for q in answering:
            sup = nonneg_pursuit(h[q], atoms, ids=ids, n_atoms=n_atoms, tol=0.05)
            supports.append(sup)
            resid.append(sup.residual_fraction)
            certs.append(certify_must_hit(silences_for(q), sup.directions, ids))
        bound = disjoint_lower_bound(certs)
        cc = certified_closure(obj_id, len(star), bound, len(answering),
                               workload=f"{len(answering)} phrasings")

        obj_atom = ids[0]
        rest_idx = list(range(1, len(ids)))
        rand_hold, rand_obj_hold, rand_bound = [], [], []
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

        probe_before = probe_under(None)
        probe_after = probe_under(atoms[[ids.index(x) for x in star]])
        p.dirs = None

        row = {
            **base, "excluded": None,
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
            "probe_before": probe_before, "probe_after": probe_after,
            "probe_drop": probe_before - probe_after,
            "summary": cc.summary(),
        }
        rows.append(row)
        if verbose:
            print(f"  {s:<10} {cc.summary()} | greedy {len(greedy_key)}, TRUE optimum {len(star)} | "
                  f"must-hit {row['musthit_rate']:.2f} vs random {row['musthit_rate_random']:.2f} vs "
                  f"random+obj {row['musthit_rate_random_with_object']:.2f} | collateral "
                  f"{row['collateral_at_optimum']:.2f} from {row['collateral_before']:.2f} | PROBE "
                  f"{probe_before:.2f} -> {probe_after:.2f}  ({time.time() - t0:.0f}s)", flush=True)

    good = [r for r in rows if r.get("excluded") is None]
    m: Dict[str, Any] = {"layer": layer, "pool_size": pool_size, "n_held": len(held),
                         "n_attempted": len(rows), "n_measured": len(good),
                         "silenceable_rate": float(np.mean([r["silenceable"] for r in rows]))
                         if rows else float("nan"),
                         "held/answer_before": float(np.mean([held_rate[s] for s in held])),
                         "positive_control": positive_control()}
    m["control_bound_can_exceed_one"] = m["positive_control"]["ok"]
    if len(good) < 3:
        m["finding"] = ("the paper-faithful J-lens ablation, with the paper's own guard against "
                        "touching tokens in the clean output, does not silence these facts at this "
                        "pool size -- so there is no closure to bound, and that is the result")
        m["per_fact"] = rows
        m["seconds"] = time.time() - t0
        return m
    for k in ("optimum", "greedy", "greedy_excess", "collateral_at_optimum", "collateral_before",
              "support_size", "support_residual", "musthit_rate", "musthit_exhaustive",
              "musthit_vacuous", "musthit_subsets_tested", "lower_bound", "bound_certified",
              "shared_atoms", "bound_sound", "tightness", "musthit_rate_random",
              "musthit_rate_random_with_object", "object_atom_in_support", "lower_bound_random",
              "probe_before", "probe_after", "probe_drop"):
        m[k] = float(np.mean([r[k] for r in good]))
    m["bound_sound_min"] = float(np.min([r["bound_sound"] for r in good]))
    m["musthit_advantage"] = m["musthit_rate"] - m["musthit_rate_random"]
    m["musthit_advantage_over_object"] = m["musthit_rate"] - m["musthit_rate_random_with_object"]
    cores = {r["subject"]: set(r["core_atoms"]) for r in good}
    m["pod_rate"] = float(np.mean([len(c) > 0 for c in cores.values()]))
    m["core_size"] = float(np.mean([len(c) for c in cores.values()]))
    shares = [len(cores[a] & cores[b]) / len(cores[a])
              for a in cores for b in cores if a != b and cores[a]]
    m["cross_fact_core_overlap"] = float(np.mean(shares)) if shares else float("nan")
    m["per_fact"] = rows
    m["seconds"] = time.time() - t0
    return m


KEYS = ["n_held", "n_attempted", "n_measured", "silenceable_rate", "held/answer_before",
        "control_bound_can_exceed_one", "optimum", "greedy", "greedy_excess",
        "collateral_at_optimum", "collateral_before", "support_size", "support_residual",
        "musthit_rate", "musthit_exhaustive", "musthit_vacuous", "musthit_subsets_tested",
        "lower_bound", "bound_certified", "shared_atoms", "bound_sound", "bound_sound_min",
        "tightness", "musthit_rate_random", "musthit_rate_random_with_object",
        "object_atom_in_support", "lower_bound_random", "musthit_advantage",
        "musthit_advantage_over_object", "pod_rate", "core_size", "cross_fact_core_overlap",
        "probe_before", "probe_after", "probe_drop"]

CRITERIA = {
    "held/answer_before": (">=", 0.75),
    "control_bound_can_exceed_one": (">=", 1.0),
    # soundness, checked against the true optimum found by enumeration; one violation falsifies it
    "bound_sound_min": (">=", 1.0),
    "musthit_exhaustive": (">=", 1.0),
    "musthit_vacuous": ("<=", 0.0),
    "musthit_rate": (">=", 0.70),
    "musthit_rate_random": ("<=", 0.40),
    "musthit_advantage": (">=", 0.30),
    "musthit_advantage_over_object": (">=", 0.15),
    # THE DELETION MUST BE A DELETION. If a probe refitted on the ablated states still reads the
    # object, the closure removed a readout path and not a fact.
    "probe_after": ("<=", 0.40),
}

DECISION_RULE = (
    "The bound is reported as CERTIFIED only where every support in the disjoint family passed an "
    "exhaustive, non-vacuous must-hit test. Where the pool cannot silence a fact there is no closure "
    "to bound, and that is reported as the finding rather than as a void experiment.")


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--pool", type=int, default=POOL)
    ap.add_argument("--max-facts", type=int, default=6)
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
        "pool_size": args.pool, "seeds": args.seeds, "decision_rule": DECISION_RULE,
        "seeds_vary": "the random-support controls only; the ablation table is exhaustive and the "
                      "model is frozen, so the optimum, the supports and the bound are deterministic",
        "result": m, "aggregate": agg, "criteria": check}

    md = [f"# E-000042 — {record['title']}", "",
          f"Frozen {MODEL}, layer {args.layer}, no training. Directions are J-lens vectors -- rows of",
          "`W_U J_l`, one vector-Jacobian product each (`so/jlens.py`) -- and the pool never contains a",
          f"token in the clean top-{CLEAN_GUARD} output, which is the workspace paper's own guard",
          "against ablating the readout instead of the fact. Every subset of the pool is ablated, so",
          "the true optimum comes from enumeration and the bound is compared against the answer.", ""]
    if "finding" in m:
        md += ["## The pool cannot silence these facts", "", m["finding"] + ".", "",
               ledger.table(["measure", "value"],
                            [["facts the model answers at >= 0.75", f"{m['n_held']}"],
                             ["facts attempted", f"{m['n_attempted']}"],
                             ["facts any subset of the pool could silence",
                              f"{m['silenceable_rate']:.4f}"]]), "",
               "This is a result, not a failure to get one: a J-lens ablation that respects the",
               "paper's guard does not remove a capital fact from GPT-2 small at this pool size. The",
               "eight-direction ablation that DID silence these facts in an earlier run removed the",
               "unembedding rows of the candidate answers themselves, which blinds the readout rather",
               "than deleting anything.", ""]
        record["finding"] = m["finding"]
    else:
        md += ["## The interval, against the optimum found by enumeration", "",
               ledger.table(["measure", "mean over facts"],
                            [["facts the pool can silence at all", f"{m['silenceable_rate']:.4f}"],
                             ["TRUE optimum, by exhaustive enumeration", f"{m['optimum']:.2f}"],
                             ["greedy upper bound", f"{m['greedy']:.2f}"],
                             ["certified lower bound from disjoint J-lens supports",
                              f"{m['lower_bound']:.2f}"],
                             ["bound / optimum (tightness)", f"{m['tightness']:.4f}"],
                             ["**bound <= optimum, worst fact**", f"**{m['bound_sound_min']:.4f}**"]]), "",
               "## Is it a deletion, or only a blinding", "",
               "An ablation that moves the argmax need not have removed anything. A linear probe is",
               "refitted on the ABLATED states, leave-one-phrasing-out, and asked for the object.", "",
               ledger.table(["measure", "mean over facts"],
                            [["the model answers, before", f"{m['collateral_before']:.4f}"],
                             ["probe reads the object, before", f"{m['probe_before']:.4f}"],
                             ["**probe reads the object, after the optimum**",
                              f"**{m['probe_after']:.4f}**"],
                             ["bystander facts at the optimum", f"{m['collateral_at_optimum']:.4f}"]]), "",
               "## Is the support really a must-hit set", "",
               ledger.table(["measure", "mean over facts"],
                            [["atoms in the support", f"{m['support_size']:.2f}"],
                             ["disjoint ablations tried per phrasing",
                              f"{m['musthit_subsets_tested']:.1f}"],
                             ["support passes the exhaustive must-hit test", f"{m['musthit_rate']:.4f}"],
                             ["a random support of the same size does (control 1)",
                              f"{m['musthit_rate_random']:.4f}"],
                             ["a random support that ALSO holds the object direction (control 2)",
                              f"{m['musthit_rate_random_with_object']:.4f}"],
                             ["**advantage over control 2**",
                              f"**{m['musthit_advantage_over_object']:+.4f}**"],
                             ["atoms every phrasing runs through (the pod core)",
                              f"{m['core_size']:.2f}"],
                             ["share of a core that is another fact's core too",
                              f"{m['cross_fact_core_overlap']:.4f}"]]), "",
               "Control 2 is the one that can really kill the claim: any support containing the",
               "object's own direction is must-hit for a nearly trivial reason, so if a random support",
               "that also contains it passes as often, the decomposition has added nothing.", "",
               "## The positive control", "",
               f"A synthetic table with {m['positive_control']['expected']} pairwise-disjoint supports "
               f"by construction: the bound reports {m['positive_control']['bound']}. Without it a "
               "bound that returned 1 unconditionally would read as a finding about the model.", ""]
    md += ["## Pre-registered criteria", "", ledger.criteria_table(check), ""]
    path = ledger.save("e000042_certified_closure", record, "\n".join(md))
    print("\n".join(md[1:]))
    print(f"\nwritten: {path}")
    return record


if __name__ == "__main__":
    main()
