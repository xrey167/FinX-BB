"""E-000062 -- can a causal J-space signature serve as a better scope/address ABI?

This is the first experiment in the novelty track that can actually falsify the J-space part of WNVI.
It does NOT test external memory yet. It asks a narrower prerequisite on a public frozen GPT-2:
when a router has entries for only half the facts, can a Jacobian-lens workspace signature route
held-out paraphrases to the right identity while rejecting same-form questions about UNINDEXED facts
and unrelated prose better than non-causal representations?

Why explicit negatives matter: Singh (arXiv:2608.26292, Aug 2026) shows current counterfactual-edit
benchmarks often cannot measure scope abstention at all. We therefore make withheld facts and generic
prose first-class negatives. No test negative is used to select a threshold.

Representations, identical downstream nearest-centroid router:
  raw      full layer residual (strong, high-dimensional control)
  random   matched-dimensional random projection of the residual
  embed    matched-dimensional projection onto country token embeddings (semantic-key control)
  jspace   matched-dimensional projection onto COUNTRY J-lens vectors estimated from an independent
           prompt corpus; this is the causal workspace candidate

Each split uses three of four train phrasings to build pod centroids and the fourth only to select the
abstention threshold. Four disjoint held-out phrasings are the test positives/withheld negatives.
Generic validation and test sentences are disjoint. Subject split and validation template rotate by
seed. GPT-2's EOS token is prepended to every evaluated prompt to remove the position-0 artifact
already established in E-000050; all arms receive the same prefix.

Pre-registered J-space ABI screen (worst seed):
  jspace joint accuracy >= .75
  jspace positive correct-route rate >= .70
  jspace specificity >= .80
  jspace joint - embed joint >= .05
  jspace joint - random joint >= .05
  jspace scope balanced accuracy - raw >= .02

Failing the last three means J-space has not earned an architectural role even if its absolute
numbers look good. A positive screen still needs another backbone and the composed symlink test.

Run: python -m so.experiments.e000062_jspace_address_abi --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from so.jlens import jlens_vectors

MODEL = "gpt2"
LAYER = 7

PAIRS = [("France", " Paris"), ("Japan", " Tokyo"), ("Italy", " Rome"), ("Germany", " Berlin"),
         ("Russia", " Moscow"), ("China", " Beijing"), ("Spain", " Madrid"), ("Egypt", " Cairo"),
         ("Canada", " Ottawa"), ("Greece", " Athens"), ("Cuba", " Havana"), ("Iran", " Tehran"),
         ("Poland", " Warsaw"), ("Austria", " Vienna"), ("Norway", " Oslo"), ("Kenya", " Nairobi")]

TRAIN_T = ["The capital of {s} is", "{s}'s capital city is", "Q: What is the capital of {s}? A:",
           "In {s}, the capital is"]
TEST_T = ["The city that serves as the capital of {s} is", "People say the capital of {s} is",
          "The seat of government of {s} is located in", "Everyone knows that the capital of {s} is"]

GEN_VAL = [
    "The river moved slowly through the valley after the storm.",
    "She opened the window because the room had become warm.",
    "A compiler translates source code before the program is executed.",
    "The meeting was moved to Thursday afternoon after lunch.",
    "Several musicians waited backstage before the concert began.",
    "The telescope collected light from a very distant galaxy.",
    "Coffee beans are roasted before they are ground and brewed.",
    "He checked the battery level before leaving the house.",
]
GEN_TEST = [
    "The old bridge was repaired during the summer months.",
    "A small garden grew behind the library near the courtyard.",
    "The software stores temporary files in a separate directory.",
    "Clouds formed over the mountains shortly before sunset.",
    "The violinist adjusted the instrument between two pieces.",
    "A microscope can reveal structures that are too small to see directly.",
    "They packed the equipment carefully before the return journey.",
    "The report contains several tables and a short appendix.",
]

CORPUS = [
    "The capital of France is Paris, and the capital of Japan is Tokyo.",
    "Germany, Italy, Spain and Greece are countries in Europe.",
    "China and Japan are countries in Asia with large cities.",
    "Canada is north of the United States and has several major cities.",
    "A government may move offices while a city remains its capital.",
    "People ask geographical questions in many different ways.",
    "The company reported earnings above what analysts had expected.",
    "Water boils at one hundred degrees Celsius at sea level pressure.",
    "She opened the book and began reading the first chapter slowly.",
    "Machine learning models are trained on large collections of text.",
    "The train crossed the river and continued toward the coast.",
    "A telescope gathers light and forms an image of distant objects.",
]


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


class Probe:
    def __init__(self, layer: int, threads: int):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if threads:
            torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.tok.pad_token = self.tok.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(MODEL).eval()
        self.layer = layer
        self.prefix = self.tok.eos_token or ""
        self.w_in = self.lm.get_input_embeddings().weight.detach()
        self.w_out = self.lm.get_output_embeddings().weight.detach()

    def encode(self, texts: Sequence[str]):
        return self.tok([self.prefix + t for t in texts], return_tensors="pt", padding=True)

    @torch.no_grad()
    def state(self, texts: Sequence[str]) -> torch.Tensor:
        e = self.encode(texts)
        last = e["attention_mask"].sum(1) - 1
        hs = self.lm(**e, output_hidden_states=True).hidden_states[self.layer]
        return hs[torch.arange(len(texts)), last].detach().float()


def threshold(scores_pos: np.ndarray, scores_neg: np.ndarray) -> Tuple[float, float]:
    """Choose tau using validation only; deterministic highest-tau tie break favors specificity."""
    vals = np.unique(np.concatenate([scores_pos, scores_neg]))
    candidates = np.concatenate([[vals.min() - 1e-6], (vals[:-1] + vals[1:]) / 2, [vals.max() + 1e-6]])
    best = (-1.0, -1e9)
    for tau in candidates:
        tpr = float(np.mean(scores_pos >= tau)) if len(scores_pos) else 0.0
        tnr = float(np.mean(scores_neg < tau)) if len(scores_neg) else 0.0
        bal = 0.5 * (tpr + tnr)
        key = (bal, float(tau))
        if key > best:
            best = key
    return float(best[1]), float(best[0])


def route(x: torch.Tensor, centroids: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    x = normalize(x); c = normalize(centroids)
    s = x @ c.t()
    mx, pred = s.max(dim=-1)
    return mx.cpu().numpy(), pred.cpu().numpy()


def one_rep(name: str, train: Dict[int,torch.Tensor], val_pos: Dict[int,torch.Tensor],
            val_neg: torch.Tensor, test_pos: Dict[int,torch.Tensor], withheld_test: torch.Tensor,
            generic_test: torch.Tensor) -> Dict[str,float]:
    ids = sorted(train)
    cent = torch.stack([normalize(train[i]).mean(0) for i in ids])
    valp = torch.cat([val_pos[i] for i in ids])
    valp_scores, _ = route(valp, cent)
    valn_scores, _ = route(val_neg, cent)
    tau, val_bal = threshold(valp_scores, valn_scores)

    xp = torch.cat([test_pos[i] for i in ids])
    y = np.concatenate([[j] * len(test_pos[i]) for j, i in enumerate(ids)])
    ps, pp = route(xp, cent)
    accepted = ps >= tau
    pos_correct = float(np.mean(accepted & (pp == y)))
    pos_accept = float(np.mean(accepted))
    route_given_accept = float(np.mean(pp[accepted] == y[accepted])) if accepted.any() else 0.0

    xw_scores, _ = route(withheld_test, cent)
    xg_scores, _ = route(generic_test, cent)
    neg_scores = np.concatenate([xw_scores, xg_scores])
    specificity = float(np.mean(neg_scores < tau))
    withheld_specificity = float(np.mean(xw_scores < tau))
    generic_specificity = float(np.mean(xg_scores < tau))
    scope_bal = 0.5 * (pos_accept + specificity)
    joint = float((np.sum(accepted & (pp == y)) + np.sum(neg_scores < tau)) / (len(y) + len(neg_scores)))
    return {
        "tau": tau, "val_bal": val_bal, "pos_accept": pos_accept,
        "pos_correct_route": pos_correct, "route_given_accept": route_given_accept,
        "specificity": specificity, "withheld_specificity": withheld_specificity,
        "generic_specificity": generic_specificity, "scope_bal": scope_bal, "joint": joint,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0,1,2,3,4])
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--jlens-batch", type=int, default=6)
    ap.add_argument("--results-dir", default="so/results")
    a = ap.parse_args()
    p = Probe(a.layer, a.threads)

    # Keep only single-token subject/object pairs so every J-lens atom has an unambiguous vocabulary row.
    pairs = [(s,o) for s,o in PAIRS if len(p.tok.encode(" " + s)) == 1 and len(p.tok.encode(o)) == 1]
    if len(pairs) < 10:
        raise RuntimeError(f"too few single-token pairs for a scope experiment: {len(pairs)}")
    subjects = [s for s,_ in pairs]
    subject_ids = [p.tok.encode(" " + s)[0] for s in subjects]

    corpus_text = CORPUS + [TRAIN_T[0].format(s=s) for s in subjects]
    corpus = p.encode(corpus_text)
    jl = jlens_vectors(p.lm, a.layer, subject_ids, corpus["input_ids"], corpus["attention_mask"],
                       p.w_out, batch=a.jlens_batch)

    # Cache every hidden state once. Seed changes only the index split and random control.
    train_states = {s: p.state([t.format(s=s) for t in TRAIN_T]) for s in subjects}
    test_states = {s: p.state([t.format(s=s) for t in TEST_T]) for s in subjects}
    gen_val_h = p.state(GEN_VAL); gen_test_h = p.state(GEN_TEST)

    def transforms(seed: int):
        g = torch.Generator().manual_seed(61000 + seed)
        d = train_states[subjects[0]].shape[-1]
        m = len(subjects)
        rand = torch.randn(d, m, generator=g) / (d ** 0.5)
        emb = normalize(p.w_in[torch.as_tensor(subject_ids)]).t().contiguous()  # d x m
        jmat = jl.vectors.t().contiguous()                                      # d x m
        return {
            "raw": lambda h: h,
            "random": lambda h, R=rand: h @ R,
            "embed": lambda h, E=emb: h @ E,
            "jspace": lambda h, J=jmat: h @ J,
        }

    rows: List[Dict[str,object]] = []
    for seed in a.seeds:
        rng = np.random.default_rng(62000 + seed)
        perm = rng.permutation(len(subjects))
        n_idx = len(subjects)//2
        indexed = sorted(int(x) for x in perm[:n_idx])
        withheld = sorted(int(x) for x in perm[n_idx:])
        val_t = seed % len(TRAIN_T)
        centroid_t = [t for t in range(len(TRAIN_T)) if t != val_t]
        reps = transforms(seed)
        metrics: Dict[str,Dict[str,float]] = {}
        for name, fn in reps.items():
            tr = {i: fn(train_states[subjects[i]][centroid_t]) for i in indexed}
            vp = {i: fn(train_states[subjects[i]][val_t:val_t+1]) for i in indexed}
            vn_withheld = torch.cat([fn(train_states[subjects[i]][val_t:val_t+1]) for i in withheld])
            vn = torch.cat([vn_withheld, fn(gen_val_h)])
            tp = {i: fn(test_states[subjects[i]]) for i in indexed}
            wt = torch.cat([fn(test_states[subjects[i]]) for i in withheld])
            gt = fn(gen_test_h)
            metrics[name] = one_rep(name, tr, vp, vn, tp, wt, gt)
        row: Dict[str,object] = {"seed":seed, "indexed":indexed, "withheld":withheld,
                                "val_template":val_t, "metrics":metrics}
        row["jspace_joint_minus_embed"] = metrics["jspace"]["joint"] - metrics["embed"]["joint"]
        row["jspace_joint_minus_random"] = metrics["jspace"]["joint"] - metrics["random"]["joint"]
        row["jspace_scope_minus_raw"] = metrics["jspace"]["scope_bal"] - metrics["raw"]["scope_bal"]
        rows.append(row)
        print(f"seed {seed}: raw joint={metrics['raw']['joint']:.3f} scope={metrics['raw']['scope_bal']:.3f} | "
              f"embed joint={metrics['embed']['joint']:.3f} | random joint={metrics['random']['joint']:.3f} | "
              f"J joint={metrics['jspace']['joint']:.3f} scope={metrics['jspace']['scope_bal']:.3f} "
              f"pos={metrics['jspace']['pos_correct_route']:.3f} spec={metrics['jspace']['specificity']:.3f}")

    def worst(path: str, high: bool=True) -> float:
        vals=[]
        for r in rows:
            if path.startswith("metrics/"):
                _, rep, key = path.split("/")
                vals.append(float(r["metrics"][rep][key]))
            else:
                vals.append(float(r[path]))
        return min(vals) if high else max(vals)

    criteria = {
        "jspace_joint": {"observed":worst("metrics/jspace/joint"), "op":">=", "bar":.75},
        "jspace_pos_correct": {"observed":worst("metrics/jspace/pos_correct_route"), "op":">=", "bar":.70},
        "jspace_specificity": {"observed":worst("metrics/jspace/specificity"), "op":">=", "bar":.80},
        "jspace_minus_embed": {"observed":worst("jspace_joint_minus_embed"), "op":">=", "bar":.05},
        "jspace_minus_random": {"observed":worst("jspace_joint_minus_random"), "op":">=", "bar":.05},
        "jspace_scope_minus_raw": {"observed":worst("jspace_scope_minus_raw"), "op":">=", "bar":.02},
    }
    for c in criteria.values():
        c["pass"] = bool(c["observed"] >= c["bar"])
    screen = all(bool(c["pass"]) for c in criteria.values())
    rec = {"experiment":"E-000062", "candidate_only":True, "model":MODEL, "layer":a.layer,
           "n_pairs":len(pairs), "subjects":subjects, "jlens":{
               "n_prompts":jl.n_prompts, "n_positions":jl.n_positions,
               "raw_norm_min":float(jl.raw_norms.min()), "raw_norm_max":float(jl.raw_norms.max())},
           "criteria":criteria, "screening_pass":screen, "rows":rows}
    out=Path(a.results_dir); out.mkdir(parents=True, exist_ok=True)
    (out/"e000062_jspace_address_abi.json").write_text(json.dumps(rec,indent=2),encoding="utf-8")
    print(json.dumps({"screening_pass":screen,"criteria":criteria},indent=2))

if __name__ == "__main__":
    main()
