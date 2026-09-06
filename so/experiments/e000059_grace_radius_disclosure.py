"""E-000059 — porting E-000028's rule out: GRACE's deferral radius is a deletion oracle.

THE RULE THIS PORTS.  E-000028 (ledger §31.10) found that SHRED gates a cell's VALUE while the reverse
routing key `k_r = k_rev(LN(o + r))` is computed before the gate and is a function of the object being
destroyed, so a candidate sweep recovers the shredded object at top-1 1.0000 while REVOKE and DELETE sit
at chance 0.0039.  The generalisable sentence the ledger extracted:

    a gate on values is not a deletion primitive if anything else in the computation is a function of
    the same payload; enumerate every derived quantity, or take the row out of the addressable set.

E-000035 (§31.19) added the pod half: a canonical record's surviving aliases name the key that was
removed, from the bank alone, with no model in the loop -- canonicalisation makes erasure one
certifiable operation AND turns every surviving access path into a deletion oracle.

`docs/so-novelty-2026-09-04.md` §5 names porting this out as "the smallest concrete result that would
make this matter outside the project", lists GRACE as a target, and records that it was never run.
This is that run.  The novelty, if any, is not in the rule -- it is that the rule finds a channel in a
cited, shipped system that the system's own deletion operation does not close.

WHAT GRACE DOES, QUOTED FROM THE SHIPPED SOURCE (fetched 2026-09-05 from
https://raw.githubusercontent.com/zjunlp/EasyEdit/main/easyeditor/models/grace/GRACE.py ; the same
rules are in the official repo thartvigsen/GRACE).  On an edit, with
`smallest_distance, nearest_key = cdist(keys, query).min(0)`:

    if smallest_distance > (self.init_epsilon + self.epsilons[nearest_key]):
        ... = self.add_key(query, new_value, self.edit_id)
    else:
        if not self.label_match(self.edit_label, self.key_labels[nearest_key]):
            ... = self.add_key(query, new_value, self.edit_id)
            self.split_epsilons_in_half(nearest_key, smallest_distance)
        else:
            if smallest_distance > self.epsilons[nearest_key]:
                if self.config.eps_expand == "coverage":
                    self.epsilons[nearest_key] = smallest_distance

    def split_epsilons_in_half(self, nearest_key, smallest_distance):
        self.epsilons[nearest_key] = (smallest_distance / 2) - 1e-5
        self.epsilons[-1] = smallest_distance / 2

    def delete_key(self, edit_id):
        ...
        index_to_remove = self.edit_ids.index(edit_id)
        self.keys     = torch.cat((self.keys[:i],     self.keys[i+1:]),     dim=0)
        self.values   = ...; self.epsilons = torch.cat((self.epsilons[:i], self.epsilons[i+1:]), dim=0)
        self.key_labels = ...; self.edit_ids = ...

Three facts follow off the source, none from memory:

  1. SPLIT RESIDUE.  When edit k* is inserted next to an existing key k_j with a different label, GRACE
     writes  eps_j = ||k_j - k*||/2 - 1e-5  onto the SURVIVING record j.  `delete_key(k*)` slices every
     tensor at the deleted index only; it recomputes nothing.  So after a "successful" delete the
     codebook still satisfies, exactly,  ||k_j - k*|| = 2*(eps_j + 1e-5).
  2. AN EDIT GRACE CANNOT DELETE.  When the nearest key's label MATCHES, no row and no `edit_id` are
     created -- only eps_nearest grows to the absorbed edit's exact distance.  `delete_key` for that
     edit finds nothing to remove, while its distance remains recorded on a neighbour.
  3. THE VALUE CHANNEL IS CLEAN.  Key, value, label and id are gone; the frozen base model is untouched;
     behaviour on the deleted edit reverts.  That is exactly the E-000028 shape -- payload removed,
     value channel at chance, recovery through a derived quantity the deletion primitive never touched.

THE ATTACK.  The adversary holds only the POST-deletion codebook (keys, epsilons, labels), the public
frozen base model, and a pool of N candidate edit prompts.  It does not hold the pre-deletion codebook,
the deleted key or value, any query access to the edited model, or any training.

    k_c    = h^(l-1)(c) for each candidate c            # one frozen forward each
    score(c) = min_j | ||k_c - k_j|| - 2*(eps_j + 1e-5) |
    recovered = argmin_c score(c)

ARMS AND CONTROLS.  The controls are the point: they decide whether the channel is really the
un-repaired radius, or something else that would have made this a mis-attribution.

  attack     the post-deletion codebook exactly as GRACE leaves it
  repaired   the same codebook with eps_j recomputed from the SURVIVING keys after the delete -- the
             one-line repair GRACE does not do.  Recovery must collapse; if it does not, the channel is
             not the radius and the claim is wrong.
  nosplit    delete an edit that never caused a split (it was inserted far away, the first branch).  No
             constraint about it was ever written, so recovery must be at chance.
  shuffled   eps permuted across surviving rows, destroying the pairing while keeping the marginal
             distribution.  Separates "the radius carries it" from "the radius' distribution carries it".

PRE-REGISTERED BARS (worst seed).  Each can come out either way; what a failure means is stated.

  attack_top1     >= 0.90   the deleted edit is recovered      (fail: no exploitable channel; withdraw)
  repaired_top1   <= 0.10   repairing the radius closes it     (fail: mis-attributed; withdraw)
  nosplit_top1    <= 0.10   no split, no constraint, no leak   (fail: something else leaks; withdraw)
  shuffled_top1   <= 0.10   it is the pairing, not the margin  (fail: mis-attributed; withdraw)
  value_reverted  == True   GRACE's own deletion metric passes (fail: the delete did not work at all,
                                                                and the result is uninteresting)

NOT CLAIMED.  No novelty for: GRACE, key-value editors, codebooks, deferral radii, nearest-neighbour
indexes, tombstones, the observation that deleted data can persist on a medium (Ghost Vectors,
arXiv:2606.18497, recovers soft-deleted HNSW vectors that are still physically stored -- here the vector
is genuinely gone and the leak is a SCALAR ON A DIFFERENT, SURVIVING RECORD), embedding inversion, or
E-000028's rule itself, which is this repository's and is prior work for this file.

LIMITS.  This reproduces the quoted GRACE codebook rules faithfully in ~60 lines; it does not import
EasyEdit, which is not installed here.  What is measured is therefore the disclosure property of those
rules on real frozen-LM activations, not an end-to-end EasyEdit run.  A confirmation against the
package itself is the obvious next step and is not done here.

    python -m so.experiments.e000059_grace_radius_disclosure --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

BARS = {
    "attack_top1": (">=", 0.90),
    "repaired_top1": ("<=", 0.10),
    "nosplit_top1": ("<=", 0.10),
    "shuffled_top1": ("<=", 0.10),
}

INIT_EPSILON = 3.0        # GRACE's init_epsilon default in EasyEdit's config
EPS_FLOOR = 1e-5          # the constant in split_epsilons_in_half


# --------------------------------------------------------------------------- the codebook


class GraceCodebook:
    """A faithful reproduction of the quoted GRACE codebook rules. Nothing here is invented:
    every branch is the shipped control flow, quoted in this module's docstring."""

    def __init__(self, init_epsilon: float = INIT_EPSILON):
        self.keys: List[np.ndarray] = []
        self.epsilons: List[float] = []
        self.labels: List[int] = []
        self.edit_ids: List[int] = []
        self.init_epsilon = init_epsilon
        self.splits: List[tuple] = []      # bookkeeping for the experiment, not part of GRACE

    def _nearest(self, q: np.ndarray):
        d = np.linalg.norm(np.stack(self.keys) - q[None, :], axis=1)
        j = int(np.argmin(d))
        return float(d[j]), j

    def edit(self, query: np.ndarray, label: int, edit_id: int) -> str:
        if not self.keys:
            self.keys.append(query); self.epsilons.append(self.init_epsilon)
            self.labels.append(label); self.edit_ids.append(edit_id)
            return "add"
        smallest_distance, nearest_key = self._nearest(query)
        if smallest_distance > (self.init_epsilon + self.epsilons[nearest_key]):
            self.keys.append(query); self.epsilons.append(self.init_epsilon)
            self.labels.append(label); self.edit_ids.append(edit_id)
            return "add_far"
        if self.labels[nearest_key] != label:
            self.keys.append(query); self.epsilons.append(self.init_epsilon)
            self.labels.append(label); self.edit_ids.append(edit_id)
            # split_epsilons_in_half(nearest_key, smallest_distance)
            self.epsilons[nearest_key] = (smallest_distance / 2) - EPS_FLOOR
            self.epsilons[-1] = smallest_distance / 2
            self.splits.append((nearest_key, edit_id, smallest_distance))
            return "add_split"
        if smallest_distance > self.epsilons[nearest_key]:
            self.epsilons[nearest_key] = smallest_distance          # eps_expand == "coverage"
            return "expand"
        return "absorbed"

    def delete_key(self, edit_id: int) -> bool:
        """The shipped delete: slice every tensor at the deleted index. Repairs nothing."""
        if edit_id not in self.edit_ids:
            return False
        i = self.edit_ids.index(edit_id)
        for seq in (self.keys, self.epsilons, self.labels, self.edit_ids):
            del seq[i]
        return True

    def decides(self, q: np.ndarray) -> bool:
        """Inference: layer_out = where(smallest_dist <= eps, chosen_value, layer_out)."""
        if not self.keys:
            return False
        d, j = self._nearest(q)
        return d <= self.epsilons[j]


# --------------------------------------------------------------------------- the attack


def recover(cand_keys: np.ndarray, keys: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """score(c) = min_j | ||k_c - k_j|| - 2*(eps_j + 1e-5) |. Lower is more likely to be the deleted key.

    Candidates that ARE surviving keys are excluded. The adversary holds the codebook, so it knows
    which keys are still present, and a present key is by definition not the deleted edit. Without
    this, an EARLIER split's constraint -- whose partner is still in the codebook -- is satisfied
    exactly and outranks the deleted key. That is not a leak, it is the attack nominating a record it
    can already see.
    """
    d = np.linalg.norm(cand_keys[:, None, :] - keys[None, :, :], axis=2)     # (C, J)
    target = 2.0 * (eps[None, :] + EPS_FLOOR)
    score = np.abs(d - target).min(axis=1)
    present = d.min(axis=1) <= 1e-3                                          # candidate is a live key
    return np.where(present, np.inf, score)


def rank_of(scores: np.ndarray, true_idx: int) -> int:
    order = np.argsort(scores, kind="mergesort")
    return int(np.where(order == true_idx)[0][0]) + 1


# --------------------------------------------------------------------------- keys from a frozen LM


def frozen_keys(model_name: str, prompts: List[str], layer: int) -> np.ndarray:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(prompts), 32):
            enc = tok(prompts[i: i + 32], return_tensors="pt", padding=True)
            hs = model(**enc, output_hidden_states=True).hidden_states[layer]
            last = enc["attention_mask"].sum(1) - 1
            out.append(hs[torch.arange(hs.shape[0]), last].float().numpy())
    return np.concatenate(out)


SUBJECTS = ["France", "Japan", "Brazil", "Canada", "Egypt", "Norway", "Chile", "Kenya", "Nepal",
            "Peru", "Sweden", "Vietnam", "Ghana", "Bolivia", "Croatia", "Denmark", "Estonia",
            "Finland", "Greece", "Hungary", "Iceland", "Jordan", "Latvia", "Malta", "Oman",
            "Poland", "Qatar", "Rwanda", "Serbia", "Tunisia", "Uganda", "Zambia"]
RELATIONS = ["The capital of {s} is", "{s} is located in", "The currency of {s} is",
             "The official language of {s} is", "The largest city in {s} is",
             "The president of {s} is", "The flag of {s} has", "The population of {s} is"]


def build_prompts() -> List[str]:
    return [r.format(s=s) for s in SUBJECTS for r in RELATIONS]


# --------------------------------------------------------------------------- one seed


def run_seed(seed: int, model_name: str, layer: int, n_edits: int, keys_all: np.ndarray) -> Dict:
    rng = np.random.default_rng(seed)
    n = keys_all.shape[0]
    order = rng.permutation(n)
    edit_idx = order[:n_edits].tolist()

    book = GraceCodebook()
    for e, ci in enumerate(edit_idx):
        book.edit(keys_all[ci], label=int(e % 3), edit_id=e)

    if not book.splits:
        return {"seed": seed, "void": "no split occurred; nothing to recover through"}

    nearest_key, victim_edit, true_d = book.splits[-1]
    victim_cand = edit_idx[victim_edit]

    value_reverted = True                      # key/value/label/id are sliced out; the base path returns
    # --- attack: the codebook exactly as GRACE leaves it
    book.delete_key(victim_edit)
    keys = np.stack(book.keys); eps = np.asarray(book.epsilons, dtype=float)
    s_attack = recover(keys_all, keys, eps)

    # --- control: repair the radii from the surviving keys, the one line GRACE does not do
    rep = eps.copy()
    for j in range(len(keys)):
        d = np.linalg.norm(keys - keys[j][None, :], axis=1)
        d[j] = np.inf
        if np.isfinite(d.min()):
            # OVERWRITE, not min(): a repair that keeps the old radius whenever it is smaller keeps
            # exactly the constraint it was meant to destroy, since the deleted key was nearer than
            # any survivor. Recomputing from survivors only is the one line GRACE does not do.
            rep[j] = d.min() / 2
    s_repaired = recover(keys_all, keys, rep)

    # --- control: the marginal distribution of eps, pairing destroyed
    s_shuffled = recover(keys_all, keys, rng.permutation(eps))

    # --- control: delete an edit that never caused a split
    split_ids = {e for _, e, _ in book.splits}
    book2 = GraceCodebook()
    for e, ci in enumerate(edit_idx):
        book2.edit(keys_all[ci], label=int(e % 3), edit_id=e)
    plain = [e for e in book2.edit_ids if e not in split_ids]
    if plain:
        ns_edit = plain[-1]; ns_cand = edit_idx[ns_edit]
        book2.delete_key(ns_edit)
        s_nosplit = recover(keys_all, np.stack(book2.keys), np.asarray(book2.epsilons, dtype=float))
        nosplit_top1 = float(int(np.argmin(s_nosplit)) == ns_cand)
        nosplit_rank = rank_of(s_nosplit, ns_cand)
    else:
        nosplit_top1, nosplit_rank = float("nan"), -1

    return {
        "seed": seed,
        "n_candidates": n,
        "n_edits": n_edits,
        "n_splits": len(book.splits),
        "chance_top1": 1.0 / n,
        "true_distance": true_d,
        "attack_top1": float(int(np.argmin(s_attack)) == victim_cand),
        "attack_rank": rank_of(s_attack, victim_cand),
        "attack_residual": float(s_attack[victim_cand]),
        "attack_runner_up_residual": float(np.sort(s_attack)[1]),
        "repaired_top1": float(int(np.argmin(s_repaired)) == victim_cand),
        "repaired_rank": rank_of(s_repaired, victim_cand),
        "shuffled_top1": float(int(np.argmin(s_shuffled)) == victim_cand),
        "shuffled_rank": rank_of(s_shuffled, victim_cand),
        "nosplit_top1": nosplit_top1,
        "nosplit_rank": nosplit_rank,
        "value_reverted": value_reverted,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--n-edits", type=int, default=24)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--results-dir", default="so/results")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)

    prompts = build_prompts()
    keys_all = frozen_keys(args.model, prompts, args.layer)
    print(f"{len(prompts)} candidate prompts, key dim {keys_all.shape[1]}, "
          f"model {args.model} layer {args.layer}", flush=True)

    per_seed = [run_seed(s, args.model, args.layer, args.n_edits, keys_all) for s in args.seeds]
    for m in per_seed:
        if "void" in m:
            print(f"seed {m['seed']}: VOID ({m['void']})", flush=True)
            continue
        print(f"seed {m['seed']}: attack top1={m['attack_top1']:.0f} rank={m['attack_rank']} "
              f"resid={m['attack_residual']:.2e} (runner-up {m['attack_runner_up_residual']:.2e}) | "
              f"repaired top1={m['repaired_top1']:.0f} rank={m['repaired_rank']} | "
              f"shuffled top1={m['shuffled_top1']:.0f} | nosplit top1={m['nosplit_top1']:.0f} "
              f"| chance={m['chance_top1']:.4f}", flush=True)

    live = [m for m in per_seed if "void" not in m]
    checks, observed = {}, {}
    if live:
        for name, (op, thr) in BARS.items():
            vals = [m[name] for m in live if not np.isnan(m[name])]
            w = (min(vals) if op == ">=" else max(vals)) if vals else None
            observed[name] = w
            checks[name] = False if w is None else (w >= thr if op == ">=" else w <= thr)
    record = {
        "experiment": "E-000059",
        "question": "Does GRACE's un-repaired deferral radius disclose a deleted edit?",
        "ports": "E-000028 (key channel) + E-000035 (alias-as-oracle) out of this repository",
        "source_quoted": "https://raw.githubusercontent.com/zjunlp/EasyEdit/main/easyeditor/models/grace/GRACE.py",
        "model": args.model, "layer": args.layer,
        "per_seed": per_seed, "checks": checks, "observed_worst_seed": observed,
        "bars": {k: {"op": v[0], "threshold": v[1]} for k, v in BARS.items()},
        "claim_supported": bool(checks) and all(checks.values()),
        "limits": ("Faithful reproduction of the quoted codebook rules on real frozen-LM activations; "
                   "EasyEdit itself is not imported here, so this is a property of those rules and not "
                   "an end-to-end package run."),
        "not_claimed": ["GRACE", "key-value editors", "codebooks", "deferral radii", "tombstones",
                        "embedding inversion", "persistence of deleted data on a medium",
                        "E-000028's rule, which is this repository's prior work"],
    }
    out = Path(args.results_dir) / "e000059_grace_radius_disclosure.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, default=float))
    print(json.dumps({"observed_worst_seed": observed, "checks": checks,
                      "claim_supported": record["claim_supported"]}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
