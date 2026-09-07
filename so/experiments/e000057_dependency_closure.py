"""Experiment E-000057 -- what a deleted row still contributes to a bystander's answer, with its null.

THE QUESTION, AND WHERE IT COMES FROM. The parallel branch's last surviving observation (ledger 31.49)
is this: a pointer has ONE referent, but the reader's answer is a softmax over the whole bank, so the
answer to a question about pod A also depends, with some coefficient, on rows belonging to a pod that
has been deleted. Their measurement: a deleted row carried routing coefficient 0.0118 / 0.0011 and
ablating it moved the logits by 0.2251 / 0.0151 max-abs, top-1 unchanged. Their conclusion: a
capability covering only the queried pod does not cover everything that shaped the answer, so the
revocation unit should be the whole dependency closure.

WHAT IS MISSING FROM THAT. No null. In a dense mixture every row has a non-zero coefficient and
ablating any of them moves the logits; a number for a deleted row means nothing until it is compared
with a LIVE row of the same coefficient. That is §31.41's and §31.45's lesson: the reader's arithmetic
supplies floors, and an uncalibrated number reports the arithmetic.

WHAT THE SUBSTRATE ACTUALLY DOES, MEASURED BEFORE THE DESIGN WAS FIXED (probe on the recorded
``e000010_seed0`` checkpoint, one row, three store states):

  state    gate      routable   value vector versus the live row
  live     0.998363  yes        --
  SHRED    0.000007  YES        max-abs 4.086 (the gate has closed; the row still routes)
  REVOKE   0.998363  no         max-abs 0.000 (the value is UNTOUCHED; the row leaves routing)

So the two lifecycle operations act on different channels, and neither is what a reader of the ledger
would guess: a REVOKED row keeps its payload bit-identical in the exported bank and is removed from
routing, while a SHREDDED row keeps its payload in the bank (``obj`` is unchanged), stays routable, and
has only its VALUE gated -- to 7e-06 of the payload plus the UNKNOWN direction. E-000028 recorded the
key half of this; the value half and the routability asymmetry are stated here.

That fixes the arms. A shredded row is the only lifecycle state that still competes for routing mass,
so it is the only one that can shape a bystander's answer, and the question is whether it shapes it
differently from a live row of the same mass.

ARMS, per pod, coefficient-matched to within ``MATCH_TOL``; one row is silenced through the reader's
own ``cell_mask``, holding the prompt, the bank, the weights and every other row fixed. Both compared
arms silence EXACTLY ONE row, so the row-count channel that dominates §31.41 and §31.45 (ADD2 at
0.965-1.000) cannot act here by construction:

  DEL    silence a routable row whose pod was SHREDded (mass + the gated UNKNOWN injection)
  LIVE   silence a live row of matched coefficient -- THE NULL the other branch's measurement lacks
  LIVE2  silence a second live row of matched coefficient -- the floor: two nulls against each other
  REV    silence a REVOKED row -- the zero control: it is not routable, so the forward must not move
         by a single bit, and if it does the mask or the routability flag is wrong
  TOP    silence the highest-coefficient row -- validity: a one-row ablation must be visible at all

MEASURED per arm: max-abs logit change, mean KL to the unablated forward, top-1 flip rate and the mean
routing coefficient; and, per pair, the PAIRED dominance over pods -- how often silencing one row moves
the forward more than silencing its coefficient-matched partner. E-000051's probe AUC was tried first
and removed: its own floor (two matched live rows) came back at 1.000, because a probe separates any
two distinct rows whatever their status. The control caught the instrument, and the instrument went.

Prior art, so no mechanism is claimed: dependency tracking, taint propagation and revoking a
capability together with everything derived from it are established (Redell 1974; CHERIvoke 2019;
Cornucopia, S&P 2020; Cornucopia Reloaded, ASPLOS 2024, which performs the check at the point of use);
PAMSPEC (``draft-infantado-agent-memory-architecture-00``) and Wu and Canedo (arXiv:2609.00243) own the
agent-memory instantiation. What is measured is one reader's mixture, with its null.

Run:  python -m so.experiments.e000057_dependency_closure [--seeds 0 1 2] [--n-pods 100]
      python -m so.experiments.e000057_dependency_closure --seeds 0 --n-pods 6 --quick \
          --results-dir /path/to/scratch                                     (a smoke run)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from so import ledger
from so.data import bank_from_store
from so.experiments import e000015_symlink_cells as E15
from so.experiments.e000051_residue_reader import Setting

ARMS = ("DEL", "LIVE", "LIVE2", "REV", "TOP")
# NO AUC HERE, AND THE FLOOR IS WHY. The first version scored E-000051's five-feature probe on the
# ablated logits, arm against arm. Its floor -- two coefficient-matched LIVE rows against each other --
# came back at 1.000 on the smoke: the probe separates any two DISTINCT rows, because silencing
# different rows produces different logit patterns whatever their status. That is the instrument
# leaking, caught by the control written to catch it, so the AUC is removed rather than reported. What
# survives is the paired magnitude comparison the question actually asks: per pod, the effect of
# silencing the deleted row against the effect of silencing a live row of the same routing mass.
PAIRED = (("del_vs_live", "DEL", "LIVE"), ("floor", "LIVE", "LIVE2"))
MATCH_TOL = 0.10
N_SHRED_PODS = 6
N_REVOKE_ROWS = 6


def _kl(p_logits: np.ndarray, q_logits: np.ndarray) -> np.ndarray:
    a = p_logits - p_logits.max(-1, keepdims=True)
    b = q_logits - q_logits.max(-1, keepdims=True)
    pa = np.exp(a); pa /= pa.sum(-1, keepdims=True)
    lb = b - np.log(np.exp(b).sum(-1, keepdims=True))
    return (pa * (np.log(np.clip(pa, 1e-30, None)) - lb)).sum(-1)


class Ablation(Setting):
    """E-000051's setting, plus a state with deleted and revoked rows, and one-row ablation."""

    def deleted_state(self):
        """One store: the targets of ``N_SHRED_PODS`` pods SHREDded, ``N_REVOKE_ROWS`` base rows REVOKED."""
        s = self._clone()
        shredded, revoked = [], []
        for p in self.pods[:N_SHRED_PODS]:
            s.shred(self.kids[p.target]); shredded.append(p.target)
        for k in self.bystander_base[-N_REVOKE_ROWS:]:
            s.revoke(self.kids[k]); revoked.append(k)
        return s, shredded, revoked

    def read_masked(self, bank, qs, mask: Optional[np.ndarray]) -> np.ndarray:
        return E15.predict(self.model, bank, self.world, list(qs), cell_mask=mask).logits

    def coefficients(self, bank, qs) -> np.ndarray:
        r = np.asarray(E15.predict(self.model, bank, self.world, list(qs)).routing)
        while r.ndim > 2:
            r = r.mean(0)
        return r.mean(0)[: bank.size]


def _pick_two(coef: np.ndarray, pool: np.ndarray, target: float, rng: np.random.Generator
              ) -> Optional[Tuple[int, int]]:
    """The two closest coefficient matches, then a coin flip for which is LIVE and which is LIVE2.

    Taking "closest" for LIVE and "second closest" for LIVE2 makes the floor asymmetric by
    construction -- LIVE is always the better match -- and the first smoke duly returned a floor
    dominance of 1.000. The assignment is randomised per unit so the floor is a real null.
    """
    tol = MATCH_TOL * max(abs(target), 1e-12)
    cand = [int(i) for i in pool if abs(coef[i] - target) <= tol]
    if len(cand) < 2:
        return None
    cand.sort(key=lambda i: abs(coef[i] - target))
    a, b = cand[0], cand[1]
    return (a, b) if rng.random() < 0.5 else (b, a)


def run_seed(seed: int, n_pods: int, threads: int, verbose: bool = True) -> Dict[str, Any]:
    t0 = time.time()
    S = Ablation("syn", seed, n_pods, threads)
    store, shredded, revoked = S.deleted_state()
    bank = bank_from_store(store)
    pos = {(int(bank.subject[i]), int(bank.relation[i])): i for i in range(bank.size)}
    del_rows = np.array([pos[k] for k in shredded if k in pos], dtype=int)
    rev_rows = np.array([pos[k] for k in revoked if k in pos], dtype=int)
    routable = bank.routable if bank.routable is not None else bank.active
    live_rows = np.array([i for i in range(bank.size)
                          if i not in set(del_rows.tolist()) and i not in set(rev_rows.tolist())
                          and bool(bank.active[i]) and bool(bank.marker_valid[i])], dtype=int)
    m: Dict[str, Any] = {"seed": seed, "n_pods": len(S.pods), "checkpoint_sha256": S.sha,
                         "n_del_rows": int(del_rows.size), "n_rev_rows": int(rev_rows.size),
                         "bank_size": int(bank.size),
                         "del_rows_routable": float(np.mean([bool(routable[i]) for i in del_rows])) if del_rows.size else float("nan"),
                         "rev_rows_routable": float(np.mean([bool(routable[i]) for i in rev_rows])) if rev_rows.size else float("nan")}
    mags: Dict[str, List[Tuple[float, float, float]]] = {a: [] for a in ARMS}
    coefs: Dict[str, List[float]] = {a: [] for a in ARMS}
    skipped = 0
    rng = np.random.default_rng(57_000 + seed)
    for p in S.pods[N_SHRED_PODS:]:                      # bystander pods: never deleted, never revoked
        qs, objs = S.queries(p, "ii")
        if not len(qs):
            continue
        coef = S.coefficients(bank, qs)
        base = S.read_masked(bank, qs, None)
        d = int(del_rows[np.argmax(coef[del_rows])]) if del_rows.size else None
        if d is None:
            continue
        target = float(coef[d])
        two = _pick_two(coef, live_rows, target, rng)
        if two is None:
            skipped += 1
            continue
        l1, l2 = two
        r = int(rev_rows[np.argmax(coef[rev_rows])]) if rev_rows.size else None
        top = int(np.argmax(coef))
        chosen = {"DEL": d, "LIVE": l1, "LIVE2": l2, "TOP": top}
        if r is not None:
            chosen["REV"] = r
        for arm, row in chosen.items():
            mask = np.ones(bank.size, dtype=bool); mask[row] = False
            lg = S.read_masked(bank, qs, mask)
            mags[arm].append((float(np.abs(lg - base).max()), float(_kl(base, lg).mean()),
                              float((lg.argmax(-1) != base.argmax(-1)).mean())))
            coefs[arm].append(float(coef[row]))
    for arm in ARMS:
        if mags[arm]:
            a = np.asarray(mags[arm])
            m[f"{arm}/maxabs"] = float(a[:, 0].mean()); m[f"{arm}/maxabs_max"] = float(a[:, 0].max())
            m[f"{arm}/kl"] = float(a[:, 1].mean()); m[f"{arm}/kl_max"] = float(a[:, 1].max())
            m[f"{arm}/flip"] = float(a[:, 2].mean()); m[f"{arm}/coef"] = float(np.mean(coefs[arm]))
    for name, a, b in PAIRED:
        if mags[a] and mags[b] and len(mags[a]) == len(mags[b]):
            x = np.asarray(mags[a])[:, 0]; y = np.asarray(mags[b])[:, 0]
            n = len(x)
            wins = float((x > y).mean())                       # paired dominance over pods
            m[f"{name}/dominance"] = wins
            m[f"{name}/median_ratio"] = float(np.median(x / np.clip(y, 1e-30, None)))
            # two-sided sign test against 0.5, normal approximation (n >= 20); reported, not scored
            m[f"{name}/sign_z"] = float((wins - 0.5) * 2.0 * np.sqrt(n)) if n else float("nan")
    m["n_units"] = int(len(mags["DEL"])); m["n_skipped_unmatched"] = int(skipped)
    m["seconds"] = time.time() - t0
    if verbose:
        print(f"  seed {seed}: units {m['n_units']} (skipped {skipped}) | DEL maxabs {m.get('DEL/maxabs', float('nan')):.4f} "
              f"vs LIVE {m.get('LIVE/maxabs', float('nan')):.4f} | REV maxabs {m.get('REV/maxabs', float('nan')):.2e} "
              f"| dominance del/live {m.get('del_vs_live/dominance', float('nan')):.3f} floor {m.get('floor/dominance', float('nan')):.3f} "
              f"| TOP maxabs {m.get('TOP/maxabs', float('nan')):.2f}  ({m['seconds']:.0f}s)", flush=True)
    return m


# Worst seed. Fixed before the run.
CRITERIA: Dict[str, Tuple[str, float]] = {
    "REV/maxabs_max": ("<=", 1e-6),        # zero control: a non-routable row cannot move the forward
    "TOP/maxabs": (">=", 1.0),             # validity: silencing the top row must move the forward
    "floor/dominance": ("<=", 0.60),       # the instrument's floor: two matched live rows, paired
    "del_vs_live/dominance": (">=", 0.75), # the other branch's effect, against its null
    "DEL/flip": (">=", 0.02),              # behavioural: does it reach answers, or only logits
}

DECISION_RULE = (
    "Worst seed. Every row is a PAIRED comparison over pods: the effect of silencing one row against "
    "the effect of silencing another of the same routing mass, on the same queries. VOID if the REV "
    "zero control moves the forward at all (the mask or the routability flag is wrong and nothing else "
    "is readable), if silencing the top-coefficient row does not move the forward (a one-row ablation "
    "is invisible, so no row below it can be read), or if the FLOOR dominance exceeds 0.60 (two "
    "coefficient-matched live rows already dominate one another and the pairing is not matched). With "
    "all three holding: NO-EFFECT if DEL vs LIVE dominance is at or below 0.60 -- a deleted row that is still routable shapes a bystander's "
    "answer exactly as a live row of the same routing mass does, the other branch's dependency-closure "
    "requirement has no measurable basis on this substrate, and the entry is a refutation of it. "
    "SUB-BEHAVIOURAL if DEL vs LIVE is at least 0.75 while the flip rate stays under 0.02 and the KL "
    "under 0.05 nats: the deleted row's contribution is distinguishable in the logits and does not "
    "reach answers, which is the same shape as §31.45's residue and is reported with the magnitudes "
    "beside it. BEHAVIOURAL if both fire: a deleted row changes what the model answers about other "
    "pods, at a measured rate, and the revocation unit for this reader is the routable set and not the "
    "queried pod. Anything in (0.60, 0.75) is inconclusive at this n. The magnitudes (max-abs, KL, "
    "flip rate, mean coefficient) are recorded for every arm whatever the reading, because the number "
    "the other branch published has no null beside it. Fixed before the run.")


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-pods", type=int, default=100)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args(argv)
    torch.set_num_threads(max(1, args.threads))
    if args.quick:
        os.environ["SO_RESULT_SUFFIX"] = "-smoke"
    per = [run_seed(s, args.n_pods, args.threads) for s in args.seeds]
    keys = sorted(k for k in per[0] if isinstance(per[0][k], (int, float)) and k != "seed")
    agg = ledger.aggregate(per, keys)
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})
    rows = [[a, f"{agg.get(f'{a}/coef', {}).get('mean', float('nan')):.5f}",
             f"{agg.get(f'{a}/maxabs', {}).get('max', float('nan')):.4f}",
             f"{agg.get(f'{a}/kl', {}).get('max', float('nan')):.4f}",
             f"{agg.get(f'{a}/flip', {}).get('max', float('nan')):.4f}"] for a in ARMS]
    tbl = ledger.table(["arm (one row silenced)", "mean routing coefficient", "max-abs logit change",
                        "mean KL to the unablated forward", "top-1 flip rate"], rows)
    auc = ledger.table(["paired comparison", "dominance (worst seed)", "median ratio", "sign-test z"],
                       [[n, f"{agg.get(f'{n}/dominance', {}).get('min', float('nan')):.3f}",
                         f"{agg.get(f'{n}/median_ratio', {}).get('mean', float('nan')):.3f}",
                         f"{agg.get(f'{n}/sign_z', {}).get('mean', float('nan')):.2f}"] for n, _, _ in PAIRED])
    record = {"experiment": "E-000057", "title": "what a deleted row still contributes to a bystander's answer",
              "evidence_level": "E4", "seeds": args.seeds, "n_pods": args.n_pods, "quick": args.quick,
              "trains_nothing": True, "decision_rule": DECISION_RULE, "per_seed": per, "aggregate": agg,
              "criteria": check,
              "control": "E-000051's setting, reader, probe and AUC unchanged; the null is a live row "
                         "matched on routing coefficient to within 10%"}
    md = [f"# E-000057 — {record['title']}", "",
          f"Synthetic E-000015 reader, seeds {args.seeds}, {args.n_pods} pods, trains nothing. One row is",
          "silenced through the reader's own cell mask; both compared arms silence exactly one row, so the",
          "row-count channel of §31.41 cannot act. Worst seed.", "", tbl, "", auc, "",
          "## The rule, fixed before the run", "", DECISION_RULE, "",
          "## Pre-registered criteria", "", ledger.criteria_table(check), ""]
    text = "\n".join(md)
    path = None
    if not args.quick:
        path = ledger.save("e000057_dependency_closure", record, text)
    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        name = "e000057_dependency_closure" + os.environ.get("SO_RESULT_SUFFIX", "")
        record.setdefault("environment", ledger.environment())
        with open(os.path.join(args.results_dir, name + ".json"), "w") as f:
            json.dump(ledger._to_jsonable(record), f, indent=1, sort_keys=True)
        with open(os.path.join(args.results_dir, name + ".md"), "w") as f:
            f.write(text.rstrip("\n") + "\n")
        path = path or os.path.join(args.results_dir, name + ".md")
    print("\n".join(md[1:])); print(f"\nwritten: {path}")
    return record


if __name__ == "__main__":
    main()
