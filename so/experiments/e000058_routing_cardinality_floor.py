"""E-000058 — is the cardinality floor a property of DENSE routing?

WHERE THIS COMES FROM.  `docs/so-claim-residue-floors.md` records a candidate claim: an off-target
deletion-residue detector over this store separates conditions by the store's structural footprint
rather than by residual knowledge.  Its two load-bearing numbers are E-000051's:

    cascade vs never   AUC(ii) 0.869   max KL 0.000   top-1 agree 1.000   (a COMPLETE deletion,
                                                                           bit-identical outputs)
    add2   vs perm     AUC(ii) 0.977   worst seed 1.0000                  (NO deletion at all)

and E-000051's own pre-registered rule already names the mechanism it suspects:

    "If add2 AUC > 0.60 the ROW-COUNT reading is recorded: the reader's off-pod outputs carry the
     number of active rows, a property of dense routing and not of history."

That sentence is an attribution, not a measurement.  It is also withdrawal condition 2 of the claim
document: *the add2 channel turns out to be an artefact of dense routing that vanishes under a sparse
or top-k reader -- that would localise it rather than kill the protocol, and must be measured.*

THE QUESTION.  `KnowledgeLayer.read` (`so/model.py:78`) routes with a softmax over EVERY cell:

    p = softmax(masked_scores)            # (B, C) over all C cells
    read = p @ v                          # every live row contributes

so the number of live rows enters the normaliser of every read, including reads that have nothing to
do with the changed pod.  If that is the channel, restricting the softmax to the top-k scoring cells
should remove it: under top-k the cells that do not win the competition cannot contribute, and adding
two unrelated rows cannot move an off-target read unless they enter the top k.

ARMS.  Routing is patched at the layer, nothing is retrained, and E-000051's own `run_reader_seed`
computes every arm and AUC, so the dense arm is a reproduction of the record rather than a
reimplementation of it.

    dense    the recorded reader, softmax over all C cells        (must reproduce E-000051)
    top16 / top4 / top1   the same weights, the same store, the same queries, the same statistic,
                          with the routing softmax restricted to the k highest-scoring allowed cells

WHAT EACH OUTCOME MEANS -- both are informative, which is why it is worth running:

  * add2 AUC collapses to the null band under top-k while the reader still reads
    (`present/auc_i >= 0.95`):  the floor is LOCALISED to dense attention over the whole store.  The
    protocol in the claim document then becomes precise about which architectures need the
    cardinality null -- memory-layer / KV-memory / dense-attention external memories -- and top-k
    retrieval systems are exempt, because their store size never reaches the forward pass.
  * add2 AUC survives under top-k:  the floor is NOT about dense attention, the claim document's
    mechanism sentence is wrong and must be rewritten, and the protocol generalises further than it
    currently claims -- the stronger result.
  * the reader stops reading under top-k (`present/auc_i < 0.95`):  VOID for that k.  A reader that
    cannot read is not a reader, and its collapsed add2 says nothing.  This is the arm that stops a
    top-1 null being mistaken for a repair.

PRE-REGISTERED BARS.  Worst seed.  Every bar can come out either way; what a failing value looks
like is stated beside it.

    dense/present/auc_i        >= 0.95   the dense reader reads          (fail: < 0.95 -> VOID, the
                                                                         checkpoint is not the record's)
    dense/add2/auc_ii          >= 0.90   dense reproduces E-000051's 0.960-0.969 (fail: < 0.90 ->
                                                                         VOID, this is not that setting)
    topk/present/auc_i         >= 0.95   the top-k reader still reads    (fail: VOID for that k)
    topk/add2/auc_ii           <= 0.60   the floor is a dense-routing property
                                         (fail: > 0.60 with the reader still reading -> the mechanism
                                          sentence is WRONG and the claim doc must be corrected)
    topk/cascade_soft/auc_ii   reported, not barred: the provenance channel is a marker property and
                                         E-000053 closes it by content-derived markers, so top-k is
                                         not expected to move it.  A large move is a finding and is
                                         recorded rather than scored.

NOT CLAIMED.  Nothing here claims novelty for top-k attention, sparse routing, sparse mixture-of-
experts, attention sparsification, retrieval top-k, or the observation that a softmax normaliser
depends on its support.  The measured object is narrower: whether the specific off-target residue
channel E-000051 recorded is carried by the dense normaliser, and therefore which memory
architectures a deletion-residue statistic must be calibrated against.

    python -m so.experiments.e000058_routing_cardinality_floor --seeds 0 --n-pods 100
    python -m so.experiments.e000058_routing_cardinality_floor --seeds 0 --n-pods 6 --quick   (smoke)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from so import ledger
from so.experiments import e000051_residue_reader as E51
from so.model import DerefBlock, HopBlock

MODES = ("dense", "top16", "top4", "top1")

BARS = {
    "dense/present/auc_i": (">=", 0.95),
    "dense/add2/auc_ii": (">=", 0.90),
    "top16/present/auc_i": (">=", 0.95),
    "top16/add2/auc_ii": ("<=", 0.60),
    "top4/present/auc_i": (">=", 0.95),
    "top4/add2/auc_ii": ("<=", 0.60),
    "top1/present/auc_i": (">=", 0.95),
    "top1/add2/auc_ii": ("<=", 0.60),
}

REPORTED = ("cascade_soft/auc_ii", "blank_matched/auc_ii", "dangle_matched/auc_ii", "perm/auc_ii")


def _sparsify(scores: torch.Tensor, k: int, keep_last: bool = False) -> torch.Tensor:
    """Keep only the k highest-scoring columns per row; -inf the rest, so they leave the normaliser.

    `keep_last` protects the dereference passthrough column, which is architectural rather than a
    store row ("the passthrough itself is architectural", `so/model.py` DerefBlock): the cardinality
    question is about the number of LIVE ROWS, so the null column competes in neither direction.
    """
    body = scores[:, :-1] if keep_last else scores
    kk = min(k, body.shape[-1])
    if kk < body.shape[-1]:
        cut = body.topk(kk, dim=-1).values[:, -1:]
        body = body.masked_fill(body < cut, float("-inf"))
    return torch.cat([body, scores[:, -1:]], dim=-1) if keep_last else body


@contextlib.contextmanager
def topk_routing(k: Optional[int]):
    """Restrict both routing softmaxes to the k highest-scoring ALLOWED cells. Nothing is retrained.

    Patched immediately before each softmax, so `allowed` masking, the forward/reverse selection and
    the value mixtures are untouched; only the support of the normaliser changes. Both read sites are
    patched -- `HopBlock.read` (the question's read) and `DerefBlock.forward` (the pointer's read) --
    because the claim under test names the DEREFERENCE pass-through bias specifically.
    k=None restores the recorded dense behaviour.
    """
    if k is None:
        yield
        return
    hop_original, deref_original = HopBlock.read, DerefBlock.forward

    def hop_read(self, h, rel, hop_emb, k_f, v_f, k_r, v_r, is_fwd, allowed):
        q = self.q(self.ln_q(h + rel + hop_emb))
        scores = torch.where(is_fwd[:, None], q @ k_f.t(), q @ k_r.t())
        scores = scores * (self.scale / k_f.shape[-1] ** 0.5)
        scores = scores.masked_fill(~allowed[None], float("-inf"))
        p = torch.softmax(_sparsify(scores, k), dim=-1)
        return torch.where(is_fwd[:, None], p @ v_f, p @ v_r), p

    def deref_forward(self, read, state, k_f, v_f, allowed):
        q = self.q(self.ln(read if (state is None or not self.use_state) else read + state))
        scores = (q @ k_f.t()) * (self.scale / k_f.shape[-1] ** 0.5)
        scores = scores.masked_fill(~allowed[None], float("-inf"))
        p = torch.softmax(_sparsify(scores, k, keep_last=self.use_passthrough), dim=-1)
        if self.use_passthrough:
            out = p[:, :-1] @ v_f[:-1] + p[:, -1:] * read
        else:
            out = p @ v_f
        return out, p

    HopBlock.read, DerefBlock.forward = hop_read, deref_forward
    try:
        yield
    finally:
        HopBlock.read, DerefBlock.forward = hop_original, deref_original


def run_seed(seed: int, n_pods: int, threads: int, n_hardgate: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {"seed": seed}
    for mode in MODES:
        k = None if mode == "dense" else int(mode[3:])
        t0 = time.time()
        with topk_routing(k):
            m = E51.run_reader_seed("syn", seed, n_pods, threads, n_hardgate, verbose=False)
        for key, val in m.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                out[f"{mode}/{key}"] = float(val)
        out[f"{mode}/seconds"] = time.time() - t0
        print(f"  {mode:>6}: present/auc_i={out.get(f'{mode}/present/auc_i', float('nan')):.4f} "
              f"add2/auc_ii={out.get(f'{mode}/add2/auc_ii', float('nan')):.4f} "
              f"cascade/auc_ii={out.get(f'{mode}/cascade_soft/auc_ii', float('nan')):.4f} "
              f"({out[f'{mode}/seconds']:.0f}s)", flush=True)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-pods", type=int, default=100)
    ap.add_argument("--n-hardgate", type=int, default=20)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--quick", action="store_true", help="reduced sizes: not a record")
    ap.add_argument("--results-dir", default="so/results")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.quick:
        args.n_hardgate = min(args.n_hardgate, 3)

    per_seed = []
    for seed in args.seeds:
        print(f"seed {seed}:", flush=True)
        per_seed.append(run_seed(seed, args.n_pods, args.threads, args.n_hardgate))

    keys = sorted(set.intersection(*[set(r) for r in per_seed]))
    agg = ledger.aggregate([{k: v for k, v in r.items() if k != "seed"} for r in per_seed], keys)
    check = ledger.check_criteria(agg, {k: v for k, v in BARS.items() if k in agg})

    voids = [m for m in MODES if not check["criteria"].get(f"{m}/present/auc_i", {}).get("pass", False)]
    localised = all(
        check["criteria"].get(f"{m}/add2/auc_ii", {}).get("pass", False)
        for m in MODES if m != "dense" and m not in voids
    ) and any(m not in voids for m in MODES if m != "dense")

    record = {
        "experiment": "E-000058",
        "question": "Is the E-000051 cardinality floor a property of dense routing?",
        "reader": "syn",
        "seeds": args.seeds,
        "n_pods": args.n_pods,
        "quick": args.quick,
        "per_seed": per_seed,
        "aggregate": agg,
        **check,
        "void_modes": voids,
        "floor_is_dense_routing": bool(localised),
        "reported_not_scored": {f"{m}/{r}": agg[f"{m}/{r}"] for m in MODES for r in REPORTED
                                if f"{m}/{r}" in agg},
        "interpretation_limit": (
            "Synthetic E-000015 reader only; the GPT-2 half of E-000051 is not re-run here. Routing is "
            "patched at the layer and nothing is retrained, so a top-k arm measures a reader that was "
            "TRAINED dense and is READ sparse -- it is a probe of where the channel lives, not a "
            "proposal to train sparse readers. A VOID arm says only that this reader stops reading at "
            "that k."
        ),
        "not_claimed": ["top-k attention", "sparse routing", "sparse mixture-of-experts",
                        "retrieval top-k", "that a softmax normaliser depends on its support"],
    }
    suffix = "-quick" if args.quick else ""
    out = Path(args.results_dir) / f"e000058_routing_cardinality_floor{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, default=float))
    print(json.dumps({"checks": check["criteria"], "claim_supported": check["claim_supported"],
                      "void_modes": voids, "floor_is_dense_routing": localised}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
