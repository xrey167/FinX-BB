"""Experiment E-000027 — the knowledge layer on a model that does not tie its embeddings.

Every result in this programme that involves a pretrained core was measured on GPT-2, and GPT-2 ties
its input and output embedding matrices.  The layer builds its payload from a row of that matrix and
adds it to the residual stream, on the reasoning that this raises the object's logit at the unchanged
LM head — which is only sound *because* the head scores against the same matrix.  Llama, Qwen, OLMo
and Pythia all default to untied.  If the mechanism only works on tied models, then every claim here
is a claim about GPT-2 and the E6 scale-out would fail for a reason that has nothing to do with the
architecture.

Pythia-160m settles it on a CPU: same shape as GPT-2 small (768 wide, 12 blocks), a different
tokenizer, and ``tie_word_embeddings: false``.  Two arms, identical in every other respect:

  output   the payload is built from the OUTPUT embedding — the rows the head actually scores
  input    the payload is built from the INPUT embedding — what the code did before, and what a
           naive port would do

On GPT-2 the two arms are the same computation.  Here they are not, and the difference between them
is the whole question.  A third possibility is worth naming in advance: ``v_proj`` is a learned
linear map, so it *could* in principle learn to carry input-embedding rows to output-embedding rows.
If the input arm also reads well, the distinction is real in the code but not load-bearing in
practice, and that is the finding.

Run:  python -m so.experiments.e000027_untied_model [--arm output|input] [--seeds 0 1]
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import numpy as np
import torch

from so import ledger
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, CKPT_SUFFIX, guard_recorded_checkpoint, _sha256
from so.llm_adapter import AdapterConfig

MODEL = "EleutherAI/pythia-160m"
ARMS = ("output", "input")
READ_LAYERS = (8, 10)          # the same depth as the GPT-2 runs: 12 blocks, read at 8 and 10


def train_or_load(gk: E8.GPT2Knowledge, arm: str, seed: int, steps: int, force: bool = False) -> Dict[str, Any]:
    path = CHECKPOINTS / f"e000027_{arm}{CKPT_SUFFIX}_seed{seed}.pt"
    if path.exists() and not force:
        ck = torch.load(path, weights_only=False)
        gk.model.load_state_dict(ck["adapter"], strict=False)
        gk.model.eval()
        return {"centre": np.asarray(ck["centre"]), "history": ck["history"],
                "train_seconds": ck["train_seconds"], "loaded": True, "checkpoint_sha256": _sha256(path)}
    out = E8.train_adapter(gk, seed, steps)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    guard_recorded_checkpoint(path)
    torch.save({"adapter": E8.adapter_state(gk.model), "centre": out["centre"], "history": out["history"],
                "train_seconds": out["train_seconds"], "adapter_config": gk.model.cfg.to_dict(),
                "model_name": MODEL, "arm": arm}, path)
    out["loaded"] = False
    out["checkpoint_sha256"] = _sha256(path)
    return out


KEYS = ["prior_direct_acc", "bank_masked_direct_acc", "bank_masked_full_vocab_top1_equals_prior",
        "direct", "direct_full_vocab_top1", "paraphrase", "provenance_direct", "hop2",
        "broken1_unknown", "broken2_unknown", "lifecycle_all", "update", "revoke", "shred",
        "locality", "probe_calibration_top1"]

CRITERIA = {
    # the mechanism ports: the layer reads through a core that does not tie its matrices
    "direct": (">=", 0.70),
    "paraphrase": (">=", 0.60),
    # and the copy bound still holds: mask every cell and the adapter must add nothing
    "bank_masked_direct_acc": ("<=", 0.01),
}
CONTROL_CRITERIA = {
    # the control arm is expected to FAIL to read; if it does not, the distinction is not load-bearing
    "direct": ("<=", 0.20),
}


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, default="output")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    per_seed: List[Dict[str, Any]] = []
    ties = None
    for seed in args.seeds:
        cfg = AdapterConfig(read_layers=READ_LAYERS, payload_from=args.arm)
        gk = E8.GPT2Knowledge(cfg, model_name=args.model)
        ties = bool(gk.model.ties_embeddings)
        if ties:
            print(f"note: {args.model} TIES its embeddings, so both arms are the same computation here",
                  flush=True)
        print(f"=== {args.model}, arm {args.arm}, seed {seed}: training ===", flush=True)
        out = train_or_load(gk, args.arm, seed, args.steps, args.force)
        print(f"=== {args.model}, arm {args.arm}, seed {seed}: evaluating ===", flush=True)
        m = E8.evaluate(gk, 800 + seed, out["centre"])
        m["train_seconds"] = out["train_seconds"]
        m["checkpoint_sha256"] = out["checkpoint_sha256"]
        per_seed.append(m)
        print(f"  direct {m['direct']:.4f}  paraphrase {m['paraphrase']:.4f}  "
              f"masked {m['bank_masked_direct_acc']:.4f}  prior {m['prior_direct_acc']:.4f}  "
              f"lifecycle {m['lifecycle_all']:.4f}", flush=True)

    keys = [k for k in KEYS if all(k in s for s in per_seed)]
    agg = ledger.aggregate(per_seed, keys)
    crit = CRITERIA if args.arm == "output" else CONTROL_CRITERIA
    check = ledger.check_criteria(agg, {k: v for k, v in crit.items() if k in agg})
    sizes = {"direct": 1000, "paraphrase": 1000, "bank_masked_direct_acc": 1000,
             "prior_direct_acc": 1000, "provenance_direct": 1000}
    rows = ledger.ci_rows(per_seed, keys, sizes,
                          lower_is_better=["bank_masked_direct_acc", "broken1_unknown", "broken2_unknown"])

    record = {"experiment": "E-000027", "arm": args.arm, "model": args.model,
              "title": f"the knowledge layer on {args.model} ({'tied' if ties else 'untied'} embeddings), "
                       f"payload from the {args.arm} embedding",
              "ties_embeddings": ties, "seeds": args.seeds, "steps": args.steps,
              "read_layers": list(READ_LAYERS), "eval": E8.EVAL,
              "per_seed": per_seed, "aggregate": agg, "criteria": check}
    md = [f"# E-000027 ({args.arm}) — {record['title']}", "",
          f"Model `{args.model}`, `tie_word_embeddings` = {ties}. Seeds {args.seeds}, {args.steps} steps,",
          f"read at blocks {READ_LAYERS}. Everything else is E-000008's protocol unchanged: worlds are",
          "re-sampled every step, the core is frozen, and only the adapter is trained.", "",
          "## Pre-registered criteria", "", ledger.criteria_table(check), "",
          ("The `output` arm is the corrected mechanism. " if args.arm == "output" else
           "The `input` arm is the CONTROL: it builds the payload from the matrix the LM head does not "
           "score against, which is what the code did before this was noticed and what a naive port "
           "would do. It is expected to fail; if it reads, the distinction is not load-bearing."), "",
          "## All measures", "", ledger.table(ledger.CI_HEADERS, rows), ""]
    path = ledger.save(f"e000027_untied_{args.arm}", record, "\n".join(md))
    print(f"\nwritten: {path}")
    print(ledger.criteria_table(check))
    return record


if __name__ == "__main__":
    main()
