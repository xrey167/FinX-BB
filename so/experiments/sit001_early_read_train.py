"""SIT-001, substrate only -- the E-000020 symlink adapter with its writes moved EARLY.

WHY A SECOND ADAPTER EXISTS. WSC-001 and E-000063 (ledger 31.57) leave one question open, and it is
the only one whose answer is a design decision rather than a measurement of GPT-2. An accessibility
audit of an external memory has to be read at a site where two things are true at once:

    ARRIVAL         the pod's content has reached this state, so "the audit sees nothing" can mean
                    something -- the mediator, not the readout, decides this
    DISTINCTNESS    the lens at this site is not already the unembedding, so the audit is a different
                    instrument from reading the logits

On the recorded adapter, ``read_layers=(8, 10)``, those two conditions are anti-aligned and no site
satisfies both: arrival at the first read site is 0.032 of the second (WSC-001), and at the second the
J-lens vector's cosine to its own unembedding row is 1.000 (ledger 31.56 measures the whole curve:
0.310 at hidden state 1, 0.698 at 7, 0.783 at 9 and 10, 1.000 at 11). The audit window is empty
BECAUSE THE WRITE LANDS TWO BLOCKS FROM THE OUTPUT, which is a property of where this adapter was
told to read, not of the lens and not of the pod.

So this script trains the identical adapter -- same trainer, same steps, same budget, same BOS -- with
``read_layers=(4, 6)`` instead. Four blocks of depth remain after the second write, so a site can be
past the write and still upstream of the point where the lens collapses onto ``W_U``. The choice was
fixed before any number was produced from these checkpoints and is argued in
``docs/novelty/sit001-preregister.md``: (4, 6) rather than (2, 4) because the routing query is read
from the residual at the read layer, and a query read before the subject is resolved is the arm's
most likely way to die -- capability is the pre-registered validity bar, and the point of the arm is
lost if it is bought by breaking the memory.

Like E-000052's trainer, THIS SCRIPT RECORDS NOTHING. The battery that reads these checkpoints is
``so/experiments/sit001_audit_window.py``, whose criteria exist in the pre-registration before any
checkpoint here is read, so that no number is produced before the bar it is read against.

Run:  SO_BOS=1 SO_CKPT_SUFFIX=_bos_early python -m so.experiments.sit001_early_read_train [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

import torch

READ_LAYERS = (4, 6)          # the one design change under test; (8, 10) is the recorded adapter
CKPT_SUFFIX_REQUIRED = "_bos_early"


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    args = ap.parse_args(argv)
    if os.environ.get("SO_BOS") != "1" or os.environ.get("SO_CKPT_SUFFIX", "") != CKPT_SUFFIX_REQUIRED:
        sys.exit(f"run with SO_BOS=1 SO_CKPT_SUFFIX={CKPT_SUFFIX_REQUIRED}: the BOS must match the adapter this "
                 f"arm is compared against (E-000052's checkpoints), and the suffix must not collide with them")
    if args.threads:
        torch.set_num_threads(args.threads)
    from so.experiments import e000008_gpt2_adapter as E8
    from so.experiments import e000020_symlink_gpt2 as E20
    from so.llm_adapter import AdapterConfig
    for seed in args.seeds:
        t0 = time.time()
        gk = E8.GPT2Knowledge(AdapterConfig(status_gated=True, use_links=True, n_deref=E20.N_DEREF,
                                            read_layers=READ_LAYERS))
        print(f"=== seed {seed}: symlink adapter, read_layers={READ_LAYERS}, BOS, {args.steps} steps ===",
              flush=True)
        out = E20.train_or_load(gk, seed, args.steps)
        print(f"  seed {seed} done: train_seconds {out['train_seconds']:.0f}, "
              f"sha {out['checkpoint_sha256'][:12]} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
