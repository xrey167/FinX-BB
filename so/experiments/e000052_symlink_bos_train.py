"""E-000052, substrate only -- the E-000020 symlink adapter trained with a BOS on every prompt.

E-000050 (ledger 31.38) showed that every held-out-phrasing number in this repository was measured on
an adapter whose subject sat on GPT-2's position-0 attention sink, because the tokenizer prepends no
BOS, and that an adapter trained WITH a BOS reads unseen phrasings at 0.9712 (worst seed) against
0.7288. The symlink adapter of E-000020 was trained the same way. This script trains it again with
``SO_BOS=1`` -- E-000020's trainer unchanged, same steps, same budget -- and saves the checkpoints as
``e000020_gpt2_bos_seed{s}.pt``. It records NOTHING: the battery that reads these checkpoints is
E-000052 proper, pre-registered separately, so that no number here is produced before its criteria
exist.

Run:  SO_BOS=1 SO_CKPT_SUFFIX=_bos python -m so.experiments.e000052_symlink_bos_train [--seeds 0 1 2] [--steps 3000]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

import torch


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    args = ap.parse_args(argv)
    if os.environ.get("SO_BOS") != "1" or os.environ.get("SO_CKPT_SUFFIX", "") != "_bos":
        sys.exit("run with SO_BOS=1 SO_CKPT_SUFFIX=_bos: the BOS must be on for training and the checkpoint "
                 "name must not collide with the recorded E-000020 checkpoints")
    if args.threads:
        torch.set_num_threads(args.threads)
    from so.experiments import e000008_gpt2_adapter as E8
    from so.experiments import e000020_symlink_gpt2 as E20
    from so.llm_adapter import AdapterConfig
    for seed in args.seeds:
        t0 = time.time()
        gk = E8.GPT2Knowledge(AdapterConfig(status_gated=True, use_links=True, n_deref=E20.N_DEREF))
        print(f"=== seed {seed}: E-000020 symlink adapter, BOS on every prompt, {args.steps} steps ===", flush=True)
        out = E20.train_or_load(gk, seed, args.steps)
        print(f"  seed {seed} done: train_seconds {out['train_seconds']:.0f}, sha {out['checkpoint_sha256'][:12]} "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
