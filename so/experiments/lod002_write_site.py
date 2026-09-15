"""LOD-002 -- does the audit's blind region follow the model's depth, or the memory's write site?

THE QUESTION, AND WHY IT DECIDES SOMETHING. Gurnee et al. (transformer-circuits.pub/2026/workspace/)
state, of the lens this programme's audit uses, that "in roughly the first third of the model, the
readouts are noisy and largely uninterpretable", and they state the exact ambiguity WSC-001's mediator
was built to break: an early absence "could indicate that either (1) the J-lens is degenerate at these
depths and fails to resolve content that is in fact present, or (2) the early-layer residual stream
genuinely carries no linearly accessible and causally relevant verbalizable content." That is a rule
about DEPTH IN THE MODEL, and it is published.

`docs/novelty/audit-siting-correction.md` withdraws this programme's claim to that rule and keeps one
different sentence: for an EXTERNAL memory the binding constraint is not depth but the site of the
WRITE, which is a configuration parameter (`AdapterConfig.read_layers`) and can be moved without
touching the model. That sentence is currently an inference from one adapter. This experiment is the
measurement that decides it, and it can come out the other way.

THE DESIGN. The identical frozen GPT-2 small, the identical recipe, the identical world, the identical
ladder -- and one field changed: the adapter is trained to read at blocks (5, 7) instead of (8, 10).
Block 8 is then DOWNSTREAM of both read layers instead of being the first of them. Depth is held
exactly fixed; only where the memory is written moves.

  If the blind region follows the WRITE:  block 8 is blind on the (8,10) adapter and SIGHTED on the
                                          (5,7) adapter. Same block, same model, opposite verdicts.
  If the blind region follows DEPTH:      block 8 is blind on both, the depth explanation survives,
                                          and the surviving sentence of the correction is WITHDRAWN.

Neither outcome is guaranteed and the second is a real possibility: a shallower reader may simply fail
to place a readable memory at all, which is why the validity row VOIDs the experiment rather than
letting a failure to read masquerade as a blind audit.

Run:  SO_BOS=1 SO_CKPT_SUFFIX=_bos_L57 python -m so.experiments.lod002_write_site --seeds 0 --steps 3000
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments import lod001_detection_limit as LOD
from so.llm_adapter import AdapterConfig

# The moved read layers, and the capture set that must move with them. Block 11 is not a capture site:
# its output IS hidden_states[12] and the lens target hidden_states[-2] is upstream of it, so the
# Jacobian is undefined there (wsc001_workspace_share.py, the comment above SITES).
READ_LAYERS: Tuple[int, ...] = (5, 7)
SITES: Tuple[int, ...] = (5, 6, 7, 8, 9, 10)


def load_or_train(seed: int, steps: int, verbose: bool = True) -> Tuple[E8.GPT2Knowledge, np.ndarray, str]:
    """The E-000020 symlink adapter at the moved read layers, from a checkpoint if one exists.

    ``SO_CKPT_SUFFIX`` must name this arm so the checkpoint cannot collide with the recorded (8, 10)
    ones; ``E20.train_or_load`` reads that env var and ``guard_recorded_checkpoint`` refuses to
    overwrite a recorded name.
    """
    suffix = os.environ.get("SO_CKPT_SUFFIX", "")
    if "_L" not in suffix:
        raise SystemExit("set SO_CKPT_SUFFIX to a name carrying the read layers, e.g. _bos_L57, so this "
                         "arm's checkpoint cannot be confused with the recorded (8, 10) ones")
    cfg = AdapterConfig(status_gated=True, use_links=True, n_deref=E20.N_DEREF, read_layers=READ_LAYERS)
    gk = E8.GPT2Knowledge(cfg)
    t0 = time.time()
    out = E20.train_or_load(gk, seed, steps)
    if verbose:
        print(f"  read_layers {READ_LAYERS}, seed {seed}: {time.time() - t0:.0f}s, "
              f"sha {out['checkpoint_sha256'][:12]}", flush=True)
    return gk, np.asarray(out["centre"]), out["checkpoint_sha256"]


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-groups", type=int, default=16)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--modes", nargs="*", default=["alias", "direct"])
    ap.add_argument("--results-dir", default="so/results/lod002")
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    if os.environ.get("SO_BOS") != "1":
        raise SystemExit("run with SO_BOS=1: the comparison arm (LOD-001) is the BOS-trained adapter, "
                         "and a prefix difference would confound the read-layer difference")
    t0 = time.time()
    per_seed = []
    for s in a.seeds:
        gk, centre, sha = load_or_train(s, a.steps)
        print(f"=== LOD-002 seed {s}, read_layers {READ_LAYERS}, sites {SITES} ===", flush=True)
        r = LOD.run_seed(gk, centre, s, a.n_groups, a.modes, sites=SITES)
        r["checkpoint_sha256"] = sha
        per_seed.append(r)
    rec = {"experiment": "LOD-002", "seeds": list(a.seeds), "n_groups": a.n_groups,
           "modes": list(a.modes), "delta": LOD.DELTA, "read_layers": list(READ_LAYERS),
           "sites": list(SITES), "steps": a.steps,
           "bos": os.environ.get("SO_BOS", "") == "1",
           "alpha_ladder": list(LOD.ALPHA_LADDER), "chord_ladder": list(LOD.CHORD_LADDER),
           "comparison_arm": "so/results/lod001/lod001_detection_limit.json (read_layers (8, 10))",
           "per_seed": per_seed, "seconds": time.time() - t0}
    p = Path(a.results_dir); p.mkdir(parents=True, exist_ok=True)
    (p / "lod002_write_site.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    (p / "lod002_write_site.md").write_text(LOD.summarise(rec), encoding="utf-8")
    print(f"wrote {p / 'lod002_write_site.json'} in {rec['seconds']:.0f}s", flush=True)
    return rec


if __name__ == "__main__":
    main()
