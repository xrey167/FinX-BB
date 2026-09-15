"""Run the complete SO experiment chain in order.

    python -m so.experiments.run_all            # everything, default seeds/steps
    python -m so.experiments.run_all --quick    # reduced seeds/steps for a smoke run

Results land in so/results/ (JSON + Markdown per experiment); trained models
are cached in so/results/checkpoints/ (not committed).
"""

from __future__ import annotations

import argparse
import importlib
import time

CHAIN = [
    ("so.experiments.e000001a_reference", [], ["--seeds", "0", "1"]),
    ("so.experiments.e000001b_mini_transformer", [], ["--seeds", "0", "1", "--steps", "1500"]),
    ("so.experiments.e000002_memorization_control", [], ["--seeds", "0", "--steps", "800"]),
    ("so.experiments.e000003_retention_generalization", [], ["--seeds", "0", "1"]),
    ("so.experiments.e000004_reconstruction_attacks", [], ["--seeds", "0", "1"]),
    ("so.experiments.e000005_causal_interventions", [], ["--seeds", "0", "1"]),
    ("so.experiments.e000006_ablations", [], ["--seeds", "0", "--steps", "800"]),
    ("so.experiments.e000007_biomarker", [], ["--seeds", "0", "1"]),
    ("so.experiments.e000009_verification_gate", [], ["--seeds", "0", "1", "--steps", "800"]),
    ("so.experiments.e000009_verification_gate",
     ["--gate-weight", "5.0", "--balanced", "--name", "e000010_balanced_gate", "--experiment", "E-000010"],
     ["--seeds", "0", "1", "--steps", "800", "--gate-weight", "5.0", "--balanced", "--name", "e000010_balanced_gate", "--experiment", "E-000010"]),
    ("so.experiments.e000014_bank_10k", [], ["--seeds", "0", "--steps", "800"]),
    ("so.experiments.e000015_symlink_cells", ["--skip-deref2"], ["--seeds", "0", "--steps", "800", "--skip-deref2"]),
    ("so.experiments.e000016_alias_chains", [], ["--seeds", "0", "--steps", "800"]),
    ("so.experiments.e000019_fresh_seed_chance", ["--seeds", "5", "6", "7"], ["--seeds", "5", "--steps", "800"]),
    ("so.experiments.e000021_gate_error_rates", [], ["--n", "20000"]),
]
# The frozen-GPT-2 experiments are NOT in this chain: they need `transformers`, download GPT-2 once,
# and cost between 20 and 90 minutes per seed on a CPU. Run them with `make gpt2`, or one at a time.


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced seeds and steps; records get a -quick suffix")
    ap.add_argument("--only", nargs="*", default=None, help="run only these modules (substring match)")
    args = ap.parse_args()
    if args.quick:
        import os
        # reduced runs get their own result AND checkpoint namespace, so a smoke run can never load
        # a model trained at the full budget, and can never overwrite one
        os.environ["SO_RESULT_SUFFIX"] = "-quick"
        os.environ.setdefault("SO_CKPT_SUFFIX", "-quick")
    t0 = time.time()
    chain = [c for c in CHAIN if not args.only or any(o in c[0] for o in args.only)]
    for i, (module, full_args, quick_args) in enumerate(chain, 1):
        print(f"\n################ [{i}/{len(chain)}] {module} ################", flush=True)
        t = time.time()
        importlib.import_module(module).main(quick_args if args.quick else full_args)
        print(f"################ {module} took {time.time() - t:.0f}s", flush=True)
    print(f"\nchain finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
