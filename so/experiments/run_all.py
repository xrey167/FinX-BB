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
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    for module, full_args, quick_args in CHAIN:
        print(f"\n################ {module} ################", flush=True)
        importlib.import_module(module).main(quick_args if args.quick else full_args)
    print(f"\nchain finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
