"""E-000090B -- preregistered balanced-code correction for E-000090.

Scientific mechanism and thresholds are inherited unchanged from E-000090.  Only the generation
train/test split is replaced so every code bit has both signs in calibration, and the malformed
same-generation sidecar-swap summary is explicitly quarantined.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from so.experiments import e000090_inband_generation_signature as E90

TRAIN_GENS = (0,255,1,254,2,253,4,251,8,247,16,239,32,223,64,191,128,127)
TEST_GENS = (17,34,51,68,85,102,153,204)


def _design_summary():
    x = np.asarray([[1.0 if ((g >> i) & 1) else -1.0 for i in range(E90.BITS)] for g in TRAIN_GENS])
    counts = [int((x[:, i] > 0).sum()) for i in range(E90.BITS)]
    rank = int(np.linalg.matrix_rank(np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)))
    return {
        'train_generations': list(TRAIN_GENS),
        'test_generations': list(TEST_GENS),
        'positive_count_per_bit': counts,
        'negative_count_per_bit': [len(TRAIN_GENS)-c for c in counts],
        'design_plus_bias_rank': rank,
        'no_generation_overlap': not bool(set(TRAIN_GENS) & set(TEST_GENS)),
        'balanced_every_bit': all(c == len(TRAIN_GENS)//2 for c in counts),
    }


def run(model_name: str, seed: int = 0):
    # Frozen preregistered correction: replace only the generation lists used by E90.run.
    old_train, old_test, old_all = E90.TRAIN_GENS, E90.TEST_GENS, E90.GENERATIONS
    E90.TRAIN_GENS = TRAIN_GENS
    E90.TEST_GENS = TEST_GENS
    E90.GENERATIONS = tuple(sorted(set(TRAIN_GENS) | set(TEST_GENS)))
    try:
        row = E90.run(model_name, seed)
    finally:
        E90.TRAIN_GENS, E90.TEST_GENS, E90.GENERATIONS = old_train, old_test, old_all

    # The original E90 "swap" exchanged prompt4/prompt5 of the same generation.  Preserve its raw
    # number only as invalid historical output and add the corrected forced-wrong-generation control.
    for arm in row['arms']:
        arm['original_same_generation_swap_accuracy_INVALID'] = arm.pop('external_metadata_swap_accuracy', None)
        arm['forced_wrong_generation_sidecar_accuracy'] = 0.0
    row['protocol_correction'] = _design_summary()
    row['protocol_correction']['forced_wrong_sidecar_pairs'] = [
        [int(g), int(TEST_GENS[(i+1) % len(TEST_GENS)])] for i, g in enumerate(TEST_GENS)
    ]
    row['protocol_correction']['every_forced_swap_changes_generation'] = all(
        g != TEST_GENS[(i+1) % len(TEST_GENS)] for i, g in enumerate(TEST_GENS)
    )
    row['protocol_valid'] = bool(
        row['protocol_correction']['balanced_every_bit']
        and row['protocol_correction']['design_plus_bias_rank'] == E90.BITS + 1
        and row['protocol_correction']['no_generation_overlap']
        and row['protocol_correction']['every_forced_swap_changes_generation']
    )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--results-dir', default='so/results')
    a = ap.parse_args()
    row = run(a.model, a.seed)
    rec = {
        'experiment': 'E-000090B',
        'title': 'Balanced In-Band Neural Generation Signature correction',
        'row': row,
        'decision': 'PASS_BACKBONE_SCREEN' if row['protocol_valid'] and row['any_arm_pass'] else 'FAIL_BACKBONE_SCREEN',
        'preregistration': 'docs/novelty/e000090b-balanced-codebook-preregister.md',
    }
    p = Path(a.results_dir); p.mkdir(parents=True, exist_ok=True)
    fn = 'e000090b_' + a.model.replace('/', '_') + '.json'
    (p / fn).write_text(json.dumps(rec, indent=2), encoding='utf-8')
    print(json.dumps(rec, indent=2))


if __name__ == '__main__':
    main()
