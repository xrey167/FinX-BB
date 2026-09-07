"""E-000096 -- nonlinear state-dependent revision receipt capacity screen.

Flexible Gaussian-RBF kernel receipt fitted only on calibration sessions.
The receipt itself receives zero novelty credit; this is an exact held-out transport kill screen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from so.llm_adapter import transformer_blocks
from so.experiments.e000095_cross_context_pod_revision_receipt import (
    _forward,
    _lm_head_logits,
    _metrics,
    _payload,
)

SIGMA_MULTS = (0.25, 0.5, 1.0, 2.0, 4.0)
RIDGES = (0.0, 1e-10, 1e-8, 1e-6, 1e-4)


def _sqdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.double()
    b = b.double()
    aa = (a * a).sum(-1, keepdim=True)
    bb = (b * b).sum(-1).unsqueeze(0)
    return (aa + bb - 2.0 * (a @ b.T)).clamp_min(0.0)


def _median_nonzero_distance(x: torch.Tensor) -> float:
    d2 = _sqdist(x, x)
    n = d2.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    vals = torch.sqrt(d2[mask].clamp_min(0.0))
    vals = vals[vals > 0]
    if vals.numel() == 0:
        return 1.0
    return float(vals.median().clamp_min(1e-12))


def _kernel(a: torch.Tensor, b: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.exp(-_sqdist(a, b) / (2.0 * float(sigma) ** 2))


def _fit_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float, ridge: float) -> torch.Tensor:
    k = _kernel(x, x, sigma)
    eye = torch.eye(k.shape[0], dtype=torch.float64)
    mat = k + float(ridge) * eye
    yd = y.double()
    try:
        alpha = torch.linalg.solve(mat, yd)
    except RuntimeError:
        alpha = torch.linalg.lstsq(mat, yd).solution
    return alpha


def _predict_rbf(x: torch.Tensor, anchors: torch.Tensor, alpha: torch.Tensor, sigma: float) -> torch.Tensor:
    return (_kernel(x, anchors, sigma) @ alpha).float()


def _select_hparams(xcal: torch.Tensor, dcal: torch.Tensor) -> Tuple[float, float, Dict[str, float]]:
    # Deterministic calibration-only split: first 3/4 fit, last 1/4 validation.
    n = xcal.shape[0]
    nfit = max(2, (3 * n) // 4)
    xfit, xval = xcal[:nfit], xcal[nfit:]
    dfit, dval = dcal[:nfit], dcal[nfit:]
    base = _median_nonzero_distance(xfit)
    best = None
    records = []
    for mult in SIGMA_MULTS:
        sigma = base * mult
        for ridge in RIDGES:
            alpha = _fit_rbf(xfit, dfit, sigma, ridge)
            pred = _predict_rbf(xval, xfit, alpha, sigma)
            err = float((pred - dval).abs().max())
            mean = float((pred - dval).abs().mean())
            rec = {'sigma_mult': mult, 'sigma': sigma, 'ridge': ridge, 'val_delta_maxabs': err, 'val_delta_meanabs': mean}
            records.append(rec)
            key = (err, mean, ridge, mult)
            if best is None or key < best[0]:
                best = (key, sigma, ridge, rec)
    assert best is not None
    return float(best[1]), float(best[2]), {'fit_median_distance': base, 'selected': best[3], 'candidate_count': len(records)}


def run(model_name: str, seed: int, n_cal: int, n_test: int, seq_len: int, payload_rms: float) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    blocks = transformer_blocks(model)
    layer = max(0, len(blocks) - 3)
    vocab = int(model.get_input_embeddings().weight.shape[0])
    old_p = _payload(model, (41 + 7 * seed) % vocab, payload_rms)
    new_p = _payload(model, (313 + 11 * seed) % vocab, payload_rms)

    n = n_cal + n_test
    g = torch.Generator().manual_seed(96000 + seed)
    ids = torch.randint(0, vocab, (n, seq_len), generator=g, dtype=torch.long)

    old = _forward(model, blocks, layer, ids, old_p)
    new = _forward(model, blocks, layer, ids, new_p)
    delta = new['final'] - old['final']
    logit_delta = (new['logits'] - old['logits']).abs().max(-1).values
    material_rate = float((logit_delta > 1e-4).float().mean())

    tr = slice(0, n_cal)
    te = slice(n_cal, n)
    xcal, dcal = old['final'][tr], delta[tr]
    xtest = old['final'][te]

    sigma, ridge, selection = _select_hparams(xcal, dcal)
    alpha = _fit_rbf(xcal, dcal, sigma, ridge)
    pred_delta = _predict_rbf(xtest, xcal, alpha, sigma)
    pred_h = xtest + pred_delta
    pred_logits = _lm_head_logits(model, pred_h)
    metrics = _metrics(pred_h, new['final'][te], pred_logits, new['logits'][te])

    # Capacity diagnostic on the full calibration set after refit.
    cal_delta_hat = _predict_rbf(xcal, xcal, alpha, sigma)
    cal_delta_maxabs = float((cal_delta_hat - dcal).abs().max())

    # Genuine state-dependence control: collapse every test input to one mean state.
    mean_state = xtest.mean(0, keepdim=True).expand_as(xtest)
    mean_pred = _predict_rbf(mean_state, xcal, alpha, sigma)
    input_dependence_rms = float((pred_delta - mean_pred).pow(2).mean().sqrt())

    centered = delta[te] - delta[te].mean(0, keepdim=True)
    context_var_rms = float(centered.pow(2).mean().sqrt())
    correction_rms = float(delta[te].pow(2).mean().sqrt())

    exact = (
        metrics['hidden_maxabs'] <= 1e-6
        and metrics['logit_maxabs'] <= 1e-5
        and metrics['top1_agree'] == 1.0
    )
    checks = {
        'V1_material_edit_rate_ge_095': material_rate >= 0.95,
        'V2_disjoint_cal_test': n_cal > 0 and n_test > 0,
        'V3_no_test_target_in_selection': True,
        'V4_real_nonlinear_suffix_blocks_ge_2': len(blocks) - layer - 1 >= 2,
        'V5_state_dependence_rms_gt_1e6': input_dependence_rms > 1e-6,
        'exact_transport': bool(exact),
    }
    valid = all(checks[k] for k in checks if k != 'exact_transport')
    decision = 'SURVIVE_NONLINEAR_RECEIPT' if (valid and exact) else ('KILL_REGISTERED_NONLINEAR_RECEIPT' if valid else 'VOID_VALIDITY')

    d = int(xtest.shape[-1])
    receipt_bytes = int((2 * n_cal * d) * 4 + 16)
    return {
        'model': model_name,
        'seed': seed,
        'n_cal': n_cal,
        'n_test': n_test,
        'seq_len': seq_len,
        'read_layer': layer,
        'n_blocks': len(blocks),
        'suffix_blocks': len(blocks) - layer - 1,
        'payload_rms': payload_rms,
        'material_edit_rate': material_rate,
        'exact_final_delta_context_variation_rms': context_var_rms,
        'exact_final_delta_rms': correction_rms,
        'selection': selection,
        'selected_sigma': sigma,
        'selected_ridge': ridge,
        'calibration_delta_maxabs_after_refit': cal_delta_maxabs,
        'input_dependence_rms': input_dependence_rms,
        'rbf_receipt': metrics,
        'receipt_bytes_fp32_estimate': receipt_bytes,
        'checks': checks,
        'decision': decision,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--seeds', type=int, nargs='*', default=[0, 1, 2])
    ap.add_argument('--n-cal', type=int, default=64)
    ap.add_argument('--n-test', type=int, default=64)
    ap.add_argument('--seq-len', type=int, default=16)
    ap.add_argument('--payload-rms', type=float, default=2.0)
    ap.add_argument('--results-dir', default='so/results')
    a = ap.parse_args()

    rows = [run(a.model, s, a.n_cal, a.n_test, a.seq_len, a.payload_rms) for s in a.seeds]
    rec = {
        'experiment': 'E-000096',
        'title': 'Nonlinear state-dependent revision receipt capacity screen',
        'rows': rows,
        'all_registered_cells_survive': all(r['decision'] == 'SURVIVE_NONLINEAR_RECEIPT' for r in rows),
        'not_claimed': 'RBF/kernel repair is a generic nonlinear baseline, not the invention. Approximate task preservation is not exact lifecycle transport.',
    }
    out = Path(a.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ('e000096_' + a.model.replace('/', '_') + '.json')
    path.write_text(json.dumps(rec, indent=2), encoding='utf-8')
    print(json.dumps(rec, indent=2))


if __name__ == '__main__':
    main()
