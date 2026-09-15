"""E-000098 -- cross-context exact correction-span kill screen.

For one fixed old->new controlled Pod edit, ask whether exact final-state corrections
across many contexts occupy a shared low-dimensional linear span.  Held-out projection
uses ORACLE coefficients from the true fresh correction, so this is an intentionally
favorable representation-existence test rather than a deployable repair algorithm.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from so.llm_adapter import transformer_blocks


RANKS = (1, 2, 4, 8, 16, 32, 48, 64)
HIDDEN_EPS = 1e-6
LOGIT_EPS = 1e-5


def _hidden(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output: Any, h: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (h,) + tuple(output[1:])
    return h


def _payload(model, token_id: int, rms: float) -> torch.Tensor:
    emb = model.get_output_embeddings() or model.get_input_embeddings()
    v = emb.weight[int(token_id)].detach().float().clone()
    denom = float(v.pow(2).mean().sqrt().clamp_min(1e-8))
    return v * (float(rms) / denom)


def _forward(model, blocks, layer: int, ids: torch.Tensor, payload: torch.Tensor) -> Dict[str, torch.Tensor]:
    box: Dict[str, torch.Tensor] = {}

    def inject(module, inputs, output):
        h = _hidden(output)
        h2 = h.clone()
        h2[:, -1, :] = h2[:, -1, :] + payload.to(h.device, h.dtype)
        box["post_read"] = h2[:, -1, :].detach().float().cpu()
        return _replace_hidden(output, h2)

    handle = blocks[layer].register_forward_hook(inject)
    try:
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    finally:
        handle.remove()

    return {
        "post_read": box["post_read"],
        "final": out.hidden_states[-1][:, -1, :].detach().float().cpu(),
        "logits": out.logits[:, -1, :].detach().float().cpu(),
    }


def _lm_head_logits(model, final_hidden: torch.Tensor) -> torch.Tensor:
    head = model.get_output_embeddings()
    dev = head.weight.device
    h = final_hidden.to(device=dev, dtype=head.weight.dtype)
    with torch.no_grad():
        y = head(h)
    return y.detach().float().cpu()


def _candidate_metrics(
    pred_h: torch.Tensor,
    gold_h: torch.Tensor,
    pred_logits: torch.Tensor,
    gold_logits: torch.Tensor,
) -> Dict[str, float]:
    dh = (pred_h.float() - gold_h.float()).abs()
    dl = (pred_logits.float() - gold_logits.float()).abs()
    top = (pred_logits.argmax(-1) == gold_logits.argmax(-1)).float()
    p = torch.softmax(gold_logits.float(), -1)
    kl = (p * (torch.log_softmax(gold_logits.float(), -1) - torch.log_softmax(pred_logits.float(), -1))).sum(-1)
    per_h = dh.max(-1).values
    per_l = dl.max(-1).values
    return {
        "hidden_maxabs": float(dh.max()),
        "hidden_meanabs": float(dh.mean()),
        "logit_maxabs": float(dl.max()),
        "logit_meanabs": float(dl.mean()),
        "top1_agree": float(top.mean()),
        "kl_mean_nats": float(kl.mean()),
        "exact_hidden_fraction_1e6": float((per_h <= HIDDEN_EPS).float().mean()),
        "exact_logit_fraction_1e5": float((per_l <= LOGIT_EPS).float().mean()),
        "all_heldout_exact": bool(
            float(dh.max()) <= HIDDEN_EPS
            and float(dl.max()) <= LOGIT_EPS
            and float(top.mean()) == 1.0
        ),
    }


def _calibration_tail_stats(s: torch.Tensor, n_cal: int, hidden_dim: int) -> Dict[str, Dict[str, float | bool]]:
    s2 = s.double().pow(2)
    bound = HIDDEN_EPS * math.sqrt(float(n_cal * hidden_dim))
    out: Dict[str, Dict[str, float | bool]] = {}
    for requested in RANKS:
        r = min(int(requested), int(s.numel()))
        tail = float(torch.sqrt(s2[r:].sum())) if r < s.numel() else 0.0
        out[str(requested)] = {
            "effective_rank": r,
            "optimal_frobenius_tail": tail,
            "entrywise_1e6_not_ruled_out_by_fro_bound": bool(tail <= bound),
            "frobenius_bound_for_all_entries_1e6": float(bound),
        }
    return out


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
    suffix_blocks = len(blocks) - layer - 1
    vocab = int(model.get_input_embeddings().weight.shape[0])
    old_p = _payload(model, (41 + 7 * seed) % vocab, payload_rms)
    new_p = _payload(model, (313 + 11 * seed) % vocab, payload_rms)

    n = n_cal + n_test
    g = torch.Generator().manual_seed(98000 + seed)
    ids = torch.randint(0, vocab, (n, seq_len), generator=g, dtype=torch.long)

    old = _forward(model, blocks, layer, ids, old_p)
    new = _forward(model, blocks, layer, ids, new_p)
    delta = new["final"] - old["final"]
    logit_delta = (new["logits"] - old["logits"]).abs().max(-1).values
    material_rate = float((logit_delta > 1e-4).float().mean())

    d_cal = delta[:n_cal].double()
    d_test = delta[n_cal:].double()
    old_test = old["final"][n_cal:].double()
    gold_test_h = new["final"][n_cal:].double()
    gold_test_logits = new["logits"][n_cal:]

    # Row-space basis in hidden-feature coordinates. full_matrices=False gives at most n_cal rows.
    _, singular, vh = torch.linalg.svd(d_cal, full_matrices=False)
    cal_tail = _calibration_tail_stats(singular, n_cal, int(d_cal.shape[1]))

    total_energy = float(singular.double().pow(2).sum())
    rank_rows: List[Dict[str, object]] = []
    seen_eff = set()
    for requested in RANKS:
        r = min(int(requested), int(vh.shape[0]))
        if r in seen_eff:
            continue
        seen_eff.add(r)
        basis = vh[:r, :]  # orthonormal rows

        # Oracle held-out coefficients: the true fresh correction itself is projected.
        # This favors the candidate and tests only whether the shared span exists.
        proj = (d_test @ basis.T) @ basis
        pred_h64 = old_test + proj
        pred_h = pred_h64.float()
        pred_logits = _lm_head_logits(model, pred_h)
        metrics = _candidate_metrics(pred_h, gold_test_h.float(), pred_logits, gold_test_logits)

        residual = d_test - proj
        denom = float(torch.linalg.vector_norm(d_test).clamp_min(1e-30))
        metrics["oracle_delta_relative_fro_error"] = float(torch.linalg.vector_norm(residual) / denom)
        metrics["oracle_delta_fro_error"] = float(torch.linalg.vector_norm(residual))
        rank_rows.append({
            "requested_rank": int(requested),
            "effective_rank": r,
            "basis_fp32_bytes": int(r * d_cal.shape[1] * 4),
            "metrics": metrics,
        })

    max_available_rank = int(vh.shape[0])
    full_row = next(r for r in rank_rows if int(r["effective_rank"]) == max_available_rank)
    any_exact = any(bool(r["metrics"]["all_heldout_exact"]) for r in rank_rows)

    # Tolerance-grounded calibration rank lower bound: smallest r for which the
    # Eckart-Young Frobenius tail no longer rules out all-entry <= 1e-6 reconstruction.
    smallest_not_ruled_out = None
    for r in range(0, int(singular.numel()) + 1):
        tail = float(torch.sqrt(singular.double().pow(2)[r:].sum())) if r < singular.numel() else 0.0
        bound = HIDDEN_EPS * math.sqrt(float(n_cal * d_cal.shape[1]))
        if tail <= bound:
            smallest_not_ruled_out = r
            break

    checks = {
        "V1_material_edit_rate_ge_095": material_rate >= 0.95,
        "V2_disjoint_cal_test": n_cal > 0 and n_test > 0,
        "V3_real_nonlinear_suffix_blocks_ge_2": suffix_blocks >= 2,
        "V4_full_calibration_row_span_tested": int(full_row["effective_rank"]) == max_available_rank,
        "any_registered_span_exact": bool(any_exact),
        "full_calibration_span_exact": bool(full_row["metrics"]["all_heldout_exact"]),
    }

    decision = (
        "KILL_SHARED_LINEAR_EXACT_CORRECTION_SPAN"
        if checks["V1_material_edit_rate_ge_095"] and checks["V2_disjoint_cal_test"]
        and checks["V3_real_nonlinear_suffix_blocks_ge_2"] and not any_exact
        else "DO_NOT_KILL_SHARED_LINEAR_SPAN"
    )

    return {
        "model": model_name,
        "seed": seed,
        "n_cal": n_cal,
        "n_test": n_test,
        "seq_len": seq_len,
        "read_layer": layer,
        "n_blocks": len(blocks),
        "suffix_blocks": suffix_blocks,
        "payload_rms": payload_rms,
        "hidden_dim": int(d_cal.shape[1]),
        "material_edit_rate": material_rate,
        "calibration_singular_values": [float(x) for x in singular.cpu()],
        "calibration_frobenius_tail_by_registered_rank": cal_tail,
        "smallest_rank_not_ruled_out_by_calibration_entrywise_1e6_fro_bound": smallest_not_ruled_out,
        "max_calibration_row_space_rank": max_available_rank,
        "rank_results": rank_rows,
        "checks": checks,
        "decision": decision,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-cal", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--payload-rms", type=float, default=2.0)
    ap.add_argument("--results-dir", default="so/results")
    a = ap.parse_args()

    rows = [run(a.model, s, a.n_cal, a.n_test, a.seq_len, a.payload_rms) for s in a.seeds]
    record = {
        "experiment": "E-000098",
        "title": "Cross-context exact correction span",
        "oracle_advantage": "Held-out coefficients use the true fresh correction; only shared-span existence is tested.",
        "rows": rows,
        "all_cells_kill_shared_linear_span": all(r["decision"] == "KILL_SHARED_LINEAR_EXACT_CORRECTION_SPAN" for r in rows),
        "not_claimed": "No general impossibility theorem; no novelty credit for SVD/PCA or low-rank KV/cache methods.",
    }

    out = Path(a.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / ("e000098_" + a.model.replace("/", "_") + ".json")
    p.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
