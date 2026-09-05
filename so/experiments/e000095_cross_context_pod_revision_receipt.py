"""E-000095 -- cross-context Pod revision receipt kill-screen.

A single old->new controlled Pod edit is applied at one internal layer of a frozen causal LM.
We ask whether a compact receipt fitted on calibration contexts can actively transport many
held-out final neural states to the exact fresh-new counterfactual state without suffix replay.

This is an existence/falsification assay, not a novelty claim.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from so.llm_adapter import transformer_blocks


def _hidden(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output: Any, h: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (h,) + tuple(output[1:])
    return h


def _payload(model, token_id: int, rms: float) -> torch.Tensor:
    emb = model.get_output_embeddings() or model.get_input_embeddings()
    v = emb.weight[int(token_id)].detach().float().clone()
    return v * (float(rms) / float(v.pow(2).mean().sqrt().clamp_min(1e-8)))


def _forward(model, blocks, layer: int, ids: torch.Tensor, payload: torch.Tensor) -> Dict[str, torch.Tensor]:
    box: Dict[str, torch.Tensor] = {}
    def inject(module, inputs, output):
        h = _hidden(output)
        h2 = h.clone()
        h2[:, -1, :] = h2[:, -1, :] + payload.to(h.device, h.dtype)
        box['post_read'] = h2[:, -1, :].detach().float().cpu()
        return _replace_hidden(output, h2)
    handle = blocks[layer].register_forward_hook(inject)
    try:
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    finally:
        handle.remove()
    final = out.hidden_states[-1][:, -1, :].detach().float().cpu()
    logits = out.logits[:, -1, :].detach().float().cpu()
    return {'post_read': box['post_read'], 'final': final, 'logits': logits}


def _lm_head_logits(model, final_hidden: torch.Tensor) -> torch.Tensor:
    # HF causal-LM hidden_states[-1] is the state consumed by the output head for the tested families.
    # Reconstructed states are stored in fp32 on CPU for measurement, but some backbones expose
    # an fp16 output head even on CPU. Cast only for the head application, then return fp32 metrics.
    head = model.get_output_embeddings()
    dev = head.weight.device
    h = final_hidden.to(device=dev, dtype=head.weight.dtype)
    with torch.no_grad():
        y = head(h)
    return y.detach().float().cpu()


def _fit_diag_affine(x: torch.Tensor, y_delta: torch.Tensor):
    # Per dimension: delta = a + b*x. Closed-form least squares on calibration contexts only.
    xm = x.mean(0); ym = y_delta.mean(0)
    xc = x - xm; yc = y_delta - ym
    var = (xc * xc).sum(0)
    b = (xc * yc).sum(0) / var.clamp_min(1e-12)
    a = ym - b * xm
    return a, b


def _metrics(pred_h: torch.Tensor, gold_h: torch.Tensor, pred_logits: torch.Tensor, gold_logits: torch.Tensor) -> Dict[str, float]:
    dh = (pred_h - gold_h).abs()
    dl = (pred_logits - gold_logits).abs()
    p = torch.softmax(gold_logits, -1)
    kl = (p * (torch.log_softmax(gold_logits, -1) - torch.log_softmax(pred_logits, -1))).sum(-1)
    top = (pred_logits.argmax(-1) == gold_logits.argmax(-1)).float()
    return {
        'hidden_maxabs': float(dh.max()),
        'hidden_meanabs': float(dh.mean()),
        'logit_maxabs': float(dl.max()),
        'kl_mean_nats': float(kl.mean()),
        'top1_agree': float(top.mean()),
        'exact_hidden_fraction_1e6': float((dh.max(-1).values <= 1e-6).float().mean()),
        'exact_logit_fraction_1e5': float((dl.max(-1).values <= 1e-5).float().mean()),
    }


def run(model_name: str, seed: int, n_cal: int, n_test: int, seq_len: int, payload_rms: float) -> Dict[str, object]:
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    blocks = transformer_blocks(model)
    layer = max(0, len(blocks) - 3)  # leave a real nonlinear suffix
    vocab = int(model.get_input_embeddings().weight.shape[0])
    old_p = _payload(model, (41 + 7*seed) % vocab, payload_rms)
    new_p = _payload(model, (313 + 11*seed) % vocab, payload_rms)
    n = n_cal + n_test
    g = torch.Generator().manual_seed(95000 + seed)
    ids = torch.randint(0, vocab, (n, seq_len), generator=g, dtype=torch.long)

    old = _forward(model, blocks, layer, ids, old_p)
    new = _forward(model, blocks, layer, ids, new_p)
    final_delta = new['final'] - old['final']
    logit_delta = (new['logits'] - old['logits']).abs().max(-1).values
    material_rate = float((logit_delta > 1e-4).float().mean())

    tr = slice(0, n_cal); te = slice(n_cal, n)
    # Receipt 1: one translation vector for the Pod edit.
    dbar = final_delta[tr].mean(0, keepdim=True)
    trans_h = old['final'][te] + dbar
    trans_logits = _lm_head_logits(model, trans_h)

    # Receipt 2: compact context-conditioned diagonal affine transport.
    a,b = _fit_diag_affine(old['final'][tr], final_delta[tr])
    diag_h = old['final'][te] + a + b * old['final'][te]
    diag_logits = _lm_head_logits(model, diag_h)

    trans_m = _metrics(trans_h, new['final'][te], trans_logits, new['logits'][te])
    diag_m = _metrics(diag_h, new['final'][te], diag_logits, new['logits'][te])

    # Context-dependence diagnostic: how much the exact correction varies across sessions.
    centered = final_delta[te] - final_delta[te].mean(0, keepdim=True)
    context_var_rms = float(centered.pow(2).mean().sqrt())
    correction_rms = float(final_delta[te].pow(2).mean().sqrt())

    exact_trans = trans_m['hidden_maxabs'] <= 1e-6 and trans_m['logit_maxabs'] <= 1e-5 and trans_m['top1_agree'] == 1.0
    exact_diag = diag_m['hidden_maxabs'] <= 1e-6 and diag_m['logit_maxabs'] <= 1e-5 and diag_m['top1_agree'] == 1.0
    checks = {
        'V1_material_edit_rate_ge_095': material_rate >= 0.95,
        'V2_disjoint_cal_test': n_cal > 0 and n_test > 0,
        'V3_real_nonlinear_suffix_blocks_ge_2': len(blocks) - layer - 1 >= 2,
        'translation_exact': bool(exact_trans),
        'diagonal_affine_exact': bool(exact_diag),
    }
    decision = 'SURVIVE_COMPACT_RECEIPT' if (checks['V1_material_edit_rate_ge_095'] and (exact_trans or exact_diag)) else 'KILL_REGISTERED_COMPACT_RECEIPTS'
    return {
        'model': model_name, 'seed': seed, 'n_cal': n_cal, 'n_test': n_test, 'seq_len': seq_len,
        'read_layer': layer, 'n_blocks': len(blocks), 'suffix_blocks': len(blocks)-layer-1,
        'payload_rms': payload_rms, 'material_edit_rate': material_rate,
        'exact_final_delta_context_variation_rms': context_var_rms,
        'exact_final_delta_rms': correction_rms,
        'translation_receipt': trans_m,
        'diagonal_affine_receipt': diag_m,
        'receipt_bytes_translation_fp32': int(old['final'].shape[-1] * 4),
        'receipt_bytes_diagonal_fp32': int(old['final'].shape[-1] * 8),
        'checks': checks, 'decision': decision,
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True)
    ap.add_argument('--seeds',type=int,nargs='*',default=[0,1,2]); ap.add_argument('--n-cal',type=int,default=64)
    ap.add_argument('--n-test',type=int,default=64); ap.add_argument('--seq-len',type=int,default=16)
    ap.add_argument('--payload-rms',type=float,default=2.0); ap.add_argument('--results-dir',default='so/results')
    a=ap.parse_args(); rows=[run(a.model,s,a.n_cal,a.n_test,a.seq_len,a.payload_rms) for s in a.seeds]
    rec={'experiment':'E-000095','title':'Cross-context Pod revision receipt','rows':rows,
         'all_registered_receipts_exact':all(r['decision']=='SURVIVE_COMPACT_RECEIPT' for r in rows),
         'not_claimed':'No novelty for cache editing, selective recomputation, JVP/Jacobian methods, low-rank adapters, or approximate steering.'}
    out=Path(a.results_dir); out.mkdir(parents=True,exist_ok=True)
    p=out/('e000095_'+a.model.replace('/','_')+'.json'); p.write_text(json.dumps(rec,indent=2),encoding='utf-8')
    print(json.dumps(rec,indent=2))

if __name__=='__main__': main()
