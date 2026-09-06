"""E-000085 -- can a frozen model transport a knowledge-free reference?

E-000084 arm E puts a handle, not a payload, into the residual stream at a read layer and binds the
value to it after the last cache-writing block.  Whether that can work at all rests on one mechanism
question that needs no training to answer:

    injected at block L, at the last position, does a random handle remain LINEARLY RECOVERABLE from
    the residual at the boundary, after the frozen blocks above L have processed it?

If it does not, arm E cannot hold its capability gate and the reason is transport, not optimisation.
If it does, the remaining risk in arm E is the routing and the training, not the carrier.

The experiment asks that question two ways, and the second is the one that decides the design.  Holding
out PROMPTS asks whether the readout survives a new context for an identity it has already seen.  A
mutable memory needs more: its identity set changes, so the readout must work for an identity it has
NEVER seen.  Both are reported.

Read the history of this file's own numbers before trusting any of them.  The first version ran only
the prompt split and read 1.0000.  Adding the identity split then read 0.0000 -- below chance -- and
that was taken as proof that a frozen model cannot carry an arbitrary reference.  BOTH readings were
about the handle family, not the model: `handles_for` was expanding the raw bits of the identity, so
128 handles had mean pairwise |cos| 0.878 and rank 8, i.e. they were nearly the same vector.  With
handles rebuilt as hashed full-dimensional directions (mean |cos| 0.041, full rank) the same
experiment reads 0.9834 with NO learning at all and 0.9844 on identities never seen.  The frozen model
transports an arbitrary reference; a crowded carrier alphabet was the whole of the earlier result.

This is a diagnostic on frozen weights with a closed-form readout.  No adapter is trained, no world is
used, and nothing here is a capability result or a novelty claim.  The readout is deliberately the same
shape as the one arm E learns: a linear map of the boundary residual scored against the handle table,
so the number is an upper bound on what the trained boundary decoder can achieve at that placement.

Run: python -m so.experiments.e000085_reference_transport [--model gpt2] [--layers 8 10] [--n-ids 64]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from so.llm_adapter import AdapterConfig, KnowledgeAdapterLM, transformer_blocks

PROMPTS = [
    "The capital of the country is", "She asked whether the answer was", "In the report it says the value is",
    "According to the record the name is", "He wrote that the city is", "The document lists the owner as",
    "When asked, the guide replied that it was", "The label on the box reads", "Their notes say the place is",
    "The final entry in the ledger is", "It turned out that the person was", "The archive gives the location as",
    "A short summary of the file says", "The witness stated the object was", "On the map the region is",
    "The catalogue entry for it reads",
]


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


@torch.no_grad()
def transport(model_name: str, read_layers: List[int], n_ids: int, seed: int,
              rms_scale: float) -> Dict[str, object]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    lm = AutoModelForCausalLM.from_pretrained(model_name).float().eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    blocks = transformer_blocks(lm)
    n_blocks = len(blocks)

    # Borrow the adapter's own handle construction so the diagnostic measures the real carrier.
    cfg = AdapterConfig(read_layers=tuple(read_layers), write_layer=n_blocks - 1, reference_carrier=True)
    adapter = KnowledgeAdapterLM(lm, cfg, list(range(10, 10 + 8)), 11)
    ids = torch.arange(n_ids, dtype=torch.long)
    handles = adapter.handles_for(ids)                      # (n_ids, d)
    handles = handles / handles.norm(dim=-1, keepdim=True)  # compare directions, not magnitudes

    enc = tok(PROMPTS, return_tensors="pt", padding=True)
    last = enc["attention_mask"].sum(-1) - 1
    ar = torch.arange(len(PROMPTS))

    state = {"handle": None}

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            if state["handle"] is None:
                return None
            h = _hidden(output)
            hl = h[ar, last]
            rms_h = hl.pow(2).mean(-1, keepdim=True).sqrt()
            v = state["handle"][None, :].expand(h.shape[0], -1)
            v = v / v.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
            delta = torch.zeros_like(h)
            delta[ar, last] = v * rms_h * rms_scale
            h2 = h + delta
            return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2
        return hook

    hooks = [blocks[l].register_forward_hook(make_hook(l)) for l in read_layers]
    try:
        # boundary residual for every identity, on every prompt
        feats, labels = [], []
        for i in range(n_ids):
            state["handle"] = handles[i]
            out = lm(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True)
            feats.append(out.hidden_states[-1][ar, last].clone())
            labels.append(torch.full((len(PROMPTS),), i, dtype=torch.long))
        state["handle"] = None
        clean = lm(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                   output_hidden_states=True).hidden_states[-1][ar, last].clone()
    finally:
        for h in hooks:
            h.remove()

    X = torch.cat(feats)                                   # (n_ids * P, d)
    y = torch.cat(labels)
    # Held-out PROMPTS, not held-out identities: the readout must generalise across contexts, which is
    # what the boundary decoder faces. Prompts are split by index so no prompt appears in both halves.
    P = len(PROMPTS)
    tr = torch.cat([torch.arange(i * P, i * P + P // 2) for i in range(n_ids)])
    te = torch.cat([torch.arange(i * P + P // 2, (i + 1) * P) for i in range(n_ids)])

    # Closed-form linear map from residual to handle direction (ridge), then score against the table.
    def fit(idx):
        A = X[idx]
        B = handles[y[idx]]
        lam = 1e-3 * float(A.pow(2).mean())
        return torch.linalg.solve(A.t() @ A + lam * torch.eye(A.shape[1]), A.t() @ B)   # (d, d)

    W = fit(tr)

    def top1(idx, w=None):
        pred = X[idx] @ (W if w is None else w)             # (n, d)
        pred = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return float((pred @ handles.t()).argmax(-1).eq(y[idx]).float().mean())

    # The split that matters, and that this experiment originally did not run. Holding out PROMPTS asks
    # whether the readout survives a new context for an identity it already knows. A mutable memory needs
    # more than that: its identities change, so the readout must work for an identity it has never seen.
    # Fit on half the identities and score the other half against the full table.
    id_tr = torch.arange(n_ids // 2)
    id_te = torch.arange(n_ids // 2, n_ids)
    rows_tr = torch.cat([torch.arange(int(i) * P, (int(i) + 1) * P) for i in id_tr])
    rows_te = torch.cat([torch.arange(int(i) * P, (int(i) + 1) * P) for i in id_te])
    W_id = fit(rows_tr)
    seen_identity_top1 = top1(rows_tr, W_id)
    unseen_identity_top1 = top1(rows_te, W_id)

    # Control: the same readout applied to the CLEAN residual, where no handle was injected. It must be
    # at chance, otherwise the readout is reading the prompt rather than the carrier.
    clean_rep = clean.repeat(n_ids, 1)
    pred_clean = clean_rep @ W
    pred_clean = pred_clean / pred_clean.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    clean_top1 = float((pred_clean @ handles.t()).argmax(-1).eq(y).float().mean())

    return {
        "model": model_name,
        "read_layers": list(read_layers),
        "n_blocks": n_blocks,
        "n_ids": n_ids,
        "n_prompts": P,
        "rms_scale": rms_scale,
        "seed": seed,
        "train_top1": top1(tr),
        "heldout_prompt_top1": top1(te),
        "seen_identity_top1": seen_identity_top1,
        "heldout_identity_top1": unseen_identity_top1,
        "clean_control_top1": clean_top1,
        "chance": 1.0 / n_ids,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 10])
    ap.add_argument("--n-ids", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rms-scale", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--results-dir", default="so/results")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rows = [transport(a.model, a.layers, a.n_ids, a.seed, s) for s in a.rms_scale]
    rec = {
        "experiment": "E-000085",
        "question": "is an injected knowledge-free handle linearly recoverable at the boundary of a frozen model?",
        "rows": rows,
        "not_claimed": "no capability result, no novelty claim; a frozen-weights diagnostic with a closed-form readout",
    }
    out = Path(a.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"e000085_reference_transport_{a.model.replace('/', '_')}.json").write_text(
        json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
