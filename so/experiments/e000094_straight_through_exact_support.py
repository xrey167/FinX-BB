"""E-000094 — train the real symlink reader with exact forward support.

Preregistered in docs/novelty/e000094-preregister.md before implementation.
Straight-through hard routing is an established baseline and receives zero novelty credit.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments import e000081_symlink_consistency_reader as E81
from so.experiments import e000091_real_reader_lineage_density as E91
from so.experiments import e000092_exact_support_reader as E92
from so.llm_adapter import AdapterConfig, transformer_blocks


def _straight_through_one_hot(scores: torch.Tensor) -> torch.Tensor:
    """Exactly one-hot in forward; softmax surrogate in backward."""
    soft = torch.softmax(scores, dim=-1)
    idx = scores.argmax(dim=-1, keepdim=True)
    hard = torch.zeros_like(scores).scatter_(-1, idx, 1.0)
    return hard + soft - soft.detach()


def _make_st_hook(model, read_index: int, layer: int):
    cfg = model.cfg
    if not (cfg.status_gated and cfg.use_links and cfg.n_deref == 1 and
            cfg.fallback == "unknown" and not cfg.match_gate and not cfg.two_channel_null):
        raise ValueError("E94 ST hook is pinned to the E81 config")

    def hook(module, inputs, output):
        if model._ctx is None:
            return None
        h = output[0] if isinstance(output, tuple) else output
        ctx = model._ctx
        B = h.shape[0]
        ar = torch.arange(B, device=h.device)
        hl = h[ar, ctx["last_idx"]]
        q = model.q_proj[str(layer)](model.q_ln[str(layer)](hl))
        ctx.setdefault("query", []).append(q)
        keys = torch.cat([ctx["keys"], model.null_key[read_index][None]])
        values = torch.cat([ctx["values"], model.null_value[read_index][None]])
        allowed = torch.cat([ctx["allowed"], torch.ones(1, dtype=torch.bool, device=h.device)])
        scores = (q @ keys.t()) * (model.scale / cfg.d_key ** 0.5)
        scores = scores.masked_fill(~allowed[None], float("-inf"))
        p = _straight_through_one_hot(scores)
        val = p @ values
        w_null = p[:, -1:]
        ctx["routing"].append(p)

        qd = model.q_deref[str(layer)](model.deref_ln[str(layer)](val))
        sd = (qd @ keys.t()) * (model.deref_scale[read_index] / cfg.d_key ** 0.5)
        sd = sd.masked_fill(~allowed[None], float("-inf"))
        n_cells = max(int(ctx["allowed"].sum().item()), 1)
        bias = model.deref_pass_bias[read_index] + float(np.log(n_cells))
        sd = torch.cat([sd[:, :-1], sd[:, -1:] + bias], dim=-1)
        pd = _straight_through_one_hot(sd)
        val = pd[:, :-1] @ values[:-1] + pd[:, -1:] * val
        w_null = w_null * pd[:, -1:]
        ctx["routing"].append(pd)

        null_c = w_null * values[-1][None]
        cell_c = val - null_c
        read = model.o_proj[str(layer)](cell_c + null_c)
        rms_h = hl.detach().pow(2).mean(-1, keepdim=True).sqrt()
        ref = model.o_proj[str(layer)](val)
        rms_r = ref.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-3 * rms_h + 1e-6)
        read = read * (rms_h / rms_r) * model.inject_gain[read_index]
        delta = torch.zeros_like(h)
        delta[ar, ctx["last_idx"]] = read
        h2 = h + delta
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2
    return hook


def install_st_support(model) -> None:
    for handle in model._hooks:
        handle.remove()
    blocks = transformer_blocks(model.lm)
    model._hooks = [
        blocks[l].register_forward_hook(_make_st_hook(model, i, l))
        for i, l in enumerate(model.cfg.read_layers)
    ]


def _executed_support_is_exact(gk, centre: np.ndarray, seed: int, groups: int) -> Dict[str, Any]:
    """Audit the actual routing tensor emitted by an intervention run."""
    intervention = E91._lineage_intervention(gk, centre, seed, groups)
    support = intervention.get("support_before", {})
    # E91's support summary counts strictly positive rows. With ST one-hot forward,
    # each slot should expose at most one positive real row; the selected null row is
    # not included in real_rows. This is a forward-value test, not a gradient test.
    counts = support.get("strictly_positive_real_rows_per_slot") or []
    exact = bool(counts and all(int(c) <= 1 for c in counts))
    return {"intervention": intervention, "real_positive_counts": counts, "exact_forward_support": exact}


def run(seed: int, steps: int, groups: int) -> Dict[str, Any]:
    E91.install_strict_contract()
    os.environ["SO_BOS"] = "1"
    torch.manual_seed(seed)
    cfg = AdapterConfig(status_gated=True, use_links=True, n_deref=E20.N_DEREF)
    gk = E8.GPT2Knowledge(cfg)
    install_st_support(gk.model)
    trained = E81.train_symlink_consistent(
        gk, seed, steps, consistency=0.15, alt_supervision=0.5,
        n_groups=max(100, groups), verbose=True,
    )
    gk.model.eval()
    centre = np.asarray(trained["centre"])
    capability = E92._eval_world(gk, centre, seed, max(100, groups))
    bypass = E92._bypass(gk)
    audited = _executed_support_is_exact(gk, centre, seed, max(100, groups))
    loc = audited["intervention"]
    routing_same = bool(loc["routing_before_vs_after"]["byte_identical"])
    neural_same = bool(
        loc["full_logits_before_vs_after"]["byte_identical"]
        and loc["hidden_before_vs_after"]["byte_identical"]
        and (not loc["kv_before_vs_after"]["available"] or loc["kv_before_vs_after"]["byte_identical"])
    )
    pass_seed = bool(
        capability["strict_every_template_ge_095"]
        and bypass == 0.0
        and audited["exact_forward_support"]
        and ((not routing_same) or neural_same)
    )
    return {
        "seed": seed,
        "steps": steps,
        "capability": capability,
        "exact_no_memory_bypass_maxabs": bypass,
        "support_audit": audited,
        "hard_route_unchanged_after_b": routing_same,
        "neural_state_unchanged_when_hard_route_unchanged": bool(routing_same and neural_same),
        "seed_feasible": pass_seed,
        "novelty_claim": False,
        "boundary": "ST/hard routing is baseline only; no lifecycle/J-space/CAVI promotion from E94 alone.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--groups", type=int, default=100)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--results-dir", default="so/results")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rows = [run(s, args.steps, args.groups) for s in args.seeds]
    rec = {
        "experiment": "E-000094",
        "title": "straight-through exact-support real-symlink reader",
        "rows": rows,
        "all_requested_seeds_feasible": all(bool(r["seed_feasible"]) for r in rows),
        "breakthrough": False,
        "novelty_claim": False,
    }
    out = Path(args.results_dir) / "e000094_straight_through_exact_support.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    if not rec["all_requested_seeds_feasible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
