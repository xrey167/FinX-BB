"""E-000092 — post-hoc exact-support evaluation of the proven symlink reader.

Preregistered before implementation in docs/novelty/e000092-preregister.md.
Hard routing is an established baseline and receives zero novelty credit.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from so.data import bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments import e000081_symlink_consistency_reader as E81
from so.experiments import e000091_real_reader_lineage_density as E91
from so.llm_adapter import AdapterConfig, transformer_blocks


def _one_hot_argmax(scores: torch.Tensor) -> torch.Tensor:
    """Exact one-hot support for evaluation; ties follow torch.argmax's first-index rule."""
    idx = scores.argmax(dim=-1, keepdim=True)
    return torch.zeros_like(scores).scatter_(-1, idx, 1.0)


def _make_exact_hook(model, read_index: int, layer: int):
    """Specialised exact-support form of the historical E81 hook.

    E92's preregistered model config is status_gated=True,use_links=True,n_deref=1,
    fallback='unknown',match_gate=False,two_channel_null=False. Refuse silently
    broadening this baseline if that config changes.
    """
    cfg = model.cfg
    if not (cfg.status_gated and cfg.use_links and cfg.n_deref == 1 and
            cfg.fallback == "unknown" and not cfg.match_gate and not cfg.two_channel_null):
        raise ValueError("E92 exact hook is pinned to the E81 config")

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
        p = _one_hot_argmax(scores)
        val = p @ values
        w_null = p[:, -1:]
        ctx["routing"].append(p)

        qd = model.q_deref[str(layer)](model.deref_ln[str(layer)](val))
        sd = (qd @ keys.t()) * (model.deref_scale[read_index] / cfg.d_key ** 0.5)
        sd = sd.masked_fill(~allowed[None], float("-inf"))
        n_cells = max(int(ctx["allowed"].sum().item()), 1)
        bias = model.deref_pass_bias[read_index] + float(np.log(n_cells))
        sd = torch.cat([sd[:, :-1], sd[:, -1:] + bias], dim=-1)
        pd = _one_hot_argmax(sd)
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


def install_exact_support(model) -> None:
    """Replace only the adapter's registered read hooks; frozen LM internals are untouched."""
    for handle in model._hooks:
        handle.remove()
    blocks = transformer_blocks(model.lm)
    model._hooks = [
        blocks[l].register_forward_hook(_make_exact_hook(model, i, l))
        for i, l in enumerate(model.cfg.read_layers)
    ]


def _eval_world(gk, centre: np.ndarray, seed: int, groups: int) -> Dict[str, Any]:
    rng = np.random.default_rng(70000 + seed)
    world, spec = E15.sample_alias_world(rng, 180, groups, 2, gk.n_entities, 4, E20.N_TRAIN_TEMPLATES)
    store, _ = E15.load_arm(world, spec, centre, 71000 + seed, symlink=True)
    bank = bank_from_store(store)
    per = {str(t): E81._evaluate_template(gk, bank, spec.alias_keys, world, t) for t in range(8, 12)}
    rates = [float(per[str(t)]["candidate_correct"]) for t in range(8, 12)]
    full = [float(per[str(t)]["full_vocab_top1_correct"]) for t in range(8, 12)]
    return {
        "per_template": per,
        "candidate_min": float(min(rates)),
        "candidate_mean": float(np.mean(rates)),
        "full_vocab_min": float(min(full)),
        "full_vocab_mean": float(np.mean(full)),
        "strict_every_template_ge_095": bool(min(rates) >= 0.95),
    }


def _bypass(gk) -> float:
    text = "The telescope was adjusted carefully before sunset."
    ids, am, last = E8.encode_texts(gk.tok, [text])
    with torch.no_grad():
        _, full_a, _, _ = gk.model(None, ids, am, last)
        out = gk.model.lm(input_ids=ids, attention_mask=am)
        ar = torch.arange(ids.shape[0])
        full_b = out.logits[ar, last]
    return float((full_a - full_b).abs().max().item())


def run(seed: int, steps: int, groups: int) -> Dict[str, Any]:
    E91.install_strict_contract()
    os.environ["SO_BOS"] = "1"
    torch.manual_seed(seed)
    cfg = AdapterConfig(status_gated=True, use_links=True, n_deref=E20.N_DEREF)
    gk = E8.GPT2Knowledge(cfg)
    trained = E81.train_symlink_consistent(
        gk, seed, steps, consistency=0.15, alt_supervision=0.5,
        n_groups=max(100, groups), verbose=True,
    )
    gk.model.eval()
    centre = np.asarray(trained["centre"])

    soft = _eval_world(gk, centre, seed, max(100, groups))
    install_exact_support(gk.model)
    exact = _eval_world(gk, centre, seed, max(100, groups))
    bypass = _bypass(gk)

    # Reuse the independently preregistered real-reader unrelated-B intervention,
    # now under exact-support hooks. It reports hidden/logit/routing/KV diffs and
    # keeps A's witness current while making B's old witness stale.
    locality = E91._lineage_intervention(gk, centre, seed, max(100, groups))
    routing = locality.get("routing_before_vs_after")
    same_route = bool(routing and routing.get("byte_identical"))
    neural_same = (
        bool(locality["full_logits_before_vs_after"]["byte_identical"])
        and bool(locality["hidden_before_vs_after"]["byte_identical"])
        and (not locality["kv_before_vs_after"]["available"] or bool(locality["kv_before_vs_after"]["byte_identical"]))
    )
    capability_drop = max(
        float(soft["per_template"][str(t)]["candidate_correct"] - exact["per_template"][str(t)]["candidate_correct"])
        for t in range(8, 12)
    )
    return {
        "seed": seed,
        "steps": steps,
        "soft_reader": soft,
        "exact_support_reader": exact,
        "max_template_candidate_drop_soft_minus_exact": capability_drop,
        "exact_no_memory_bypass_maxabs": bypass,
        "unrelated_b_intervention": locality,
        "hard_route_unchanged_after_b": same_route,
        "neural_state_unchanged_when_hard_route_unchanged": bool(same_route and neural_same),
        "posthoc_exact_support_feasible": bool(
            exact["strict_every_template_ge_095"] and capability_drop <= 0.02 and bypass == 0.0
        ),
        "novelty_claim": False,
        "boundary": "Hard routing is prior-art baseline. No CAVI/lifecycle/J-space promotion from E92 alone.",
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
        "experiment": "E-000092",
        "title": "exact-support real-symlink reader feasibility",
        "rows": rows,
        "all_requested_seeds_posthoc_exact_support_feasible": all(bool(r["posthoc_exact_support_feasible"]) for r in rows),
        "all_unchanged_hard_routes_have_unchanged_neural_state": all(
            (not r["hard_route_unchanged_after_b"]) or bool(r["neural_state_unchanged_when_hard_route_unchanged"])
            for r in rows
        ),
        "breakthrough": False,
        "novelty_claim": False,
    }
    out = Path(args.results_dir) / "e000092_exact_support_reader.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
