"""E-000093 — train the real symlink reader for counterfactual mutation locality.

Preregistered in docs/novelty/e000093-preregister.md before implementation.
A positive result is a mechanism result, not a novelty award.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from so.data import bank_from_store, sample_training_queries
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments import e000081_symlink_consistency_reader as E81
from so.experiments import e000091_real_reader_lineage_density as E91
from so.experiments import e000092_exact_support_reader as E92
from so.llm_adapter import AdapterConfig
from so.train import TrainConfig, lr_at, make_centre, routing_loss


def _sym_kl_log(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        F.kl_div(logp, logq, log_target=True, reduction="batchmean")
        + F.kl_div(logq, logp, log_target=True, reduction="batchmean")
    )


def _routing_sym_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    lp = torch.log(p.clamp_min(1e-9)).reshape(-1, p.shape[-1])
    lq = torch.log(q.clamp_min(1e-9)).reshape(-1, q.shape[-1])
    return _sym_kl_log(lp, lq)


def _counterfactual_locality(
    gk: E8.GPT2Knowledge,
    tensors: Dict[str, torch.Tensor],
    ids: torch.Tensor,
    am: torch.Tensor,
    last: torch.Tensor,
    cand: torch.Tensor,
    routing: torch.Tensor,
    hidden: torch.Tensor,
    route: torch.Tensor,
    queries,
    rng: np.random.Generator,
    max_rows: int = 8,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Mutate one active non-link bystander row and require selected A queries to stay invariant."""
    active = tensors["active"].bool()
    is_link = tensors.get("is_link")
    canonical = active if is_link is None else (active & ~is_link.bool())
    candidates = torch.nonzero(canonical, as_tuple=False).flatten().tolist()
    if not candidates:
        zero = cand.sum() * 0
        return zero, {"locality_rows": 0.0}

    rng.shuffle(candidates)
    b_idx = None
    selected: List[int] = []
    for b in candidates:
        eligible = []
        for i, q in enumerate(queries):
            if q.hops != 1:
                continue
            deps = route[i]
            if not bool((deps == int(b)).any().item()):
                eligible.append(i)
        if eligible:
            rng.shuffle(eligible)
            selected = eligible[: min(max_rows, len(eligible))]
            b_idx = int(b)
            break
    if b_idx is None or not selected:
        zero = cand.sum() * 0
        return zero, {"locality_rows": 0.0}

    idx = torch.as_tensor(selected, dtype=torch.long, device=ids.device)
    cf = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in tensors.items()}
    old_obj = int(cf["obj"][b_idx].item())
    new_obj = int((old_obj + 17) % gk.n_entities)
    if new_obj == old_obj:
        new_obj = int((new_obj + 1) % gk.n_entities)
    cf["obj"][b_idx] = new_obj

    cand_cf, _, routing_cf, hidden_cf = gk.model(
        cf,
        ids.index_select(0, idx),
        am.index_select(0, idx),
        last.index_select(0, idx),
    )
    cand_a = cand.index_select(0, idx)
    route_a = routing.index_select(0, idx)
    hidden_a = hidden.index_select(0, idx)

    cand_kl = _sym_kl_log(torch.log_softmax(cand_a, -1), torch.log_softmax(cand_cf, -1))
    route_kl = _routing_sym_kl(route_a, routing_cf)
    mse = (hidden_a - hidden_cf).pow(2).mean()
    denom = hidden_a.detach().pow(2).mean().clamp_min(1e-6)
    hidden_nmse = mse / denom
    loss = cand_kl + route_kl + hidden_nmse
    return loss, {
        "locality_rows": float(len(selected)),
        "locality_b_row": float(b_idx),
        "locality_candidate_kl": float(cand_kl.detach().item()),
        "locality_routing_kl": float(route_kl.detach().item()),
        "locality_hidden_nmse": float(hidden_nmse.detach().item()),
    }


def train_locality(
    gk: E8.GPT2Knowledge,
    seed: int,
    steps: int,
    *,
    locality_weight: float = 0.25,
    locality_rows: int = 8,
    consistency: float = 0.15,
    alt_supervision: float = 0.5,
    batch_size: int = 32,
    consistency_rows: int = 8,
    route_weight: float = 1.0,
    gate_weight: float = 5.0,
    lr: float = 2e-3,
    route_only_steps: int = 400,
    p_revoked: float = 0.20,
    p_shred: float = 0.10,
    p_dangling: float = 0.05,
    n_groups: int = 100,
    extra_unanswerable: float = 0.2,
    verbose: bool = True,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    centre = make_centre(seed, gk.model.cfg.marker_dim)
    model = gk.model
    params = model.adapter_parameters()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    tcfg = TrainConfig(seed=seed, n_steps=steps, lr=lr, warmup=50)
    mix = {"fwd1": 0.7, "fwd2": 0.3}
    history: List[Dict[str, Any]] = []
    t0 = time.time()
    n_extra = int(round(batch_size * extra_unanswerable))
    n_reads = len(model.cfg.read_layers)

    for step in range(steps):
        model.train()
        route_only = step < route_only_steps
        n_base = int(rng.integers(150, 301)) if route_only else int(rng.integers(500, 701))
        world, spec = E15.sample_alias_world(rng, n_base, n_groups, 2, gk.n_entities, 4, E20.N_TRAIN_TEMPLATES)
        bank = E15.bank_with_links(rng, world, spec, centre, p_revoked, p_shred, 0.05, p_dangling)
        queries = sample_training_queries(rng, world, bank, batch_size - n_extra, mix)
        queries += world.sample_queries(rng, n_extra, 1, "fwd", require_answer=False, index=bank.index_view)
        ids, am, last = E8.encode_texts(gk.tok, [E17.query_text_pc(q, gk.names, E20.N_TRAIN_TEMPLATES) for q in queries])
        target = E8.targets_of(queries, bank, world)
        route = E20.route_targets_slots(queries, bank, world, n_reads, model.cfg.n_deref)
        for group in opt.param_groups:
            group["lr"] = lr_at(step, tcfg)
        tensors = bank.tensors()
        cand, _, routing, hidden = model(tensors, ids, am, last)
        loss_ans = F.cross_entropy(cand, target)
        loss_route = routing_loss(routing, route)

        valid = tensors["marker_valid"].float()
        per_cell = F.binary_cross_entropy_with_logits(model.gate_logits(tensors["marker"]).squeeze(-1), valid, reduction="none")
        n_pos = valid.sum().clamp_min(1)
        n_neg = (1 - valid).sum().clamp_min(1)
        loss_gate = 0.5 * (per_cell * valid).sum() / n_pos + 0.5 * (per_cell * (1 - valid)).sum() / n_neg

        loss_alt = cand.sum() * 0
        loss_cons = cand.sum() * 0
        paired = 0
        if not route_only and (consistency > 0 or alt_supervision > 0):
            eligible = [i for i, q in enumerate(queries) if q.hops == 1]
            if eligible:
                rng.shuffle(eligible)
                idx_list = eligible[: min(consistency_rows, len(eligible))]
                idx = torch.as_tensor(idx_list, dtype=torch.long)
                paired_queries = [queries[i] for i in idx_list]
                alt_templates = []
                for q in paired_queries:
                    original = int(q.surface[0] % E20.N_TRAIN_TEMPLATES)
                    alt = int(rng.integers(0, E20.N_TRAIN_TEMPLATES - 1))
                    if alt >= original:
                        alt += 1
                    alt_templates.append(alt)
                ids2, am2, last2 = E8.encode_texts(gk.tok, [E17.query_text_pc(q, gk.names, E20.N_TRAIN_TEMPLATES, t) for q, t in zip(paired_queries, alt_templates)])
                cand2, _, routing2, _ = model(tensors, ids2, am2, last2)
                target2 = target.index_select(0, idx)
                route2 = route.index_select(0, idx)
                loss_alt = F.cross_entropy(cand2, target2) + route_weight * routing_loss(routing2, route2)
                primary_cand = cand.index_select(0, idx)
                primary_route = routing.index_select(0, idx)
                ans_kl = _sym_kl_log(torch.log_softmax(primary_cand, -1), torch.log_softmax(cand2, -1))
                route_kl = _routing_sym_kl(primary_route, routing2)
                loss_cons = ans_kl + route_kl
                paired = len(idx_list)

        locality = cand.sum() * 0
        locality_stats: Dict[str, float] = {"locality_rows": 0.0}
        if not route_only and locality_weight > 0:
            locality, locality_stats = _counterfactual_locality(
                gk, tensors, ids, am, last, cand, routing, hidden, route, queries, rng, locality_rows
            )

        if route_only:
            loss = loss_route + gate_weight * loss_gate
        else:
            loss = (
                loss_ans + route_weight * loss_route + gate_weight * loss_gate
                + alt_supervision * loss_alt + consistency * loss_cons
                + locality_weight * locality
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if (step + 1) % 100 == 0 or step == 0:
            rec = {
                "step": step + 1,
                "loss": float(loss.item()),
                "answer_loss": float(loss_ans.item()),
                "route_loss": float(loss_route.item()),
                "gate_loss": float(loss_gate.item()),
                "alt_supervised_loss": float(loss_alt.item()),
                "consistency_loss": float(loss_cons.item()),
                "locality_loss": float(locality.detach().item()),
                "paired_rows": paired,
                "batch_acc": float((cand.argmax(-1) == target).float().mean().item()),
                "elapsed_s": time.time() - t0,
                **locality_stats,
            }
            history.append(rec)
            if verbose:
                print(f"  step {rec['step']:5d} loss {rec['loss']:.4f} acc {rec['batch_acc']:.3f} local {rec['locality_loss']:.5f} {rec['elapsed_s']:.0f}s", flush=True)
    model.eval()
    return {"centre": centre, "history": history, "train_seconds": time.time() - t0}


def _eval(gk, centre: np.ndarray, seed: int, groups: int) -> Dict[str, Any]:
    rng = np.random.default_rng(70000 + seed)
    world, spec = E15.sample_alias_world(rng, 180, groups, 2, gk.n_entities, 4, E20.N_TRAIN_TEMPLATES)
    store, _ = E15.load_arm(world, spec, centre, 71000 + seed, symlink=True)
    bank = bank_from_store(store)
    per = {str(t): E81._evaluate_template(gk, bank, spec.alias_keys, world, t) for t in range(8, 12)}
    rates = [float(per[str(t)]["candidate_correct"]) for t in range(8, 12)]
    return {
        "per_template": per,
        "candidate_min": float(min(rates)),
        "strict_every_template_ge_095": bool(min(rates) >= 0.95),
        "no_memory_bypass_maxabs": E92._bypass(gk),
    }


def _arm(seed: int, steps: int, groups: int, locality: bool) -> Dict[str, Any]:
    torch.manual_seed(seed)
    cfg = AdapterConfig(status_gated=True, use_links=True, n_deref=E20.N_DEREF)
    gk = E8.GPT2Knowledge(cfg)
    if locality:
        trained = train_locality(gk, seed, steps, n_groups=max(100, groups), verbose=True)
    else:
        trained = E81.train_symlink_consistent(
            gk, seed, steps, consistency=0.15, alt_supervision=0.5,
            n_groups=max(100, groups), verbose=True,
        )
    centre = np.asarray(trained["centre"])
    gk.model.eval()
    capability = _eval(gk, centre, seed, max(100, groups))
    intervention = E91._lineage_intervention(gk, centre, seed, max(100, groups))
    return {
        "capability": capability,
        "intervention": intervention,
        "train_seconds": float(trained["train_seconds"]),
    }


def _maxabs(d: Dict[str, Any]) -> float:
    v = d.get("maxabs")
    return float(v) if v is not None else float("nan")


def run(seed: int, steps: int, groups: int) -> Dict[str, Any]:
    E91.install_strict_contract()
    os.environ["SO_BOS"] = "1"
    control = _arm(seed, steps, groups, locality=False)
    local = _arm(seed, steps, groups, locality=True)
    ci, li = control["intervention"], local["intervention"]
    c_h = _maxabs(ci["hidden_before_vs_after"])
    l_h = _maxabs(li["hidden_before_vs_after"])
    c_f = _maxabs(ci["full_logits_before_vs_after"])
    l_f = _maxabs(li["full_logits_before_vs_after"])
    h_gain = float(c_h / max(l_h, 1e-30))
    f_gain = float(c_f / max(l_f, 1e-30))
    stale_c = _maxabs(ci["stale_kv_continuation_vs_fresh"])
    stale_l = _maxabs(li["stale_kv_continuation_vs_fresh"])
    cap = bool(local["capability"]["strict_every_template_ge_095"] and local["capability"]["no_memory_bypass_maxabs"] == 0.0)
    authority = bool(li["a_only_lineage_still_current_after_b_update"] and li["a_witness_still_current_after_b_update"] and li["b_old_witness_stale_after_b_update"])
    success = bool(cap and authority and h_gain >= 10.0 and f_gain >= 10.0 and stale_l <= stale_c + 1e-12)
    return {
        "seed": seed,
        "steps": steps,
        "control": control,
        "locality_arm": local,
        "hidden_locality_gain_control_over_locality": h_gain,
        "full_logit_locality_gain_control_over_locality": f_gain,
        "stale_continuation_control_maxabs": stale_c,
        "stale_continuation_locality_maxabs": stale_l,
        "locality_capability_gate": cap,
        "authority_controls_ok": authority,
        "preregistered_success": success,
        "novelty_claim": False,
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
    interpretable = [r for r in rows if r["locality_capability_gate"]]
    rec = {
        "experiment": "E-000093",
        "title": "counterfactual mutation-locality objective",
        "rows": rows,
        "interpretable_seeds": len(interpretable),
        "successful_interpretable_seeds": sum(bool(r["preregistered_success"]) for r in interpretable),
        "all_requested_seeds_success": len(interpretable) == len(rows) and all(bool(r["preregistered_success"]) for r in rows),
        "breakthrough": False,
        "novelty_claim": False,
    }
    out = Path(args.results_dir) / "e000093_mutation_locality_objective.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
