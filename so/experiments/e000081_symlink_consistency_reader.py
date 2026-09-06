"""E-000081 -- targeted capability repair for the strict real-symlink reader gate.

E-000077 reached fresh alias correctness 0.99 / 0.95 / 0.94 on seeds 0/1/2
with BOS + the dense 100-group symlink regime.  Because one seed missed the
>=0.95 prerequisite, no positive CAVI attack may be interpreted from that run.

This experiment does NOT weaken or cherry-pick that gate.  Instead it ports the
useful idea from E-000017's paraphrase-consistency trainer into the *real link +
dereference* training regime: the same semantic query is rendered under a
second independently chosen TRAINING surface form and receives both supervised
answer/route loss and a symmetric consistency loss over candidate answers and
all resolve/dereference routing slots.

Important correction versus copying E-000017 literally: paired rows are indexed
explicitly.  We never assume that filtering one-hop queries leaves them in the
first k batch positions.

Evaluation is on a fresh independent symlink world and all four held-out forms
(templates 8..11), including the historical strict CAVI form template 9.  It
records both candidate-set correctness and full-vocabulary top-1 correctness.

This is a capability experiment only.  It makes no novelty claim and does not
change CAVI semantics, lifecycle authority, thresholds, or the adversarial
battery.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from so.data import bank_from_store, sample_training_queries
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
from so.llm_adapter import AdapterConfig
from so.train import TrainConfig, lr_at, make_centre, routing_loss


def _sym_kl_log_probs(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        F.kl_div(logp, logq, log_target=True, reduction="batchmean")
        + F.kl_div(logq, logp, log_target=True, reduction="batchmean")
    )


def train_symlink_consistent(
    gk: E8.GPT2Knowledge,
    seed: int,
    steps: int,
    *,
    consistency: float,
    alt_supervision: float = 0.5,
    bind_supervision: float = 0.0,
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
        world, spec = E15.sample_alias_world(
            rng, n_base, n_groups, 2, gk.n_entities, 4, E20.N_TRAIN_TEMPLATES
        )
        bank = E15.bank_with_links(
            rng, world, spec, centre, p_revoked, p_shred, 0.05, p_dangling
        )
        queries = sample_training_queries(rng, world, bank, batch_size - n_extra, mix)
        queries += world.sample_queries(
            rng, n_extra, 1, "fwd", require_answer=False, index=bank.index_view
        )
        ids, am, last = E8.encode_texts(
            gk.tok,
            [E17.query_text_pc(q, gk.names, E20.N_TRAIN_TEMPLATES) for q in queries],
        )
        target = E8.targets_of(queries, bank, world)
        route = E20.route_targets_slots(
            queries, bank, world, n_reads, model.cfg.n_deref
        )
        for group in opt.param_groups:
            group["lr"] = lr_at(step, tcfg)
        tensors = bank.tensors()
        cand, _, routing, _ = model(tensors, ids, am, last)
        loss_ans = F.cross_entropy(cand, target)
        loss_route = routing_loss(routing, route)

        # E-000084 arm E only (bind_supervision > 0, default off, so every earlier configuration is
        # unchanged). The reference carrier has one addressing decision the other arms do not: at the
        # boundary it must decide WHICH row the handle it transported names. In arms A/C/D every
        # addressing slot is supervised by `loss_route`; leaving this one unsupervised handicaps arm E
        # rather than testing it, and the first arm E run showed exactly that failure — a boundary
        # distribution that never concentrated, an answer independent of the payload, and 0.0 correct.
        # The target is the row the query resolves to: the last dereference slot, or the resolve slot
        # where the dereference was a passthrough because the row was not a pointer.
        loss_bind = cand.sum() * 0
        if bind_supervision > 0 and getattr(model, "last_bind", None) is not None:
            last_deref, last_resolve = route[:, -1], route[:, -2]
            bind_target = torch.where(last_deref >= 0, last_deref, last_resolve)
            loss_bind = routing_loss(model.last_bind, bind_target[:, None])

        valid = tensors["marker_valid"].float()
        per_cell = F.binary_cross_entropy_with_logits(
            model.gate_logits(tensors["marker"]).squeeze(-1), valid, reduction="none"
        )
        n_pos = valid.sum().clamp_min(1)
        n_neg = (1 - valid).sum().clamp_min(1)
        loss_gate = (
            0.5 * (per_cell * valid).sum() / n_pos
            + 0.5 * (per_cell * (1 - valid)).sum() / n_neg
        )

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
                ids2, am2, last2 = E8.encode_texts(
                    gk.tok,
                    [
                        E17.query_text_pc(q, gk.names, E20.N_TRAIN_TEMPLATES, t)
                        for q, t in zip(paired_queries, alt_templates)
                    ],
                )
                cand2, _, routing2, _ = model(tensors, ids2, am2, last2)
                target2 = target.index_select(0, idx)
                route2 = route.index_select(0, idx)
                loss_alt = F.cross_entropy(cand2, target2) + route_weight * routing_loss(routing2, route2)

                primary_cand = cand.index_select(0, idx)
                primary_route = routing.index_select(0, idx)
                ans_kl = _sym_kl_log_probs(
                    torch.log_softmax(primary_cand, -1),
                    torch.log_softmax(cand2, -1),
                )
                # Routing tensors are probabilities.  Flatten all neural read and
                # dereference slots so consistency covers the whole address path.
                r1 = torch.log(primary_route.clamp_min(1e-9)).reshape(-1, primary_route.shape[-1])
                r2 = torch.log(routing2.clamp_min(1e-9)).reshape(-1, routing2.shape[-1])
                route_kl = _sym_kl_log_probs(r1, r2)
                loss_cons = ans_kl + route_kl
                paired = len(idx_list)

        if route_only:
            loss = loss_route + gate_weight * loss_gate + bind_supervision * loss_bind
        else:
            loss = (
                loss_ans
                + route_weight * loss_route
                + gate_weight * loss_gate
                + alt_supervision * loss_alt
                + consistency * loss_cons
                + bind_supervision * loss_bind
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
                "bind_loss": float(loss_bind.item()),
                "route_loss": float(loss_route.item()),
                "gate_loss": float(loss_gate.item()),
                "alt_supervised_loss": float(loss_alt.item()),
                "consistency_loss": float(loss_cons.item()),
                "paired_rows": paired,
                "batch_acc": float((cand.argmax(-1) == target).float().mean().item()),
                "elapsed_s": time.time() - t0,
            }
            history.append(rec)
            if verbose:
                print(
                    f"  step {rec['step']:5d} loss {rec['loss']:.4f} ans {rec['answer_loss']:.4f} "
                    f"route {rec['route_loss']:.4f} alt {rec['alt_supervised_loss']:.4f} "
                    f"cons {rec['consistency_loss']:.4f} acc {rec['batch_acc']:.3f} "
                    f"{rec['elapsed_s']:.0f}s",
                    flush=True,
                )
    model.eval()
    return {"centre": centre, "history": history, "train_seconds": time.time() - t0}


def _evaluate_template(gk, bank, alias_keys: Sequence, world, template: int) -> Dict[str, float]:
    cand_ok = []
    full_ok = []
    for key in alias_keys:
        s, r = key
        text = E17.TEMPLATES12[r][template].format(s=gk.names[s])
        ids, am, last = E8.encode_texts(gk.tok, [text])
        with torch.no_grad():
            cand, full, _, _ = gk.model(bank.tensors(), ids, am, last)
        truth_entity = int(world.index[key])
        cand_ok.append(int(cand.argmax(-1)[0]) == truth_entity)
        expected_token = int(gk.entity_ids[truth_entity])
        full_ok.append(int(full.argmax(-1)[0]) == expected_token)
    return {
        "candidate_correct": float(np.mean(cand_ok)),
        "full_vocab_top1_correct": float(np.mean(full_ok)),
    }


def run(seed: int, steps: int, consistency: float, alt_supervision: float, n_groups: int) -> Dict[str, object]:
    torch.manual_seed(seed)
    cfg = AdapterConfig(status_gated=True, use_links=True, n_deref=E20.N_DEREF)
    gk = E8.GPT2Knowledge(cfg)
    trained = train_symlink_consistent(
        gk,
        seed,
        steps,
        consistency=consistency,
        alt_supervision=alt_supervision,
        n_groups=max(24, n_groups),
        verbose=True,
    )
    centre = np.asarray(trained["centre"])

    # Identical independent-world construction family as E-000070/E-000077.
    rng = np.random.default_rng(70000 + seed)
    world, spec = E15.sample_alias_world(
        rng, 180, n_groups, 2, gk.n_entities, 4, E20.N_TRAIN_TEMPLATES
    )
    store, _kids = E15.load_arm(world, spec, centre, 71000 + seed, symlink=True)
    bank = bank_from_store(store)

    per_template = {
        str(t): _evaluate_template(gk, bank, spec.alias_keys, world, t)
        for t in range(E20.N_TRAIN_TEMPLATES, E20.N_TRAIN_TEMPLATES + E17.N_HELDOUT)
    }
    candidate_rates = [per_template[str(t)]["candidate_correct"] for t in range(8, 12)]
    full_rates = [per_template[str(t)]["full_vocab_top1_correct"] for t in range(8, 12)]
    template9 = per_template["9"]["candidate_correct"]
    heldout_mean = float(np.mean(candidate_rates))
    heldout_min = float(np.min(candidate_rates))
    checks = {
        "strict_template9_real_symlink_gate": template9 >= 0.95,
        "heldout_paraphrase_mean_ge_095": heldout_mean >= 0.95,
        "heldout_every_template_ge_095": heldout_min >= 0.95,
    }
    return {
        "seed": seed,
        "steps": steps,
        "consistency": consistency,
        "alt_supervision": alt_supervision,
        "bos_enabled": E8.bos_enabled(),
        "groups": n_groups,
        "n_alias_eval": len(spec.alias_keys),
        "per_heldout_template": per_template,
        "template9_candidate_correct": template9,
        "heldout_candidate_mean": heldout_mean,
        "heldout_candidate_min": heldout_min,
        "heldout_full_vocab_mean": float(np.mean(full_rates)),
        "heldout_full_vocab_min": float(np.min(full_rates)),
        "checks": checks,
        "strict_pass": all(checks.values()),
        "train_seconds": float(trained["train_seconds"]),
        "last_training_record": trained["history"][-1] if trained["history"] else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--groups", type=int, default=100)
    ap.add_argument("--consistency", type=float, default=0.15)
    ap.add_argument("--alt-supervision", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--results-dir", default="so/results")
    a = ap.parse_args()
    os.environ.setdefault("SO_BOS", "1")
    torch.set_num_threads(a.threads)
    rows = [run(s, a.steps, a.consistency, a.alt_supervision, a.groups) for s in a.seeds]
    rec = {
        "experiment": "E-000081",
        "candidate_only": True,
        "rows": rows,
        "all_strict_pass": all(bool(r["strict_pass"]) for r in rows),
        "gate_unchanged": "template9 >=0.95 per seed; joint held-out target >=0.95; no CAVI semantics changed",
        "not_claimed": "paraphrase consistency, symlinks, routing supervision, BOS handling, or capability training as novelty",
    }
    out_dir = Path(a.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"e000081_symlink_consistency_c{a.consistency:g}_a{a.alt_supervision:g}.json"
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    if not rec["all_strict_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
