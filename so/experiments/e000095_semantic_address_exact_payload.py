"""E-000095 — dense semantic address plane, exact mutable payload plane.

Preregistered in docs/novelty/e000095-preregister.md before implementation.
This is a strong ordinary baseline. Decoupled addressing, straight-through routing,
pointer following and versioning receive ZERO novelty credit.
"""
from __future__ import annotations

import argparse
import json
import os
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from so.data import bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments import e000081_symlink_consistency_reader as E81
from so.experiments import e000091_real_reader_lineage_density as E91
from so.experiments import e000092_exact_support_reader as E92
from so.llm_adapter import AdapterConfig, transformer_blocks


def _st_one_hot(scores: torch.Tensor) -> torch.Tensor:
    soft = torch.softmax(scores, dim=-1)
    idx = scores.argmax(dim=-1, keepdim=True)
    hard = torch.zeros_like(scores).scatter_(-1, idx, 1.0)
    return hard + soft - soft.detach()


def _pointer_matrix(model, bank: Dict[str, torch.Tensor], enc: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Exact selected-row -> current target-row mapping, NULL for dangling targets.

    The matrix is control-plane state. It is never averaged by semantic attention.
    Direct rows map to themselves; link rows map by the target's subject/relation key.
    """
    subject = bank["subject"].detach().cpu().tolist()
    relation = bank["relation"].detach().cpu().tolist()
    C = len(subject)
    lookup: Dict[Tuple[int, int], int] = {}
    for i, (s, r) in enumerate(zip(subject, relation)):
        lookup[(int(s), int(r))] = i
    target = list(range(C))
    if "is_link" in bank:
        is_link = bank["is_link"].detach().cpu().bool().tolist()
        ls = bank["link_subject"].detach().cpu().tolist()
        lr = bank["link_relation"].detach().cpu().tolist()
        for i, flag in enumerate(is_link):
            if flag:
                target[i] = lookup.get((int(ls[i]), int(lr[i])), C)
    M = torch.zeros((C + 1, C + 1), dtype=enc["keys"].dtype, device=enc["keys"].device)
    if C:
        rows = torch.arange(C, device=M.device)
        cols = torch.as_tensor(target, dtype=torch.long, device=M.device)
        M[rows, cols] = 1.0
    M[C, C] = 1.0
    return M


def _prepare_ctx(model, bank: Dict[str, torch.Tensor], last_idx: torch.Tensor,
                 cell_mask: torch.Tensor | None = None) -> None:
    enc = model.encode_bank(bank)
    allowed = enc["active"] if cell_mask is None else enc["active"] & cell_mask
    # Own-row identity only. Crucially, aliases do NOT place their mutable target in this plane.
    own_address = model.v_link(enc["keys"])
    model._ctx = {
        "keys": enc["keys"],
        "values": enc["values"],
        "address_values": own_address,
        "pointer_matrix": _pointer_matrix(model, bank, enc),
        "allowed": allowed,
        "last_idx": last_idx,
        "routing": [],
    }


def _make_hook(model, read_index: int, layer: int):
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
        allowed = torch.cat([ctx["allowed"], torch.ones(1, dtype=torch.bool, device=h.device)])

        # Stage 1: dense LANGUAGE/IDENTITY matching. Values here are only own-row identities.
        scores = (q @ keys.t()) * (model.scale / model.cfg.d_key ** 0.5)
        scores = scores.masked_fill(~allowed[None], float("-inf"))
        p_addr = torch.softmax(scores, dim=-1)
        null_addr = model.v_link(model.null_key[read_index][None])
        addr_values = torch.cat([ctx["address_values"], null_addr], dim=0)
        addr = p_addr @ addr_values
        ctx["routing"].append(p_addr)

        # Stage 2: exact executed ROW identity, differentiable only through ST surrogate.
        qid = model.q_deref[str(layer)](model.deref_ln[str(layer)](addr))
        sid = (qid @ keys.t()) * (model.deref_scale[read_index] / model.cfg.d_key ** 0.5)
        sid = sid.masked_fill(~allowed[None], float("-inf"))
        p_id = _st_one_hot(sid)
        ctx["routing"].append(p_id)

        # Stage 3: deterministic current Symlink jump; stage 4: exact current payload.
        p_target = p_id @ ctx["pointer_matrix"]
        values = torch.cat([ctx["values"], model.null_value[read_index][None]], dim=0)
        val = p_target @ values
        w_null = p_target[:, -1:]
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


def _forward(self, bank: Dict[str, torch.Tensor] | None, input_ids: torch.Tensor,
             attention_mask: torch.Tensor, last_idx: torch.Tensor,
             cell_mask: torch.Tensor | None = None):
    if bank is not None:
        _prepare_ctx(self, bank, last_idx, cell_mask)
    else:
        self._ctx = None
    try:
        out = self.lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        ar = torch.arange(input_ids.shape[0], device=input_ids.device)
        full = out.logits[ar, last_idx]
        hidden = out.hidden_states[-1][ar, last_idx]
        cand = full[:, self.candidate_ids]
        routing = torch.stack(self._ctx["routing"], dim=1) if self._ctx is not None and self._ctx["routing"] else None
        self.last_query = (torch.stack(self._ctx["query"], dim=1)
                           if self._ctx is not None and self._ctx.get("query") else None)
        return cand, full, routing, hidden
    finally:
        self._ctx = None


def _identity_route_targets(queries, bank, world, n_reads: int, n_deref: int) -> torch.Tensor:
    """Train stage 1 and stage 2 on the same exact semantic row identity.

    Pointer following is deterministic after identity selection and needs no neural route label.
    """
    base = E20.route_targets_slots(queries, bank, world, n_reads, n_deref).clone()
    if n_deref != 1:
        raise ValueError("E95 is pinned to one identity-selection slot per read")
    for rr in range(n_reads):
        resolve = base[:, 2 * rr]
        base[:, 2 * rr + 1] = resolve
    return base


def install(model) -> None:
    if not (model.cfg.status_gated and model.cfg.use_links and model.cfg.n_deref == 1):
        raise ValueError("E95 requires strict link reader config")
    for handle in model._hooks:
        handle.remove()
    blocks = transformer_blocks(model.lm)
    model._hooks = [blocks[l].register_forward_hook(_make_hook(model, i, l))
                    for i, l in enumerate(model.cfg.read_layers)]
    model.forward = types.MethodType(_forward, model)


def _forward_cache(gk, bank, text: str) -> Dict[str, Any]:
    ids, am, last = E8.encode_texts(gk.tok, [text])
    _prepare_ctx(gk.model, bank.tensors(), last)
    try:
        with torch.no_grad():
            out = gk.model.lm(input_ids=ids, attention_mask=am, output_hidden_states=True, use_cache=True)
        ctx = gk.model._ctx
        routing = torch.stack(ctx["routing"], dim=1).detach().cpu()
        ar = torch.arange(ids.shape[0], device=ids.device)
        full = out.logits[ar, last].detach().cpu()
        hidden = out.hidden_states[-1][ar, last].detach().cpu()
        return {"ids": ids.detach().cpu(), "full": full, "hidden": hidden,
                "candidate": full[:, gk.model.candidate_ids.detach().cpu()],
                "routing": routing, "past": out.past_key_values}
    finally:
        gk.model._ctx = None


def _neural_diff(gk, before: Dict[str, Any], after: Dict[str, Any], continuation_id: int) -> Dict[str, Any]:
    stale = E91._continue(gk, before["past"], int(before["ids"].shape[1]), continuation_id)
    fresh = E91._continue(gk, after["past"], int(after["ids"].shape[1]), continuation_id)
    return {
        "routing": E91._tensor_diff(before["routing"], after["routing"]),
        "hidden": E91._tensor_diff(before["hidden"], after["hidden"]),
        "full_logits": E91._tensor_diff(before["full"], after["full"]),
        "kv": E91._cache_diff(before["past"], after["past"]),
        "stale_kv_continuation_vs_fresh": E91._tensor_diff(stale, fresh),
    }


def _all_byte_identical(d: Dict[str, Any]) -> bool:
    return bool(d["routing"]["byte_identical"] and d["hidden"]["byte_identical"]
                and d["full_logits"]["byte_identical"]
                and (not d["kv"]["available"] or d["kv"]["byte_identical"])
                and d["stale_kv_continuation_vs_fresh"]["byte_identical"])


def _interventions(gk, centre: np.ndarray, seed: int, groups: int) -> Dict[str, Any]:
    def fresh_arm(offset: int):
        rng = np.random.default_rng(195000 + seed + offset)
        world, spec = E15.sample_alias_world(rng, 220, groups, 2, gk.n_entities, 4, E20.N_TRAIN_TEMPLATES)
        store, kids = E15.load_arm(world, spec, centre, 196000 + seed + offset, symlink=True)
        return world, spec, store, kids

    # --- unrelated target payload UPDATE
    world, spec, store, kids = fresh_arm(0)
    ta, aa = spec.groups[0]; tb, ab = spec.groups[1]
    key_a = aa[0]; text = E91._text(gk, key_a, 9); truth_a = int(world.index[key_a])
    before = _forward_cache(gk, bank_from_store(store), text)
    old_b = int(world.index[tb]); new_b = int((old_b + 17) % gk.n_entities)
    store.update(kids[tb], new_b)
    after = _forward_cache(gk, bank_from_store(store), text)
    d_update_b = _neural_diff(gk, before, after, int(gk.entity_ids[truth_a]))

    # --- unrelated alias RELINK: B alias moves, A query and target stay untouched
    world2, spec2, store2, kids2 = fresh_arm(1000)
    ta2, aa2 = spec2.groups[0]; tb2, ab2 = spec2.groups[1]; tc2, _ac2 = spec2.groups[2]
    key_a2 = aa2[0]; text2 = E91._text(gk, key_a2, 9); truth_a2 = int(world2.index[key_a2])
    before2 = _forward_cache(gk, bank_from_store(store2), text2)
    store2.relink(kids2[ab2[0]], kids2[tc2])
    after2 = _forward_cache(gk, bank_from_store(store2), text2)
    d_relink_b = _neural_diff(gk, before2, after2, int(gk.entity_ids[truth_a2]))

    # --- relevant target UPDATE must produce the fresh new object
    world3, spec3, store3, kids3 = fresh_arm(2000)
    ta3, aa3 = spec3.groups[0]; key_a3 = aa3[0]; text3 = E91._text(gk, key_a3, 9)
    old_a3 = int(world3.index[ta3]); new_a3 = int((old_a3 + 23) % gk.n_entities)
    if new_a3 == old_a3: new_a3 = (new_a3 + 1) % gk.n_entities
    before3 = _forward_cache(gk, bank_from_store(store3), text3)
    store3.update(kids3[ta3], new_a3)
    after3 = _forward_cache(gk, bank_from_store(store3), text3)
    pred_update_a = int(after3["candidate"].argmax(-1)[0])

    # --- queried alias RELINK must follow the current new target
    world4, spec4, store4, kids4 = fresh_arm(3000)
    ta4, aa4 = spec4.groups[0]; tc4, _ac4 = spec4.groups[2]
    key_a4 = aa4[0]; text4 = E91._text(gk, key_a4, 9); expected_relink = int(world4.index[tc4])
    before4 = _forward_cache(gk, bank_from_store(store4), text4)
    store4.relink(kids4[key_a4], kids4[tc4])
    after4 = _forward_cache(gk, bank_from_store(store4), text4)
    pred_relink_a = int(after4["candidate"].argmax(-1)[0])

    # Exact executed mutable identity support is the hard second slot of each pair.
    id_slots = before["routing"][0, 1::2, :-1]
    positive = (id_slots > 0).sum(-1)
    return {
        "unrelated_payload_update": {"diff": d_update_b, "byte_identical": _all_byte_identical(d_update_b)},
        "unrelated_alias_relink": {"diff": d_relink_b, "byte_identical": _all_byte_identical(d_relink_b)},
        "relevant_payload_update": {"expected": new_a3, "predicted": pred_update_a,
                                    "current_correct": pred_update_a == new_a3,
                                    "state_changed": not bool(torch.equal(before3["full"], after3["full"]))},
        "queried_alias_relink": {"expected": expected_relink, "predicted": pred_relink_a,
                                  "current_correct": pred_relink_a == expected_relink,
                                  "state_changed": not bool(torch.equal(before4["full"], after4["full"]))},
        "mutable_identity_positive_real_rows_per_read": [int(x) for x in positive.tolist()],
        "exact_mutable_identity_support": bool(all(int(x) <= 1 for x in positive.tolist())),
    }


def run(seed: int, steps: int, groups: int) -> Dict[str, Any]:
    E91.install_strict_contract()
    os.environ["SO_BOS"] = "1"
    torch.manual_seed(seed)
    cfg = AdapterConfig(status_gated=True, use_links=True, n_deref=1)
    gk = E8.GPT2Knowledge(cfg)
    install(gk.model)
    original_routes = E20.route_targets_slots
    E20.route_targets_slots = _identity_route_targets
    try:
        trained = E81.train_symlink_consistent(gk, seed, steps, consistency=0.15,
                                               alt_supervision=0.5, n_groups=max(100, groups), verbose=True)
    finally:
        E20.route_targets_slots = original_routes
    gk.model.eval()
    centre = np.asarray(trained["centre"])
    capability = E92._eval_world(gk, centre, seed, max(100, groups))
    bypass = E92._bypass(gk)
    iv = _interventions(gk, centre, seed, max(100, groups))
    strict_cap = bool(capability["strict_every_template_ge_095"]
                      and capability["candidate_min"] >= 0.95 and capability["full_vocab_min"] >= 0.95)
    feasible = bool(strict_cap and bypass == 0.0 and iv["exact_mutable_identity_support"]
                    and iv["unrelated_payload_update"]["byte_identical"]
                    and iv["unrelated_alias_relink"]["byte_identical"]
                    and iv["relevant_payload_update"]["current_correct"]
                    and iv["queried_alias_relink"]["current_correct"])
    return {"seed": seed, "steps": steps, "capability": capability,
            "exact_no_memory_bypass_maxabs": bypass, "interventions": iv,
            "seed_feasible": feasible, "novelty_claim": False,
            "boundary": "Decoupled semantic addressing/exact payload is baseline only; no CAVI/J-space/SHRED promotion."}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--groups", type=int, default=100)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--results-dir", default="so/results")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rows = [run(s, args.steps, args.groups) for s in args.seeds]
    rec = {"experiment": "E-000095", "title": "semantic address plane + exact mutable payload plane",
           "rows": rows, "all_requested_seeds_feasible": all(r["seed_feasible"] for r in rows),
           "breakthrough": False, "novelty_claim": False}
    out = Path(args.results_dir) / "e000095_semantic_address_exact_payload.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    if not rec["all_requested_seeds_feasible"]:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
