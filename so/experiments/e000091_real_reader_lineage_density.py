"""E-000091 — exact-lineage density in the strong real-symlink reader.

Preregistered in docs/novelty/e000091-preregister.md before this implementation.
This is a falsification screen, not a novelty claim.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

import so.data as data
from so.data import bank_from_store
from so.derived_lineage import DerivedLineage
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments import e000081_symlink_consistency_reader as E81
from so.experiments.e000070_cavi_live_symlink_boundary import _authority_and_manifest
from so.llm_adapter import AdapterConfig
from so.mvcc import MVCCStore


# --------------------------------------------------------------------------- strict marker contract

def _mechanical_valid(markers: np.ndarray, centre: np.ndarray, radius: float = 0.35) -> np.ndarray:
    if markers.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    return np.linalg.norm(markers.astype(float) - centre[None, :], axis=1) <= radius


def _strict_valid_markers(rng: np.random.Generator, centre: np.ndarray, n: int,
                          scale: float = 0.05, valid_radius: float = 0.35,
                          max_rounds: int = 10000) -> np.ndarray:
    n = int(n)
    if n == 0:
        return np.empty((0, centre.shape[0]), dtype=float)
    out = np.empty((n, centre.shape[0]), dtype=float)
    remaining = np.arange(n)
    for _ in range(max_rounds):
        if remaining.size == 0:
            return out
        m = centre[None, :] + rng.normal(scale=scale, size=(remaining.size, centre.shape[0]))
        m = m / np.linalg.norm(m, axis=1, keepdims=True)
        ok = _mechanical_valid(m, centre, valid_radius)
        if ok.any():
            out[remaining[ok]] = m[ok]
            remaining = remaining[~ok]
    raise RuntimeError(f"strict marker sampler failed for {remaining.size}/{n} rows")


def _strict_store_marker(self: MVCCStore) -> np.ndarray:
    return _strict_valid_markers(self.rng, self.marker_centre, 1,
                                 scale=0.05, valid_radius=self.valid_radius)[0]


def install_strict_contract() -> None:
    def training_sampler(rng, centre, n, scale=0.05):
        return _strict_valid_markers(rng, centre, n, scale=scale, valid_radius=0.35)
    data.valid_markers = training_sampler
    MVCCStore.new_valid_marker = _strict_store_marker


# --------------------------------------------------------------------------- neural instrumentation

def _text(gk, key: Tuple[int, int], template: int) -> str:
    s, r = key
    return E17.TEMPLATES12[r][template].format(s=gk.names[s])


def _prepare_context(gk, bank, last_idx: torch.Tensor) -> None:
    enc = gk.model.encode_bank(bank.tensors())
    gk.model._ctx = {
        "keys": enc["keys"],
        "values": enc["values"],
        "allowed": enc["active"],
        "last_idx": last_idx,
        "routing": [],
    }


def _forward_with_cache(gk, bank, text: str) -> Dict[str, Any]:
    ids, am, last = E8.encode_texts(gk.tok, [text])
    _prepare_context(gk, bank, last)
    try:
        with torch.no_grad():
            out = gk.model.lm(
                input_ids=ids,
                attention_mask=am,
                output_hidden_states=True,
                use_cache=True,
            )
        ctx = gk.model._ctx
        routing = torch.stack(ctx["routing"], dim=1).detach().cpu() if ctx and ctx["routing"] else None
        queries = torch.stack(ctx["query"], dim=1).detach().cpu() if ctx and ctx.get("query") else None
        ar = torch.arange(ids.shape[0], device=ids.device)
        full = out.logits[ar, last].detach().cpu()
        hidden = out.hidden_states[-1][ar, last].detach().cpu()
        cand = full[:, gk.model.candidate_ids.detach().cpu()]
        return {
            "ids": ids.detach().cpu(),
            "attention_mask": am.detach().cpu(),
            "last": last.detach().cpu(),
            "full": full,
            "candidate": cand,
            "hidden": hidden,
            "routing": routing,
            "queries": queries,
            "past": out.past_key_values,
        }
    finally:
        gk.model._ctx = None


def _legacy_cache_tensors(past: Any) -> List[torch.Tensor]:
    obj = past
    if hasattr(obj, "to_legacy_cache"):
        try:
            obj = obj.to_legacy_cache()
        except Exception:
            pass
    tensors: List[torch.Tensor] = []
    if isinstance(obj, (tuple, list)):
        for layer in obj:
            if isinstance(layer, (tuple, list)):
                tensors.extend(t.detach().cpu() for t in layer[:2] if torch.is_tensor(t))
            elif torch.is_tensor(layer):
                tensors.append(layer.detach().cpu())
    if tensors:
        return tensors
    layers = getattr(obj, "layers", None)
    if layers is not None:
        for layer in layers:
            for name in ("keys", "values", "key_cache", "value_cache"):
                t = getattr(layer, name, None)
                if torch.is_tensor(t):
                    tensors.append(t.detach().cpu())
    return tensors


def _tensor_diff(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    if a.shape != b.shape:
        return {"same_shape": False, "byte_identical": False, "maxabs": None}
    return {
        "same_shape": True,
        "byte_identical": bool(torch.equal(a, b)),
        "maxabs": float((a.float() - b.float()).abs().max().item()) if a.numel() else 0.0,
    }


def _cache_diff(a: Any, b: Any) -> Dict[str, Any]:
    aa, bb = _legacy_cache_tensors(a), _legacy_cache_tensors(b)
    if not aa or not bb or len(aa) != len(bb):
        return {
            "available": False,
            "tensor_count_a": len(aa),
            "tensor_count_b": len(bb),
            "byte_identical": None,
            "maxabs": None,
            "unequal_tensors": None,
        }
    diffs = [_tensor_diff(x, y) for x, y in zip(aa, bb)]
    return {
        "available": True,
        "tensor_count_a": len(aa),
        "tensor_count_b": len(bb),
        "byte_identical": all(bool(d["byte_identical"]) for d in diffs),
        "maxabs": max(float(d["maxabs"] or 0.0) for d in diffs),
        "unequal_tensors": sum(not bool(d["byte_identical"]) for d in diffs),
    }


def _continue(gk, past: Any, prompt_len: int, continuation_id: int) -> torch.Tensor:
    token = torch.tensor([[int(continuation_id)]], dtype=torch.long)
    mask = torch.ones((1, prompt_len + 1), dtype=torch.long)
    with torch.no_grad():
        out = gk.model.lm(
            input_ids=token,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
        )
    return out.logits[:, -1, :].detach().cpu()


def _support(routing: torch.Tensor | None) -> Dict[str, Any]:
    if routing is None:
        return {"available": False}
    real = routing[0, :, :-1]
    positive = (real > 0).sum(-1)
    return {
        "available": True,
        "slots": int(real.shape[0]),
        "real_rows": int(real.shape[-1]),
        "strictly_positive_real_rows_per_slot": [int(x) for x in positive.tolist()],
        "min_positive_probability_per_slot": [float(x) for x in real.min(-1).values.tolist()],
        "max_probability_per_slot": [float(x) for x in real.max(-1).values.tolist()],
    }


# --------------------------------------------------------------------------- strict reader + intervention

def _strict_reader_gate(gk, centre: np.ndarray, seed: int, groups: int) -> Dict[str, Any]:
    rng = np.random.default_rng(70000 + seed)
    world, spec = E15.sample_alias_world(
        rng, 180, groups, 2, gk.n_entities, 4, E20.N_TRAIN_TEMPLATES
    )
    store, _kids = E15.load_arm(world, spec, centre, 71000 + seed, symlink=True)
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
        "template9": float(per["9"]["candidate_correct"]),
        "strict_every_template_ge_095": bool(min(rates) >= 0.95),
    }


def _lineage_intervention(gk, centre: np.ndarray, seed: int, groups: int) -> Dict[str, Any]:
    rng = np.random.default_rng(191000 + seed)
    world, spec = E15.sample_alias_world(
        rng, 220, groups, 2, gk.n_entities, 4, E20.N_TRAIN_TEMPLATES
    )
    store, kids = E15.load_arm(world, spec, centre, 192000 + seed, symlink=True)
    bank0 = bank_from_store(store)
    auth, _manifest = _authority_and_manifest(store, bank0)

    target_a, aliases_a = spec.groups[0]
    target_b, aliases_b = spec.groups[1]
    key_a = aliases_a[0]
    aid_a, pid_a = kids[key_a], kids[target_a]
    aid_b, pid_b = kids[aliases_b[0]], kids[target_b]
    witness_a = auth.witness(aid_a)
    witness_b = auth.witness(aid_b)
    lineage_a = DerivedLineage.of(witness_a)
    text = _text(gk, key_a, 9)
    truth_a = int(world.index[key_a])
    old_b_obj = int(world.index[target_b])
    new_b_obj = int((old_b_obj + 17) % gk.n_entities)
    if new_b_obj == truth_a:
        new_b_obj = int((new_b_obj + 1) % gk.n_entities)

    before = _forward_with_cache(gk, bank0, text)
    # Deterministic no-op rebuild control before mutation.
    bank0b = bank_from_store(store)
    noop = _forward_with_cache(gk, bank0b, text)

    # Canonical B payload changes. A's alias+pod authority state is untouched.
    store.update(pid_b, new_b_obj)
    auth.update_pod(pid_b)
    bank1 = bank_from_store(store)
    after = _forward_with_cache(gk, bank1, text)

    cand_before = int(before["candidate"].argmax(-1)[0])
    cand_after = int(after["candidate"].argmax(-1)[0])
    continuation_id = int(gk.entity_ids[truth_a])

    cache_before_after = _cache_diff(before["past"], after["past"])
    cache_noop = _cache_diff(before["past"], noop["past"])
    stale_cont = _continue(gk, before["past"], int(before["ids"].shape[1]), continuation_id)
    fresh_cont = _continue(gk, after["past"], int(after["ids"].shape[1]), continuation_id)

    routing_diff = _tensor_diff(before["routing"], after["routing"]) if before["routing"] is not None and after["routing"] is not None else None
    noop_routing_diff = _tensor_diff(before["routing"], noop["routing"]) if before["routing"] is not None and noop["routing"] is not None else None

    return {
        "queried_alias_a": int(aid_a),
        "queried_pod_a": int(pid_a),
        "mutated_unrelated_alias_b": int(aid_b),
        "mutated_unrelated_pod_b": int(pid_b),
        "truth_a": truth_a,
        "old_b_object": old_b_obj,
        "new_b_object": new_b_obj,
        "a_only_lineage_dependencies": lineage_a.dependency_count,
        "a_only_lineage_still_current_after_b_update": bool(lineage_a.is_current(auth)),
        "a_witness_still_current_after_b_update": bool(auth.validate_witness(witness_a)),
        "b_old_witness_stale_after_b_update": not bool(auth.validate_witness(witness_b)),
        "candidate_before": cand_before,
        "candidate_after": cand_after,
        "candidate_before_correct": cand_before == truth_a,
        "candidate_after_correct": cand_after == truth_a,
        "full_logits_before_vs_after": _tensor_diff(before["full"], after["full"]),
        "hidden_before_vs_after": _tensor_diff(before["hidden"], after["hidden"]),
        "routing_before_vs_after": routing_diff,
        "kv_before_vs_after": cache_before_after,
        "stale_kv_continuation_vs_fresh": _tensor_diff(stale_cont, fresh_cont),
        "support_before": _support(before["routing"]),
        "support_after": _support(after["routing"]),
        "noop_full_logits": _tensor_diff(before["full"], noop["full"]),
        "noop_hidden": _tensor_diff(before["hidden"], noop["hidden"]),
        "noop_routing": noop_routing_diff,
        "noop_kv": cache_noop,
        "intervention_is_unrelated_canonical_payload_update": True,
    }


def run(seed: int, steps: int, groups: int) -> Dict[str, Any]:
    install_strict_contract()
    os.environ["SO_BOS"] = "1"
    torch.manual_seed(seed)
    cfg = AdapterConfig(status_gated=True, use_links=True, n_deref=E20.N_DEREF)
    gk = E8.GPT2Knowledge(cfg)
    trained = E81.train_symlink_consistent(
        gk,
        seed,
        steps,
        consistency=0.15,
        alt_supervision=0.5,
        n_groups=max(100, groups),
        verbose=True,
    )
    gk.model.eval()
    centre = np.asarray(trained["centre"])
    gate = _strict_reader_gate(gk, centre, seed, max(100, groups))
    row: Dict[str, Any] = {
        "seed": seed,
        "steps": steps,
        "bos": True,
        "consistency": 0.15,
        "alt_supervision": 0.5,
        "strict_marker_radius": 0.35,
        "reader_gate": gate,
        "attack_interpretable": bool(gate["strict_every_template_ge_095"]),
    }
    if not row["attack_interpretable"]:
        row["lineage_intervention"] = None
        row["decision"] = "CAPABILITY_GATE_FAILED_DO_NOT_INTERPRET_LINEAGE_ATTACK"
        return row

    attack = _lineage_intervention(gk, centre, seed, max(100, groups))
    row["lineage_intervention"] = attack
    neural_changed = (
        not bool(attack["full_logits_before_vs_after"]["byte_identical"])
        or not bool(attack["hidden_before_vs_after"]["byte_identical"])
        or (attack["kv_before_vs_after"]["available"] and not bool(attack["kv_before_vs_after"]["byte_identical"]))
        or not bool(attack["stale_kv_continuation_vs_fresh"]["byte_identical"])
    )
    controls_ok = (
        bool(attack["a_only_lineage_still_current_after_b_update"])
        and bool(attack["a_witness_still_current_after_b_update"])
        and bool(attack["b_old_witness_stale_after_b_update"])
        and bool(attack["noop_full_logits"]["byte_identical"])
        and bool(attack["noop_hidden"]["byte_identical"])
        and (attack["noop_routing"] is None or bool(attack["noop_routing"]["byte_identical"]))
        and (not attack["noop_kv"]["available"] or bool(attack["noop_kv"]["byte_identical"]))
    )
    row["selected_object_only_lineage_falsified"] = bool(neural_changed and controls_ok)
    row["controls_ok"] = bool(controls_ok)
    row["decision"] = (
        "SELECTED_OBJECT_ONLY_LINEAGE_UNSOUND_FOR_CURRENT_REAL_READER"
        if row["selected_object_only_lineage_falsified"]
        else "NOT_FALSIFIED_IN_THIS_SEED"
    )
    return row


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
    all_reader = all(bool(r["reader_gate"]["strict_every_template_ge_095"]) for r in rows)
    all_falsified = all(bool(r.get("selected_object_only_lineage_falsified", False)) for r in rows) if all_reader else False
    rec = {
        "experiment": "E-000091",
        "title": "real-reader exact-lineage density audit",
        "rows": rows,
        "all_requested_reader_seeds_strict_pass": all_reader,
        "all_interpretable_seeds_falsify_selected_object_only_lineage": all_falsified,
        "major_invention": False,
        "novelty_claim": False,
        "boundary": (
            "A positive falsification says only that A-only selected-object lineage is unsound for exact reuse under the current softmax reader. "
            "It does not prove that every sparse neural architecture is impossible and it does not award novelty to hard/top-k routing."
        ),
    }
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("e000091_real_reader_lineage_density_s" + "-".join(str(s) for s in args.seeds) + ".json")
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    if not all_reader:
        raise SystemExit(2)
    if not all_falsified:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
