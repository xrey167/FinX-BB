"""E-000080 -- object-scoped derived-state lineage after the E-000079 falsification.

Question
--------
Can we repair the stale-KV lifecycle hole without falling back to a global
cache epoch that discards every session whenever any knowledge object changes?

This experiment is a systems/correctness prototype, not a novelty claim.  It
compares two policies after one pod UPDATE:

* global epoch: every cached neural state is stale;
* object-scoped lineage: only state whose recorded alias+pod witness is no
  longer current is stale.

On two public causal-LM backbones, two independent prefills are constructed:
cache A depends on pod A and cache B depends on pod B.  Pod A is updated.  We
require that:

1. old A lineage becomes stale and old B lineage stays current;
2. unguarded reuse of old A KV materially differs from current-A recomputation;
3. rejecting/recomputing old A reproduces current-A logits;
4. reusing unchanged B KV reproduces a freshly rebuilt B cache;
5. therefore one unrelated object mutation need not globally flush B;
6. fan-out scaling separately shows one canonical pod mutation invalidates all
   k alias-derived witnesses without k alias edits;
7. locality scaling shows one pod mutation invalidates only its dependent state
   among many independent pods.

The payload injection is controlled and deterministic so this experiment does
not depend on trained symlink-reader capability and must not be counted as a
positive CAVI breakthrough.  Generic cache tags, dependency sets, generations,
validation and recomputation are established techniques and explicitly excluded
from novelty.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM

from so.cavi import CAVIAuthority
from so.derived_lineage import DerivedLineage, LineagedState
from so.llm_adapter import transformer_blocks


def _hidden(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden,) + tuple(output[1:])
    if torch.is_tensor(output):
        return hidden
    raise TypeError(f"unsupported block output type: {type(output).__name__}")


def _payload(model, target_id: int, rms: float) -> torch.Tensor:
    emb = model.get_output_embeddings()
    if emb is None:
        emb = model.get_input_embeddings()
    v = emb.weight[target_id].detach().float().clone()
    return v * (float(rms) / float(v.pow(2).mean().sqrt().clamp_min(1e-8)))


def _prefill(model, blocks, read_layers: Tuple[int, int], prompt_ids: torch.Tensor,
             payload: torch.Tensor):
    count = {"n": 0}
    handles = []
    for layer in read_layers:
        def inject(module, inputs, output, _layer=layer):
            h = _hidden(output)
            h2 = h.clone()
            h2[:, -1, :] = h2[:, -1, :] + payload.to(device=h.device, dtype=h.dtype)
            count["n"] += 1
            return _replace_hidden(output, h2)
        handles.append(blocks[layer].register_forward_hook(inject))
    try:
        with torch.no_grad():
            out = model(input_ids=prompt_ids, use_cache=True)
    finally:
        for h in handles:
            h.remove()
    if count["n"] != len(read_layers):
        raise RuntimeError(f"expected {len(read_layers)} memory writes, got {count['n']}")
    if out.past_key_values is None:
        raise RuntimeError("backbone did not return past_key_values")
    return out.past_key_values


def _prefill_no_memory(model, prompt_ids: torch.Tensor):
    with torch.no_grad():
        out = model(input_ids=prompt_ids, use_cache=True)
    if out.past_key_values is None:
        raise RuntimeError("backbone did not return past_key_values")
    return out.past_key_values


def _continue(model, past, prompt_len: int, continuation_id: torch.Tensor) -> torch.Tensor:
    mask = torch.ones((1, prompt_len + 1), dtype=torch.long)
    with torch.no_grad():
        out = model(
            input_ids=continuation_id,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
        )
    return out.logits[:, -1, :].detach().float()


def _maxabs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def _kl(reference: torch.Tensor, observed: torch.Tensor) -> float:
    p = torch.softmax(reference.float(), -1)
    return float((p * (torch.log_softmax(reference.float(), -1) - torch.log_softmax(observed.float(), -1))).sum(-1).mean())


def _fanout_scaling() -> List[Dict[str, object]]:
    rows = []
    for k in (1, 10, 100, 1000, 10000):
        auth = CAVIAuthority()
        auth.create_pod(1)
        for i in range(k):
            auth.create_alias(10_000 + i, 1)
        witnesses = [auth.witness(10_000 + i) for i in range(k)]
        alias_inc_before = auth.alias_incarnation(10_000)
        timings = []
        for _ in range(1000):
            t0 = time.perf_counter_ns()
            auth.update_pod(1)
            timings.append(time.perf_counter_ns() - t0)
        stale = sum(not auth.validate_witness(w) for w in witnesses)
        rows.append({
            "aliases_k": k,
            "canonical_lifecycle_operations": 1,
            "duplicated_edit_baseline_operations": k,
            "all_alias_derived_witnesses_stale": stale == k,
            "stale_witnesses": stale,
            "alias_incarnation_unchanged": auth.alias_incarnation(10_000) == alias_inc_before,
            "median_authority_update_us": statistics.median(timings) / 1e3,
            "packed_lineage_bytes_for_k_cached_alias_states": 32 * k,
        })
    return rows


def _locality_scaling() -> List[Dict[str, object]]:
    rows = []
    for n in (2, 10, 100, 1000, 5000):
        auth = CAVIAuthority()
        lineages = []
        for i in range(n):
            pid = i + 1
            aid = 100_000 + i
            auth.create_pod(pid)
            auth.create_alias(aid, pid)
            lineages.append(DerivedLineage.of(auth.witness(aid)))
        auth.update_pod(1)
        stale = sum(not lineage.is_current(auth) for lineage in lineages)
        rows.append({
            "independent_pods_and_cached_states": n,
            "object_scoped_stale_states": stale,
            "object_scoped_reusable_states": n - stale,
            "global_epoch_stale_states": n,
            "recompute_fraction_object_scoped": stale / n,
            "recompute_fraction_global_epoch": 1.0,
        })
    return rows


def run(model_name: str, seed: int, payload_rms: float) -> Dict[str, object]:
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    blocks = transformer_blocks(model)
    if len(blocks) < 5:
        raise ValueError(f"need >=5 blocks, got {len(blocks)}")
    read_layers = (1, min(3, len(blocks) - 2))
    if read_layers[0] == read_layers[1]:
        read_layers = (0, len(blocks) - 2)

    vocab = int(model.get_input_embeddings().weight.shape[0])
    prompt_a = torch.tensor([[17 % vocab, 103 % vocab, 227 % vocab, 331 % vocab, 443 % vocab]], dtype=torch.long)
    prompt_b = torch.tensor([[19 % vocab, 107 % vocab, 229 % vocab, 337 % vocab, 449 % vocab]], dtype=torch.long)
    cont_a = torch.tensor([[521 % vocab]], dtype=torch.long)
    cont_b = torch.tensor([[523 % vocab]], dtype=torch.long)
    old_a = _payload(model, 42 % vocab, payload_rms)
    new_a = _payload(model, 314 % vocab, payload_rms)
    b_payload = _payload(model, 271 % vocab, payload_rms)

    auth = CAVIAuthority()
    auth.create_pod(1)
    auth.create_pod(2)
    auth.create_alias(101, 1)
    auth.create_alias(201, 2)

    # Materialise old-generation neural state for two independent objects.
    cache_a = LineagedState(
        _prefill(model, blocks, read_layers, prompt_a, old_a),
        DerivedLineage.of(auth.witness(101)),
    )
    cache_b = LineagedState(
        _prefill(model, blocks, read_layers, prompt_b, b_payload),
        DerivedLineage.of(auth.witness(201)),
    )

    # One canonical object changes; no global epoch is advanced.
    auth.update_pod(1)
    a_stale = not cache_a.reusable(auth)
    b_still_current = cache_b.reusable(auth)
    new_a_lineage = DerivedLineage.of(auth.witness(101))

    # Attack: ignore A's stale lineage and reuse its old KV anyway.
    stale_a_logits = _continue(model, cache_a.payload, prompt_a.shape[1], cont_a)

    # Current-A gold and guarded repair: both use the new pod payload.  Separate
    # cache objects are required because HF DynamicCache may be mutated on decode.
    t0 = time.perf_counter_ns()
    current_a_gold_cache = _prefill(model, blocks, read_layers, prompt_a, new_a)
    current_a_logits = _continue(model, current_a_gold_cache, prompt_a.shape[1], cont_a)
    current_a_recompute_ms = (time.perf_counter_ns() - t0) / 1e6

    t0 = time.perf_counter_ns()
    repaired_a_cache = LineagedState(
        _prefill(model, blocks, read_layers, prompt_a, new_a), new_a_lineage
    )
    repaired_a_logits = _continue(model, repaired_a_cache.payload, prompt_a.shape[1], cont_a)
    repaired_a_ms = (time.perf_counter_ns() - t0) / 1e6

    # B is unrelated and its old cache remains authorized.  Compare direct reuse
    # with a separately rebuilt B cache under exactly the same current state.
    t0 = time.perf_counter_ns()
    reused_b_logits = _continue(model, cache_b.payload, prompt_b.shape[1], cont_b)
    b_reuse_ms = (time.perf_counter_ns() - t0) / 1e6

    t0 = time.perf_counter_ns()
    fresh_b_cache = _prefill(model, blocks, read_layers, prompt_b, b_payload)
    fresh_b_logits = _continue(model, fresh_b_cache, prompt_b.shape[1], cont_b)
    b_recompute_ms = (time.perf_counter_ns() - t0) / 1e6

    stale_a_vs_current = _maxabs(stale_a_logits, current_a_logits)
    repaired_a_vs_current = _maxabs(repaired_a_logits, current_a_logits)
    reused_b_vs_fresh = _maxabs(reused_b_logits, fresh_b_logits)
    checks = {
        "updated_object_lineage_stale": a_stale,
        "unrelated_object_lineage_current": b_still_current,
        "stale_A_KV_materially_differs_from_current_A": stale_a_vs_current > 1e-4,
        "lineage_reject_plus_recompute_matches_current_A": repaired_a_vs_current < 5e-3,
        "unchanged_B_KV_reuse_matches_fresh_B": reused_b_vs_fresh < 5e-3,
        "fresh_A_lineage_current": repaired_a_cache.reusable(auth),
        "selective_policy_avoids_unrelated_B_recompute": b_still_current,
    }
    return {
        "model": model_name,
        "seed": seed,
        "read_layers": list(read_layers),
        "checks": checks,
        "pass": all(checks.values()),
        "stale_A_vs_current_A_maxabs": stale_a_vs_current,
        "stale_A_vs_current_A_kl_nats": _kl(current_a_logits, stale_a_logits),
        "repaired_A_vs_current_A_maxabs": repaired_a_vs_current,
        "reused_B_vs_fresh_B_maxabs": reused_b_vs_fresh,
        "current_A_top1": int(current_a_logits.argmax(-1)[0]),
        "stale_A_top1": int(stale_a_logits.argmax(-1)[0]),
        "reused_B_top1": int(reused_b_logits.argmax(-1)[0]),
        "fresh_B_top1": int(fresh_b_logits.argmax(-1)[0]),
        "B_cache_reuse_ms": b_reuse_ms,
        "B_full_recompute_plus_decode_ms": b_recompute_ms,
        "B_recompute_over_reuse_ratio": b_recompute_ms / max(b_reuse_ms, 1e-9),
        "A_current_recompute_ms": current_a_recompute_ms,
        "A_guarded_repair_ms": repaired_a_ms,
        "lineage_metadata_bytes_per_single_dependency_cache": cache_b.lineage.packed_metadata_bytes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--payload-rms", type=float, default=4.0)
    ap.add_argument("--results-dir", default="so/results")
    a = ap.parse_args()
    rows = [run(a.model, seed, a.payload_rms) for seed in a.seeds]
    fanout = _fanout_scaling()
    locality = _locality_scaling()
    all_pass = all(bool(row["pass"]) for row in rows)
    rec = {
        "experiment": "E-000080",
        "all_pass": all_pass,
        "model_rows": rows,
        "fanout_scaling": fanout,
        "locality_scaling": locality,
        "interpretation": (
            "Object-scoped alias+pod lineage repairs the E-000079 stale-KV hole in this controlled setting while preserving "
            "unrelated KV reuse. One canonical pod update also makes every alias-derived old witness stale without editing aliases."
        ),
        "not_claimed": (
            "No novelty claim for dependency/version tags, cache validation, invalidation, recomputation, snapshots, generations, "
            "capabilities, pointers or KV editing. Controlled residual payloads mean this is not positive real-symlink evidence."
        ),
    }
    out_dir = Path(a.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / ("e000080_object_scoped_kv_lineage_" + a.model.replace("/", "_") + ".json")
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
