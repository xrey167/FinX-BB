"""E-000079 -- autoregressive KV-lineage race against the E-000078 boundary.

Purpose
-------
E-000075 showed that validating authority independently at multiple neural read
sites permits a torn generation inside one forward. E-000078 therefore holds a
single authority snapshot from the first configured memory read through the
last configured memory read.

That is necessary, but a live decoder materialises *derived neural state* after
those reads and reuses it in later token forwards. In particular, a residual
memory write after block i changes hidden state entering later blocks, so their
K/V tensors can encode that memory. If the lifecycle mutates after the final
memory-read hook but before the downstream blocks finish, the current E-000078
guard has already released its authority lock. The forward can therefore return
KV state *after revocation* that was causally derived from the revoked memory.
A later token forward may reuse that KV while performing no memory read at all.

This experiment tests exactly that boundary on public causal-LM backbones. It
uses a controlled residual payload instead of a trained symlink reader so the
result is independent of reader accuracy. It is a contract falsification only:
cache epochs, invalidation, recomputation, snapshots and generations are
established mechanisms and are NOT claimed as novelty.

Registered prediction for the current E-000078 contract:
  1. two residual memory reads occur under one ForwardSnapshotConsumptionGuard;
  2. a lifecycle mutation launched immediately after the final read hook can
     commit before the next transformer block executes;
  3. the prefill returns past_key_values only after that mutation;
  4. a subsequent post-revoke forward injects no memory, yet reuse of that stale
     KV changes its logits relative to a full current-state recomputation;
  5. a clean, current-generation KV cache matches full recomputation (control).

If these hold, E-000078 must NOT be promoted as a sufficient live-generation
transaction boundary. The architecture must extend pod/incarnation lineage to
all reusable neural-derived state crossing a forward boundary (KV, hidden or
activation caches) and force current-lineage recomputation/refresh when needed.

Run examples:
  python -m so.experiments.e000079_generation_kv_lineage_race --model distilgpt2
  python -m so.experiments.e000079_generation_kv_lineage_race --model EleutherAI/pythia-70m-deduped
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from so.cavi_snapshot import ForwardSnapshotConsumptionGuard
from so.llm_adapter import transformer_blocks


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden,) + tuple(output[1:])
    if torch.is_tensor(output):
        return hidden
    # Modern HF model blocks used here return tensors/tuples. Fail loudly rather
    # than silently not injecting on an unknown backbone contract.
    raise TypeError(f"unsupported block output type: {type(output).__name__}")


def _hidden(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _kl_nats(reference_logits: torch.Tensor, observed_logits: torch.Tensor) -> float:
    p = torch.softmax(reference_logits.float(), dim=-1)
    logp = torch.log_softmax(reference_logits.float(), dim=-1)
    logq = torch.log_softmax(observed_logits.float(), dim=-1)
    return float((p * (logp - logq)).sum(-1).mean())


def run(model_name: str, seed: int, payload_rms: float) -> Dict[str, object]:
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    blocks = transformer_blocks(model)
    if len(blocks) < 5:
        raise ValueError(f"{model_name}: need >=5 decoder blocks, got {len(blocks)}")
    # Leave at least one downstream block after the final memory read so it can
    # materialise KV from the injected residual after the authority lock releases.
    read_layers = (1, min(3, len(blocks) - 2))
    if read_layers[0] == read_layers[1]:
        read_layers = (0, len(blocks) - 2)

    vocab = int(model.get_input_embeddings().weight.shape[0])
    prompt_ids = torch.tensor([[11 % vocab, 101 % vocab, 211 % vocab, 307 % vocab, 401 % vocab]], dtype=torch.long)
    continuation_id = torch.tensor([[509 % vocab]], dtype=torch.long)
    target_id = int(42 % vocab)

    out_emb = model.get_output_embeddings()
    if out_emb is None:
        payload = model.get_input_embeddings().weight[target_id].detach().float().clone()
    else:
        payload = out_emb.weight[target_id].detach().float().clone()
    rms = payload.pow(2).mean().sqrt().clamp_min(1e-8)
    payload = payload * (float(payload_rms) / float(rms))

    lock = threading.RLock()
    authority = {"live": True, "generation": 1, "mutation_ns": None}
    mutated = threading.Event()
    downstream_seen = threading.Event()
    timing = {"downstream_ns": None, "prefill_return_ns": None}
    adapter = SimpleNamespace(
        lm=model,
        cfg=SimpleNamespace(read_layers=read_layers),
        _ctx={"allowed": torch.tensor([True], dtype=torch.bool)},
    )
    injection_count = {"n": 0}

    # Register the real memory writes BEFORE the guard, matching KnowledgeAdapterLM
    # construction followed later by ForwardSnapshotConsumptionGuard construction.
    inject_handles = []
    for layer in read_layers:
        def inject(module, inputs, output, _layer=layer):
            ctx = adapter._ctx
            if ctx is None or not bool(ctx["allowed"][0]):
                return None
            h = _hidden(output)
            h2 = h.clone()
            h2[:, -1, :] = h2[:, -1, :] + payload.to(device=h.device, dtype=h.dtype)
            injection_count["n"] += 1
            return _replace_hidden(output, h2)
        inject_handles.append(blocks[layer].register_forward_hook(inject))

    def mask_fn() -> np.ndarray:
        return np.asarray([bool(authority["live"])], dtype=bool)

    guard = ForwardSnapshotConsumptionGuard(adapter, mask_fn, lock)

    def mutate() -> None:
        with lock:
            authority["live"] = False
            authority["generation"] += 1
            authority["mutation_ns"] = time.perf_counter_ns()
            mutated.set()

    mutator_box: Dict[str, threading.Thread] = {}

    # Registered AFTER guard._end on the final read layer: by the time this hook
    # runs the current E-000078 lock has been released. Wait for the mutation to
    # make the race deterministic, then let later blocks run.
    def trigger_after_last_read(module, inputs, output):
        th = threading.Thread(target=mutate, daemon=True)
        mutator_box["thread"] = th
        th.start()
        if not mutated.wait(5.0):
            raise RuntimeError("lifecycle mutation could not commit after final read")
        return None

    trigger_handle = blocks[read_layers[-1]].register_forward_hook(trigger_after_last_read)

    def mark_downstream(module, inputs):
        timing["downstream_ns"] = time.perf_counter_ns()
        downstream_seen.set()
        return None

    downstream_handle = blocks[read_layers[-1] + 1].register_forward_pre_hook(mark_downstream)

    try:
        with torch.no_grad():
            # Old generation is read atomically at two sites. The mutation commits
            # immediately after the final read, before downstream cache materialisation.
            live_prefill = model(input_ids=prompt_ids, use_cache=True)
            timing["prefill_return_ns"] = time.perf_counter_ns()
        th = mutator_box.get("thread")
        if th is not None:
            th.join(timeout=5.0)
            if th.is_alive():
                raise RuntimeError("mutator did not finish")

        stale_past = live_prefill.past_key_values
        if stale_past is None:
            raise RuntimeError(f"{model_name} did not return past_key_values")

        # REVOKED/current state. No neural-memory read occurs in either continuation.
        adapter._ctx = None
        full_mask = torch.ones((1, prompt_ids.shape[1] + 1), dtype=torch.long)
        with torch.no_grad():
            stale_cont = model(
                input_ids=continuation_id,
                attention_mask=full_mask,
                past_key_values=stale_past,
                use_cache=True,
            )
            # Current-state gold: recompute the whole context after revoke.
            full_ids = torch.cat([prompt_ids, continuation_id], dim=1)
            fresh_current = model(input_ids=full_ids, attention_mask=full_mask, use_cache=True)

            # Control: a cache built under the current (no-memory) state should agree
            # with full recomputation to normal numerical tolerance.
            clean_prefill = model(input_ids=prompt_ids, use_cache=True)
            clean_cont = model(
                input_ids=continuation_id,
                attention_mask=full_mask,
                past_key_values=clean_prefill.past_key_values,
                use_cache=True,
            )

        stale_logits = stale_cont.logits[:, -1, :].float()
        fresh_logits = fresh_current.logits[:, -1, :].float()
        clean_logits = clean_cont.logits[:, -1, :].float()
        stale_vs_fresh = float((stale_logits - fresh_logits).abs().max())
        clean_vs_fresh = float((clean_logits - fresh_logits).abs().max())
        target_delta = float(stale_logits[0, target_id] - fresh_logits[0, target_id])
        target_prob_stale = float(torch.softmax(stale_logits, -1)[0, target_id])
        target_prob_fresh = float(torch.softmax(fresh_logits, -1)[0, target_id])
        mutation_ns = authority["mutation_ns"]
        downstream_ns = timing["downstream_ns"]
        return_ns = timing["prefill_return_ns"]

        checks = {
            "two_memory_reads_one_forward": injection_count["n"] == 2,
            "mutation_committed": bool(mutated.is_set()),
            "mutation_after_guard_before_downstream": (
                mutation_ns is not None and downstream_ns is not None and mutation_ns <= downstream_ns
            ),
            "mutation_before_prefill_return": (
                mutation_ns is not None and return_ns is not None and mutation_ns < return_ns
            ),
            "post_revoke_forward_has_no_memory_context": adapter._ctx is None,
            "stale_kv_changes_post_revoke_logits": stale_vs_fresh > 1e-4,
            "clean_current_kv_matches_full_recompute": clean_vs_fresh < 5e-3,
        }
        return {
            "model": model_name,
            "seed": seed,
            "n_blocks": len(blocks),
            "read_layers": list(read_layers),
            "payload_rms": payload_rms,
            "target_id": target_id,
            "checks": checks,
            "pass": all(checks.values()),
            "post_revoke_stale_kv_vs_current_maxabs": stale_vs_fresh,
            "post_revoke_stale_kv_vs_current_kl_nats": _kl_nats(fresh_logits, stale_logits),
            "clean_kv_vs_full_recompute_maxabs": clean_vs_fresh,
            "target_logit_delta_stale_minus_current": target_delta,
            "target_probability_stale": target_prob_stale,
            "target_probability_current": target_prob_fresh,
            "stale_top1": int(stale_logits.argmax(-1)[0]),
            "current_top1": int(fresh_logits.argmax(-1)[0]),
            "mutation_generation": int(authority["generation"]),
            "mutation_to_downstream_us": (
                None if mutation_ns is None or downstream_ns is None else (downstream_ns - mutation_ns) / 1e3
            ),
            "mutation_to_prefill_return_us": (
                None if mutation_ns is None or return_ns is None else (return_ns - mutation_ns) / 1e3
            ),
        }
    finally:
        trigger_handle.remove()
        downstream_handle.remove()
        guard.close()
        for h in inject_handles:
            h.remove()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--payload-rms", type=float, default=4.0)
    ap.add_argument("--results-dir", default="so/results")
    a = ap.parse_args()
    rows: List[Dict[str, object]] = [run(a.model, s, a.payload_rms) for s in a.seeds]
    all_pass = all(bool(r["pass"]) for r in rows)
    rec = {
        "experiment": "E-000079",
        "result": "E-000078 boundary falsified for autoregressive derived-state reuse" if all_pass else "inconclusive",
        "all_pass": all_pass,
        "rows": rows,
        "interpretation": (
            "Per-forward first-read-to-last-read snapshot isolation is necessary but not sufficient for live autoregressive generation: "
            "memory-derived KV can cross a lifecycle boundary and be consumed by a later no-memory forward. The required architecture "
            "must attach pod/incarnation lineage to reusable neural-derived state and recompute/refresh state whose lineage is no longer current."
        ),
        "not_claimed": (
            "No novelty claim for snapshot isolation, cache invalidation, generations, capabilities, KV caching, or recomputation. "
            "This is a negative/falsification result and does not bypass the >=0.95 three-seed real-symlink capability gate for positive CAVI claims."
        ),
    }
    p = Path(a.results_dir)
    p.mkdir(parents=True, exist_ok=True)
    out = p / ("e000079_generation_kv_lineage_race_" + a.model.replace("/", "_") + ".json")
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
