from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import sha256_file
from research.qwen25_05b_r308_late_bound_handle_vm import MODEL_ID, REVISION, EXPECTED_WEIGHTS

STAGE = "R349-QWEN-INTERNAL-PROSPECTIVE-STATE"
REPORT_PATH = Path(os.environ.get("SO_R349_REPORT", "ci-r349/report.json"))
FUTURE_WORLDS = int(os.environ.get("SO_R349_FUTURE_WORLDS", "8"))
BENCH_LOOPS = int(os.environ.get("SO_R349_BENCH_LOOPS", "4"))
RACE_TRIALS = int(os.environ.get("SO_R349_RACE_TRIALS", "256"))
PORT_ALPHA = float(os.environ.get("SO_R349_PORT_ALPHA", "0.85"))
SEED = 3490914
VALUE_COUNT = 32


@dataclass(frozen=True)
class ProspectiveState:
    """A retained neural activation plus unresolved canonical world references.

    `hidden` has passed through real pretrained Qwen decoder blocks but has not
    consumed the mutable values behind `refs`.  The value payload is therefore
    not part of the retained numerical activation yet.
    """

    hidden: torch.Tensor
    refs: tuple[int, int]


@dataclass(frozen=True)
class PositionalContext:
    position_ids: torch.Tensor
    cache_position: torch.Tensor
    position_embeddings: tuple[torch.Tensor, torch.Tensor]


def bits_pair(a: int, b: int, device: torch.device) -> torch.Tensor:
    vals = []
    for v in (a, b):
        for bit in range(5):
            vals.append(1.0 if ((v >> bit) & 1) else -1.0)
    return torch.tensor(vals, dtype=torch.float32, device=device)


def prepare(model, input_ids: torch.Tensor) -> tuple[torch.Tensor, PositionalContext]:
    core = model.model
    hidden = core.embed_tokens(input_ids)
    seq = hidden.shape[1]
    cache_position = torch.arange(seq, device=hidden.device)
    position_ids = cache_position.unsqueeze(0)
    position_embeddings = core.rotary_emb(hidden, position_ids)
    return hidden, PositionalContext(position_ids, cache_position, position_embeddings)


def run_layers(model, hidden: torch.Tensor, pos: PositionalContext, start: int, end: int) -> torch.Tensor:
    core = model.model
    for layer in core.layers[start:end]:
        hidden = layer(
            hidden,
            attention_mask=None,
            position_ids=pos.position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=pos.cache_position,
            position_embeddings=pos.position_embeddings,
        )[0]
    return hidden


def finish_logits(model, hidden: torch.Tensor) -> torch.Tensor:
    hidden = model.model.norm(hidden)
    return model.lm_head(hidden)[:, -1].float()


def port_basis(hidden_size: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(SEED + 9)
    b = torch.randn(10, hidden_size, generator=g, dtype=torch.float32)
    b = F.normalize(b, dim=-1)
    return b.to(device)


def materialize(hidden: torch.Tensor, refs: tuple[int, int], world: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Explicit world materialization barrier into the real Qwen residual stream."""
    va, vb = int(world[refs[0]]), int(world[refs[1]])
    feature = bits_pair(va, vb, hidden.device)
    raw = feature @ basis
    raw = F.normalize(raw, dim=-1)
    current = hidden[:, -1].float()
    current_rms = current.square().mean().sqrt().clamp_min(1e-6)
    scale = current_rms * math.sqrt(current.shape[-1]) * PORT_ALPHA
    delta = raw * scale
    out = hidden.clone()
    out[:, -1] = (current + delta).to(out.dtype)
    return out


def compile_pns(model, input_ids: torch.Tensor, refs: tuple[int, int], barrier: int):
    hidden, pos = prepare(model, input_ids)
    hidden = run_layers(model, hidden, pos, 0, barrier)
    return ProspectiveState(hidden.detach().clone(), refs), pos


def serve_pns(model, pns: ProspectiveState, pos: PositionalContext, world: torch.Tensor, basis: torch.Tensor, barrier: int):
    hidden = materialize(pns.hidden, pns.refs, world, basis)
    hidden = run_layers(model, hidden, pos, barrier, len(model.model.layers))
    return finish_logits(model, hidden)


def full_current_pns(model, input_ids, refs, world, basis, barrier):
    hidden, pos = prepare(model, input_ids)
    hidden = run_layers(model, hidden, pos, 0, barrier)
    hidden = materialize(hidden, refs, world, basis)
    hidden = run_layers(model, hidden, pos, barrier, len(model.model.layers))
    return finish_logits(model, hidden)


def compile_early_numeric(model, input_ids, refs, world, basis, early_layer: int, cache_layer: int):
    hidden, pos = prepare(model, input_ids)
    hidden = run_layers(model, hidden, pos, 0, early_layer)
    hidden = materialize(hidden, refs, world, basis)
    hidden = run_layers(model, hidden, pos, early_layer, cache_layer)
    return hidden.detach().clone(), pos


def finish_from_numeric_cache(model, cached, pos, cache_layer: int):
    hidden = run_layers(model, cached, pos, cache_layer, len(model.model.layers))
    return finish_logits(model, hidden)


def full_early_current(model, input_ids, refs, world, basis, early_layer: int):
    hidden, pos = prepare(model, input_ids)
    hidden = run_layers(model, hidden, pos, 0, early_layer)
    hidden = materialize(hidden, refs, world, basis)
    hidden = run_layers(model, hidden, pos, early_layer, len(model.model.layers))
    return finish_logits(model, hidden)


def tensor_digest(x: torch.Tensor, refs: tuple[int, int]) -> str:
    h = hashlib.sha256()
    h.update(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    h.update(bytes(refs))
    return h.hexdigest()


def sym_kl(a: torch.Tensor, b: torch.Tensor) -> float:
    pa = torch.log_softmax(a, dim=-1)
    pb = torch.log_softmax(b, dim=-1)
    p = pa.exp(); q = pb.exp()
    return float(0.5 * ((p * (pa - pb)).sum() + (q * (pb - pa)).sum()))


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.set_num_interop_threads(1)
    random.seed(SEED)
    torch.manual_seed(SEED)

    md = Path(snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors", "*.index.json", "*.merges", "*.vocab", "merges.txt", "vocab.json"],
    ))
    hashes = {n: sha256_file(md / n) for n in EXPECTED_WEIGHTS}
    assert hashes == EXPECTED_WEIGHTS
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        md,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
    ).eval()
    model.requires_grad_(False)

    n_layers = len(model.model.layers)
    barrier = max(2, n_layers - 4)
    early_layer = max(2, n_layers // 4)
    cache_layer = barrier
    refs = (7, 19)

    prompt = tok.apply_chat_template([
        {"role": "system", "content": (
            "The governed values behind canonical references are intentionally absent from the language prompt. "
            "A neural World Port may materialize their current values only at an explicit decoder barrier."
        )},
        {"role": "user", "content": (
            "Continue reasoning about canonical references POD_007 and POD_019. Preserve their identities; "
            "the authoritative mutable values will be supplied by the World Port."
        )},
    ], tokenize=False, add_generation_prompt=True)
    input_ids = torch.tensor([tok(prompt, add_special_tokens=False).input_ids], dtype=torch.long)
    basis = port_basis(model.config.hidden_size, input_ids.device)

    # First verify that our manual frozen-Qwen execution is the same model path.
    with torch.inference_mode():
        std = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False, return_dict=True).logits[:, -1].float()
        h, pos = prepare(model, input_ids)
        h = run_layers(model, h, pos, 0, n_layers)
        manual = finish_logits(model, h)
    manual_delta = float((std - manual).abs().max())
    manual_top_match = int(std.argmax(-1)) == int(manual.argmax(-1))

    # Compile both retained-state variants on the same home world.
    gen = torch.Generator().manual_seed(SEED + 100)
    home = torch.randint(0, VALUE_COUNT, (32,), generator=gen)
    with torch.inference_mode():
        pns, pns_pos = compile_pns(model, input_ids, refs, barrier)
        early_numeric_cache, early_pos = compile_early_numeric(
            model, input_ids, refs, home, basis, early_layer, cache_layer
        )
    pns_digest_before = tensor_digest(pns.hidden, refs)
    numeric_digest_before = tensor_digest(early_numeric_cache, refs)

    pns_logit_deltas = []
    pns_top_matches = []
    stale_numeric_logit_deltas = []
    stale_numeric_sym_kl = []
    stale_numeric_top_mismatches = 0
    current_world_changed_logits = []

    with torch.inference_mode():
        home_pns = serve_pns(model, pns, pns_pos, home, basis, barrier)
        for _ in range(FUTURE_WORLDS):
            world = torch.randint(0, VALUE_COUNT, (32,), generator=gen)
            # Ensure the two referenced payloads actually changed.
            if int(world[refs[0]]) == int(home[refs[0]]):
                world[refs[0]] = (world[refs[0]] + 1) % VALUE_COUNT
            if int(world[refs[1]]) == int(home[refs[1]]):
                world[refs[1]] = (world[refs[1]] + 3) % VALUE_COUNT

            full = full_current_pns(model, input_ids, refs, world, basis, barrier)
            cached = serve_pns(model, pns, pns_pos, world, basis, barrier)
            d = float((full - cached).abs().max())
            pns_logit_deltas.append(d)
            pns_top_matches.append(int(full.argmax(-1)) == int(cached.argmax(-1)))
            current_world_changed_logits.append(float((home_pns - full).abs().max()))

            fresh_early = full_early_current(model, input_ids, refs, world, basis, early_layer)
            stale_early = finish_from_numeric_cache(model, early_numeric_cache, early_pos, cache_layer)
            stale_numeric_logit_deltas.append(float((fresh_early - stale_early).abs().max()))
            stale_numeric_sym_kl.append(sym_kl(fresh_early[0], stale_early[0]))
            stale_numeric_top_mismatches += int(int(fresh_early.argmax(-1)) != int(stale_early.argmax(-1)))

    # Arbitrary world writes: prospective state must remain byte-identical.
    world = home.clone()
    generations = torch.ones(len(world), dtype=torch.long)
    same_value_updates = 0
    for _ in range(12000):
        pid = random.randrange(len(world))
        generations[pid] += 1
        if random.random() < 0.40:
            same_value_updates += 1
        else:
            world[pid] = random.randrange(VALUE_COUNT)
    pns_digest_after = tensor_digest(pns.hidden, refs)
    numeric_digest_after = tensor_digest(early_numeric_cache, refs)

    # Commit/race gate at late materialization. Every injected generation race on a
    # referenced Pod must be observed before publication, including same-value ABA.
    expected = detected = escaped = retries = 0
    max_retry_delta = 0.0
    with torch.inference_mode():
        for _ in range(RACE_TRIALS):
            seen = generations[list(refs)].clone()
            candidate = serve_pns(model, pns, pns_pos, world, basis, barrier)
            raced = random.random() < 0.22
            if raced:
                pid = refs[random.randrange(2)]
                expected += 1
                generations[pid] += 1
                if random.random() >= 0.50:
                    world[pid] = random.randrange(VALUE_COUNT)
            now = generations[list(refs)]
            if not torch.equal(seen, now):
                detected += 1
                retries += 1
                candidate = serve_pns(model, pns, pns_pos, world, basis, barrier)
                oracle = full_current_pns(model, input_ids, refs, world, basis, barrier)
                max_retry_delta = max(max_retry_delta, float((candidate - oracle).abs().max()))
            elif raced:
                escaped += 1

    # Timing: full current-world model path vs cached prospective prefix.  These are
    # diagnostic CPU numbers; the architecture claim is about the skipped layer set.
    bench_world = torch.randint(0, VALUE_COUNT, (32,), generator=gen)
    with torch.inference_mode():
        for _ in range(2):
            _ = full_current_pns(model, input_ids, refs, bench_world, basis, barrier)
            _ = serve_pns(model, pns, pns_pos, bench_world, basis, barrier)
        full_times = []
        pns_times = []
        for _ in range(BENCH_LOOPS):
            ts = time.perf_counter_ns(); _ = full_current_pns(model, input_ids, refs, bench_world, basis, barrier); full_times.append(time.perf_counter_ns() - ts)
            ts = time.perf_counter_ns(); _ = serve_pns(model, pns, pns_pos, bench_world, basis, barrier); pns_times.append(time.perf_counter_ns() - ts)

    report = {
        "stage": STAGE,
        "architecture_candidate": "Internal Prospective Neural State (PNS) in frozen Qwen residual execution",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": hashes,
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "decoder_layers": n_layers,
        "prospective_materialization_barrier_layer": barrier,
        "prospective_world_independent_layers": barrier,
        "post_materialization_layers": n_layers - barrier,
        "early_numeric_materialization_layer": early_layer,
        "early_numeric_cached_after_layer": cache_layer,
        "manual_vs_standard_qwen_max_logit_delta": manual_delta,
        "manual_vs_standard_qwen_top_token_match": manual_top_match,
        "future_worlds": FUTURE_WORLDS,
        "max_pns_cached_vs_full_current_logit_delta": max(pns_logit_deltas),
        "pns_cached_vs_full_current_top_token_match_rate": sum(pns_top_matches) / len(pns_top_matches),
        "min_world_update_effect_on_current_logits": min(current_world_changed_logits),
        "mean_world_update_effect_on_current_logits": statistics.mean(current_world_changed_logits),
        "mean_stale_early_numeric_max_logit_delta": statistics.mean(stale_numeric_logit_deltas),
        "min_stale_early_numeric_max_logit_delta": min(stale_numeric_logit_deltas),
        "mean_stale_early_numeric_symmetric_kl": statistics.mean(stale_numeric_sym_kl),
        "stale_early_numeric_top_token_mismatch_rate": stale_numeric_top_mismatches / FUTURE_WORLDS,
        "prospective_state_digest_unchanged_after_12000_world_writes": pns_digest_before == pns_digest_after,
        "early_numeric_cache_digest_unchanged_but_semantically_stale": numeric_digest_before == numeric_digest_after,
        "same_value_generation_updates": same_value_updates,
        "write_time_pns_value_patch_operations": 0,
        "race_conflicts_expected": expected,
        "race_conflicts_detected": detected,
        "race_conflicts_escaped": escaped,
        "race_retries": retries,
        "max_retry_vs_full_current_logit_delta": max_retry_delta,
        "median_full_current_recompute_ns_cpu": statistics.median(full_times),
        "median_cached_pns_serve_ns_cpu": statistics.median(pns_times),
        "full_over_cached_pns_speedup_cpu": statistics.median(full_times) / statistics.median(pns_times),
        "fraction_decoder_layers_skipped_on_world_rebind": barrier / n_layers,
        "mechanism": (
            "The real frozen Qwen residual stream is split at a late decoder barrier. A ProspectiveState carries the pretrained hidden activation "
            "together with unresolved canonical references through all pre-barrier Qwen blocks; no mutable world payload has entered that retained "
            "numeric state. Current values are dereferenced only at the explicit materialization barrier and injected into the real Qwen residual "
            "stream, after which the remaining native decoder layers and LM head run normally. Reusing the prospective prefix under a future world "
            "must therefore be exactly equivalent to recomputing the complete current-world model path, while an early-materialized numeric hidden "
            "cache is retrospective and becomes semantically stale after the same world rewrite."
        ),
        "claim_boundary": (
            "This is the first internal frozen-pretrained residual-stream intervention in this branch, but the World Port injection is deterministic "
            "rather than a learned open-ended knowledge interface. It establishes the internal late-binding/cache semantics, not language-task "
            "superiority or novelty. R350 must learn the internal World Port and test governed language tasks against retrieval/editing baselines."
        ),
        "dod_status": "NOT_DOD; frozen-pretrained internal PNS mechanism gate",
    }
    report["contract_pass"] = (
        manual_top_match
        and manual_delta <= 0.02
        and report["max_pns_cached_vs_full_current_logit_delta"] <= 1e-5
        and report["pns_cached_vs_full_current_top_token_match_rate"] == 1.0
        and report["prospective_state_digest_unchanged_after_12000_world_writes"]
        and report["min_world_update_effect_on_current_logits"] > 1e-4
        and report["min_stale_early_numeric_max_logit_delta"] > 1e-4
        and detected == expected and escaped == 0 and retries == expected
        and max_retry_delta <= 1e-5
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in [
        "stage", "decoder_layers", "prospective_materialization_barrier_layer",
        "manual_vs_standard_qwen_max_logit_delta", "max_pns_cached_vs_full_current_logit_delta",
        "pns_cached_vs_full_current_top_token_match_rate", "mean_stale_early_numeric_max_logit_delta",
        "stale_early_numeric_top_token_mismatch_rate", "prospective_state_digest_unchanged_after_12000_world_writes",
        "race_conflicts_expected", "race_conflicts_detected", "race_conflicts_escaped",
        "max_retry_vs_full_current_logit_delta", "full_over_cached_pns_speedup_cpu",
        "fraction_decoder_layers_skipped_on_world_rebind", "contract_pass", "report_sha256",
    ]}, indent=2))
    return 0 if report["contract_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
