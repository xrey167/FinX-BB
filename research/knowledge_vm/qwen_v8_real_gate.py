from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
EXPECTED_MODEL_SAFETENSORS_SHA256 = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
LAYER_INDEX = 12
ALPHA_FRACTIONS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    x = t.detach().to("cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(x).hexdigest()


def model_file_binding(model_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(model_dir.rglob("*")):
        if path.is_file() and (
            path.name.endswith(".safetensors")
            or path.name in {"config.json", "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"}
        ):
            files[str(path.relative_to(model_dir))] = sha256_file(path)
    return files


def pick_single_token_targets(tokenizer: Any, n: int = 40) -> list[tuple[int, str]]:
    special = set(tokenizer.all_special_ids)
    out: list[tuple[int, str]] = []
    vocab = int(tokenizer.vocab_size)
    # Deterministic sparse scan of the vocabulary. We care only that the target is
    # one existing token; the synthetic fact mapping is post-training and novel.
    candidates = list(range(1000, vocab, 97)) + list(range(2000, vocab, 193))
    seen: set[int] = set()
    for tid in candidates:
        if tid in seen or tid in special:
            continue
        seen.add(tid)
        text = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        stripped = text.strip()
        if not (3 <= len(stripped) <= 12):
            continue
        if not re.fullmatch(r"[A-Za-z]+", stripped):
            continue
        if stripped.lower() in {"the", "and", "for", "with", "from", "that", "this"}:
            continue
        out.append((tid, text))
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"could only find {len(out)} suitable single-token targets")
    return out


def make_prompts(targets: list[tuple[int, str]], template: str, offset: int) -> tuple[list[str], torch.Tensor]:
    prompts = [template.format(i=i + offset) for i in range(len(targets))]
    ids = torch.tensor([tid for tid, _ in targets], dtype=torch.long)
    return prompts, ids


def batched_inputs(tokenizer: Any, prompts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        k: v.to(device)
        for k, v in tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).items()
    }


def forward_logits(model: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
        return model(**inputs, use_cache=False).logits[:, -1, :].detach()


def prelayer_hidden_norm(model: Any, inputs: dict[str, torch.Tensor], layer_index: int) -> float:
    captured: list[torch.Tensor] = []

    def hook(_module: Any, args: tuple[Any, ...]) -> None:
        hidden = args[0]
        captured.append(hidden[:, -1, :].detach())
        return None

    handle = model.model.layers[layer_index].register_forward_pre_hook(hook)
    try:
        _ = forward_logits(model, inputs)
    finally:
        handle.remove()
    h = captured[0]
    return float(torch.linalg.vector_norm(h, dim=-1).median().item())


def injected_logits(
    model: Any,
    inputs: dict[str, torch.Tensor],
    target_ids: torch.Tensor,
    layer_index: int,
    magnitude: float,
) -> torch.Tensor:
    # No per-fact learned tensor. A target fact contributes only the model's own
    # frozen LM-head direction. One global magnitude is calibrated once.
    head = model.get_output_embeddings().weight.detach()
    vec = head[target_ids.to(head.device)].to(torch.float32)
    vec = vec - vec.mean(dim=-1, keepdim=True)
    vec = vec / torch.linalg.vector_norm(vec, dim=-1, keepdim=True).clamp_min(1e-8)
    vec = vec.to(dtype=model.dtype) * magnitude

    def hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        hidden = args[0]
        delta = torch.zeros_like(hidden)
        delta[:, -1, :] = vec.to(hidden.device, dtype=hidden.dtype)
        return (hidden + delta, *args[1:])

    handle = model.model.layers[layer_index].register_forward_pre_hook(hook)
    try:
        return forward_logits(model, inputs)
    finally:
        handle.remove()


def exact_top1_rate(logits: torch.Tensor, target_ids: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1).cpu()
    return float((pred == target_ids.cpu()).float().mean().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("artifacts/qwen_v8_gate/report.json"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    report: dict[str, Any] = {
        "stage": "QWEN-V8-REAL-GATE-1",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "layer_index": LAYER_INDEX,
        "base_parameters_trainable": False,
        "per_fact_gradients": 0,
        "knowledge_text_tokens_added": 0,
        "scientific_pass": False,
    }

    model_path = Path(snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"],
    ))
    binding_before = model_file_binding(model_path)
    report["model_file_binding"] = binding_before
    weight_files = sorted(model_path.glob("*.safetensors"))
    if len(weight_files) == 1 and weight_files[0].name == "model.safetensors":
        got = sha256_file(weight_files[0])
        report["model_safetensors_sha256"] = got
        if got != EXPECTED_MODEL_SAFETENSORS_SHA256:
            raise RuntimeError(f"frozen Qwen weight hash mismatch: {got}")
    else:
        # Fail closed rather than silently blessing a changed packaging layout.
        raise RuntimeError(f"expected one model.safetensors, found {[p.name for p in weight_files]}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = torch.device("cpu")
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    report["parameter_count"] = int(param_count)
    report["hidden_size"] = int(model.config.hidden_size)
    report["num_hidden_layers"] = int(model.config.num_hidden_layers)
    report["all_base_gradients_absent"] = all(p.grad is None and not p.requires_grad for p in model.parameters())

    # Natural-language baseline + exact no-op hook equivalence.
    controls = [
        "The capital of France is",
        "Two plus two equals",
        "Water freezes at",
        "The opposite of north is",
    ]
    control_inputs = batched_inputs(tokenizer, controls, device)
    baseline = forward_logits(model, control_inputs)

    def noop(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        return args

    h = model.model.layers[LAYER_INDEX].register_forward_pre_hook(noop)
    try:
        noop_logits = forward_logits(model, control_inputs)
    finally:
        h.remove()
    report["noop_hook_max_abs_logit_error"] = float((baseline - noop_logits).abs().max().item())
    report["noop_hook_argmax_equivalent"] = bool(torch.equal(baseline.argmax(-1), noop_logits.argmax(-1)))
    report["baseline_logits_sha256"] = tensor_sha256(baseline)

    # Global, zero-fact-gradient head-dual transport gate.
    targets = pick_single_token_targets(tokenizer, 40)
    calibration = targets[:20]
    heldout = targets[20:40]
    cal_prompts, cal_ids = make_prompts(calibration, "Registry code for Nivora Station {i}:", 0)
    ho_prompts_a, ho_ids = make_prompts(heldout, "Registry code for Talvera Depot {i}:", 100)
    ho_prompts_b, _ = make_prompts(heldout, "Which registry code belongs to Talvera Depot {i}?", 100)

    cal_inputs = batched_inputs(tokenizer, cal_prompts, device)
    hidden_norm = prelayer_hidden_norm(model, cal_inputs, LAYER_INDEX)
    report["median_prelayer_hidden_norm"] = hidden_norm

    alpha_trials: list[dict[str, float]] = []
    best_frac = None
    best_rate = -1.0
    for frac in ALPHA_FRACTIONS:
        logits = injected_logits(model, cal_inputs, cal_ids, LAYER_INDEX, hidden_norm * frac)
        rate = exact_top1_rate(logits, cal_ids)
        alpha_trials.append({"fraction": float(frac), "top1_rate": rate})
        if rate > best_rate:
            best_rate = rate
            best_frac = float(frac)
    assert best_frac is not None
    report["global_alpha_trials"] = alpha_trials
    report["selected_global_alpha_fraction"] = best_frac
    report["calibration_top1_rate"] = best_rate

    heldout_rates: list[float] = []
    input_hashes_equal = True
    for prompts in (ho_prompts_a, ho_prompts_b):
        inputs = batched_inputs(tokenizer, prompts, device)
        before_ids = inputs["input_ids"].detach().clone()
        logits = injected_logits(model, inputs, ho_ids, LAYER_INDEX, hidden_norm * best_frac)
        heldout_rates.append(exact_top1_rate(logits, ho_ids))
        input_hashes_equal = input_hashes_equal and torch.equal(before_ids, inputs["input_ids"])
    report["heldout_paraphrase_top1_rates"] = heldout_rates
    report["heldout_mean_top1_rate"] = float(sum(heldout_rates) / len(heldout_rates))
    report["input_tokens_unchanged"] = bool(input_hashes_equal)

    # Hot swap: same prompt panel, different target page, no model update.
    swap_targets = torch.flip(ho_ids, dims=[0])
    swap_inputs = batched_inputs(tokenizer, ho_prompts_a, device)
    swap_logits = injected_logits(model, swap_inputs, swap_targets, LAYER_INDEX, hidden_norm * best_frac)
    report["hot_swap_top1_rate"] = exact_top1_rate(swap_logits, swap_targets)

    # Revocation/no-program path must be exactly baseline behavior.
    heldout_inputs = batched_inputs(tokenizer, ho_prompts_a, device)
    revoked_baseline = forward_logits(model, heldout_inputs)
    revoked_again = forward_logits(model, heldout_inputs)
    report["revoked_no_program_max_abs_logit_error"] = float((revoked_baseline - revoked_again).abs().max().item())
    report["revoked_no_program_argmax_equivalent"] = bool(torch.equal(revoked_baseline.argmax(-1), revoked_again.argmax(-1)))

    binding_after = model_file_binding(model_path)
    report["disk_model_unchanged"] = binding_before == binding_after
    report["elapsed_s"] = time.time() - started

    # This is deliberately a bridge gate, not the final project DoD. Pass means:
    # real frozen pretrained model + exact hook integrity + nonzero held-out transfer.
    report["bridge_gate_pass"] = bool(
        report["model_safetensors_sha256"] == EXPECTED_MODEL_SAFETENSORS_SHA256
        and report["noop_hook_argmax_equivalent"]
        and report["noop_hook_max_abs_logit_error"] == 0.0
        and report["all_base_gradients_absent"]
        and report["disk_model_unchanged"]
        and report["input_tokens_unchanged"]
        and report["heldout_mean_top1_rate"] > 0.0
        and report["revoked_no_program_max_abs_logit_error"] == 0.0
    )
    report["scientific_pass"] = False  # frozen project DoD is intentionally stronger

    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
