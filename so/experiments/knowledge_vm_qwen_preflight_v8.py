from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
EXPECTED_WEIGHT_SHA256 = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
TARGET_LAYER = 12


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    x = t.detach().contiguous().cpu()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


def ids_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def run_forward(model: Any, tokenizer: Any, prompt: str) -> dict[str, Any]:
    batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    with torch.inference_mode():
        out = model(**batch, use_cache=False, output_hidden_states=True, return_dict=True)
    logits = out.logits[:, -1, :].float().cpu()
    token_id = int(torch.argmax(logits, dim=-1).item())
    return {
        "input_ids": batch["input_ids"].cpu(),
        "attention_mask": batch.get("attention_mask", None).cpu() if batch.get("attention_mask", None) is not None else None,
        "logits": logits,
        "token_id": token_id,
        "token_text": tokenizer.decode([token_id]),
        "hidden_states": tuple(x.detach().cpu() for x in out.hidden_states),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path(".cache/qwen2.5-0.5b-pinned"))
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.model_dir.parent.mkdir(parents=True, exist_ok=True)

    snapshot_path = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=str(args.model_dir),
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "merges.txt",
                "vocab.json",
                "model.safetensors",
            ],
        )
    )

    weight_path = snapshot_path / "model.safetensors"
    if not weight_path.is_file():
        raise RuntimeError(f"missing pinned weight file: {weight_path}")
    weight_sha = sha256_file(weight_path)
    if weight_sha != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(
            f"weight sha mismatch: expected {EXPECTED_WEIGHT_SHA256}, got {weight_sha}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot_path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("base model is not fully frozen")

    cfg = model.config
    hidden_size = int(getattr(cfg, "hidden_size"))
    num_layers = int(getattr(cfg, "num_hidden_layers"))
    if hidden_size != 896 or num_layers != 24:
        raise RuntimeError(f"unexpected pinned architecture hidden={hidden_size}, layers={num_layers}")

    layers = model.model.layers
    if len(layers) != num_layers:
        raise RuntimeError("decoder-layer inventory mismatch")

    prompts = [
        "The capital of France is",
        "Water freezes at",
        "Two plus two equals",
        "The opposite of north is",
        "A triangle has three",
    ]

    baseline_records: list[dict[str, Any]] = []
    for prompt in prompts:
        r = run_forward(model, tokenizer, prompt)
        baseline_records.append(
            {
                "prompt": prompt,
                "input_ids_sha256": ids_sha256(r["input_ids"]),
                "logits_sha256": tensor_sha256(r["logits"]),
                "argmax_token_id": r["token_id"],
                "argmax_text": r["token_text"],
                "target_layer_hidden_norm": float(r["hidden_states"][TARGET_LAYER].float().norm().item()),
            }
        )

    capture: dict[str, Any] = {}

    def noop_hook(module: torch.nn.Module, hook_args: tuple[Any, ...], hook_kwargs: dict[str, Any]):
        if not hook_args:
            raise RuntimeError("target layer received no positional hidden state")
        hidden = hook_args[0]
        capture["shape"] = list(hidden.shape)
        capture["dtype"] = str(hidden.dtype)
        capture["norm"] = float(hidden.detach().float().norm().cpu().item())
        capture["sha256_before"] = tensor_sha256(hidden)
        identical = hidden + torch.zeros_like(hidden)
        capture["sha256_after_zero"] = tensor_sha256(identical)
        return (identical,) + tuple(hook_args[1:]), hook_kwargs

    handle = layers[TARGET_LAYER].register_forward_pre_hook(noop_hook, with_kwargs=True)
    try:
        hooked = run_forward(model, tokenizer, prompts[0])
    finally:
        handle.remove()

    reference = run_forward(model, tokenizer, prompts[0])
    max_abs = float((hooked["logits"] - reference["logits"]).abs().max().item())
    argmax_same = hooked["token_id"] == reference["token_id"]
    logits_hash_same = tensor_sha256(hooked["logits"]) == tensor_sha256(reference["logits"])
    input_hash_same = ids_sha256(hooked["input_ids"]) == ids_sha256(reference["input_ids"])

    if max_abs != 0.0 or not argmax_same or not logits_hash_same or not input_hash_same:
        raise RuntimeError(
            f"no-op pre-layer hook changed behavior: max_abs={max_abs}, argmax_same={argmax_same}, "
            f"logits_hash_same={logits_hash_same}, input_hash_same={input_hash_same}"
        )

    if capture.get("sha256_before") != capture.get("sha256_after_zero"):
        raise RuntimeError("zero knowledge program changed target-layer hidden bytes")

    result = {
        "stage": "QWEN-REAL-PREFLIGHT-V8",
        "scientific_result": "real_pretrained_preflight_pass",
        "breakthrough_claim": False,
        "dod_satisfied": False,
        "model": {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weight_sha256": weight_sha,
            "parameter_count": count_parameters(model),
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "vocab_size": int(getattr(cfg, "vocab_size")),
            "all_parameters_frozen": not any(p.requires_grad for p in model.parameters()),
        },
        "hook": {
            "target_layer": TARGET_LAYER,
            "capture": capture,
            "noop_max_abs_logit_error": max_abs,
            "noop_argmax_same": argmax_same,
            "noop_logits_hash_same": logits_hash_same,
            "input_tokens_unchanged": input_hash_same,
        },
        "baseline": baseline_records,
        "next_gate": "request-resident KCIR coefficients + globally calibrated low-rank ABI on the same frozen model",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
