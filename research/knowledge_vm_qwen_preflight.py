from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
EXPECTED_MODEL_SAFETENSORS_SHA256 = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
OUT = Path(os.environ.get("SO_QWEN_REPORT", "ci-qwen-preflight/report.json"))


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_sha(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    import torch
    import transformers
    import tokenizers
    import safetensors
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    OUT.parent.mkdir(parents=True, exist_ok=True)
    model_dir = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=[
                "*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors",
            ],
        )
    ).resolve()

    weight_files = sorted(model_dir.glob("*.safetensors"))
    if not weight_files:
        raise RuntimeError("No safetensors weights were downloaded")
    weight_hashes = {p.name: sha256_file(p) for p in weight_files}
    model_weight_hash = weight_hashes.get("model.safetensors")
    expected_weight_match = model_weight_hash == EXPECTED_MODEL_SAFETENSORS_SHA256

    tok = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config = model.config
    hidden_size = int(getattr(config, "hidden_size", -1))
    layer_count = int(getattr(config, "num_hidden_layers", -1))
    vocab_size = int(getattr(config, "vocab_size", -1))

    prompts = [
        "The capital of France is",
        "Two plus two equals",
        "Water freezes at",
        "The opposite of north is",
        "A short story begins with",
    ]
    observations = []
    with torch.inference_mode():
        for prompt in prompts:
            batch = tok(prompt, return_tensors="pt")
            out = model(**batch, use_cache=True, output_hidden_states=True, return_dict=True)
            logits = out.logits[:, -1, :]
            next_id = int(logits.argmax(dim=-1).item())
            topv, topi = torch.topk(logits[0], k=5)
            generated = model.generate(
                **batch,
                max_new_tokens=8,
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.eos_token_id,
            )
            observations.append(
                {
                    "prompt": prompt,
                    "input_ids": batch["input_ids"].tolist()[0],
                    "next_token_id": next_id,
                    "next_token_text": tok.decode([next_id]),
                    "top5_ids": topi.tolist(),
                    "top5_logits": [float(x) for x in topv.tolist()],
                    "generated_text": tok.decode(generated[0], skip_special_tokens=True),
                    "past_layers": len(out.past_key_values),
                    "hidden_states": len(out.hidden_states),
                    "last_hidden_shape": list(out.hidden_states[-1].shape),
                }
            )

    # Architecture inventory for the later Knowledge-port hook. Qwen2/Qwen2.5 normally
    # exposes model.model.layers; fail closed if that assumption does not hold.
    decoder = getattr(model, "model", None)
    layers = getattr(decoder, "layers", None)
    hookable = layers is not None and len(layers) == layer_count
    layer_types = sorted({type(layer).__name__ for layer in layers}) if hookable else []

    report = {
        "stage": "QWEN-REAL-PREFLIGHT-V1",
        "scientific_result": "real_pretrained_preflight_pass" if expected_weight_match and hookable and trainable_count == 0 else "fail",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "expected_model_safetensors_sha256": EXPECTED_MODEL_SAFETENSORS_SHA256,
        "weight_hashes": weight_hashes,
        "expected_weight_match": expected_weight_match,
        "model_dir": str(model_dir),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "hidden_size": hidden_size,
        "layer_count": layer_count,
        "vocab_size": vocab_size,
        "hookable_decoder_layers": hookable,
        "layer_types": layer_types,
        "observations": observations,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "safetensors": safetensors.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    report["report_sha256"] = canonical_sha(report)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))

    if not expected_weight_match:
        print("Pinned weight SHA-256 mismatch", file=sys.stderr)
        return 2
    if trainable_count != 0:
        print("Base model unexpectedly has trainable parameters", file=sys.stderr)
        return 3
    if not hookable:
        print("Could not identify the expected decoder-layer hook surface", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
