from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

MODEL_ID = "Qwen/Qwen3-0.6B"
REVISION = "a08cec3036ee1085a4863a0b730e5d3f1c2f8d04"
WEIGHT_SHA = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
OUT = Path(os.environ.get("SO_R285_REPORT", "ci-qwen-r285/report.json"))
ENTITY = "Luma Harbor"
VALUES = ("red", "green", "blue", "yellow")
FEATURES = (8, 16, 32, 64, 128)
QUESTIONS = (
    "What is Luma Harbor's registry color? Answer only with the color word.",
    "Is Luma Harbor's registry color red? Answer only yes or no.",
    "Is Luma Harbor's registry color either red or yellow? Answer only yes or no.",
    "Encode Luma Harbor's registry color as A if red or green, otherwise B. Answer only A or B.",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def ids(tok, messages, generation=False):
    return tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation, enable_thinking=False)


def render(tok, value: str, question: str):
    pm = [{"role": "system", "content": f"Authoritative knowledge register: {ENTITY}'s registry color is {value}. Use this fact exactly."}]
    rm = [{"role": "user", "content": question}]
    pids = ids(tok, pm, False)
    fids = ids(tok, pm + rm, True)
    if fids[: len(pids)] != pids:
        raise RuntimeError("unstable causal prefix boundary")
    return pids, fids[len(pids) :]


def cache_kv(cache, idx: int):
    if hasattr(cache, "layers"):
        layer = cache.layers[idx]
        return layer.keys, layer.values
    return cache.key_cache[idx], cache.value_cache[idx]


def orthogonal_gaussian(rows: int, dim: int, seed: int) -> torch.Tensor:
    """Deterministic approximately Gaussian orthogonal random features.

    Rows are built in orthogonal blocks and rescaled by independent chi-like
    Gaussian row norms. This is a fixed model-global feature map, never fit to a
    fact or answer.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    blocks = []
    remaining = rows
    while remaining > 0:
        a = torch.randn((dim, dim), generator=g, dtype=torch.float64)
        q, _ = torch.linalg.qr(a)
        take = min(remaining, dim)
        # Match N(0,I) radial scale per row.
        radial = torch.linalg.norm(torch.randn((take, dim), generator=g, dtype=torch.float64), dim=-1, keepdim=True)
        blocks.append(q[:take] * radial)
        remaining -= take
    return torch.cat(blocks, dim=0)


def positive_features(x: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    # FAVOR+-style positive random features for exp(x dot y).
    # Use float64; norms in Qwen3's normalized attention geometry are moderate.
    proj = x.double() @ omega.T
    norm = 0.5 * torch.sum(x.double() * x.double(), dim=-1, keepdim=True)
    return torch.exp(torch.clamp(proj - norm, -40.0, 40.0)) / math.sqrt(omega.shape[0])


def compile_moments(k: torch.Tensor, v: torch.Tensor, omega: torch.Tensor, scaling: float):
    ks = k.double() * math.sqrt(scaling)
    phik = positive_features(ks, omega)
    # S: [m,dv], z: [m]. Additivity makes install/edit/delete naturally patchable.
    return {
        "S": phik.T @ v.double(),
        "z": phik.sum(dim=0),
    }


def exact_response(q, k, v, scaling):
    s = (q.double() @ k.double().T) * scaling
    return torch.softmax(s, dim=-1) @ v.double()


def capsule_response(q, cap, omega, scaling):
    qs = q.double() * math.sqrt(scaling)
    phiq = positive_features(qs.unsqueeze(0), omega)[0]
    num = phiq @ cap["S"]
    den = phiq @ cap["z"]
    if abs(float(den.item())) < 1e-12:
        return torch.full_like(num, float("nan"))
    return num / den


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    md = Path(snapshot_download(repo_id=MODEL_ID, revision=REVISION, allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors"]))
    assert sha256_file(md / "model.safetensors") == WEIGHT_SHA
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(md, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager", trust_remote_code=False).eval()
    model.requires_grad_(False)

    # Fixed feature maps are model/layer/KV-head ABI state, not per-fact state.
    omegas = {}
    head_dim = int(model.model.layers[0].self_attn.head_dim)
    for li, layer in enumerate(model.model.layers):
        kvh = int(layer.self_attn.num_key_value_heads)
        for kh in range(kvh):
            full = orthogonal_gaussian(max(FEATURES), head_dim, seed=700001 + 1009 * li + 17 * kh)
            for m in FEATURES:
                omegas[(li, kh, m)] = full[:m].contiguous()

    rows = []
    prefix_lengths = []
    for value in VALUES:
        pids, _ = render(tok, value, QUESTIONS[0])
        p = torch.tensor([pids], dtype=torch.long)
        with torch.inference_mode():
            po = model(input_ids=p, attention_mask=torch.ones_like(p), use_cache=True, return_dict=True)
        prefix_cache = po.past_key_values
        L = len(pids)
        prefix_lengths.append(L)

        compiled = {}
        for li, layer in enumerate(model.model.layers):
            kcache, vcache = cache_kv(prefix_cache, li)
            kvh = kcache.shape[1]
            scaling = float(layer.self_attn.scaling)
            for kh in range(kvh):
                k = kcache[0, kh, :L, :].float().cpu()
                v = vcache[0, kh, :L, :].float().cpu()
                for m in FEATURES:
                    compiled[(li, kh, m)] = compile_moments(k, v, omegas[(li, kh, m)], scaling)

        for qi, question in enumerate(QUESTIONS):
            p2, qids = render(tok, value, question)
            assert p2 == pids
            q = torch.tensor([qids], dtype=torch.long)
            pos_ids = torch.arange(L, L + q.shape[1], dtype=torch.long).unsqueeze(0)
            mask = torch.ones((1, L + q.shape[1]), dtype=torch.long)
            cache_pos = torch.arange(L, L + q.shape[1], dtype=torch.long)
            qraw = {}
            handles = []
            for li, layer in enumerate(model.model.layers):
                def hook(_module, _inputs, output, li=li):
                    qraw[li] = output.detach().float().cpu()
                handles.append(layer.self_attn.q_proj.register_forward_hook(hook))
            with torch.inference_mode():
                model(input_ids=q, attention_mask=mask, position_ids=pos_ids, cache_position=cache_pos,
                      past_key_values=copy.deepcopy(prefix_cache), use_cache=True, return_dict=True)
            for h in handles:
                h.remove()

            dummy = torch.zeros((1, q.shape[1], model.config.hidden_size), dtype=torch.float32)
            with torch.inference_mode():
                cos, sin = model.model.rotary_emb(dummy, pos_ids)

            for li, layer in enumerate(model.model.layers):
                attn = layer.self_attn
                kcache, vcache = cache_kv(prefix_cache, li)
                kvh = kcache.shape[1]
                d = int(attn.head_dim)
                raw = qraw[li]
                qh = raw.view(1, q.shape[1], -1, d).transpose(1, 2)
                if hasattr(attn, "q_norm"):
                    qh = attn.q_norm(qh)
                qrot, _ = apply_rotary_pos_emb(qh, qh, cos, sin)
                qlast = qrot[0, :, -1, :].float().cpu()
                qheads = qlast.shape[0]
                rep = qheads // kvh
                scaling = float(attn.scaling)
                for hidx in range(qheads):
                    kh = min(kvh - 1, hidx // rep)
                    kk = kcache[0, kh, :L, :].float().cpu()
                    vv = vcache[0, kh, :L, :].float().cpu()
                    exact = exact_response(qlast[hidx], kk, vv, scaling)
                    enorm = max(float(torch.linalg.norm(exact).item()), 1e-12)
                    for m in FEATURES:
                        approx = capsule_response(qlast[hidx], compiled[(li, kh, m)], omegas[(li, kh, m)], scaling)
                        if torch.isnan(approx).any() or torch.isinf(approx).any():
                            rel, cosv = float("inf"), -1.0
                        else:
                            rel = float(torch.linalg.norm(approx - exact).item() / enorm)
                            cosv = float(F.cosine_similarity(approx.float(), exact.float(), dim=0).item())
                        rows.append({"value": value, "question_index": qi, "layer": li, "query_head": hidx,
                                     "kv_head": kh, "features": m, "rel_error": rel, "cosine": cosv})

    by_m = {}
    Lmed = int(torch.tensor(prefix_lengths).median().item())
    for m in FEATURES:
        rr = [r for r in rows if r["features"] == m]
        e = torch.tensor([r["rel_error"] for r in rr], dtype=torch.float64)
        c = torch.tensor([r["cosine"] for r in rr], dtype=torch.float64)
        # Per KV-head/layer float counts: full = L*(dk+dv), capsule = m*dv + m + model-global omega (not per fact).
        # omega is ABI state and deliberately excluded from per-fact bytes.
        per_fact_ratio = (m * head_dim + m) / max(1, (Lmed * head_dim * 2))
        by_m[str(m)] = {
            "n": len(rr),
            "median_rel_error": float(e.median().item()),
            "p90_rel_error": float(torch.quantile(e, 0.90).item()),
            "p99_rel_error": float(torch.quantile(e, 0.99).item()),
            "mean_cosine": float(c.mean().item()),
            "p10_cosine": float(torch.quantile(c, 0.10).item()),
            "min_cosine": float(c.min().item()),
            "per_fact_float_ratio_vs_full_kv_excluding_global_feature_map": per_fact_ratio,
        }

    report = {
        "stage": "R285-QWEN3-SOFTMAX-KERNEL-MOMENT-CAPSULE",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weight_sha256": WEIGHT_SHA,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "per_fact_gradient_steps": 0,
        "runtime_knowledge_text_tokens": 0,
        "prefix_lengths": prefix_lengths,
        "feature_counts": list(FEATURES),
        "by_features": by_m,
        "scientific_scope": "Real frozen Qwen3-0.6B. Tests additive FAVOR+-style softmax-kernel sufficient statistics compiled analytically from full fact-prefix K/V using a fixed model-global random feature ABI. Evaluated on actual future query vectors at every layer/head. No per-fact fitting, no answer-direction fitting; mechanism gate only.",
        "dod_status": "NOT_DOD; kernel-moment response capsule mechanism gate",
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(by_m, indent=2, sort_keys=True))
    ok = any(
        by_m[str(m)]["median_rel_error"] <= 0.10
        and by_m[str(m)]["p90_rel_error"] <= 0.25
        and by_m[str(m)]["p10_cosine"] >= 0.95
        and by_m[str(m)]["per_fact_float_ratio_vs_full_kv_excluding_global_feature_map"] < 1.0
        for m in FEATURES
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
