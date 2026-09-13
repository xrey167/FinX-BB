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
OUT = Path(os.environ.get("SO_R284_REPORT", "ci-qwen-r284/report.json"))
ENTITY = "Luma Harbor"
VALUES = ("red", "green", "blue", "yellow")
RANKS = (1, 2, 4, 8, 16)
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


def farthest_indices(k: torch.Tensor, rank: int) -> torch.Tensor:
    # k: [L,D], deterministic farthest-point subset in key geometry.
    L = k.shape[0]
    rank = min(rank, L)
    if rank == L:
        return torch.arange(L, dtype=torch.long)
    first = int(torch.argmax(torch.sum(k * k, dim=-1)).item())
    chosen = [first]
    min_d2 = torch.sum((k - k[first]) ** 2, dim=-1)
    for _ in range(1, rank):
        nxt = int(torch.argmax(min_d2).item())
        chosen.append(nxt)
        d2 = torch.sum((k - k[nxt]) ** 2, dim=-1)
        min_d2 = torch.minimum(min_d2, d2)
    return torch.tensor(chosen, dtype=torch.long)


def compile_nystrom(k: torch.Tensor, v: torch.Tensor, rank: int, scaling: float):
    # Analytic, query-independent compilation of the softmax-kernel response.
    # q*k*scaling = (q*sqrt(scaling)) * (k*sqrt(scaling)).
    k = k.double()
    v = v.double()
    ks = k * math.sqrt(scaling)
    idx = farthest_indices(ks, rank)
    z = ks[idx]
    gram = torch.clamp(z @ z.T, -30.0, 30.0)
    cross = torch.clamp(z @ ks.T, -30.0, 30.0)
    A = torch.exp(gram)
    B = torch.exp(cross)
    # Small Tikhonov term only stabilizes the finite prefix kernel solve.
    reg = 1e-8 * max(1.0, float(torch.linalg.norm(A, ord=2).item()))
    C = torch.linalg.pinv(A + reg * torch.eye(A.shape[0], dtype=A.dtype), rtol=1e-10, atol=1e-12) @ B
    cv = C @ v
    c1 = C.sum(dim=-1)
    return {"z": z, "cv": cv, "c1": c1, "rank": int(rank), "landmarks": idx.tolist()}


def response_exact(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scaling: float):
    q = q.double()
    k = k.double()
    v = v.double()
    scores = (q @ k.T) * scaling
    return torch.softmax(scores, dim=-1) @ v


def response_capsule(q: torch.Tensor, capsule, scaling: float):
    qs = q.double() * math.sqrt(scaling)
    logits = torch.clamp(qs @ capsule["z"].T, -30.0, 30.0)
    a = torch.exp(logits)
    num = a @ capsule["cv"]
    den = a @ capsule["c1"]
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

        # Compile response capsules once per revision/layer/KV head/rank.
        compiled = {}
        for li, attn in enumerate(model.model.layers):
            kcache, vcache = cache_kv(prefix_cache, li)
            kvh = kcache.shape[1]
            scaling = float(attn.self_attn.scaling)
            for kh in range(kvh):
                k = kcache[0, kh, :L, :].float().cpu()
                v = vcache[0, kh, :L, :].float().cpu()
                for rank in RANKS:
                    compiled[(li, kh, rank)] = compile_nystrom(k, v, min(rank, L), scaling)

        for qi, question in enumerate(QUESTIONS):
            p2, qids = render(tok, value, question)
            assert p2 == pids
            q = torch.tensor([qids], dtype=torch.long)
            past_len = L
            pos_ids = torch.arange(past_len, past_len + q.shape[1], dtype=torch.long).unsqueeze(0)
            mask = torch.ones((1, past_len + q.shape[1]), dtype=torch.long)
            cache_pos = torch.arange(past_len, past_len + q.shape[1], dtype=torch.long)

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

            # Rotary cos/sin are global for Qwen3 and depend only on query positions.
            dummy = torch.zeros((1, q.shape[1], model.config.hidden_size), dtype=torch.float32)
            with torch.inference_mode():
                cos, sin = model.model.rotary_emb(dummy, pos_ids)

            for li, layer in enumerate(model.model.layers):
                attn = layer.self_attn
                kcache, vcache = cache_kv(prefix_cache, li)
                kvh = kcache.shape[1]
                head_dim = int(attn.head_dim)
                raw = qraw[li]
                qh = raw.view(1, q.shape[1], -1, head_dim).transpose(1, 2)
                if hasattr(attn, "q_norm"):
                    qh = attn.q_norm(qh)
                qrot, _ = apply_rotary_pos_emb(qh, qh, cos, sin)
                qlast = qrot[0, :, -1, :].float().cpu()
                q_heads = qlast.shape[0]
                rep = q_heads // kvh
                scaling = float(attn.scaling)
                for hidx in range(q_heads):
                    kh = min(kvh - 1, hidx // rep)
                    kk = kcache[0, kh, :L, :].float().cpu()
                    vv = vcache[0, kh, :L, :].float().cpu()
                    exact = response_exact(qlast[hidx], kk, vv, scaling)
                    exact_norm = max(float(torch.linalg.norm(exact).item()), 1e-12)
                    for rank in RANKS:
                        cap = compiled[(li, kh, rank)]
                        approx = response_capsule(qlast[hidx], cap, scaling)
                        if torch.isnan(approx).any():
                            rel = float("inf")
                            cosine = -1.0
                        else:
                            rel = float(torch.linalg.norm(approx - exact).item() / exact_norm)
                            cosine = float(F.cosine_similarity(approx.float(), exact.float(), dim=0).item())
                        rows.append({
                            "value": value,
                            "question_index": qi,
                            "layer": li,
                            "query_head": hidx,
                            "kv_head": kh,
                            "rank": rank,
                            "rel_error": rel,
                            "cosine": cosine,
                        })

    by_rank = {}
    for rank in RANKS:
        rr = [r for r in rows if r["rank"] == rank]
        errs = torch.tensor([r["rel_error"] for r in rr], dtype=torch.float64)
        coss = torch.tensor([r["cosine"] for r in rr], dtype=torch.float64)
        by_rank[str(rank)] = {
            "n": len(rr),
            "median_rel_error": float(errs.median().item()),
            "p90_rel_error": float(torch.quantile(errs, 0.90).item()),
            "p99_rel_error": float(torch.quantile(errs, 0.99).item()),
            "mean_cosine": float(coss.mean().item()),
            "p10_cosine": float(torch.quantile(coss, 0.10).item()),
            "min_cosine": float(coss.min().item()),
        }

    report = {
        "stage": "R284-QWEN3-QUERY-CONDITIONED-RESPONSE-CAPSULE",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weight_sha256": WEIGHT_SHA,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "per_fact_gradient_steps": 0,
        "runtime_knowledge_text_tokens": 0,
        "prefix_lengths": prefix_lengths,
        "ranks": list(RANKS),
        "by_rank": by_rank,
        "scientific_scope": "Real frozen Qwen3-0.6B. Analytic Nyström softmax-kernel response capsules compiled directly from full causal fact-prefix K/V. Measures preservation of the prefix attention response on actual future query vectors across all layers/heads. This is a mechanism gate, not end-to-end generation and not DoD.",
        "dod_status": "NOT_DOD; query-conditioned response-operator mechanism gate",
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(by_rank, indent=2, sort_keys=True))
    # Gate: a genuinely compact <=16-rank response operator should preserve the
    # actual prefix response, not merely top-1 coincidence.
    ok = any(
        by_rank[str(r)]["median_rel_error"] <= 0.05
        and by_rank[str(r)]["p90_rel_error"] <= 0.15
        and by_rank[str(r)]["p10_cosine"] >= 0.98
        for r in RANKS if r <= 16
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
