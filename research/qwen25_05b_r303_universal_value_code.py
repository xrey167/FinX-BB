from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import sha256_file, candidate_token

MODEL_ID = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
EXPECTED_WEIGHTS = {"model.safetensors": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"}
OUT = Path(os.environ.get("SO_R303_REPORT", "ci-qwen-r303/report.json"))

VALUE_WORDS = (
    "Paris", "Berlin", "Rome", "Tokyo", "Beijing", "Ottawa", "Madrid", "Moscow",
    "Cairo", "Brasilia", "Athens", "Vienna", "Lisbon", "Oslo", "Helsinki", "Stockholm",
    "Dublin", "Warsaw", "Ankara", "red", "blue", "green", "yellow", "circle", "square",
    "north", "south", "east", "west", "summer", "winter", "spring", "autumn",
)


def topk_metrics(logits: torch.Tensor, target_ids: list[int]):
    top1 = logits.argmax(dim=-1).tolist()
    top5 = torch.topk(logits, k=5, dim=-1).indices.tolist()
    top20 = torch.topk(logits, k=20, dim=-1).indices.tolist()
    return {
        "top1": sum(int(a == b) for a, b in zip(top1, target_ids)) / len(target_ids),
        "top5": sum(int(b in row) for b, row in zip(target_ids, top5)) / len(target_ids),
        "top20": sum(int(b in row) for b, row in zip(target_ids, top20)) / len(target_ids),
    }, top1


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    rng = random.Random(303)

    md = Path(snapshot_download(
        repo_id=MODEL_ID, revision=REVISION,
        allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors", "*.index.json", "*.merges", "*.vocab", "merges.txt", "vocab.json"],
    ))
    hashes = {n: sha256_file(md / n) for n in EXPECTED_WEIGHTS}; assert hashes == EXPECTED_WEIGHTS
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        md, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", trust_remote_code=False,
    ).eval(); model.requires_grad_(False)

    W = model.lm_head.weight.detach().float().cpu()
    E = model.get_input_embeddings().weight.detach().float().cpu()
    vocab, d = W.shape

    words=[]; target_ids=[]
    for word in VALUE_WORDS:
        try: tid = candidate_token(tok, word)
        except RuntimeError: continue
        if tid in target_ids: continue
        words.append(word); target_ids.append(tid)
    if len(words) < 20:
        raise RuntimeError(f"too few unique one-token values: {words}")

    excluded = set(target_ids) | set(tok.all_special_ids)
    pool = [i for i in range(vocab) if i not in excluded]
    train_ids = rng.sample(pool, min(8192, len(pool)))
    Wtrain = W[train_ids]
    Etrain = E[train_ids]

    # Shared statistics are learned once from unrelated vocabulary rows; no future-value row is used.
    w_mean = Wtrain.mean(dim=0)
    w_std = Wtrain.std(dim=0).clamp_min(1e-5)

    # Cheap low-rank shared input->output embedding bridge, fit only on unrelated training tokens.
    rank = 64
    gen = torch.Generator().manual_seed(303)
    P = torch.randn((d, rank), generator=gen)
    P, _ = torch.linalg.qr(P, mode="reduced")
    Z = Etrain @ P
    ridge = 1e-2
    lhs = Z.T @ Z + ridge * torch.eye(rank)
    rhs = Z.T @ Wtrain
    B = torch.linalg.solve(lhs, rhs)

    target = torch.tensor(target_ids, dtype=torch.long)
    raw_output = W[target]
    centered_output = raw_output - w_mean
    diag_white_output = centered_output / w_std
    mapped_input = (E[target] @ P) @ B

    variants = {
        "raw_output_row": raw_output,
        "centered_output_row": centered_output,
        "diag_whitened_output_row": diag_white_output,
        "rank64_input_to_output_map": mapped_input,
    }

    variant_metrics = {}
    rows=[]
    for name,H in variants.items():
        # Chunk-free here: <= ~33 test rows, frozen LM head only.
        logits = H @ W.T
        metrics, top1 = topk_metrics(logits, target_ids)
        margins=[]
        for i,(word,tid,pred) in enumerate(zip(words,target_ids,top1)):
            target_score=float(logits[i,tid])
            tmp=logits[i].clone(); tmp[tid]=-float("inf")
            best_other=float(tmp.max())
            margins.append(target_score-best_other)
            rows.append({
                "variant":name,"word":word,"target_token_id":tid,"greedy_token_id":int(pred),
                "greedy_text":tok.decode([int(pred)]),"correct":int(pred)==tid,"target_margin_vs_best_other":target_score-best_other,
            })
        variant_metrics[name]={**metrics,"mean_target_margin_vs_best_other":sum(margins)/len(margins),"min_target_margin_vs_best_other":min(margins)}

    # Quantize the best no-per-value-training code candidate coordinate-wise to ask whether
    # a future value can be stored compactly as a model-native generation code.
    best_name=max(variant_metrics,key=lambda n:(variant_metrics[n]["top1"],variant_metrics[n]["top5"]))
    H=variants[best_name]
    maxabs=H.abs().amax(dim=1,keepdim=True).clamp_min(1e-8)
    q=torch.round(H/maxabs*127).clamp(-127,127).to(torch.int8)
    H8=q.float()/127*maxabs
    qlogits=H8@W.T
    qmetrics,qtop=topk_metrics(qlogits,target_ids)
    int8_bytes_per_value=d + 4  # int8 coordinates + fp32 scale
    bf16_bytes_per_value=d*2

    report={
        "stage":"R303-UNIVERSAL-MODEL-NATIVE-VALUE-CODE",
        "architecture_candidate":"Universal Model-Native Port Value Code",
        "model_id":MODEL_ID,"revision":REVISION,"weights_sha256":hashes,"backbone_frozen":all(not p.requires_grad for p in model.parameters()),
        "vocab_size":vocab,"hidden_dim":d,"shared_fit_token_count":len(train_ids),"future_value_words":words,"future_value_count":len(words),
        "future_value_rows_excluded_from_shared_fit":True,"per_value_gradient_steps":0,
        "variant_metrics":variant_metrics,"best_variant":best_name,
        "int8_best_variant_metrics":qmetrics,"int8_bytes_per_value":int8_bytes_per_value,"bf16_bytes_per_value":bf16_bytes_per_value,"int8_storage_ratio_vs_bf16":int8_bytes_per_value/bf16_bytes_per_value,
        "rows":rows,
        "mechanism":(
            "A future symbolic value is translated by the frozen tokenizer/model vocabulary into a model-native vector. "
            "No entity/value pair is trained. R303 tests whether such a value code can directly inhabit J-Space and be "
            "decoded by the unchanged Qwen LM head, plus whether one shared embedding-space map can generalize to values "
            "excluded from its fit."
        ),
        "claim_boundary":(
            "Embedding/output-row geometry and embedding alignment are established techniques; this is not a novelty claim. "
            "The gate probes a practical zero-per-record-write representation for CKCA and is limited to single-token values."
        ),
        "dod_status":"NOT_DOD; universal future-value code probe",
    }
    canonical=json.dumps(report,sort_keys=True,separators=(",",":")).encode(); report["report_sha256"]=hashlib.sha256(canonical).hexdigest(); OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"future_values":len(words),"best_variant":best_name,"variant_metrics":variant_metrics,"int8_metrics":qmetrics,"int8_ratio":report["int8_storage_ratio_vs_bf16"]},indent=2))
    # This is exploratory: persist all outcomes; only fail if no variant even places most values in top-20.
    if max(x["top20"] for x in variant_metrics.values()) < 0.80:
        return 2
    return 0

if __name__=="__main__": raise SystemExit(main())
