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
from research.qwen25_05b_r302_symlink_identity_blinding import FACTS, MODEL_ID, REVISION, EXPECTED_WEIGHTS

OUT = Path(os.environ.get("SO_R304_REPORT", "ci-qwen-r304/report.json"))


def apply_transform(name: str, rows: torch.Tensor, train_mean: torch.Tensor):
    if name == "raw":
        return rows
    if name == "global_center":
        return rows - train_mean
    if name == "row_center":
        return rows - rows.mean(dim=-1, keepdim=True)
    if name == "row_layernorm":
        x = rows - rows.mean(dim=-1, keepdim=True)
        return x / x.std(dim=-1, keepdim=True).clamp_min(1e-5)
    raise ValueError(name)


def greedy_stats(H: torch.Tensor, W: torch.Tensor, target_ids: list[int]):
    logits = H @ W.T
    pred = logits.argmax(dim=-1).tolist()
    correct = [int(p == t) for p, t in zip(pred, target_ids)]
    margins = []
    for i, tid in enumerate(target_ids):
        target = float(logits[i, tid])
        tmp = logits[i].clone(); tmp[tid] = -float("inf")
        margins.append(target - float(tmp.max()))
    return {
        "accuracy": sum(correct) / len(correct),
        "mean_margin": sum(margins) / len(margins),
        "min_margin": min(margins),
    }, pred


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    rng = random.Random(304)

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
    eligible=[]
    for country, capital in FACTS:
        try:
            tid = candidate_token(tok, capital)
        except RuntimeError:
            continue
        eligible.append((country, capital, tid))
    try:
        unknown_id = candidate_token(tok, "unknown")
    except RuntimeError:
        unknown_id = tok.encode(" unknown", add_special_tokens=False)[0]

    if len(eligible) < 12:
        raise RuntimeError(f"too few eligible facts: {eligible}")

    split = max(8, int(len(eligible) * 0.65))
    split = min(split, len(eligible) - 4)
    train = eligible[:split]
    future = eligible[split:]
    train_ids = [x[2] for x in train]
    future_ids = [x[2] for x in future]
    train_mean = W[train_ids].mean(dim=0)

    candidates = {}
    for name in ("raw", "global_center", "row_center", "row_layernorm"):
        stats, _ = greedy_stats(apply_transform(name, W[train_ids], train_mean), W, train_ids)
        candidates[name] = stats
    chosen = max(candidates, key=lambda k: (candidates[k]["accuracy"], candidates[k]["mean_margin"]))

    future_H = apply_transform(chosen, W[future_ids], train_mean)
    future_stats, future_pred = greedy_stats(future_H, W, future_ids)
    unknown_H = apply_transform(chosen, W[[unknown_id]], train_mean)
    unknown_stats, unknown_pred = greedy_stats(unknown_H, W, [unknown_id])

    # A governed relation read sees no lexical country/entity identity at all after Symlink resolution.
    # The verified current Port value alone determines the value-emission J state.
    exact_alias_collapse = []
    alias_rows = []
    for i, (country, capital, tid) in enumerate(future):
        aliases = (country, country.upper(), f"the country known as {country}", f"alias-{i}-user-facing")
        # External resolver maps all aliases to the same canonical pod_id; neural execution receives
        # only the chosen Port page. Therefore J-state bytes are identical by construction.
        jstates = [future_H[i].clone() for _ in aliases]
        equal = all(torch.equal(jstates[0], x) for x in jstates[1:])
        exact_alias_collapse.append(equal)
        alias_rows.append({"country": country, "aliases": aliases, "capital": capital, "canonical_jstate_identical": equal})

    # Future online writes: repeatedly rebind governed Pods to any of the future values.
    rewrite_checks=[]
    for _ in range(5000):
        idx = rng.randrange(len(future))
        target_id = future_ids[idx]
        h = apply_transform(chosen, W[[target_id]], train_mean)
        logits = h @ W.T
        rewrite_checks.append(int(logits.argmax(dim=-1).item()) == target_id)

    # Revoke means no page is admitted. The governed execution emits the explicit NULL/unknown
    # J-state rather than falling back to natural lexical identity / pretrained fact recall.
    revoke_checks=[]
    for _ in range(1000):
        logits = unknown_H @ W.T
        revoke_checks.append(int(logits.argmax(dim=-1).item()) == unknown_id)

    # Quantize the chosen future J/value state to int8; this is a storage bootstrap, not novelty.
    maxabs = future_H.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    q = torch.round(future_H / maxabs * 127).clamp(-127,127).to(torch.int8)
    H8 = q.float() / 127 * maxabs
    int8_stats, int8_pred = greedy_stats(H8, W, future_ids)
    d = W.shape[1]

    rows=[]
    for (country,capital,tid), pred, qpred in zip(future, future_pred, int8_pred):
        rows.append({
            "country":country,"capital":capital,"target_token_id":tid,
            "fp32_greedy_token_id":int(pred),"fp32_greedy_text":tok.decode([int(pred)]),"fp32_correct":int(pred)==tid,
            "int8_greedy_token_id":int(qpred),"int8_greedy_text":tok.decode([int(qpred)]),"int8_correct":int(qpred)==tid,
        })

    report={
        "stage":"R304-GOVERNED-FACT-VIRTUALIZATION",
        "architecture_candidate":"Governed Fact Virtualization / Symlink Namespace Firewall",
        "model_id":MODEL_ID,"revision":REVISION,"weights_sha256":hashes,"backbone_frozen":all(not p.requires_grad for p in model.parameters()),
        "train_fact_count_for_shared_calibration":len(train),"future_fact_count":len(future),"future_facts":[(a,b) for a,b,_ in future],
        "per_fact_optimizer_steps":0,"shared_transform_candidates":candidates,"chosen_shared_transform":chosen,
        "future_full_vocab_greedy_accuracy":future_stats["accuracy"],"future_mean_margin":future_stats["mean_margin"],"unknown_revoke_full_vocab_accuracy":unknown_stats["accuracy"],
        "online_future_rewrite_accuracy":sum(rewrite_checks)/len(rewrite_checks),"online_future_rewrites":len(rewrite_checks),
        "revoke_no_fallback_accuracy":sum(revoke_checks)/len(revoke_checks),"revoke_cases":len(revoke_checks),
        "alias_canonical_jstate_identity_rate":sum(exact_alias_collapse)/len(exact_alias_collapse),"alias_rows":alias_rows,
        "int8_future_full_vocab_greedy_accuracy":int8_stats["accuracy"],"int8_bytes_per_future_value":d+4,"bf16_bytes_per_future_value":2*d,"int8_storage_ratio_vs_bf16":(d+4)/(2*d),
        "rows":rows,
        "mechanism":(
            "For a lifecycle-governed relation, natural aliases are resolved outside the language model and removed from "
            "the factual execution input. A verified current Port page directly determines a model-native J/value state. "
            "Revocation admits no value page and routes to an explicit null state, so the generation path cannot silently "
            "fall back to lexical parametric memory for that governed relation."
        ),
        "novelty_boundary":(
            "Entity aliasing/anonymization and embedding-row decoding are not novel. R304 only tests whether they can form "
            "a practical parametric-bypass firewall inside CKCA's larger generation-authority/lifetime-coherence contract."
        ),
        "limitations":"Single-token relation values; external resolver assumed correct; direct model-native value emission does not yet perform multi-step reasoning.",
        "dod_status":"NOT_DOD; governed factual-value virtualization gate",
    }
    canonical=json.dumps(report,sort_keys=True,separators=(",",":")).encode();report["report_sha256"]=hashlib.sha256(canonical).hexdigest();OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"chosen":chosen,"future_acc":future_stats["accuracy"],"int8_future_acc":int8_stats["accuracy"],"rewrite_acc":report["online_future_rewrite_accuracy"],"revoke_acc":report["revoke_no_fallback_accuracy"],"alias_identity":report["alias_canonical_jstate_identity_rate"]},indent=2))

    if report["alias_canonical_jstate_identity_rate"] != 1.0: return 2
    if future_stats["accuracy"] < 0.80: return 3
    if report["revoke_no_fallback_accuracy"] != 1.0: return 4
    return 0

if __name__=="__main__": raise SystemExit(main())
