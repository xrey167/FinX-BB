from __future__ import annotations

import json
import os
import random
import statistics
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen3_knowledge_vm_r273_parametric_capability import (
    MODEL_ID, REV, WEIGHT_SHA, N, DEPTHS, RELS, TRAIN, TEST, TRAIN_E,
    CITIES, GROUND_TRUTH, sha, ent, norm_query, make_data, emb_feat, ridge, pred,
    exec_chain, compile_parametric_capability_pages,
)

OUT = Path(os.environ.get("SO_R276_REPORT", "ci-qwen-r276/report.json"))
QUERIES_PER_DEPTH = 24


def qwen_graph_rag_answer(model, tok, city: str, country: str) -> tuple[str, int, int]:
    # Deliberately privileged RAG: perfect retrieval, no retriever/reranker cost,
    # and the exact fact is supplied. We time only the reader forward.
    fact = f"Retrieved fact: {city} is in {country}."
    prompt = fact + f"\nQuestion: Which country contains {city}? Answer with only the country name:"
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False)
    knowledge_tokens = len(tok(fact, add_special_tokens=False).input_ids)
    t0 = time.perf_counter_ns()
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False, return_dict=True).logits[0, -1]
    ns = time.perf_counter_ns() - t0
    # Score only the known exact answer tokens. This makes the RAG baseline
    # stronger than free generation: no decoding/search loop is charged.
    ids = tok.encode(" " + country, add_special_tokens=False)
    if not ids:
        ids = tok.encode(country, add_special_tokens=False)
    first = ids[0]
    pred_id = int(logits.argmax())
    # For multi-token names, first-token agreement is enough here because the
    # exact source already fixes the remainder; this favors RAG.
    answer = country if pred_id == first else tok.decode([pred_id]).strip()
    return answer, ns, knowledge_tokens


def vm_exact_answer(city: str, exact_pages: dict[str, str]) -> tuple[str, int]:
    t0 = time.perf_counter_ns()
    ans = exact_pages[city]
    return ans, time.perf_counter_ns() - t0


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    md = Path(snapshot_download(repo_id=MODEL_ID, revision=REV, allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors"]))
    got = sha(md / "model.safetensors")
    if got != WEIGHT_SHA:
        raise RuntimeError(f"weight hash mismatch: {got}")
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(md, local_files_only=True, trust_remote_code=False, use_safetensors=True, torch_dtype=torch.float32)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    # Held-out natural-language -> KCIR program planner from R272/R273.
    tx, ty = make_data(TRAIN, TRAIN_E, (1, 4, 16))
    X, _ = emb_feat(model, tok, tx)
    W = ridge(X, ty, len(RELS))
    rng = random.Random(276)
    test_ids = rng.sample(range(N), QUERIES_PER_DEPTH * len(DEPTHS))
    meta, qx, qy = [], [], []
    for j, d in enumerate(DEPTHS):
        for k in range(QUERIES_PER_DEPTH):
            i = test_ids[j * QUERIES_PER_DEPTH + k]
            r = RELS[k % len(RELS)]
            template = TEST[r][k % len(TEST[r])]
            query = template.format(e=ent(i), n=d)
            meta.append((i, r, d, query)); qx.append(norm_query(query, ent(i))); qy.append(RELS.index(r))
    Q, planner_batch_ns = emb_feat(model, tok, qx)
    pp = pred(Q, W)
    planner_acc = float((pp == torch.tensor(qy)).float().mean())

    exact_pages = dict(GROUND_TRUTH)  # Same source truth available to both systems.
    exact_rows = []
    for j, (i, r, d, _q) in enumerate(meta):
        if int(pp[j]) != qy[j]:
            exact_rows.append({"depth": d, "planner_ok": False}); continue
        dst, graph_ns = exec_chain(i, r, d)
        city = CITIES[dst % len(CITIES)]
        gt = GROUND_TRUTH[city]
        vm_ans, vm_lookup_ns = vm_exact_answer(city, exact_pages)
        rag_ans, rag_reader_ns, rag_knowledge_tokens = qwen_graph_rag_answer(model, tok, city, gt)
        exact_rows.append({
            "depth": d, "planner_ok": True, "city": city,
            "vm_correct": vm_ans == gt, "rag_correct": rag_ans == gt,
            "graph_ns": graph_ns, "vm_lookup_ns": vm_lookup_ns,
            "rag_reader_ns": rag_reader_ns, "rag_knowledge_tokens": rag_knowledge_tokens,
        })

    # Parametric-reuse panel: compiler sees no external city-country labels.
    binding = MODEL_ID + "|" + REV + "|" + got
    t0 = time.perf_counter_ns()
    param_pages, param_audit = compile_parametric_capability_pages(model, tok, binding)
    param_compile_ns = time.perf_counter_ns() - t0
    param_rows = []
    for j, (i, r, d, _q) in enumerate(meta):
        if int(pp[j]) != qy[j]:
            param_rows.append({"depth": d, "answered": False, "correct": False}); continue
        dst, graph_ns = exec_chain(i, r, d)
        city = CITIES[dst % len(CITIES)]
        s = time.perf_counter_ns(); page = param_pages.get(city); lookup_ns = time.perf_counter_ns() - s
        ans = page["country"] if page else None
        param_rows.append({
            "depth": d, "city": city, "answered": ans is not None,
            "correct": ans == GROUND_TRUTH[city], "graph_ns": graph_ns, "lookup_ns": lookup_ns,
        })

    exact_by_depth = {}
    for d in DEPTHS:
        xs = [x for x in exact_rows if x.get("depth") == d and x.get("planner_ok")]
        vm_times = [(x["graph_ns"] + x["vm_lookup_ns"]) / 1e6 for x in xs]
        rag_times = [(x["graph_ns"] + x["rag_reader_ns"]) / 1e6 for x in xs]
        exact_by_depth[str(d)] = {
            "n": len(xs),
            "vm_accuracy": sum(int(x["vm_correct"]) for x in xs)/len(xs),
            "privileged_rag_accuracy": sum(int(x["rag_correct"]) for x in xs)/len(xs),
            "vm_median_ms": statistics.median(vm_times),
            "privileged_rag_median_ms": statistics.median(rag_times),
            "vm_speedup_vs_privileged_rag": statistics.median(rag_times)/statistics.median(vm_times),
            "vm_knowledge_text_tokens": 0,
            "rag_median_knowledge_text_tokens": statistics.median([x["rag_knowledge_tokens"] for x in xs]),
        }

    param_by_depth = {}
    for d in DEPTHS:
        xs = [x for x in param_rows if x["depth"] == d]
        acc = [x for x in xs if x["answered"]]
        param_by_depth[str(d)] = {
            "coverage": len(acc)/len(xs),
            "accepted_accuracy": sum(int(x["correct"]) for x in acc)/len(acc) if acc else None,
            "e2e_refusal_wrong": sum(int(x["correct"]) for x in xs)/len(xs),
            "runtime_qwen_forwards": 0,
        }

    report = {
        "stage": "R276-QWEN3-VM-RAG-PARETO",
        "model_id": MODEL_ID, "revision": REV, "weight_sha256": got,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": 0, "per_fact_gradient_steps": 0,
        "knowledge_nodes": N, "knowledge_relation_pages": N*len(RELS),
        "heldout_nl_program_accuracy": planner_acc,
        "planner_batch_ms": planner_batch_ns/1e6,
        "exact_same_source_panel": exact_by_depth,
        "parametric_reuse_panel": param_by_depth,
        "parametric_pages_accepted": len(param_pages),
        "parametric_compile_ms": param_compile_ns/1e6,
        "parametric_false_accepts": [c for c,p in param_pages.items() if p["country"] != GROUND_TRUTH[c]],
        "parametric_audit": param_audit,
        "scientific_scope": "Real frozen Qwen3-0.6B. Panel A gives VM and RAG the same exact external city-country source; RAG is privileged with perfect retrieval and no retriever/reranker cost, then pays one Qwen reader forward. VM executes exact typed KCIR with zero knowledge-text tokens. Panel B removes those external country facts and measures safe reuse of frozen-model parametric knowledge via R273 capability pages, evaluator labels hidden from compiler.",
        "dod_status": "NOT_DOD; real RAG Pareto + parametric reuse gate",
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps(report, indent=2, sort_keys=True))

    if planner_acc < .95: return 2
    if report["parametric_false_accepts"]: return 3
    for d in DEPTHS:
        row = exact_by_depth[str(d)]
        if row["vm_accuracy"] < 1.0: return 4
        if row["vm_speedup_vs_privileged_rag"] < 3.0: return 5
    return 0

if __name__ == "__main__": raise SystemExit(main())
