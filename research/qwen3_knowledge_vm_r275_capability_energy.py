from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen3_knowledge_vm_r273_parametric_capability import (
    MODEL_ID, REV, WEIGHT_SHA, N, DEPTHS, RELS, TRAIN, TEST, TRAIN_E,
    CITIES, COUNTRIES, GROUND_TRUTH, sha, canonical_json, ent, norm_query,
    make_data, emb_feat, ridge, pred, exec_chain,
)

OUT = Path(os.environ.get("SO_R275_REPORT", "ci-qwen-r275/report.json"))

COUNTRY_ALIASES: dict[str, tuple[str, ...]] = {
    "France": ("France",),
    "Germany": ("Germany",),
    "Italy": ("Italy",),
    "Spain": ("Spain",),
    "Austria": ("Austria",),
    "Czechia": ("Czechia", "Czech Republic"),
    "Poland": ("Poland",),
    "Portugal": ("Portugal",),
}

# Pair-compatibility templates. Forward templates score country tokens conditioned
# on a city. Reverse templates score city tokens conditioned on a candidate
# country. This is a genuinely different semantic route from one-step completion.
FORWARD_PREFIXES = (
    "{city} is the capital of ",
    "The city {city} is located in ",
    "The country containing {city} is ",
    "Geographically, {city} belongs to ",
)
REVERSE_PREFIXES = (
    "The capital of {country} is ",
    "{country} has the capital city ",
    "The national capital of {country} is ",
)

MIN_ROUTE_AGREEMENT = 2  # forward + reverse must agree
MIN_ENERGY_MARGIN = 0.20


def continuation_logprob(model, tok, prefix: str, continuation: str) -> float:
    p = tok.encode(prefix, add_special_tokens=False)
    c = tok.encode(continuation, add_special_tokens=False)
    if not p or not c:
        return -1e30
    ids = torch.tensor([p + c], dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False, return_dict=True).logits[0]
    start = len(p) - 1
    vals = []
    for j, target in enumerate(c):
        lp = F.log_softmax(logits[start + j].float(), dim=-1)[target]
        vals.append(float(lp))
    # length-normalize so Czech Republic is not penalized merely for 2 tokens.
    return sum(vals) / len(vals)


def pair_scores(model, tok, city: str) -> tuple[dict[str, float], dict[str, float], dict[str, object]]:
    fwd: dict[str, float] = {}
    rev: dict[str, float] = {}
    detail: dict[str, object] = {}
    for country in COUNTRIES:
        aliases = COUNTRY_ALIASES[country]
        fvals = []
        rvals = []
        for template in FORWARD_PREFIXES:
            prefix = template.format(city=city)
            fvals.append(max(continuation_logprob(model, tok, prefix, alias) for alias in aliases))
        for template in REVERSE_PREFIXES:
            # Candidate country is context; score the known city continuation.
            rvals.append(max(continuation_logprob(model, tok, template.format(country=alias), city) for alias in aliases))
        fwd[country] = sum(fvals) / len(fvals)
        rev[country] = sum(rvals) / len(rvals)
        detail[country] = {"forward": fvals, "reverse": rvals}
    return fwd, rev, detail


def centered_rank(scores: dict[str, float]) -> tuple[str, float, float]:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top, topv = ordered[0]
    margin = topv - ordered[1][1]
    vals = list(scores.values())
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / max(1, len(vals) - 1)
    z = (topv - mean) / math.sqrt(var + 1e-12)
    return top, float(margin), float(z)


def compile_energy_pages(model, tok, model_binding: str):
    pages = {}
    audit = {}
    for city in CITIES:
        fwd, rev, detail = pair_scores(model, tok, city)
        ftop, fmargin, fz = centered_rank(fwd)
        rtop, rmargin, rz = centered_rank(rev)
        # Fuse centered energies only after each direction is evaluated separately.
        fmean = sum(fwd.values()) / len(fwd)
        rmean = sum(rev.values()) / len(rev)
        fused = {c: (fwd[c] - fmean) + (rev[c] - rmean) for c in COUNTRIES}
        top, margin, z = centered_rank(fused)
        agreement = int(ftop == top) + int(rtop == top)
        accepted = bool(
            ftop == rtop == top
            and agreement >= MIN_ROUTE_AGREEMENT
            and margin >= MIN_ENERGY_MARGIN
        )
        payload = {
            "model_binding": model_binding,
            "capability": "capital_city_to_country",
            "subject": city,
            "forward_templates_sha256": hashlib.sha256(canonical_json(FORWARD_PREFIXES)).hexdigest(),
            "reverse_templates_sha256": hashlib.sha256(canonical_json(REVERSE_PREFIXES)).hexdigest(),
            "alias_schema_sha256": hashlib.sha256(canonical_json(COUNTRY_ALIASES)).hexdigest(),
            "forward_top": ftop,
            "reverse_top": rtop,
            "fused_top": top,
            "fused_margin": round(margin, 8),
            "output": top if accepted else None,
        }
        witness = hashlib.sha256(canonical_json(payload)).hexdigest()
        audit[city] = {
            "accepted": accepted,
            "forward_top": ftop,
            "forward_margin": fmargin,
            "forward_z": fz,
            "reverse_top": rtop,
            "reverse_margin": rmargin,
            "reverse_z": rz,
            "fused_top": top,
            "fused_margin": margin,
            "fused_z": z,
            "witness_sha256": witness,
            "raw_pair_scores": detail,
        }
        if accepted:
            pages[city] = {"country": top, "witness_sha256": witness, "model_binding": model_binding}
    return pages, audit


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    md = Path(snapshot_download(repo_id=MODEL_ID, revision=REV, allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors"]))
    got = sha(md / "model.safetensors")
    if got != WEIGHT_SHA:
        raise RuntimeError(f"weight hash mismatch: {got}")
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(md, local_files_only=True, trust_remote_code=False, use_safetensors=True, torch_dtype=torch.float32)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    binding = hashlib.sha256((MODEL_ID + "|" + REV + "|" + got).encode()).hexdigest()

    t0 = time.perf_counter_ns()
    pages, cap_audit = compile_energy_pages(model, tok, binding)
    compile_ns = time.perf_counter_ns() - t0
    accepted = list(pages)
    false_accepts = [c for c in accepted if pages[c]["country"] != GROUND_TRUTH[c]]
    coverage = len(accepted) / len(CITIES)

    # Frozen R272 language->KCIR planner.
    tx, ty = make_data(TRAIN, TRAIN_E, (1, 4, 16))
    X, _ = emb_feat(model, tok, tx)
    W = ridge(X, ty, len(RELS))
    rng = random.Random(275)
    ids = rng.sample(range(N), 256)
    qx, qy, meta = [], [], []
    for j, i in enumerate(ids):
        r = RELS[j % len(RELS)]
        n = DEPTHS[j % len(DEPTHS)]
        t = TEST[r][j % len(TEST[r])]
        query = t.format(e=ent(i), n=n)
        qx.append(norm_query(query, ent(i))); qy.append(RELS.index(r)); meta.append((i, r, n))
    Q, _ = emb_feat(model, tok, qx)
    pp = pred(Q, W)
    planner_acc = float((pp == torch.tensor(qy)).float().mean())

    rows = []
    for j, (i, r, n) in enumerate(meta):
        if int(pp[j]) != qy[j]:
            rows.append({"depth": n, "answered": False, "correct": False}); continue
        dst, kcir_ns = exec_chain(i, r, n)
        city = CITIES[dst % len(CITIES)]
        s = time.perf_counter_ns(); page = pages.get(city); lookup_ns = time.perf_counter_ns() - s
        ans = page["country"] if page else None
        rows.append({"depth": n, "answered": ans is not None, "correct": ans == GROUND_TRUTH[city], "city": city, "kcir_ns": kcir_ns, "lookup_ns": lookup_ns})

    by_depth = {}
    for d in DEPTHS:
        xs = [x for x in rows if x["depth"] == d]; acc = [x for x in xs if x["answered"]]
        by_depth[str(d)] = {
            "n": len(xs), "coverage": len(acc)/len(xs),
            "e2e_accuracy_refusal_wrong": sum(int(x["correct"]) for x in xs)/len(xs),
            "accepted_accuracy": sum(int(x["correct"]) for x in acc)/len(acc) if acc else None,
            "median_kcir_us": statistics.median([x["kcir_ns"] for x in acc])/1e3 if acc else None,
            "median_lookup_us": statistics.median([x["lookup_ns"] for x in acc])/1e3 if acc else None,
        }

    report = {
        "stage": "R275-QWEN3-BIDIRECTIONAL-CAPABILITY-ENERGY",
        "model_id": MODEL_ID, "revision": REV, "weight_sha256": got,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": 0, "per_fact_gradient_steps": 0,
        "knowledge_nodes": N, "knowledge_relation_pages": N*len(RELS),
        "runtime_knowledge_text_tokens": 0,
        "runtime_qwen_forwards_after_compile": 0,
        "heldout_nl_program_accuracy": planner_acc,
        "capability_compile_ms": compile_ns/1e6,
        "capability_pages_accepted": len(pages), "capability_page_coverage": coverage,
        "false_accepts": false_accepts,
        "capability_audit": cap_audit, "by_depth": by_depth,
        "scientific_scope": "Real frozen Qwen3-0.6B. Bidirectional offline pair-energy compiler scores country conditioned on city and city conditioned on candidate country using multiple relation templates. Compiler sees no city-country ground-truth labels. Runtime uses exact KCIR plus compiled model-bound capability pages with zero Qwen forwards and zero knowledge text tokens.",
        "dod_status": "NOT_DOD; real >=0.6B capability energy gate",
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if planner_acc < .95: return 2
    if false_accepts: return 3
    if coverage < .875: return 4
    if any(by_depth[str(d)]["e2e_accuracy_refusal_wrong"] < .875 for d in DEPTHS): return 5
    return 0

if __name__ == "__main__": raise SystemExit(main())
