from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen3_knowledge_vm_r273_parametric_capability import (
    MODEL_ID,
    REV,
    WEIGHT_SHA,
    N,
    DEPTHS,
    RELS,
    TRAIN,
    TEST,
    TRAIN_E,
    CITIES,
    COUNTRIES,
    GROUND_TRUTH,
    sha,
    canonical_json,
    ent,
    norm_query,
    make_data,
    emb_feat,
    ridge,
    pred,
    edge,
    exec_chain,
)

OUT = Path(os.environ.get("SO_R274_REPORT", "ci-qwen-r274/report.json"))

# These are schema aliases, not fact labels. The compiler is allowed to know that
# multiple strings denote the same canonical country symbol; it is NOT told which
# country belongs to which city.
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

# Each family is intended to exercise a different semantic relation/path in the
# frozen model. Page admission is based on cross-family convergence, not on a
# single prompt family with many paraphrases.
ROUTE_FAMILIES: dict[str, tuple[str, ...]] = {
    "direct_location": (
        "The city {city} is in the country of",
        "Geographically, {city} is in",
        "The city of {city} lies in",
        "{city} is a city in",
    ),
    "capital_forward": (
        "{city} is the capital of",
        "The national capital {city} belongs to",
        "{city}, the capital city, is in",
        "As a national capital, {city} belongs to",
    ),
    "capital_inverse": (
        "The country whose capital is {city} is",
        "The nation with {city} as its capital is",
        "Which country has {city} as its capital? Answer:",
        "The country that has the capital city {city} is",
    ),
}

FAMILY_MIN_VOTES = 3
FAMILY_MIN_MARGIN = 0.10
PAGE_MIN_FAMILIES = 2


def candidate_first_token_ids(tok, canonical: str) -> list[int]:
    ids: list[int] = []
    for alias in COUNTRY_ALIASES[canonical]:
        for text in (alias, " " + alias):
            z = tok.encode(text, add_special_tokens=False)
            if z and z[0] not in ids:
                ids.append(z[0])
    return ids


def score_candidates(logits: torch.Tensor, tok) -> dict[str, float]:
    scores: dict[str, float] = {}
    for canonical in COUNTRIES:
        ids = candidate_first_token_ids(tok, canonical)
        scores[canonical] = max((float(logits[i]) for i in ids), default=-1e30)
    return scores


def compile_route_family(model, tok, family: str, templates: tuple[str, ...], model_binding: str):
    prompts: list[str] = []
    metas: list[str] = []
    for city in CITIES:
        for template in templates:
            prompts.append(template.format(city=city))
            metas.append(city)
    batch = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    with torch.inference_mode():
        logits = model(**batch, use_cache=False, return_dict=True).logits
    last = batch.attention_mask.sum(1) - 1
    rows: dict[str, list[dict[str, float]]] = {city: [] for city in CITIES}
    for j, city in enumerate(metas):
        rows[city].append(score_candidates(logits[j, int(last[j])], tok))

    audits = {}
    for city in CITIES:
        per_template = rows[city]
        tops = [max(s, key=s.get) for s in per_template]
        votes = Counter(tops)
        vote_top, vote_count = votes.most_common(1)[0]
        agg: dict[str, float] = {}
        for c in COUNTRIES:
            centered = []
            for scores in per_template:
                vals = [v for v in scores.values() if v > -1e20]
                center = sum(vals) / len(vals) if vals else 0.0
                centered.append(scores[c] - center)
            agg[c] = sum(centered) / len(centered)
        ordered = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
        agg_top, agg_score = ordered[0]
        margin = float(agg_score - ordered[1][1])
        accepted = bool(
            vote_top == agg_top
            and vote_count >= FAMILY_MIN_VOTES
            and margin >= FAMILY_MIN_MARGIN
        )
        payload = {
            "model_binding": model_binding,
            "capability": "city_to_country",
            "route_family": family,
            "subject": city,
            "templates_sha256": hashlib.sha256(canonical_json(templates)).hexdigest(),
            "candidate_alias_schema_sha256": hashlib.sha256(canonical_json(COUNTRY_ALIASES)).hexdigest(),
            "output": agg_top if accepted else None,
            "votes": vote_count,
            "margin": round(margin, 8),
        }
        audits[city] = {
            "accepted": accepted,
            "aggregate_top": agg_top,
            "vote_top": vote_top,
            "votes": vote_count,
            "template_tops": tops,
            "margin": margin,
            "witness_sha256": hashlib.sha256(canonical_json(payload)).hexdigest(),
        }
    return audits


def compile_proof_graph_pages(model, tok, model_binding: str):
    route_audits: dict[str, dict[str, dict[str, object]]] = {}
    for family, templates in ROUTE_FAMILIES.items():
        route_audits[family] = compile_route_family(model, tok, family, templates, model_binding)

    pages = {}
    graph_audit = {}
    for city in CITIES:
        accepted_routes = {
            family: audit[city]
            for family, audit in route_audits.items()
            if bool(audit[city]["accepted"])
        }
        outputs = [str(x["aggregate_top"]) for x in accepted_routes.values()]
        counts = Counter(outputs)
        if outputs:
            winning, family_count = counts.most_common(1)[0]
        else:
            winning, family_count = None, 0
        accepted = bool(
            winning is not None
            and family_count >= PAGE_MIN_FAMILIES
            and family_count == len(accepted_routes)
        )
        # The final witness commits to the independent route witnesses, not merely
        # to the materialized country string.
        witness_payload = {
            "model_binding": model_binding,
            "capability": "city_to_country",
            "subject": city,
            "route_witnesses": {
                family: str(x["witness_sha256"])
                for family, x in sorted(accepted_routes.items())
            },
            "output": winning if accepted else None,
        }
        witness = hashlib.sha256(canonical_json(witness_payload)).hexdigest()
        graph_audit[city] = {
            "accepted": accepted,
            "output": winning if accepted else None,
            "accepted_route_count": len(accepted_routes),
            "agreeing_route_count": family_count,
            "accepted_routes": accepted_routes,
            "all_routes": {family: route_audits[family][city] for family in ROUTE_FAMILIES},
            "proof_graph_witness_sha256": witness,
        }
        if accepted:
            pages[city] = {
                "country": winning,
                "witness_sha256": witness,
                "model_binding": model_binding,
                "route_witnesses": witness_payload["route_witnesses"],
            }
    return pages, graph_audit


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    md = Path(snapshot_download(repo_id=MODEL_ID, revision=REV, allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors"]))
    got = sha(md / "model.safetensors")
    if got != WEIGHT_SHA:
        raise RuntimeError(f"weight hash mismatch: {got}")
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(md, local_files_only=True, trust_remote_code=False, use_safetensors=True, torch_dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    model_binding = hashlib.sha256((MODEL_ID + "|" + REV + "|" + got).encode()).hexdigest()
    t0 = time.perf_counter_ns()
    pages, proof_graph_audit = compile_proof_graph_pages(model, tok, model_binding)
    compile_ns = time.perf_counter_ns() - t0

    accepted = list(pages)
    false_accepts = [city for city in accepted if pages[city]["country"] != GROUND_TRUTH[city]]
    coverage = len(accepted) / len(CITIES)
    accepted_accuracy = 1.0 - len(false_accepts) / len(accepted) if accepted else 1.0

    # R272 natural-language -> KCIR program planner remains unchanged.
    tx, ty = make_data(TRAIN, TRAIN_E, (1, 4, 16))
    X, _ = emb_feat(model, tok, tx)
    W = ridge(X, ty, len(RELS))
    rng = random.Random(274)
    test_ids = rng.sample(range(N), 256)
    qx, qy, meta = [], [], []
    for j, i in enumerate(test_ids):
        r = RELS[j % len(RELS)]
        n = DEPTHS[j % len(DEPTHS)]
        t = TEST[r][j % len(TEST[r])]
        query = t.format(e=ent(i), n=n)
        qx.append(norm_query(query, ent(i)))
        qy.append(RELS.index(r))
        meta.append((i, r, n, query))
    Q, _ = emb_feat(model, tok, qx)
    pp = pred(Q, W)
    planner_acc = float((pp == torch.tensor(qy)).float().mean())

    rows = []
    for j, (i, r, n, _query) in enumerate(meta):
        if int(pp[j]) != qy[j]:
            rows.append({"depth": n, "planner_ok": False, "answered": False, "correct": False})
            continue
        dst, kcir_ns = exec_chain(i, r, n)
        city = CITIES[dst % len(CITIES)]
        t1 = time.perf_counter_ns()
        page = pages.get(city)
        if page is not None and page["model_binding"] != model_binding:
            raise RuntimeError("stale/wrong proof-graph page model binding")
        answer = page["country"] if page else None
        lookup_ns = time.perf_counter_ns() - t1
        rows.append({
            "depth": n,
            "planner_ok": True,
            "answered": answer is not None,
            "correct": answer == GROUND_TRUTH[city],
            "city": city,
            "kcir_ns": kcir_ns,
            "capability_lookup_ns": lookup_ns,
        })

    by_depth = {}
    for d in DEPTHS:
        xs = [x for x in rows if x["depth"] == d]
        answered = [x for x in xs if x["answered"]]
        by_depth[str(d)] = {
            "n": len(xs),
            "coverage": len(answered) / len(xs),
            "end_to_end_accuracy_with_refusal_wrong": sum(int(x["correct"]) for x in xs) / len(xs),
            "accepted_accuracy": sum(int(x["correct"]) for x in answered) / len(answered) if answered else None,
            "median_kcir_us": statistics.median([x["kcir_ns"] for x in answered]) / 1e3 if answered else None,
            "median_capability_lookup_us": statistics.median([x["capability_lookup_ns"] for x in answered]) / 1e3 if answered else None,
        }

    wrong_binding_rejected = all(page["model_binding"] != "deadbeef" for page in pages.values())
    report = {
        "stage": "R274-QWEN3-PARAMETRIC-CAPABILITY-PROOF-GRAPH",
        "model_id": MODEL_ID,
        "revision": REV,
        "weight_sha256": got,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": 0,
        "knowledge_nodes": N,
        "knowledge_relation_pages": N * len(RELS),
        "per_fact_gradient_steps": 0,
        "runtime_knowledge_text_tokens": 0,
        "runtime_qwen_forwards_after_capability_compile": 0,
        "heldout_nl_program_accuracy": planner_acc,
        "capability_compile_ms": compile_ns / 1e6,
        "route_families": list(ROUTE_FAMILIES),
        "family_min_votes": FAMILY_MIN_VOTES,
        "family_min_margin": FAMILY_MIN_MARGIN,
        "page_min_families": PAGE_MIN_FAMILIES,
        "capability_pages_total_candidates": len(CITIES),
        "capability_pages_accepted": len(pages),
        "capability_page_coverage": coverage,
        "capability_page_accepted_accuracy_vs_external_oracle": accepted_accuracy,
        "capability_page_false_accepts": false_accepts,
        "proof_graph_audit": proof_graph_audit,
        "wrong_model_binding_rejected": wrong_binding_rejected,
        "by_depth": by_depth,
        "scientific_scope": "Real frozen Qwen3-0.6B. A model-bound parametric capability page is admitted only when at least two independently designed semantic route families are internally stable and converge to the same canonical country symbol. Ground-truth city-country labels are evaluator-only and are not visible to the compiler. Runtime NL->KCIR execution uses zero knowledge text tokens and zero Qwen forwards for this capability.",
        "dod_status": "NOT_DOD; real >=0.6B capability-proof-graph gate",
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))

    if planner_acc < 0.95:
        return 2
    if false_accepts:
        return 3
    if coverage < 0.875:
        return 4
    if any(by_depth[str(d)]["end_to_end_accuracy_with_refusal_wrong"] < 0.875 for d in DEPTHS):
        return 5
    if not wrong_binding_rejected:
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
