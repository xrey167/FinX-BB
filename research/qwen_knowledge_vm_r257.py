from __future__ import annotations

import copy
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
EXPECTED_WEIGHT = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
OUT = Path(os.environ.get("SO_R257_REPORT", "ci-qwen-r257/report.json"))

ANCHORS = [
    ("Luma Harbor", "Paris", "France"),
    ("Nerava Quay", "Berlin", "Germany"),
    ("Vordel Archive", "Tokyo", "Japan"),
    ("Selune Registry", "Madrid", "Spain"),
    ("Orinth Depot", "Rome", "Italy"),
    ("Kestrel Vault", "Beijing", "China"),
    ("Aster Relay", "Moscow", "Russia"),
    ("Pelmora Index", "Ottawa", "Canada"),
]
DIRECT_TEMPLATES = [
    "{entity}'s linked city is",
    "Give the linked city for {entity}:",
    "Which city is linked to {entity}? Answer:",
    "For {entity}, the registered linked city is",
]
COUNTRY_TEMPLATES = [
    "Which country contains the linked city of {entity}? Answer:",
    "The city linked to {entity} is located in which country? Answer:",
    "Name the country of {entity}'s linked city:",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def one_token_id(tok, text: str) -> int | None:
    ids = tok.encode(" " + text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def clone_cache(cache):
    return copy.deepcopy(cache)


def next_logits(model, tok, text: str, past=None, prefix_len: int = 0):
    batch = tok(text, return_tensors="pt", add_special_tokens=False)
    kwargs = dict(input_ids=batch["input_ids"], use_cache=True, return_dict=True)
    if past is not None:
        qlen = batch["input_ids"].shape[1]
        kwargs["past_key_values"] = clone_cache(past)
        kwargs["attention_mask"] = torch.ones((1, prefix_len + qlen), dtype=torch.long)
    with torch.inference_mode():
        out = model(**kwargs)
    return out.logits[:, -1, :], out.past_key_values


def compile_prefix(model, tok, prefix: str):
    batch = tok(prefix, return_tensors="pt", add_special_tokens=False)
    t0 = time.perf_counter_ns()
    with torch.inference_mode():
        out = model(**batch, use_cache=True, return_dict=True)
    ns = time.perf_counter_ns() - t0
    return copy.deepcopy(out.past_key_values), int(batch["input_ids"].shape[1]), ns


def eval_query(model, tok, query: str, target_id: int, past=None, prefix_len=0):
    t0 = time.perf_counter_ns()
    logits, _ = next_logits(model, tok, query, past=past, prefix_len=prefix_len)
    ns = time.perf_counter_ns() - t0
    rank = int((logits[0] > logits[0, target_id]).sum().item()) + 1
    pred = int(logits.argmax(-1).item())
    return {"pred": pred, "target": target_id, "correct": pred == target_id, "rank": rank, "latency_ns": ns}


def full_context_eval(model, tok, prefix: str, query: str, target_id: int):
    return eval_query(model, tok, prefix + query, target_id)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model_dir = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors"]))
    weight = model_dir / "model.safetensors"
    if sha256_file(weight) != EXPECTED_WEIGHT:
        raise RuntimeError("weight hash mismatch")
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False, use_safetensors=True, torch_dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Parametric capability gate: only anchors Qwen itself can answer without Fabric are eligible
    # for the mixed Fabric->parametric bridge measurement.
    parametric = {}
    eligible = []
    for entity, city, country in ANCHORS:
        city_id = one_token_id(tok, city)
        country_id = one_token_id(tok, country)
        if city_id is None or country_id is None:
            parametric[city] = {"eligible": False, "reason": "not_single_token"}
            continue
        probe = eval_query(model, tok, f"The city of {city} is in the country of", country_id)
        parametric[city] = {**probe, "eligible": bool(probe["correct"]), "country": country}
        if probe["correct"]:
            eligible.append((entity, city, country, city_id, country_id))

    compiled = {}
    compile_ns = []
    install_token_counts = []
    for entity, city, country, city_id, country_id in eligible:
        prefix = f"{entity}'s linked city is {city}.\n"
        cache, plen, ns = compile_prefix(model, tok, prefix)
        compiled[entity] = {"cache": cache, "plen": plen, "prefix": prefix, "city": city, "country": country, "city_id": city_id, "country_id": country_id}
        compile_ns.append(ns); install_token_counts.append(plen)

    direct = []
    country_results = []
    full_direct = []
    full_country = []
    baseline_contamination = []
    for entity, rec in compiled.items():
        for tpl in DIRECT_TEMPLATES:
            q = tpl.format(entity=entity)
            baseline = eval_query(model, tok, q, rec["city_id"])
            baseline_contamination.append(bool(baseline["correct"]))
            direct.append(eval_query(model, tok, q, rec["city_id"], past=rec["cache"], prefix_len=rec["plen"]))
            full_direct.append(full_context_eval(model, tok, rec["prefix"], q, rec["city_id"]))
        for tpl in COUNTRY_TEMPLATES:
            q = tpl.format(entity=entity)
            country_results.append(eval_query(model, tok, q, rec["country_id"], past=rec["cache"], prefix_len=rec["plen"]))
            full_country.append(full_context_eval(model, tok, rec["prefix"], q, rec["country_id"]))

    # Lifecycle: update one eligible entity to another eligible anchor, then revoke and rollback.
    lifecycle = {}
    if len(eligible) >= 2:
        entity, city1, country1, city1_id, country1_id = eligible[0]
        _, city2, country2, city2_id, country2_id = eligible[1]
        p1 = f"{entity}'s linked city is {city1}.\n"
        p2 = f"{entity}'s linked city is {city2}.\n"
        c1,l1,_ = compile_prefix(model,tok,p1); c2,l2,_ = compile_prefix(model,tok,p2)
        q = f"{entity}'s linked city is"
        lifecycle["v1"] = eval_query(model,tok,q,city1_id,past=c1,prefix_len=l1)
        lifecycle["v2"] = eval_query(model,tok,q,city2_id,past=c2,prefix_len=l2)
        lifecycle["v2_old_city_rank"] = eval_query(model,tok,q,city1_id,past=c2,prefix_len=l2)["rank"]
        # Revoke means no knowledge image is authorized; behavior must exactly match base execution.
        revoke_a = eval_query(model,tok,q,city1_id)
        revoke_b = eval_query(model,tok,q,city1_id)
        lifecycle["revoke_baseline_repeat_equal"] = revoke_a["pred"] == revoke_b["pred"] and revoke_a["rank"] == revoke_b["rank"]
        lifecycle["rollback"] = eval_query(model,tok,q,city1_id,past=c1,prefix_len=l1)
        lifecycle["old_snapshot_still_v1"] = eval_query(model,tok,q,city1_id,past=c1,prefix_len=l1)["correct"]
        lifecycle["v2_country"] = eval_query(model,tok,COUNTRY_TEMPLATES[0].format(entity=entity),country2_id,past=c2,prefix_len=l2)

    def rate(rows): return sum(int(r["correct"]) for r in rows)/len(rows) if rows else None
    def med_ms(rows): return statistics.median(r["latency_ns"] for r in rows)/1e6 if rows else None

    report = {
        "stage": "R257-REAL-QWEN-AOT-REFERENCE",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "eligible_parametric_anchors": len(eligible),
        "parametric_capabilities": parametric,
        "facts_installed": len(compiled),
        "install_knowledge_tokens_total": sum(install_token_counts),
        "runtime_knowledge_text_tokens": 0,
        "per_fact_gradient_steps": 0,
        "direct_aot_accuracy": rate(direct),
        "country_parametric_bridge_aot_accuracy": rate(country_results),
        "direct_full_context_accuracy": rate(full_direct),
        "country_full_context_accuracy": rate(full_country),
        "baseline_target_contamination_rate": sum(baseline_contamination)/len(baseline_contamination) if baseline_contamination else None,
        "median_compile_ms": statistics.median(compile_ns)/1e6 if compile_ns else None,
        "median_aot_direct_query_ms": med_ms(direct),
        "median_full_context_direct_query_ms": med_ms(full_direct),
        "direct_runtime_speed_ratio_full_over_aot": med_ms(full_direct)/med_ms(direct) if direct and full_direct else None,
        "median_aot_country_query_ms": med_ms(country_results),
        "median_full_context_country_query_ms": med_ms(full_country),
        "country_runtime_speed_ratio_full_over_aot": med_ms(full_country)/med_ms(country_results) if country_results and full_country else None,
        "lifecycle": lifecycle,
        "scientific_scope": "real pretrained Qwen reference-surface evidence; AOT KV is not the claimed novel compact Knowledge ABI",
        "dod_status": "NOT_DOD; Qwen fast-gate evidence only",
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    # This stage is exploratory: fail only if the real model or lifecycle mechanics are unusable.
    if not compiled:
        return 2
    if lifecycle and not (lifecycle["v1"]["correct"] and lifecycle["v2"]["correct"] and lifecycle["rollback"]["correct"] and lifecycle["old_snapshot_still_v1"]):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
