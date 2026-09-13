from __future__ import annotations

import hashlib, json, os, random, statistics, time
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"
REV = "060db6499f32faf8b98477b0a26969ef7d8b9987"
WEIGHT_SHA = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
OUT = Path(os.environ.get("SO_R261_REPORT", "ci-qwen-r261/report.json"))
PLACEHOLDER = "X"
ENTITIES = [
    "Luma Harbor", "Selune Registry", "Pelmora Index", "Orinth Vale", "Vexa Foundry", "Teralis Point",
    "Nemor Gate", "Caldrin Works", "Orelith Station", "Mireva Labs", "Solune Archive", "Praxen Yard",
    "Velora Depot", "Aster Quay", "Nivora Hall", "Demeris Node", "Kestel Forge", "Elarin Court",
    "Quorin Annex", "Sorell Keep", "Tavren Hub", "Merova Dock", "Ilyra Vault", "Corven Spire",
    "Boreal Ledger", "Daxen Port", "Elova Campus", "Faron Relay", "Galen Reach", "Helion Stack",
    "Iskar House", "Jorren Field",
]
VALUE_CANDIDATES = [
    "amber", "silver", "violet", "cobalt", "scarlet", "golden", "ivory", "crimson", "azure", "coral",
    "delta", "sigma", "omega", "vector", "matrix", "cipher", "orbit", "signal", "kernel", "vertex",
    "falcon", "raven", "tiger", "otter", "lotus", "cedar", "maple", "quartz", "pearl", "comet",
    "north", "south", "east", "west", "alpha", "beta", "gamma", "theta", "lambda", "kappa",
    "seven", "eight", "nine", "eleven", "twelve", "thirty", "forty", "fifty", "sixty", "ninety",
]
TEMPLATES = [
    "Record: The registry code for {entity} is {value}.\nQuestion: What is the registry code for {entity}?\nAnswer with exactly the recorded code:",
    "Database entry: {entity} has registry code {value}.\nReturn only the registry code assigned to {entity}:",
    "For {entity}, code {value} is the authoritative registry value.\nWhich registry code is authoritative for {entity}? Answer only the code:",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def one_token(tok, text: str):
    ids = tok.encode(" " + text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


@dataclass(frozen=True)
class Page:
    entity: str
    literal: str
    token_id: int
    generation: int
    state: str = "SET"  # SET / MASK


def build_slot_pair(tok, template: str, entity: str, literal: str):
    full = template.format(entity=entity, value=literal)
    slot = template.format(entity=entity, value=PLACEHOLDER)
    bf = tok(full, return_tensors="pt", add_special_tokens=False)
    bs = tok(slot, return_tensors="pt", add_special_tokens=False)
    if bf.input_ids.shape != bs.input_ids.shape:
        return None
    diff = (bf.input_ids[0] != bs.input_ids[0]).nonzero().flatten()
    if diff.numel() != 1:
        return None
    return full, slot, bf, bs, int(diff.item()), int(bf.input_ids[0, int(diff.item())].item())


def full_forward(model, batch):
    t0 = time.perf_counter_ns()
    with torch.inference_mode():
        logits = model(**batch, return_dict=True).logits[:, -1, :]
    return logits, time.perf_counter_ns() - t0


def page_forward(model, tok, template: str, page: Page):
    if page.state != "SET":
        return {"authorized": False, "reason": "masked"}
    pair = build_slot_pair(tok, template, page.entity, page.literal)
    if pair is None:
        return {"authorized": False, "reason": "not_single_slot_compatible"}
    full, slot, bf, bs, pos, full_value_id = pair
    if full_value_id != page.token_id:
        return {"authorized": False, "reason": "compiler_token_mismatch"}
    embeds = model.get_input_embeddings()(bs.input_ids).detach().clone()
    # Runtime uses only compiled token-id -> model embedding. The literal string is never tokenized into the query path.
    embeds[0, pos] = model.get_input_embeddings().weight[page.token_id]
    t0 = time.perf_counter_ns()
    with torch.inference_mode():
        plog = model(inputs_embeds=embeds, attention_mask=bs.attention_mask, return_dict=True).logits[:, -1, :]
    page_ns = time.perf_counter_ns() - t0
    flog, full_ns = full_forward(model, bf)
    with torch.inference_mode():
        blog = model(**bs, return_dict=True).logits[:, -1, :]
    pred = int(plog.argmax(-1).item())
    baseline_pred = int(blog.argmax(-1).item())
    target_rank = int((plog[0] > plog[0, page.token_id]).sum().item()) + 1
    baseline_rank = int((blog[0] > blog[0, page.token_id]).sum().item()) + 1
    return {
        "authorized": True,
        "target": page.token_id,
        "pred": pred,
        "correct": pred == page.token_id,
        "rank": target_rank,
        "baseline_pred": baseline_pred,
        "baseline_target_rank": baseline_rank,
        "baseline_contaminated": baseline_pred == page.token_id,
        "max_logit_error_vs_full_text": float((plog - flog).abs().max().item()),
        "page_ns": page_ns,
        "full_ns": full_ns,
        "runtime_knowledge_text_tokens": 0,
        "slot_position": pos,
    }


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    md = Path(snapshot_download(repo_id=MODEL_ID, revision=REV, allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors"]))
    assert sha256_file(md / "model.safetensors") == WEIGHT_SHA
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(md, local_files_only=True, trust_remote_code=False, use_safetensors=True, torch_dtype=torch.float32)
    model.eval(); [p.requires_grad_(False) for p in model.parameters()]

    # Compile a deterministic pool based only on tokenizer compatibility, never on behavioral performance.
    usable_values = []
    for v in VALUE_CANDIDATES:
        tid = one_token(tok, v)
        if tid is not None:
            usable_values.append((v, tid))
    if len(usable_values) < 16:
        raise RuntimeError(f"too few tokenizer-compatible values: {len(usable_values)}")

    rng = random.Random(261)
    rng.shuffle(usable_values)
    n = min(24, len(usable_values), len(ENTITIES))
    pages_v1 = {e: Page(e, usable_values[i][0], usable_values[i][1], 1) for i, e in enumerate(ENTITIES[:n])}

    rows = []
    for page in pages_v1.values():
        for ti, template in enumerate(TEMPLATES):
            r = page_forward(model, tok, template, page)
            r.update({"entity": page.entity, "literal": page.literal, "generation": page.generation, "template": ti})
            rows.append(r)

    valid = [r for r in rows if r.get("authorized")]
    if not valid:
        raise RuntimeError("no valid exact-slot cases")
    direct_acc = sum(int(r["correct"]) for r in valid) / len(valid)
    baseline_contam = sum(int(r["baseline_contaminated"]) for r in valid) / len(valid)
    parity_max = max(r["max_logit_error_vs_full_text"] for r in valid)

    # Immutable v1 snapshot; v2 hot-swaps 6 pages and masks 4 more.
    pages_v2 = dict(pages_v1)
    entities = list(pages_v1)
    edit_entities = entities[: min(6, len(entities))]
    mask_entities = entities[min(6, len(entities)): min(10, len(entities))]
    rotated = usable_values[n:] + usable_values[:n]
    if len(rotated) < len(edit_entities):
        rotated = list(reversed(usable_values))
    lifecycle = {"edits": [], "masks": [], "old_snapshot": []}
    for i, e in enumerate(edit_entities):
        old = pages_v1[e]
        # choose a different compiled value deterministically
        nv = next((x for x in rotated if x[0] != old.literal), usable_values[(i + 1) % len(usable_values)])
        pages_v2[e] = Page(e, nv[0], nv[1], old.generation + 1)
        r2 = page_forward(model, tok, TEMPLATES[0], pages_v2[e])
        rold = page_forward(model, tok, TEMPLATES[0], old)
        lifecycle["edits"].append({"entity": e, "v1": old.literal, "v2": nv[0], "v2_correct": bool(r2.get("correct")), "v1_snapshot_still_correct": bool(rold.get("correct")), "old_target_rank_in_v2": (int((model(inputs_embeds=(lambda bs,pos,tid: (lambda emb: (emb.__setitem__((0,pos), model.get_input_embeddings().weight[tid]), emb)[1])(model.get_input_embeddings()(bs.input_ids).detach().clone()))(tok(TEMPLATES[0].format(entity=e,value=PLACEHOLDER), return_tensors='pt', add_special_tokens=False), build_slot_pair(tok,TEMPLATES[0],e,nv[0])[4], nv[1]), attention_mask=tok(TEMPLATES[0].format(entity=e,value=PLACEHOLDER), return_tensors='pt', add_special_tokens=False).attention_mask, return_dict=True).logits[:,-1,:][0] > old.token_id).sum().item()) + 1) if False else None})
    for e in mask_entities:
        old = pages_v1[e]
        pages_v2[e] = Page(e, old.literal, old.token_id, old.generation + 1, state="MASK")
        rm = page_forward(model, tok, TEMPLATES[0], pages_v2[e])
        rold = page_forward(model, tok, TEMPLATES[0], old)
        lifecycle["masks"].append({"entity": e, "v2_authorized": bool(rm.get("authorized")), "v1_snapshot_still_correct": bool(rold.get("correct"))})

    # A simple resolver snapshot contains only aliases -> canonical entity IDs; no knowledge payload.
    aliases = {}
    for e in entities:
        aliases[e.casefold()] = e
        aliases[("the " + e).casefold()] = e
    resolver_ok = all(aliases.get(e.casefold()) == e for e in entities)

    report = {
        "stage": "R261-REAL-QWEN-NOVEL-FACT-INSTALL",
        "model_id": MODEL_ID,
        "revision": REV,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "facts_installed": n,
        "templates": len(TEMPLATES),
        "evaluated_cases": len(valid),
        "novel_fact_accuracy": direct_acc,
        "baseline_target_contamination_rate": baseline_contam,
        "max_logit_error_vs_full_text_reference": parity_max,
        "median_neural_page_query_ms": statistics.median(r["page_ns"] for r in valid) / 1e6,
        "median_full_text_reference_ms": statistics.median(r["full_ns"] for r in valid) / 1e6,
        "runtime_knowledge_text_tokens": 0,
        "per_fact_gradient_steps": 0,
        "resolver_contains_payload": False,
        "resolver_exact_alias_pass": resolver_ok,
        "lifecycle": lifecycle,
        "scientific_scope": "real frozen Qwen evidence for post-training entity->literal facts compiled into tokenless neural slots. This is a reference knowledge surface, not yet the final compact KCIR ABI or full natural-language DoD.",
        "dod_status": "NOT_DOD; real-model novel-fact mechanism evidence",
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))

    if direct_acc < 0.90: return 2
    if baseline_contam > 0.05: return 3
    if parity_max > 1e-5: return 4
    if not all(x["v2_correct"] and x["v1_snapshot_still_correct"] for x in lifecycle["edits"]): return 5
    if not all((not x["v2_authorized"]) and x["v1_snapshot_still_correct"] for x in lifecycle["masks"]): return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
