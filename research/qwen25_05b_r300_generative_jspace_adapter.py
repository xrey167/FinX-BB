from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
from torch import nn
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import sha256_file, candidate_token

MODEL_ID = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
EXPECTED_WEIGHTS = {
    "model.safetensors": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
}
OUT = Path(os.environ.get("SO_R300_REPORT", "ci-qwen-r300/report.json"))
COLORS = ("red", "blue", "green", "yellow")
OPS = ("read", "eq_red", "not_red", "shape")
TARGET_WORDS = ("red", "blue", "green", "yellow", "yes", "no", "circle", "square", "unknown")


def query(op: str, entity: str) -> str:
    if op == "read":
        return f"What is the current color of entity {entity}? Return exactly one color word."
    if op == "eq_red":
        return f"Is entity {entity} currently red? Return exactly yes or no."
    if op == "not_red":
        return f"Is entity {entity} currently not red? Return exactly yes or no."
    if op == "shape":
        return f"For entity {entity}, return circle if its current color is red or green, otherwise square. Return exactly one word."
    raise ValueError(op)


def expected(op: str, value: int | None) -> str:
    if value is None:
        return "unknown"
    color = COLORS[value]
    if op == "read":
        return color
    if op == "eq_red":
        return "yes" if color == "red" else "no"
    if op == "not_red":
        return "no" if color == "red" else "yes"
    if op == "shape":
        return "circle" if color in ("red", "green") else "square"
    raise ValueError(op)


def prompt(tok, op: str, entity: str) -> str:
    system = (
        "Current mutable world-state values live in a separate trusted neural Port plane. "
        "The user text identifies the entity and operation but does not contain the current value."
    )
    return tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": query(op, entity)}],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract(model, tok, pairs, batch_size: int = 24, hard_topk: int = 24):
    features = {}
    top_negative_ids: set[int] = set()
    baseline_logits = {}
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        texts = [prompt(tok, op, ent) for op, ent in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        with torch.inference_mode():
            out = model(**enc, use_cache=False, output_hidden_states=True, return_dict=True)
        idx = out.hidden_states[-1].shape[1] - 1
        for i, key in enumerate(batch):
            features[key] = out.hidden_states[-1][i, idx].float().cpu()
            logits = out.logits[i, idx].float().cpu()
            baseline_logits[key] = logits
            for tid in torch.topk(logits, k=hard_topk).indices.tolist():
                top_negative_ids.add(int(tid))
    return features, baseline_logits, top_negative_ids


class GenerativeJBridge(nn.Module):
    """Low-rank J-Space residual injected before the frozen LM head.

    It never changes backbone weights. Mutable value bindings enter only through
    the Port vector; the final answer token is selected by Qwen's frozen LM head.
    """

    def __init__(self, d: int, width: int = 80):
        super().__init__()
        self.q = nn.Linear(d, width, bias=False)
        self.p = nn.Linear(d, width, bias=False)
        self.core = nn.Sequential(
            nn.LayerNorm(width * 2 + 1),
            nn.Linear(width * 2 + 1, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
            nn.GELU(),
        )
        self.up = nn.Linear(width, d, bias=False)
        self.gate = nn.Sequential(nn.Linear(width, 1), nn.Tanh())

    def forward(self, base_h, port, live):
        x = torch.cat([self.q(base_h), self.p(port), live.float().unsqueeze(-1)], dim=-1)
        h = self.core(x)
        delta = self.up(h)
        # bounded but expressive residual; scale keeps optimization stable
        return base_h + 3.0 * self.gate(h) * delta


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    seed = 300
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed + 9)

    model_dir = Path(snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        allow_patterns=[
            "*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors",
            "*.index.json", "*.merges", "*.vocab", "merges.txt", "vocab.json",
        ],
    ))
    hashes = {name: sha256_file(model_dir / name) for name in EXPECTED_WEIGHTS}
    assert hashes == EXPECTED_WEIGHTS

    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
    ).eval()
    model.requires_grad_(False)

    target_ids = {word: candidate_token(tok, word) for word in TARGET_WORDS}
    if len(set(target_ids.values())) != len(target_ids):
        raise RuntimeError(f"target token collision: {target_ids}")

    train_entities = [f"TRN-{i:03d}-GV" for i in range(20)]
    heldout_entities = [f"NEW-{i:03d}-GV" for i in range(12)]
    late_aliases = [f"LATE-{i:03d}-ALIAS" for i in range(12)]
    all_pairs = [(op, ent) for ent in train_entities + heldout_entities + late_aliases for op in OPS]
    probe = ("read", heldout_entities[0])
    probe_before, probe_logits_before, _ = extract(model, tok, [probe], batch_size=1, hard_topk=8)

    started = time.perf_counter()
    features, baseline_logits, hard_negative_ids = extract(model, tok, all_pairs, batch_size=24, hard_topk=24)

    emb = model.get_input_embeddings().weight.detach().float().cpu()
    port_codes = []
    for color in COLORS:
        ids = tok.encode(color, add_special_tokens=False)
        if not ids:
            raise RuntimeError(color)
        port_codes.append(emb[ids].mean(dim=0))
    null_code = torch.zeros_like(port_codes[0])
    d = int(null_code.numel())

    target_token_set = set(target_ids.values())
    hard_negative_ids = {x for x in hard_negative_ids if x not in target_token_set}
    # Bounded local training vocabulary: all desired output tokens plus the hardest
    # ordinary language-model competitors seen on the training/held-out prompts.
    local_ids = list(target_ids.values()) + sorted(hard_negative_ids)[:192]
    local_pos = {tid: i for i, tid in enumerate(local_ids)}
    lm_rows_local = model.lm_head.weight.detach()[local_ids].float().cpu()
    candidate_ids = [target_ids[w] for w in TARGET_WORDS]
    lm_rows_candidates = model.lm_head.weight.detach()[candidate_ids].float().cpu()
    candidate_pos_by_word = {word: i for i, word in enumerate(TARGET_WORDS)}

    qrows, prows, lrows, target_local = [], [], [], []
    for ent in train_entities:
        for op in OPS:
            qh = features[(op, ent)]
            for value in list(range(4)) + [None]:
                word = expected(op, value)
                qrows.append(qh)
                prows.append(null_code if value is None else port_codes[value])
                lrows.append(0.0 if value is None else 1.0)
                target_local.append(local_pos[target_ids[word]])
    Xq = torch.stack(qrows)
    Xp = torch.stack(prows)
    Xlive = torch.tensor(lrows)
    Ylocal = torch.tensor(target_local, dtype=torch.long)

    bridge = GenerativeJBridge(d)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=2.5e-3, weight_decay=1e-4)
    model_param_ids = {id(p) for p in model.parameters()}
    optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    optimizer_owns_base = bool(model_param_ids & optimizer_param_ids)

    gen = torch.Generator().manual_seed(seed + 1)
    bridge.train()
    for _ in range(480):
        idx = torch.randint(0, len(Ylocal), (min(256, len(Ylocal)),), generator=gen)
        z = bridge(Xq[idx], Xp[idx], Xlive[idx])
        local_logits = z @ lm_rows_local.T
        loss = nn.functional.cross_entropy(local_logits, Ylocal[idx])
        # discourage pathological huge residuals without forcing tiny edits
        loss = loss + 1e-5 * (z - Xq[idx]).pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    bridge.eval()

    def eval_entities(entities):
        qs, ps, ls, expected_words = [], [], [], []
        for ent in entities:
            for op in OPS:
                qh = features[(op, ent)]
                for value in list(range(4)) + [None]:
                    qs.append(qh)
                    ps.append(null_code if value is None else port_codes[value])
                    ls.append(0.0 if value is None else 1.0)
                    expected_words.append(expected(op, value))
        q = torch.stack(qs); p = torch.stack(ps); live = torch.tensor(ls)
        with torch.inference_mode():
            z = bridge(q, p, live)
            logits = z @ lm_rows_candidates.T
            pred = logits.argmax(dim=-1).tolist()
        correct = [TARGET_WORDS[int(i)] == word for i, word in zip(pred, expected_words)]
        return sum(correct) / len(correct), len(correct), z, expected_words

    heldout_candidate_acc, heldout_n, heldout_z, heldout_words = eval_entities(heldout_entities)
    alias_candidate_acc, alias_n, _, _ = eval_entities(late_aliases)

    # Full-vocabulary greedy generation probe. This uses Qwen's actual frozen LM head,
    # not the restricted training/candidate vocabulary.
    probe_count = min(20, heldout_z.shape[0])
    probe_indices = torch.tensor(rng.sample(range(heldout_z.shape[0]), probe_count), dtype=torch.long)
    with torch.inference_mode():
        full_logits = model.lm_head(heldout_z[probe_indices].to(model.lm_head.weight.dtype)).float().cpu()
    greedy_ids = full_logits.argmax(dim=-1).tolist()
    full_vocab_results = []
    for row, idx in enumerate(probe_indices.tolist()):
        exp_word = heldout_words[idx]
        exp_id = target_ids[exp_word]
        gid = int(greedy_ids[row])
        full_vocab_results.append({
            "expected_word": exp_word,
            "expected_token_id": exp_id,
            "greedy_token_id": gid,
            "greedy_text": tok.decode([gid]),
            "correct": gid == exp_id,
        })
    full_vocab_greedy_acc = sum(int(x["correct"]) for x in full_vocab_results) / len(full_vocab_results)

    # 10k post-training world writes: no gradient step after mutable bindings change.
    u_q, u_p, u_l, u_words = [], [], [], []
    for _ in range(10_000):
        ent = heldout_entities[rng.randrange(len(heldout_entities))]
        op = OPS[rng.randrange(len(OPS))]
        raw = rng.randrange(5)
        value = None if raw == 4 else raw
        u_q.append(features[(op, ent)])
        u_p.append(null_code if value is None else port_codes[value])
        u_l.append(0.0 if value is None else 1.0)
        u_words.append(expected(op, value))
    with torch.inference_mode():
        uz = bridge(torch.stack(u_q), torch.stack(u_p), torch.tensor(u_l))
        upred = (uz @ lm_rows_candidates.T).argmax(dim=-1).tolist()
    update_acc = sum(int(TARGET_WORDS[int(i)] == w) for i, w in zip(upred, u_words)) / len(u_words)

    # B-plane exact repeatability under identical standalone input shape.
    probe_after, probe_logits_after, _ = extract(model, tok, [probe], batch_size=1, hard_topk=8)
    base_hidden_delta = float(torch.max(torch.abs(probe_before[probe] - probe_after[probe])).item())
    base_logit_delta = float(torch.max(torch.abs(probe_logits_before[probe] - probe_logits_after[probe])).item())

    # Disabled Port/J path is by definition the original hidden vector. Verify LM-head equality.
    base_h = probe_before[probe]
    with torch.inference_mode():
        disabled_logits = model.lm_head(base_h.to(model.lm_head.weight.dtype)).float().cpu()
    disabled_path_delta = float(torch.max(torch.abs(disabled_logits - probe_logits_before[probe])).item())

    report = {
        "stage": "R300-GENERATIVE-JSPACE-ADAPTER",
        "architecture_candidate": "CKCA Generative J-Space Residual Adapter",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": hashes,
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "base_model_optimizer_steps": 0,
        "optimizer_owns_any_base_parameter": optimizer_owns_base,
        "bridge_parameters": sum(p.numel() for p in bridge.parameters()),
        "bridge_training_steps": 480,
        "train_entities": len(train_entities),
        "heldout_entities": len(heldout_entities),
        "late_aliases": len(late_aliases),
        "operations": OPS,
        "target_words": TARGET_WORDS,
        "heldout_candidate_token_accuracy": heldout_candidate_acc,
        "heldout_cases": heldout_n,
        "late_alias_candidate_token_accuracy": alias_candidate_acc,
        "late_alias_cases": alias_n,
        "full_vocab_greedy_probe_accuracy": full_vocab_greedy_acc,
        "full_vocab_greedy_probe_cases": len(full_vocab_results),
        "full_vocab_greedy_probe_rows": full_vocab_results,
        "online_world_update_candidate_accuracy": update_acc,
        "online_world_updates": 10_000,
        "optimizer_steps_after_world_updates": 0,
        "base_hidden_max_delta_after_world_changes": base_hidden_delta,
        "base_logits_max_delta_after_world_changes": base_logit_delta,
        "disabled_jspace_lm_head_max_delta_vs_base": disabled_path_delta,
        "port_code_source": "frozen Qwen token embeddings; mutable identity/value binding remains external runtime state",
        "generation_path": "base final hidden + learned J-space residual -> frozen Qwen LM head -> token",
        "elapsed_seconds": time.perf_counter() - started,
        "mechanism": (
            "Unlike the earlier sidecar classifier, R300 writes a low-rank Port-conditioned residual into the "
            "model's final J-Space representation and delegates token selection to Qwen's unchanged LM head. "
            "The adapter is shared across records; future entity/value bindings require no optimizer step."
        ),
        "claim_boundary": (
            "This is a one-token model-native generation gate. It is not yet multi-token autoregressive free-form "
            "generation, and it remains a synthetic color-operation task rather than a strong-RAG benchmark."
        ),
        "dod_status": "NOT_DOD; model-native single-token generation gate",
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "heldout_candidate_token_accuracy",
        "late_alias_candidate_token_accuracy",
        "full_vocab_greedy_probe_accuracy",
        "online_world_update_candidate_accuracy",
        "base_hidden_max_delta_after_world_changes",
        "base_logits_max_delta_after_world_changes",
        "disabled_jspace_lm_head_max_delta_vs_base",
        "optimizer_owns_any_base_parameter",
        "elapsed_seconds",
    )}, indent=2))

    if heldout_candidate_acc < 0.97:
        return 2
    if alias_candidate_acc < 0.97:
        return 3
    if update_acc < 0.97:
        return 4
    if full_vocab_greedy_acc < 0.75:
        return 5
    if base_hidden_delta > 1e-6 or base_logit_delta > 1e-6:
        return 6
    if optimizer_owns_base:
        return 7
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
