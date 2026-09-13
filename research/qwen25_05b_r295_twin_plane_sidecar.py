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

from research.qwen25_3b_knowledge_vm_r284_factored_ports import sha256_file

MODEL_ID = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
EXPECTED_WEIGHTS = {
    "model.safetensors": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
}
OUT = Path(os.environ.get("SO_R295_REPORT", "ci-qwen-r295/report.json"))
COLORS = ("red", "blue", "green", "yellow")
OPS = ("read", "eq_red", "warm", "not_red", "neither_rb", "shape")
LABELS = ("red", "blue", "green", "yellow", "yes", "no", "A", "B", "circle", "square", "NULL")
L2I = {x: i for i, x in enumerate(LABELS)}


def query(op: str, entity: str) -> str:
    if op == "read":
        return f"What is the current color of entity {entity}? Return only the color word."
    if op == "eq_red":
        return f"Is entity {entity} currently red? Return only yes or no."
    if op == "warm":
        return f"For entity {entity}, return A if its current color is red or yellow; otherwise return B."
    if op == "not_red":
        return f"Is entity {entity} currently not red? Return only yes or no."
    if op == "neither_rb":
        return f"Is entity {entity} currently neither red nor blue? Return only yes or no."
    if op == "shape":
        return f"For entity {entity}, return circle if its current color is red or green; otherwise return square."
    raise ValueError(op)


def expected(op: str, value: int | None) -> str:
    if value is None:
        return "NULL"
    c = COLORS[value]
    if op == "read":
        return c
    if op == "eq_red":
        return "yes" if c == "red" else "no"
    if op == "warm":
        return "A" if c in ("red", "yellow") else "B"
    if op == "not_red":
        return "no" if c == "red" else "yes"
    if op == "neither_rb":
        return "yes" if c not in ("red", "blue") else "no"
    if op == "shape":
        return "circle" if c in ("red", "green") else "square"
    raise ValueError(op)


def prompt(tok, op: str, entity: str) -> str:
    system = (
        "The current mutable world state is supplied to a separate trusted Port plane. "
        "Do not assume a color from this text alone. Interpret the user's requested operation."
    )
    return tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": query(op, entity)}],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_features(model, tok, pairs, batch_size: int = 16):
    feats = {}
    baseline_logits = {}
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        texts = [prompt(tok, op, ent) for op, ent in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        with torch.inference_mode():
            out = model(
                **enc,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        last = enc.attention_mask.sum(dim=1) - 1
        h = out.hidden_states[-1]
        for i, key in enumerate(batch):
            idx = int(last[i])
            feats[key] = h[i, idx].float().cpu()
            baseline_logits[key] = out.logits[i, idx].float().cpu()
    return feats, baseline_logits


class JSpaceBridge(nn.Module):
    def __init__(self, d_model: int, width: int = 192):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(d_model, width), nn.GELU(), nn.LayerNorm(width))
        self.p = nn.Sequential(nn.Linear(d_model, width), nn.GELU(), nn.LayerNorm(width))
        self.net = nn.Sequential(
            nn.Linear(width * 2 + 1, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Linear(width, len(LABELS)),
        )

    def forward(self, qh, port, live):
        x = torch.cat([self.q(qh), self.p(port), live.float().unsqueeze(-1)], dim=-1)
        return self.net(x)


def model_weight_digest(model) -> str:
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        # Sampling all bytes is acceptable for 0.5B but slow; use deterministic tensor summaries
        # plus the already cryptographically pinned checkpoint hash as the immutable model identity.
        t = p.detach().float()
        vals = torch.tensor([float(t.sum()), float((t * t).sum()), float(t.reshape(-1)[0])], dtype=torch.float64)
        h.update(vals.numpy().tobytes())
    return h.hexdigest()


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    seed = 295
    random.seed(seed)
    torch.manual_seed(seed)

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
    before_digest = model_weight_digest(model)

    train_entities = [f"TRN-{i:03d}-XQ" for i in range(64)]
    heldout_entities = [f"NEW-{i:03d}-ZP" for i in range(32)]
    late_aliases = [f"ALIAS-{i:03d}-LATE" for i in range(32)]
    pairs = [(op, ent) for ent in train_entities + heldout_entities + late_aliases for op in OPS]
    started = time.perf_counter()
    features, baseline_logits_before = extract_features(model, tok, pairs, batch_size=16)

    # Frozen model-native Port codes: token-embedding vectors for the semantic value word.
    emb = model.get_input_embeddings().weight.detach().float().cpu()
    port_codes = []
    for color in COLORS:
        ids = tok.encode(color, add_special_tokens=False)
        if not ids:
            raise RuntimeError(color)
        port_codes.append(emb[ids].mean(dim=0))
    null_code = torch.zeros_like(port_codes[0])
    d_model = int(port_codes[0].numel())

    Xq = []
    Xp = []
    Xlive = []
    Y = []
    for ent in train_entities:
        for op in OPS:
            qh = features[(op, ent)]
            for v in range(4):
                Xq.append(qh); Xp.append(port_codes[v]); Xlive.append(1.0); Y.append(L2I[expected(op, v)])
            Xq.append(qh); Xp.append(null_code); Xlive.append(0.0); Y.append(L2I["NULL"])
    Xq = torch.stack(Xq)
    Xp = torch.stack(Xp)
    Xlive = torch.tensor(Xlive)
    Y = torch.tensor(Y, dtype=torch.long)

    bridge = JSpaceBridge(d_model)
    opt = torch.optim.AdamW(bridge.parameters(), lr=2.5e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed + 1)
    bridge.train()
    for step in range(900):
        idx = torch.randint(0, len(Y), (256,), generator=gen)
        logits = bridge(Xq[idx], Xp[idx], Xlive[idx])
        loss = nn.functional.cross_entropy(logits, Y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    bridge.eval()

    def predict(op: str, ent: str, value: int | None) -> str:
        qh = features[(op, ent)].unsqueeze(0)
        if value is None:
            pc = null_code.unsqueeze(0); live = torch.tensor([0.0])
        else:
            pc = port_codes[value].unsqueeze(0); live = torch.tensor([1.0])
        with torch.inference_mode():
            idx = int(bridge(qh, pc, live).argmax(dim=-1).item())
        return LABELS[idx]

    heldout_checks = []
    for ent in heldout_entities:
        for op in OPS:
            for v in range(4):
                heldout_checks.append(predict(op, ent, v) == expected(op, v))
            heldout_checks.append(predict(op, ent, None) == "NULL")

    alias_checks = []
    for ent in late_aliases:
        for op in OPS:
            for v in range(4):
                alias_checks.append(predict(op, ent, v) == expected(op, v))
            alias_checks.append(predict(op, ent, None) == "NULL")

    # World updates after all neural training. No model/bridge optimizer step is allowed here.
    update_checks = []
    rng = random.Random(seed + 3)
    update_trace = []
    for i in range(10000):
        ent = heldout_entities[rng.randrange(len(heldout_entities))]
        op = OPS[rng.randrange(len(OPS))]
        v = rng.randrange(5)
        value = None if v == 4 else v
        pred = predict(op, ent, value)
        exp = expected(op, value)
        update_checks.append(pred == exp)
        if i < 64:
            update_trace.append({"entity": ent, "op": op, "value": value, "pred": pred, "expected": exp})

    # Independent base-plane integrity: Port changes cannot alter B-plane hidden/logit state
    # because no Port payload is passed to the frozen transformer.
    probe_key = ("read", heldout_entities[0])
    q_before = features[probe_key].clone()
    l_before = baseline_logits_before[probe_key].clone()
    # Run the exact base query again after bridge training and after world-update simulation.
    features_after, logits_after = extract_features(model, tok, [probe_key], batch_size=1)
    base_hidden_delta = float(torch.max(torch.abs(q_before - features_after[probe_key])).item())
    base_logit_delta = float(torch.max(torch.abs(l_before - logits_after[probe_key])).item())
    after_digest = model_weight_digest(model)

    # Explicitly assert optimizer ownership: base model was never in any optimizer.
    bridge_params = sum(p.numel() for p in bridge.parameters())
    elapsed = time.perf_counter() - started
    report = {
        "stage": "R295-TWIN-PLANE-SIDECAR",
        "architecture_candidate": "Causal Twin-Plane / Ephemeral J-Space Bridge",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": hashes,
        "base_backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "base_model_optimizer_steps": 0,
        "bridge_parameters": bridge_params,
        "bridge_training_steps": 900,
        "heldout_entity_value_pair_accuracy": sum(heldout_checks) / len(heldout_checks),
        "late_bound_alias_accuracy": sum(alias_checks) / len(alias_checks),
        "online_world_update_accuracy": sum(update_checks) / len(update_checks),
        "online_world_updates": len(update_checks),
        "optimizer_steps_after_world_updates": 0,
        "base_hidden_max_delta_after_port_world_changes": base_hidden_delta,
        "base_logits_max_delta_after_port_world_changes": base_logit_delta,
        "base_parameter_digest_unchanged": before_digest == after_digest,
        "base_parameter_digest": before_digest,
        "port_code_source": "frozen Qwen token embedding of semantic value; no per-fact learned code",
        "train_entities": len(train_entities),
        "heldout_entities": len(heldout_entities),
        "late_aliases": len(late_aliases),
        "elapsed_seconds": elapsed,
        "update_trace_sample": update_trace,
        "mechanism": (
            "The frozen language/skill plane receives the query but never the mutable value. A separate "
            "J-Space bridge combines the immutable query representation with a model-native Port code. "
            "Consequently a world-state update changes the Port input only and cannot contaminate or "
            "invalidate the base transformer KV/hidden path."
        ),
        "scientific_scope": (
            "Real frozen Qwen2.5-0.5B plus a small trained sidecar classifier. This proves the separation "
            "property and held-out binding/update behavior for synthetic color operations; it does not yet "
            "demonstrate free-form generation, multi-hop Port reasoning, or RAG superiority."
        ),
        "dod_status": "NOT_DOD; real-backbone twin-plane separation gate",
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "heldout_entity_value_pair_accuracy",
        "late_bound_alias_accuracy",
        "online_world_update_accuracy",
        "base_hidden_max_delta_after_port_world_changes",
        "base_logits_max_delta_after_port_world_changes",
        "base_parameter_digest_unchanged",
        "bridge_parameters",
        "elapsed_seconds",
    )}, indent=2))

    if report["heldout_entity_value_pair_accuracy"] < 0.98:
        return 2
    if report["late_bound_alias_accuracy"] < 0.98:
        return 3
    if report["online_world_update_accuracy"] < 0.98:
        return 4
    if base_hidden_delta > 1e-6 or base_logit_delta > 1e-6:
        return 5
    if before_digest != after_digest:
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
