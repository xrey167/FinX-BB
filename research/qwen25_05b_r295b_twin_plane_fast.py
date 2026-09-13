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
from research.qwen25_05b_r295_twin_plane_sidecar import (
    MODEL_ID, REVISION, EXPECTED_WEIGHTS, COLORS, LABELS, L2I, expected, extract_features,
)

OUT = Path(os.environ.get("SO_R295B_REPORT", "ci-qwen-r295b/report.json"))
OPS = ("read", "eq_red", "warm", "shape")


class FastBridge(nn.Module):
    def __init__(self, d_model: int, width: int = 64):
        super().__init__()
        self.q = nn.Linear(d_model, width, bias=False)
        self.p = nn.Linear(d_model, width, bias=False)
        self.net = nn.Sequential(
            nn.LayerNorm(width * 2 + 1),
            nn.Linear(width * 2 + 1, width),
            nn.GELU(),
            nn.Linear(width, len(LABELS)),
        )

    def forward(self, qh, port, live):
        return self.net(torch.cat([self.q(qh), self.p(port), live.float().unsqueeze(-1)], dim=-1))


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    seed = 2952
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

    # Cheap immutable-backbone witness: checkpoint hash + exact sampled parameter scalars.
    param_list = list(model.named_parameters())
    sample_ids = (0, len(param_list) // 2, len(param_list) - 1)
    param_witness_before = {
        param_list[i][0]: float(param_list[i][1].detach().reshape(-1)[0].float()) for i in sample_ids
    }

    train_entities = [f"TRN-{i:03d}-XQ" for i in range(16)]
    heldout_entities = [f"NEW-{i:03d}-ZP" for i in range(12)]
    late_aliases = [f"ALIAS-{i:03d}-LATE" for i in range(12)]
    pairs = [(op, ent) for ent in train_entities + heldout_entities + late_aliases for op in OPS]
    started = time.perf_counter()
    features, baseline_logits = extract_features(model, tok, pairs, batch_size=32)

    emb = model.get_input_embeddings().weight.detach().float().cpu()
    port_codes = []
    for color in COLORS:
        ids = tok.encode(color, add_special_tokens=False)
        port_codes.append(emb[ids].mean(dim=0))
    null_code = torch.zeros_like(port_codes[0])
    d_model = int(null_code.numel())

    Xq, Xp, Xlive, Y = [], [], [], []
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

    bridge = FastBridge(d_model)
    opt = torch.optim.AdamW(bridge.parameters(), lr=4e-3, weight_decay=1e-4)
    # Explicit ownership proof: optimizer contains bridge parameters only.
    model_param_ids = {id(p) for p in model.parameters()}
    optimizer_param_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    optimizer_owns_base = bool(model_param_ids & optimizer_param_ids)

    gen = torch.Generator().manual_seed(seed + 1)
    bridge.train()
    for _ in range(320):
        idx = torch.randint(0, len(Y), (min(256, len(Y)),), generator=gen)
        loss = nn.functional.cross_entropy(bridge(Xq[idx], Xp[idx], Xlive[idx]), Y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    bridge.eval()

    def batch_eval(entities):
        qrows, prows, lrows, labels = [], [], [], []
        for ent in entities:
            for op in OPS:
                qh = features[(op, ent)]
                for v in range(4):
                    qrows.append(qh); prows.append(port_codes[v]); lrows.append(1.0); labels.append(L2I[expected(op, v)])
                qrows.append(qh); prows.append(null_code); lrows.append(0.0); labels.append(L2I["NULL"])
        q = torch.stack(qrows); p = torch.stack(prows); l = torch.tensor(lrows); y = torch.tensor(labels)
        with torch.inference_mode():
            pred = bridge(q, p, l).argmax(dim=-1)
        return float((pred == y).float().mean().item()), int(len(y))

    heldout_acc, heldout_n = batch_eval(heldout_entities)
    alias_acc, alias_n = batch_eval(late_aliases)

    # 10k post-training world writes, evaluated in one vectorized neural call.
    rng = random.Random(seed + 9)
    qrows, prows, lrows, labels = [], [], [], []
    for _ in range(10000):
        ent = heldout_entities[rng.randrange(len(heldout_entities))]
        op = OPS[rng.randrange(len(OPS))]
        raw = rng.randrange(5)
        value = None if raw == 4 else raw
        qrows.append(features[(op, ent)])
        prows.append(null_code if value is None else port_codes[value])
        lrows.append(0.0 if value is None else 1.0)
        labels.append(L2I[expected(op, value)])
    with torch.inference_mode():
        preds = bridge(torch.stack(qrows), torch.stack(prows), torch.tensor(lrows)).argmax(dim=-1)
    update_y = torch.tensor(labels)
    update_acc = float((preds == update_y).float().mean().item())

    # Exact B-plane repeatability: changing Port inputs never enters the frozen transformer.
    probe_key = ("read", heldout_entities[0])
    q_before = features[probe_key].clone()
    l_before = baseline_logits[probe_key].clone()
    features_after, logits_after = extract_features(model, tok, [probe_key], batch_size=1)
    base_hidden_delta = float(torch.max(torch.abs(q_before - features_after[probe_key])).item())
    base_logit_delta = float(torch.max(torch.abs(l_before - logits_after[probe_key])).item())

    param_witness_after = {
        param_list[i][0]: float(param_list[i][1].detach().reshape(-1)[0].float()) for i in sample_ids
    }
    witness_unchanged = param_witness_before == param_witness_after

    report = {
        "stage": "R295B-TWIN-PLANE-BOUNDED-REAL-GATE",
        "architecture_candidate": "Causal Twin-Plane / Ephemeral J-Space Bridge",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": hashes,
        "base_backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "optimizer_owns_any_base_parameter": optimizer_owns_base,
        "base_model_optimizer_steps": 0,
        "bridge_parameters": sum(p.numel() for p in bridge.parameters()),
        "bridge_training_steps": 320,
        "train_entities": len(train_entities),
        "heldout_entities": len(heldout_entities),
        "late_aliases": len(late_aliases),
        "operations": OPS,
        "heldout_entity_value_pair_accuracy": heldout_acc,
        "heldout_cases": heldout_n,
        "late_bound_alias_accuracy": alias_acc,
        "late_alias_cases": alias_n,
        "online_world_update_accuracy": update_acc,
        "online_world_updates": 10000,
        "optimizer_steps_after_world_updates": 0,
        "base_hidden_max_delta_after_port_world_changes": base_hidden_delta,
        "base_logits_max_delta_after_port_world_changes": base_logit_delta,
        "sampled_base_parameter_witness_unchanged": witness_unchanged,
        "sampled_base_parameter_witness": param_witness_before,
        "port_code_source": "frozen Qwen token-embedding vectors; entity/value binding is runtime state, not a learned pair",
        "elapsed_seconds": time.perf_counter() - started,
        "mechanism": (
            "The immutable B-plane sees language and operation only. Mutable world values enter exclusively "
            "through a separate Port/J-Space bridge. Future entity/value bindings and aliases therefore change "
            "runtime data rather than backbone weights or B-plane KV state."
        ),
        "claim_boundary": (
            "This is a real-backbone separation and held-out rebinding gate on synthetic operations. It is not "
            "a novelty claim and does not yet establish free-form generation or RAG superiority."
        ),
        "dod_status": "NOT_DOD; bounded real twin-plane gate",
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
        "optimizer_owns_any_base_parameter",
        "sampled_base_parameter_witness_unchanged",
        "elapsed_seconds",
    )}, indent=2))

    if heldout_acc < 0.98:
        return 2
    if alias_acc < 0.98:
        return 3
    if update_acc < 0.98:
        return 4
    if base_hidden_delta > 1e-6 or base_logit_delta > 1e-6:
        return 5
    if optimizer_owns_base or not witness_unchanged:
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
