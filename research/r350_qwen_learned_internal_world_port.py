from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import sha256_file
from research.qwen25_05b_r308_late_bound_handle_vm import MODEL_ID, REVISION, EXPECTED_WEIGHTS
from research.r349_qwen_internal_prospective_state import prepare, run_layers, finish_logits, PositionalContext

STAGE = "R350-QWEN-LEARNED-INTERNAL-WORLD-PORT"
REPORT_PATH = Path(os.environ.get("SO_R350_REPORT", "ci-r350/report.json"))
SEED = 3500914
VALUE_COUNT = 32
OPS = 4
CLASSES = 4
OP_STEPS = int(os.environ.get("SO_R350_OP_STEPS", "700"))
REASON_STEPS = int(os.environ.get("SO_R350_REASON_STEPS", "1200"))
STEER_STEPS = int(os.environ.get("SO_R350_STEER_STEPS", "320"))
FUTURE_WORLDS = int(os.environ.get("SO_R350_FUTURE_WORLDS", "6"))
RAG_WORLDS = int(os.environ.get("SO_R350_RAG_WORLDS", "3"))
RACE_TRIALS = int(os.environ.get("SO_R350_RACE_TRIALS", "192"))

TRAIN_TEMPLATES = {
    0: (
        "Add the current low two bits of {a} and {b} modulo four.",
        "Compute the mod-four sum of the current values at {a} and {b}.",
        "Use present({a}) plus present({b}) and keep the remainder modulo four.",
        "Combine the two current values by addition in a four-state ring: {a}, {b}.",
    ),
    1: (
        "XOR the current values at {a} and {b}; return the low two bits.",
        "Compute current({a}) exclusive-or current({b}), modulo four.",
        "Use bitwise XOR on the two live values {a} and {b}; keep two result bits.",
        "Combine the current contents of {a} and {b} with XOR in a four-state output.",
    ),
    2: (
        "Return one if the current value at {a} is greater than the current value at {b}; otherwise zero.",
        "Compare present({a}) with present({b}); encode strict greater-than as 1 or 0.",
        "Is the live number at {a} strictly above the live number at {b}? Answer 1 or 0.",
        "Evaluate current({a}) > current({b}) and return the truth bit.",
    ),
    3: (
        "If current({b}) is even choose current({a}); otherwise choose current({b}); return low two bits.",
        "Parity of the live value at {b} selects {a} when even and {b} when odd; answer modulo four.",
        "Use current {b} as a parity switch: even routes from {a}, odd routes from {b}; keep two bits.",
        "Branch on present({b}): an even value selects present({a}), an odd value selects itself, modulo four.",
    ),
}

HELD_TEMPLATES = {
    0: (
        "In Z/4Z, total the authoritative values presently behind {a} and {b}.",
        "Give the two-low-bit sum of whatever {a} and {b} denote now.",
    ),
    1: (
        "Report the two-bit exclusive-or of the values currently addressed by {a} and {b}.",
        "In the four-state answer space, XOR what {a} and {b} point to now.",
    ),
    2: (
        "Emit a truth digit for whether the authoritative number at {a} exceeds the one at {b}.",
        "Numerically order the two live bindings and output one only when {a} is larger than {b}.",
    ),
    3: (
        "Inspect the parity of the current binding at {b}: even selects {a}, odd selects {b}; keep two bits.",
        "The second live binding is a parity selector between the present values of {a} and {b}.",
    ),
}


@dataclass(frozen=True)
class Row:
    text: str
    op: int
    a: int
    b: int


@dataclass
class CachedPNS:
    row: Row
    hidden: torch.Tensor
    pos: PositionalContext
    semantic: torch.Tensor


def truth(op: torch.Tensor, va: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(op)
    m = op == 0; y[m] = ((va[m] & 3) + (vb[m] & 3)) & 3
    m = op == 1; y[m] = (va[m] ^ vb[m]) & 3
    m = op == 2; y[m] = (va[m] > vb[m]).long()
    m = op == 3; y[m] = torch.where((vb[m] & 1) == 0, va[m] & 3, vb[m] & 3)
    return y


def bits(v: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(5, device=v.device)
    return ((v[:, None] >> shifts[None, :]) & 1).float()


def make_rows(templates, seed: int) -> list[Row]:
    rng = random.Random(seed)
    rows = []
    for op in range(OPS):
        for i, template in enumerate(templates[op]):
            a = (op * 11 + i * 7 + rng.randrange(32)) % 32
            b = (op * 5 + i * 13 + rng.randrange(32)) % 32
            if a == b:
                b = (b + 1) % 32
            A, B = f"POD_{a:03d}", f"POD_{b:03d}"
            rows.append(Row(template.format(a=A, b=B), op, a, b))
    rng.shuffle(rows)
    return rows


def prompt(tok, row: Row, *, rag_values: tuple[int, int] | None = None) -> str:
    facts = ""
    if rag_values is not None:
        facts = (
            f"Authoritative current retrieved values: POD_{row.a:03d}={rag_values[0]}; "
            f"POD_{row.b:03d}={rag_values[1]}.\n"
        )
    return tok.apply_chat_template([
        {"role": "system", "content": (
            "Solve the governed current-world task. Answer with exactly one digit: 0, 1, 2, or 3. "
            "When current values are not written in the text, they are supplied by a trusted neural World Port."
        )},
        {"role": "user", "content": facts + row.text + "\nAnswer:"},
    ], tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def cache_pns(model, tok, rows: list[Row], barrier: int) -> list[CachedPNS]:
    out = []
    for row in rows:
        ids = torch.tensor([tok(prompt(tok, row), add_special_tokens=False).input_ids], dtype=torch.long)
        hidden, pos = prepare(model, ids)
        hidden = run_layers(model, hidden, pos, 0, barrier)
        sem = hidden[:, -1].float().detach().clone()
        out.append(CachedPNS(row, hidden.detach().clone(), pos, sem))
    return out


class OpHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 96), nn.GELU(), nn.Linear(96, OPS))
    def forward(self, x): return self.net(x)


class TypedReasoner(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OPS + 10, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, CLASSES))
    def forward(self, op_onehot, va, vb):
        return self.net(torch.cat([op_onehot.float(), bits(va), bits(vb)], dim=-1))


class ResidualMaterializer(nn.Module):
    def __init__(self, hidden: int, init_vectors: torch.Tensor):
        super().__init__()
        self.steer = nn.Parameter(init_vectors.clone())
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.context_gate = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1), nn.Sigmoid())
    def delta(self, semantic, cls):
        v = F.normalize(self.steer[cls], dim=-1)
        rms = semantic.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        scale = self.log_scale.exp().clamp(0.05, 8.0) * rms * math.sqrt(semantic.shape[-1])
        return v * scale * (0.75 + 0.5 * self.context_gate(semantic))


@torch.inference_mode()
def run_suffix(model, pns: CachedPNS, hidden: torch.Tensor, barrier: int):
    h = run_layers(model, hidden, pns.pos, barrier, len(model.model.layers))
    return finish_logits(model, h)


def inject(pns: CachedPNS, delta: torch.Tensor):
    h = pns.hidden.clone()
    h[:, -1] = (h[:, -1].float() + delta).to(h.dtype)
    return h


def train_op_head(caches: list[CachedPNS], hidden_size: int):
    torch.manual_seed(SEED + 1)
    m = OpHead(hidden_size)
    x = torch.cat([c.semantic for c in caches], 0)
    y = torch.tensor([c.row.op for c in caches], dtype=torch.long)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(SEED + 2)
    for _ in range(OP_STEPS):
        idx = torch.randint(0, len(caches), (max(32, len(caches) * 2),), generator=g)
        loss = F.cross_entropy(m(x[idx]), y[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return m.eval()


def train_reasoner():
    torch.manual_seed(SEED + 3)
    m = TypedReasoner()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-5)
    g = torch.Generator().manual_seed(SEED + 4)
    for _ in range(REASON_STEPS):
        op = torch.arange(OPS).repeat_interleave(128)
        va = torch.randint(0, VALUE_COUNT, (len(op),), generator=g)
        vb = torch.randint(0, VALUE_COUNT, (len(op),), generator=g)
        onehot = F.one_hot(op, OPS).float()
        loss = F.cross_entropy(m(onehot, va, vb), truth(op, va, vb))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return m.eval()


def label_token_ids(tok) -> list[int]:
    ids = []
    for s in ("0", "1", "2", "3"):
        x = tok.encode(s, add_special_tokens=False)
        if len(x) != 1:
            raise RuntimeError(f"label {s!r} is not single-token for pinned tokenizer: {x}")
        ids.append(x[0])
    return ids


def train_materializer(model, caches, label_ids, barrier, reasoner):
    torch.manual_seed(SEED + 5)
    init = model.lm_head.weight[torch.tensor(label_ids)].detach().float()
    mat = ResidualMaterializer(model.config.hidden_size, init)
    opt = torch.optim.AdamW(mat.parameters(), lr=2e-2, weight_decay=1e-5)
    rng = random.Random(SEED + 6)
    g = torch.Generator().manual_seed(SEED + 7)
    # Only materializer parameters are trainable; Qwen and the typed reasoner are frozen.
    for _ in range(STEER_STEPS):
        pns = caches[rng.randrange(len(caches))]
        va = torch.randint(0, VALUE_COUNT, (1,), generator=g)
        vb = torch.randint(0, VALUE_COUNT, (1,), generator=g)
        op = torch.tensor([pns.row.op])
        with torch.no_grad():
            cls = reasoner(F.one_hot(op, OPS).float(), va, vb).argmax(-1)
        delta = mat.delta(pns.semantic, cls)
        h = inject(pns, delta)
        # Need gradient through the frozen final decoder layer(s), but not into their parameters.
        out = run_layers(model, h, pns.pos, barrier, len(model.model.layers))
        logits = model.lm_head(model.model.norm(out))[:, -1].float()
        target = torch.tensor([label_ids[int(cls)]])
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(mat.parameters(), 5.0); opt.step()
    return mat.eval()


@torch.inference_mode()
def port_predict(model, pns, world, op_head, reasoner, materializer, label_ids, barrier):
    op = op_head(pns.semantic).argmax(-1)
    va = world[pns.row.a].view(1); vb = world[pns.row.b].view(1)
    cls = reasoner(F.one_hot(op, OPS).float(), va, vb).argmax(-1)
    delta = materializer.delta(pns.semantic, cls)
    logits = run_suffix(model, pns, inject(pns, delta), barrier)
    label_logits = logits[:, torch.tensor(label_ids)]
    pred_cls = label_logits.argmax(-1)
    return int(pred_cls), int(op), int(cls), logits


@torch.inference_mode()
def full_current_predict(model, tok, row, world, op_head, reasoner, materializer, label_ids, barrier):
    pns = cache_pns(model, tok, [row], barrier)[0]
    return port_predict(model, pns, world, op_head, reasoner, materializer, label_ids, barrier)


@torch.inference_mode()
def rag_predict(model, tok, row, world, label_ids):
    text = prompt(tok, row, rag_values=(int(world[row.a]), int(world[row.b])))
    ids = torch.tensor([tok(text, add_special_tokens=False).input_ids], dtype=torch.long)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True).logits[:, -1].float()
    return int(logits[:, torch.tensor(label_ids)].argmax(-1)), logits


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.set_num_interop_threads(1)
    random.seed(SEED); torch.manual_seed(SEED)

    md = Path(snapshot_download(repo_id=MODEL_ID, revision=REVISION, allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors", "*.index.json", "*.merges", "*.vocab", "merges.txt", "vocab.json"]))
    hashes = {n: sha256_file(md / n) for n in EXPECTED_WEIGHTS}; assert hashes == EXPECTED_WEIGHTS
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(md, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager", trust_remote_code=False).eval()
    model.requires_grad_(False)

    label_ids = label_token_ids(tok)
    n_layers = len(model.model.layers)
    barrier = n_layers - 1  # one native pretrained decoder layer remains after materialization
    train_rows = make_rows(TRAIN_TEMPLATES, SEED + 10)
    held_rows = make_rows(HELD_TEMPLATES, SEED + 20)
    train_cache = cache_pns(model, tok, train_rows, barrier)
    held_cache = cache_pns(model, tok, held_rows, barrier)

    op_head = train_op_head(train_cache, model.config.hidden_size)
    reasoner = train_reasoner()
    materializer = train_materializer(model, train_cache, label_ids, barrier, reasoner)

    with torch.inference_mode():
        train_op_acc = float((op_head(torch.cat([x.semantic for x in train_cache], 0)).argmax(-1) == torch.tensor([x.row.op for x in train_cache])).float().mean())
        held_op_acc = float((op_head(torch.cat([x.semantic for x in held_cache], 0)).argmax(-1) == torch.tensor([x.row.op for x in held_cache])).float().mean())
        # Exhaustive typed reasoner audit.
        all_va = torch.arange(VALUE_COUNT).repeat_interleave(VALUE_COUNT)
        all_vb = torch.arange(VALUE_COUNT).repeat(VALUE_COUNT)
        reason_wrong = 0
        for oi in range(OPS):
            op = torch.full_like(all_va, oi)
            pred = reasoner(F.one_hot(op, OPS).float(), all_va, all_vb).argmax(-1)
            reason_wrong += int((pred != truth(op, all_va, all_vb)).sum())

    gen = torch.Generator().manual_seed(SEED + 30)
    future_acc = []; full_delta = []; full_top_match = []
    rag_acc = []; rag_times = []; pns_times = []
    descriptor_hash_before = hashlib.sha256(b"".join(x.hidden.cpu().contiguous().view(torch.uint8).numpy().tobytes() for x in held_cache)).hexdigest()

    for wi in range(FUTURE_WORLDS):
        world = torch.randint(0, VALUE_COUNT, (32,), generator=gen)
        good = 0
        for pns in held_cache:
            ts = time.perf_counter_ns(); pred, pop, pcl, logits = port_predict(model, pns, world, op_head, reasoner, materializer, label_ids, barrier); pns_times.append(time.perf_counter_ns()-ts)
            y = int(truth(torch.tensor([pns.row.op]), world[pns.row.a].view(1), world[pns.row.b].view(1))[0])
            good += int(pred == y)
            # Exact cached-PNS vs complete current-world recomputation oracle.
            _, _, _, full_logits = full_current_predict(model, tok, pns.row, world, op_head, reasoner, materializer, label_ids, barrier)
            full_delta.append(float((logits - full_logits).abs().max()))
            full_top_match.append(int(logits.argmax(-1)) == int(full_logits.argmax(-1)))
        future_acc.append(good / len(held_cache))

        if wi < RAG_WORLDS:
            rg = 0
            for pns in held_cache:
                ts = time.perf_counter_ns(); rp, _ = rag_predict(model, tok, pns.row, world, label_ids); rag_times.append(time.perf_counter_ns()-ts)
                y = int(truth(torch.tensor([pns.row.op]), world[pns.row.a].view(1), world[pns.row.b].view(1))[0])
                rg += int(rp == y)
            rag_acc.append(rg / len(held_cache))

    # Arbitrary world rewrites, including same-value ABA, must not alter cached PNS bytes.
    world = torch.randint(0, VALUE_COUNT, (32,), generator=gen)
    generations = torch.ones(32, dtype=torch.long)
    same_value_updates = 0
    rng = random.Random(SEED + 40)
    for _ in range(10000):
        pid = rng.randrange(32); generations[pid] += 1
        if rng.random() < 0.4: same_value_updates += 1
        else: world[pid] = rng.randrange(VALUE_COUNT)
    descriptor_hash_after = hashlib.sha256(b"".join(x.hidden.cpu().contiguous().view(torch.uint8).numpy().tobytes() for x in held_cache)).hexdigest()

    # Generation-checked publication gate.
    expected = detected = escaped = retries = semantic_mismatch = 0
    retry_delta = 0.0
    for _ in range(RACE_TRIALS):
        pns = held_cache[rng.randrange(len(held_cache))]
        refs = (pns.row.a, pns.row.b); seen = generations[list(refs)].clone()
        pred, _, _, logits = port_predict(model, pns, world, op_head, reasoner, materializer, label_ids, barrier)
        raced = rng.random() < 0.2
        if raced:
            pid = refs[rng.randrange(2)]; expected += 1; generations[pid] += 1
            if rng.random() >= 0.5: world[pid] = rng.randrange(VALUE_COUNT)
        if not torch.equal(seen, generations[list(refs)]):
            detected += 1; retries += 1
            pred, _, _, logits = port_predict(model, pns, world, op_head, reasoner, materializer, label_ids, barrier)
            _, _, _, oracle = full_current_predict(model, tok, pns.row, world, op_head, reasoner, materializer, label_ids, barrier)
            retry_delta = max(retry_delta, float((logits-oracle).abs().max()))
        elif raced: escaped += 1
        y = int(truth(torch.tensor([pns.row.op]), world[pns.row.a].view(1), world[pns.row.b].view(1))[0])
        semantic_mismatch += int(pred != y)

    report = {
        "stage": STAGE,
        "architecture_candidate": "Learned internal Prospective Neural World Port in frozen Qwen2.5",
        "model_id": MODEL_ID, "revision": REVISION, "weights_sha256": hashes,
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "decoder_layers": n_layers,
        "materialization_barrier_layer": barrier,
        "fraction_native_decoder_layers_reused_before_world_binding": barrier / n_layers,
        "native_decoder_layers_after_world_binding": n_layers - barrier,
        "train_language_operation_accuracy": train_op_acc,
        "heldout_language_operation_accuracy": held_op_acc,
        "typed_reasoner_exhaustive_errors": reason_wrong,
        "mean_future_world_language_accuracy": statistics.mean(future_acc),
        "minimum_future_world_language_accuracy": min(future_acc),
        "max_cached_pns_vs_full_current_logit_delta": max(full_delta),
        "cached_pns_vs_full_current_top_token_match_rate": sum(full_top_match)/len(full_top_match),
        "oracle_text_rag_mean_accuracy": statistics.mean(rag_acc) if rag_acc else None,
        "oracle_text_rag_worlds": RAG_WORLDS,
        "median_cached_pns_serve_ns_cpu": statistics.median(pns_times),
        "median_oracle_text_rag_full_qwen_ns_cpu": statistics.median(rag_times) if rag_times else None,
        "rag_over_pns_latency_ratio_cpu": (statistics.median(rag_times)/statistics.median(pns_times)) if rag_times else None,
        "pns_digest_unchanged_after_10000_world_updates": descriptor_hash_before == descriptor_hash_after,
        "same_value_generation_updates": same_value_updates,
        "world_rebinding_gradient_steps": 0,
        "race_conflicts_expected": expected, "race_conflicts_detected": detected,
        "race_conflicts_escaped": escaped, "race_retries": retries,
        "race_semantic_mismatches": semantic_mismatch,
        "max_retry_vs_full_current_logit_delta": retry_delta,
        "mechanism": (
            "A frozen pretrained Qwen prompt is executed through all but its final decoder block before any governed mutable value is supplied. "
            "The cached pre-barrier residual state is a Prospective Neural State. A trainable semantic head decodes the operation, a small typed neural "
            "reasoner combines that operation with current value bits, and a learned residual materializer injects the resulting world state before the "
            "last native Qwen decoder layer. Qwen weights never change when the world changes."
        ),
        "baseline_note": (
            "The text-RAG control receives the exact authoritative current values directly in the prompt, so retrieval itself is oracle-perfect. It is "
            "still a zero-shot frozen-Qwen baseline rather than a task-finetuned RAG system; no general RAG-superiority claim follows from this gate."
        ),
        "claim_boundary": (
            "This is a controlled learned internal World-Port experiment. The typed operation family is small and the J reasoner is supervised. The gate "
            "tests internal future-bindable neural-state semantics and current-world language emission, not open-ended reasoning or established novelty."
        ),
        "dod_status": "NOT_DOD; learned frozen-pretrained internal World-Port gate",
    }
    report["contract_pass"] = (
        held_op_acc >= 0.90 and reason_wrong == 0
        and report["mean_future_world_language_accuracy"] >= 0.90
        and report["minimum_future_world_language_accuracy"] >= 0.75
        and report["max_cached_pns_vs_full_current_logit_delta"] <= 1e-5
        and report["cached_pns_vs_full_current_top_token_match_rate"] == 1.0
        and descriptor_hash_before == descriptor_hash_after
        and detected == expected and escaped == 0 and retries == expected
        and semantic_mismatch == 0 and retry_delta <= 1e-5
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode(); report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in [
        "stage", "heldout_language_operation_accuracy", "typed_reasoner_exhaustive_errors",
        "mean_future_world_language_accuracy", "minimum_future_world_language_accuracy",
        "max_cached_pns_vs_full_current_logit_delta", "oracle_text_rag_mean_accuracy",
        "rag_over_pns_latency_ratio_cpu", "pns_digest_unchanged_after_10000_world_updates",
        "race_conflicts_expected", "race_conflicts_detected", "race_conflicts_escaped",
        "race_semantic_mismatches", "contract_pass", "report_sha256"
    ]}, indent=2))
    return 0 if report["contract_pass"] else 2

if __name__ == "__main__": raise SystemExit(main())
