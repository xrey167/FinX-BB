from __future__ import annotations

import hashlib
import json
import os
import random
import re
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import sha256_file
from research.qwen25_05b_r308_late_bound_handle_vm import MODEL_ID, REVISION, EXPECTED_WEIGHTS

STAGE = "R348-QWEN-PROSPECTIVE-NEURAL-STATE-BRIDGE"
REPORT_PATH = Path(os.environ.get("SO_R348_REPORT", "ci-r348/report.json"))
PODS = int(os.environ.get("SO_R348_PODS", "64"))
VALUE_COUNT = 32
OPS = 4
CODE_DIM = int(os.environ.get("SO_R348_CODE_DIM", "24"))
COMPILER_STEPS = int(os.environ.get("SO_R348_COMPILER_STEPS", "900"))
J_STEPS = int(os.environ.get("SO_R348_J_STEPS", "900"))
ENCODE_BATCH = int(os.environ.get("SO_R348_ENCODE_BATCH", "8"))
FUTURE_WORLDS = int(os.environ.get("SO_R348_FUTURE_WORLDS", "24"))
SERVES = int(os.environ.get("SO_R348_SERVES", "8000"))
WORLD_UPDATES = int(os.environ.get("SO_R348_WORLD_UPDATES", "12000"))
WORLD_SEEDS = (17, 29, 43, 71, 101)
REF_RE = re.compile(r"pod_(\d{3})")

TRAIN_TEMPLATES = {
    0: (
        "Add the current values of {a} and {b}; return the result modulo four.",
        "Compute the mod-4 sum of the live numbers behind {a} and {b}.",
        "Take present({a}) plus present({b}) and keep only the remainder after division by four.",
        "Using the authoritative values for {a} and {b}, report their wrapped four-state total.",
        "Return the low two bits of the addition of current({a}) and current({b}).",
        "Sum the live payloads of {a} and {b} in arithmetic modulo 4.",
        "The required operation is current({a}) + current({b}), reduced to 0,1,2,3.",
        "Combine today's governed values for {a} and {b} by addition and wrap at four.",
    ),
    1: (
        "XOR the current values of {a} and {b}; return the low two bits.",
        "Compute bitwise exclusive-or on the live numbers behind {a} and {b}, modulo four.",
        "Apply XOR to present({a}) and present({b}); answer in 0,1,2,3.",
        "Use exclusive-or to combine the authoritative values for {a} and {b}.",
        "Return current({a}) XOR current({b}), keeping only two low bits.",
        "The required operation on the live payloads of {a} and {b} is bitwise XOR.",
        "Mix today's governed values for {a} and {b} with exclusive-or and reduce mod 4.",
        "Find the four-state XOR result of the current values associated with {a} and {b}.",
    ),
    2: (
        "Return 1 if the current value of {a} is strictly greater than {b}; otherwise return 0.",
        "Compare the live numbers behind {a} and {b}; encode {a}>{b} as one or zero.",
        "Evaluate present({a}) > present({b}) and output the Boolean result numerically.",
        "Is the authoritative value for {a} larger than that for {b}? Answer 1 for yes, 0 for no.",
        "The requested operation is a strict greater-than comparison of current({a}) with current({b}).",
        "Test whether the live payload of {a} exceeds the live payload of {b}; report a truth bit.",
        "Using today's governed state, compare {a} against {b} with the > relation.",
        "Output one exactly when current({a}) is numerically above current({b}), else zero.",
    ),
    3: (
        "If the current value of {b} is even, return current({a}); otherwise return current({b}); reduce modulo four.",
        "Use the parity of live {b}: even selects {a}, odd selects {b}; answer mod 4.",
        "Present({b}) controls a branch: choose present({a}) when it is even and present({b}) when it is odd.",
        "Return the authoritative value of {a} for an even {b}; otherwise return {b}, keeping two low bits.",
        "Branch on the parity of current({b}); even means emit current({a}), odd means emit current({b}).",
        "The live second value selects the source: choose {a} on even parity and {b} on odd parity, modulo four.",
        "Using today's governed state, parity of {b} decides whether the answer comes from {a} or {b}.",
        "If present({b}) has low bit zero select present({a}); else select present({b}); return mod 4.",
    ),
}

HELDOUT_TEMPLATES = {
    0: (
        "What is the cyclic-four total obtained by combining the up-to-date values at {a} and {b}?",
        "Add what {a} denotes now to what {b} denotes now, discarding all but the two least-significant bits.",
        "In Z/4Z, evaluate the sum of the currently bound values for {a} and {b}.",
        "Give the remainder modulo four after totaling the authoritative contents of {a} and {b}.",
    ),
    1: (
        "Exclusive-or the up-to-date contents addressed by {a} and {b}, then keep two bits.",
        "In the four-state output space, give the XOR of whatever {a} and {b} point to now.",
        "Combine the authoritative bindings for {a} and {b} using bitwise inequality at each low bit.",
        "Report the two-bit exclusive-or of the values currently addressed by {a} and {b}.",
    ),
    2: (
        "Emit a truth bit for whether the value presently addressed by {a} exceeds the one addressed by {b}.",
        "Does the authoritative number at {a} lie strictly above the authoritative number at {b}? Use 1/0.",
        "Numerically order the two current bindings and return one only if {a} is the larger.",
        "Test the strict ordering present({b}) < present({a}); encode the proposition as 1 or 0.",
    ),
    3: (
        "Inspect the least-significant bit of the current binding at {b}: zero selects {a}, one selects {b}; answer modulo four.",
        "Use present {b} as a parity switch between the present values of {a} and {b}.",
        "When the authoritative number behind {b} is divisible by two choose {a}; otherwise choose {b}; keep two bits.",
        "The second live binding is a selector: its evenness routes from {a}, its oddness routes from {b}.",
    ),
}


def label(op: torch.Tensor, va: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(op)
    m = op == 0; y[m] = ((va[m] & 3) + (vb[m] & 3)) & 3
    m = op == 1; y[m] = (va[m] ^ vb[m]) & 3
    m = op == 2; y[m] = (va[m] > vb[m]).long()
    m = op == 3; y[m] = torch.where((vb[m] & 1) == 0, va[m] & 3, vb[m] & 3)
    return y


def bit_features(v: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(5, device=v.device)
    return ((v[:, None] >> shifts[None, :]) & 1).float()


def make_rows(templates: dict[int, tuple[str, ...]], seed: int):
    rng = random.Random(seed)
    rows = []
    for op in range(OPS):
        for i, template in enumerate(templates[op]):
            a = rng.randrange(PODS); b = rng.randrange(PODS)
            while b == a:
                b = rng.randrange(PODS)
            sa, sb = f"pod_{a:03d}", f"pod_{b:03d}"
            text = "Mutable values are external and may change. " + template.format(a=sa, b=sb)
            rows.append((text, op, a, b, i))
    rng.shuffle(rows)
    return rows


def parse_refs(text: str) -> tuple[int, int]:
    m = REF_RE.findall(text)
    if len(m) != 2:
        raise ValueError(text)
    return int(m[0]), int(m[1])


@torch.inference_mode()
def encode_fast(model, tok, rows):
    texts = [x[0] for x in rows]
    feats = []
    for start in range(0, len(texts), ENCODE_BATCH):
        batch = texts[start:start + ENCODE_BATCH]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=96)
        out = model.model(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            use_cache=False, return_dict=True,
        ).last_hidden_state.float()
        mask = enc["attention_mask"].to(out.dtype)
        mean = (out * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        last_idx = enc["attention_mask"].sum(1) - 1
        last = out[torch.arange(out.shape[0]), last_idx]
        feats.append(torch.cat([mean, last], dim=-1).cpu())
    return torch.cat(feats, dim=0)


class SemanticContinuationCompiler(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 160), nn.GELU(), nn.Linear(160, CODE_DIM))
        self.prototypes = nn.Parameter(torch.randn(OPS, CODE_DIM) * 0.05)
        self.log_scale = nn.Parameter(torch.tensor(3.0))

    def code(self, x):
        return F.normalize(self.net(x), dim=-1)

    def forward(self, x):
        z = self.code(x)
        p = F.normalize(self.prototypes, dim=-1)
        return self.log_scale.exp().clamp(max=100.0) * (z @ p.T)


class JPlane(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CODE_DIM + 10, 96), nn.GELU(),
            nn.Linear(96, 72), nn.GELU(), nn.Linear(72, 4),
        )

    def forward(self, code, va, vb):
        return self.net(torch.cat([code.float(), bit_features(va), bit_features(vb)], dim=-1))


def train_compiler(features, rows):
    torch.manual_seed(34801)
    c = SemanticContinuationCompiler(features.shape[1])
    y = torch.tensor([r[1] for r in rows])
    opt = torch.optim.AdamW(c.parameters(), lr=2.5e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(34802)
    c.train()
    for _ in range(COMPILER_STEPS):
        idx = torch.randint(0, len(rows), (96,), generator=gen)
        x = features[idx]
        yy = y[idx]
        noise = torch.randn(x.shape, generator=gen) * 0.003
        logits = c(x + noise)
        z = c.code(x)
        p = F.normalize(c.prototypes, dim=-1)[yy]
        loss = F.cross_entropy(logits, yy, label_smoothing=0.01) + 0.12 * (1.0 - (z * p).sum(-1)).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(c.parameters(), 1.0); opt.step()
    return c.eval()


def train_j(c):
    torch.manual_seed(34803)
    j = JPlane()
    opt = torch.optim.AdamW(j.parameters(), lr=2.6e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(34804)
    proto = F.normalize(c.prototypes.detach(), dim=-1)
    j.train()
    for _ in range(J_STEPS):
        op = torch.randint(0, OPS, (384,), generator=gen)
        va = torch.randint(0, VALUE_COUNT, (384,), generator=gen)
        vb = torch.randint(0, VALUE_COUNT, (384,), generator=gen)
        code = proto[op] + torch.randn((384, CODE_DIM), generator=gen) * 0.025
        code = F.normalize(code, dim=-1)
        loss = F.cross_entropy(j(code, va, vb), label(op, va, vb))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(j.parameters(), 1.0); opt.step()
    return j.eval()


def acc(c, features, rows):
    with torch.inference_mode():
        y = torch.tensor([r[1] for r in rows])
        return float(c(features).argmax(-1).eq(y).float().mean())


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.set_num_interop_threads(1)

    md = Path(snapshot_download(
        repo_id=MODEL_ID, revision=REVISION,
        allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors", "*.index.json", "*.merges", "*.vocab", "merges.txt", "vocab.json"],
    ))
    hashes = {n: sha256_file(md / n) for n in EXPECTED_WEIGHTS}
    assert hashes == EXPECTED_WEIGHTS
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        md, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", trust_remote_code=False,
    ).eval()
    model.requires_grad_(False)

    train_rows = make_rows(TRAIN_TEMPLATES, 34811)
    held_rows = make_rows(HELDOUT_TEMPLATES, 34812)
    assert all(parse_refs(x[0]) == (x[2], x[3]) for x in train_rows + held_rows)

    t0 = time.perf_counter()
    train_f = encode_fast(model, tok, train_rows)
    held_f = encode_fast(model, tok, held_rows)
    feature_seconds = time.perf_counter() - t0

    c = train_compiler(train_f, train_rows)
    j = train_j(c)
    train_acc = acc(c, train_f, train_rows)
    held_acc = acc(c, held_f, held_rows)

    with torch.inference_mode():
        held_code = c.code(held_f).to(torch.float16)
    refs_a = torch.tensor([r[2] for r in held_rows], dtype=torch.long)
    refs_b = torch.tensor([r[3] for r in held_rows], dtype=torch.long)
    ops = torch.tensor([r[1] for r in held_rows], dtype=torch.long)

    digest_before = hashlib.sha256(
        held_code.numpy().tobytes() + refs_a.to(torch.int16).numpy().tobytes() + refs_b.to(torch.int16).numpy().tobytes()
    ).hexdigest()

    future_acc = []
    stale_acc = []
    world0 = torch.randint(0, VALUE_COUNT, (PODS,), generator=torch.Generator().manual_seed(34850))
    stale_va, stale_vb = world0[refs_a].clone(), world0[refs_b].clone()
    for seed in WORLD_SEEDS:
        gen = torch.Generator().manual_seed(seed * 348 + 9)
        for _ in range(FUTURE_WORLDS):
            world = torch.randint(0, VALUE_COUNT, (PODS,), generator=gen)
            truth = label(ops, world[refs_a], world[refs_b])
            with torch.inference_mode():
                cur = j(held_code.float(), world[refs_a], world[refs_b]).argmax(-1)
                stale = j(held_code.float(), stale_va, stale_vb).argmax(-1)
            future_acc.append(float(cur.eq(truth).float().mean()))
            stale_acc.append(float(stale.eq(truth).float().mean()))

    # World writes do not mutate the retained PNS descriptor.
    world = world0.clone(); generations = torch.ones(PODS, dtype=torch.long)
    rng = random.Random(34860)
    reverse_touch = [[] for _ in range(PODS)]
    for i, (a, b) in enumerate(zip(refs_a.tolist(), refs_b.tolist())):
        reverse_touch[a].append(i); reverse_touch[b].append(i)
    numeric_patch_counterfactual = 0
    same_value_updates = 0
    for _ in range(WORLD_UPDATES):
        pid = rng.randrange(PODS)
        numeric_patch_counterfactual += len(reverse_touch[pid])
        same = rng.random() < 0.35
        same_value_updates += int(same)
        generations[pid] += 1
        if not same:
            world[pid] = rng.randrange(VALUE_COUNT)

    digest_after = hashlib.sha256(
        held_code.numpy().tobytes() + refs_a.to(torch.int16).numpy().tobytes() + refs_b.to(torch.int16).numpy().tobytes()
    ).hexdigest()

    # Exact generation validation at materialization/publication.
    expected = detected = escaped = retries = semantic_mismatches = 0
    with torch.inference_mode():
        for _ in range(SERVES):
            i = rng.randrange(len(held_rows))
            a, b, op = int(refs_a[i]), int(refs_b[i]), int(ops[i])
            seen = generations[[a, b]].clone()
            pred = int(j(held_code[i:i+1].float(), world[a].view(1), world[b].view(1)).argmax(-1))
            raced = rng.random() < 0.10
            if raced:
                pid = (a, b)[rng.randrange(2)]
                expected += 1
                generations[pid] += 1
                if rng.random() >= 0.45:
                    world[pid] = rng.randrange(VALUE_COUNT)
            if not torch.equal(seen, generations[[a, b]]):
                detected += 1; retries += 1
                pred = int(j(held_code[i:i+1].float(), world[a].view(1), world[b].view(1)).argmax(-1))
            elif raced:
                escaped += 1
            truth = int(label(torch.tensor([op]), world[a].view(1), world[b].view(1))[0])
            semantic_mismatches += int(pred != truth)

    # One tiny current-Qwen recompile timing sample, compared with cached PNS J execution.
    sample_rows = held_rows[:4]
    t0 = time.perf_counter_ns(); _ = encode_fast(model, tok, sample_rows); qwen_recompile_ns = time.perf_counter_ns() - t0
    ea = refs_a[:4]; eb = refs_b[:4]
    t0 = time.perf_counter_ns();
    with torch.inference_mode(): _ = j(held_code[:4].float(), world[ea], world[eb])
    pns_serve_ns = time.perf_counter_ns() - t0

    qwen_feature_dim = int(train_f.shape[1])
    descriptor_bytes = CODE_DIM * 2 + 4
    qwen_feature_bytes = qwen_feature_dim * 2

    report = {
        "stage": STAGE,
        "architecture_candidate": "Frozen-Qwen Prospective Neural State bridge",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": hashes,
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "train_prompts": len(train_rows),
        "heldout_language_prompts": len(held_rows),
        "qwen_feature_dimension": qwen_feature_dim,
        "continuation_code_dimension": CODE_DIM,
        "train_operation_accuracy": train_acc,
        "heldout_language_operation_accuracy": held_acc,
        "mean_future_world_accuracy": statistics.mean(future_acc),
        "minimum_future_world_accuracy": min(future_acc),
        "mean_stale_numeric_cache_accuracy": statistics.mean(stale_acc),
        "future_gain_pns_minus_stale_numeric": statistics.mean(future_acc) - statistics.mean(stale_acc),
        "descriptor_digest_unchanged_after_world_updates": digest_before == digest_after,
        "write_time_pns_patches_or_invalidations": 0,
        "numeric_cache_patches_counterfactual": numeric_patch_counterfactual,
        "same_value_updates": same_value_updates,
        "race_conflicts_expected": expected,
        "race_conflicts_detected": detected,
        "race_conflicts_escaped": escaped,
        "race_retries": retries,
        "race_semantic_mismatches": semantic_mismatches,
        "descriptor_bytes_per_query": descriptor_bytes,
        "bf16_qwen_feature_cache_bytes_per_query": qwen_feature_bytes,
        "descriptor_to_qwen_feature_cache_ratio": descriptor_bytes / qwen_feature_bytes,
        "qwen_feature_build_seconds": feature_seconds,
        "tiny_batch_qwen_recompile_ns": qwen_recompile_ns,
        "tiny_batch_cached_pns_serve_ns": pns_serve_ns,
        "tiny_batch_recompile_over_pns_ratio": qwen_recompile_ns / max(1, pns_serve_ns),
        "world_rebinding_gradient_steps": 0,
        "entity_reference_extraction": "deterministic canonical pod-ID parser; R346 separately proves trainable reference routing",
        "mechanism": (
            "A real frozen Qwen2.5-0.5B representation is compiled once into a compact semantic continuation code while canonical Pod refs remain live. "
            "No mutable payload is inserted into Qwen or the cached PNS. At the materialization barrier, a shared J-plane dereferences current values "
            "and validates exact generations before publication. Arbitrary world rewrites therefore change the denotation of the retained state without "
            "changing the retained state bytes."
        ),
        "claim_boundary": (
            "This is a deliberately small real-pretrained bridge experiment. Entity IDs are parsed canonically rather than learned in this gate, and "
            "the J-plane is a small neural executor. The result does not establish open-ended language reasoning, strong-RAG superiority or novelty."
        ),
        "dod_status": "NOT_DOD; real-pretrained PNS semantic-bridge gate",
    }
    report["contract_pass"] = (
        train_acc >= 0.99 and held_acc >= 0.90
        and report["mean_future_world_accuracy"] >= 0.97
        and report["minimum_future_world_accuracy"] >= 0.90
        and digest_before == digest_after
        and detected == expected and escaped == 0 and retries == expected
        and semantic_mismatches == 0
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in [
        "stage", "train_operation_accuracy", "heldout_language_operation_accuracy",
        "mean_future_world_accuracy", "minimum_future_world_accuracy", "mean_stale_numeric_cache_accuracy",
        "future_gain_pns_minus_stale_numeric", "descriptor_to_qwen_feature_cache_ratio",
        "descriptor_digest_unchanged_after_world_updates", "race_conflicts_expected", "race_conflicts_detected",
        "race_conflicts_escaped", "race_semantic_mismatches", "tiny_batch_recompile_over_pns_ratio",
        "contract_pass", "report_sha256",
    ]}, indent=2))
    return 0 if report["contract_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
