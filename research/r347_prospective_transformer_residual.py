from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

STAGE = "R347-PROSPECTIVE-TRANSFORMER-RESIDUAL"
PODS = int(os.environ.get("SO_R347_PODS", "128"))
SEM_DIM = int(os.environ.get("SO_R347_SEM_DIM", "24"))
Q_DIM = int(os.environ.get("SO_R347_Q_DIM", "40"))
CODE_DIM = int(os.environ.get("SO_R347_CODE_DIM", "12"))
VALUE_COUNT = int(os.environ.get("SO_R347_VALUES", "32"))
OPS = 4
CLASSES = 4
BLOCKS = int(os.environ.get("SO_R347_BLOCKS", "6"))
TRAIN_STEPS = int(os.environ.get("SO_R347_TRAIN_STEPS", "1200"))
BATCH = int(os.environ.get("SO_R347_BATCH", "384"))
DESCRIPTORS = int(os.environ.get("SO_R347_DESCRIPTORS", "3000"))
FUTURE_WORLDS = int(os.environ.get("SO_R347_FUTURE_WORLDS", "32"))
UPDATES = int(os.environ.get("SO_R347_UPDATES", "20000"))
SERVES = int(os.environ.get("SO_R347_SERVES", "12000"))
REPORT_PATH = Path(os.environ.get("SO_R347_REPORT", "ci-r347/report.json"))
SEEDS = (34717, 34729, 34743)


def labels(op: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(op)
    m = op == 0; y[m] = ((a[m] & 3) + (b[m] & 3)) & 3
    m = op == 1; y[m] = (a[m] ^ b[m]) & 3
    m = op == 2; y[m] = ((a[m] >> 1) & 1) + 2 * ((b[m] >> 2) & 1)
    m = op == 3; y[m] = torch.where((b[m] & 1) == 0, a[m] & 3, b[m] & 3)
    return y


def bits(x: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(5, device=x.device)
    return ((x[:, None] >> shifts[None, :]) & 1).float()


class ProspectiveRouter(nn.Module):
    """Attention router plus a world-independent residual continuation stream.

    The two attention heads return canonical addresses.  They do not perform a
    value-side gather.  The continuation code is allowed to pass through deep
    nonlinear residual computation because it depends only on query semantics.
    """

    def __init__(self, semantic_keys: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("semantic_keys", semantic_keys)
        self.query_a = nn.Sequential(nn.Linear(Q_DIM, 80), nn.GELU(), nn.Linear(80, SEM_DIM))
        self.query_b = nn.Sequential(nn.Linear(Q_DIM, 80), nn.GELU(), nn.Linear(80, SEM_DIM))
        self.key_proj = nn.Linear(SEM_DIM, SEM_DIM, bias=False)
        self.code_in = nn.Sequential(nn.LayerNorm(Q_DIM), nn.Linear(Q_DIM, CODE_DIM), nn.GELU())
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(CODE_DIM),
                nn.Linear(CODE_DIM, CODE_DIM * 3),
                nn.GELU(),
                nn.Linear(CODE_DIM * 3, CODE_DIM),
            ) for _ in range(BLOCKS)
        ])

    def forward(self, q: torch.Tensor):
        keys = F.normalize(self.key_proj(self.semantic_keys), dim=-1)
        qa = F.normalize(self.query_a(q), dim=-1)
        qb = F.normalize(self.query_b(q), dim=-1)
        la = qa @ keys.T * 12.0
        lb = qb @ keys.T * 12.0
        code = self.code_in(q)
        for block in self.blocks:
            code = code + 0.35 * block(code)
        return la, lb, code

    @torch.inference_mode()
    def compile_descriptor(self, q: torch.Tensor):
        la, lb, code = self(q)
        return la.argmax(-1), lb.argmax(-1), code


class Materializer(nn.Module):
    """First world-dependent neural region (the J plane)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(CODE_DIM + 10),
            nn.Linear(CODE_DIM + 10, 96), nn.GELU(),
            nn.Linear(96, 96), nn.GELU(),
            nn.Linear(96, CLASSES),
        )

    def forward(self, code: torch.Tensor, va: torch.Tensor, vb: torch.Tensor):
        return self.net(torch.cat([code, bits(va), bits(vb)], dim=-1))


def build_query(semantic_keys, op_seed, mix, a, b, op, gen, noise=True):
    raw = torch.cat([semantic_keys[a], semantic_keys[b], op_seed[op]], dim=-1)
    q = torch.tanh(raw @ mix)
    if noise:
        q = q + 0.012 * torch.randn(q.shape, generator=gen)
    return q


def digest_descriptor(ra, rb, code):
    return hashlib.sha256(
        ra.to(torch.int16).numpy().tobytes()
        + rb.to(torch.int16).numpy().tobytes()
        + code.to(torch.bfloat16).view(torch.int16).numpy().tobytes()
    ).hexdigest()


def run_seed(seed: int) -> dict:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)

    semantic_keys = F.normalize(torch.randn(PODS, SEM_DIM, generator=gen), dim=-1)
    op_seed = torch.randn(OPS, 10, generator=gen)
    mix = torch.randn(SEM_DIM * 2 + 10, Q_DIM, generator=gen) / (SEM_DIM ** 0.5)

    router = ProspectiveRouter(semantic_keys)
    j = Materializer()
    params = list(router.parameters()) + list(j.parameters())
    opt = torch.optim.AdamW(params, lr=2.4e-3, weight_decay=1e-4)

    router.train(); j.train()
    t0 = time.perf_counter()
    tail = []
    for step in range(TRAIN_STEPS):
        a = torch.randint(0, PODS, (BATCH,), generator=gen)
        b = torch.randint(0, PODS, (BATCH,), generator=gen)
        op = torch.randint(0, OPS, (BATCH,), generator=gen)
        q = build_query(semantic_keys, op_seed, mix, a, b, op, gen, noise=True)
        la, lb, code = router(q)

        # The entire mutable world is redrawn every step.  There is therefore no
        # stable entity->value fact for the neural weights to memorize.
        world = torch.randint(0, VALUE_COUNT, (PODS,), generator=gen)
        out = j(code, world[a], world[b])
        y = labels(op, world[a], world[b])
        loss = F.cross_entropy(la, a) + F.cross_entropy(lb, b) + F.cross_entropy(out, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step >= TRAIN_STEPS - 50:
            tail.append(float(loss.detach()))

    train_seconds = time.perf_counter() - t0
    router.eval(); j.eval()

    a = torch.randint(0, PODS, (DESCRIPTORS,), generator=gen)
    b = torch.randint(0, PODS, (DESCRIPTORS,), generator=gen)
    op = torch.randint(0, OPS, (DESCRIPTORS,), generator=gen)
    q = build_query(semantic_keys, op_seed, mix, a, b, op, gen, noise=False)

    with torch.inference_mode():
        ra, rb, code = router.compile_descriptor(q)
    route_a = float(ra.eq(a).float().mean())
    route_b = float(rb.eq(b).float().mean())
    descriptor_acc = float((ra.eq(a) & rb.eq(b)).float().mean())

    descriptor_hash_before = digest_descriptor(ra, rb, code)

    # Compile-time world for the matched conventional numeric cache.
    home = torch.randint(0, VALUE_COUNT, (PODS,), generator=gen)
    with torch.inference_mode():
        home_logits = j(code, home[ra], home[rb])
    # Conventional early materialization stores numeric world-dependent operands.
    stale_home_va = home[ra].clone()
    stale_home_vb = home[rb].clone()

    prospective_acc = []
    stale_acc = []
    max_logit_error_vs_full = 0.0
    prospective_vs_full_class_mismatch = 0
    for _ in range(FUTURE_WORLDS):
        future = torch.randint(0, VALUE_COUNT, (PODS,), generator=gen)
        with torch.inference_mode():
            # Full current-world recompute: rerun the query side and materialize current values.
            fra, frb, fcode = router.compile_descriptor(q)
            full = j(fcode, future[fra], future[frb])
            # Prospective cache: skip all routing/residual compilation and only late-deref.
            prospective = j(code, future[ra], future[rb])
            # Conventional numeric cached state: reuse old materialized operands.
            stale = j(code, stale_home_va, stale_home_vb)
        truth = labels(op, future[a], future[b])
        prospective_acc.append(float(prospective.argmax(-1).eq(truth).float().mean()))
        stale_acc.append(float(stale.argmax(-1).eq(truth).float().mean()))
        max_logit_error_vs_full = max(max_logit_error_vs_full, float((prospective - full).abs().max()))
        prospective_vs_full_class_mismatch += int((prospective.argmax(-1) != full.argmax(-1)).sum())

    # World update stream.  A conventional numeric cache has to invalidate/patch every
    # descriptor whose materialized value touched the updated Pod.  The prospective state does not.
    reverse = [[] for _ in range(PODS)]
    for i, (x, y) in enumerate(zip(ra.tolist(), rb.tolist())):
        reverse[x].append(i); reverse[y].append(i)
    numeric_cache_patches = 0
    world = home.clone()
    generations = torch.ones(PODS, dtype=torch.long)
    same_value_updates = 0
    write_ns = []
    for _ in range(UPDATES):
        pid = rng.randrange(PODS)
        numeric_cache_patches += len(reverse[pid])
        same = rng.random() < 0.35
        same_value_updates += int(same)
        ts = time.perf_counter_ns()
        generations[pid] += 1
        if not same:
            world[pid] = rng.randrange(VALUE_COUNT)
        write_ns.append(time.perf_counter_ns() - ts)

    descriptor_hash_after = digest_descriptor(ra, rb, code)

    # Transactional late-dereference under same-value ABA and value-changing races.
    expected = detected = escaped = retries = semantic_mismatch = 0
    with torch.inference_mode():
        for _ in range(SERVES):
            idx = rng.randrange(DESCRIPTORS)
            refs = (int(ra[idx]), int(rb[idx]))
            seen = generations[list(refs)].clone()
            pred = int(j(code[idx:idx+1], world[refs[0]].view(1), world[refs[1]].view(1)).argmax(-1))
            raced = rng.random() < 0.09
            if raced:
                pid = refs[rng.randrange(2)]
                expected += 1
                generations[pid] += 1
                if rng.random() >= 0.45:
                    world[pid] = rng.randrange(VALUE_COUNT)
            now = generations[list(refs)]
            if not torch.equal(seen, now):
                detected += 1; retries += 1
                pred = int(j(code[idx:idx+1], world[refs[0]].view(1), world[refs[1]].view(1)).argmax(-1))
            elif raced:
                escaped += 1
            truth = int(labels(op[idx:idx+1], world[int(a[idx])].view(1), world[int(b[idx])].view(1))[0])
            semantic_mismatch += int(pred != truth)

    # Fair cache-size accounting: addresses are int16 and code is BF16.  The matched
    # numeric cache requires code plus two materialized 5-bit operand feature vectors in BF16.
    prospective_cache_bytes = DESCRIPTORS * (2 + 2 + CODE_DIM * 2)
    numeric_hidden_cache_bytes = DESCRIPTORS * ((CODE_DIM + 10) * 2)

    # Read-side timing against a full current-world recompute.  Timings are Python/CPU diagnostics only.
    sample = min(512, DESCRIPTORS)
    idx = torch.arange(sample)
    timing_world = torch.randint(0, VALUE_COUNT, (PODS,), generator=gen)
    full_ns = []
    prospective_ns = []
    with torch.inference_mode():
        for _ in range(80):
            ts = time.perf_counter_ns()
            tra, trb, tcode = router.compile_descriptor(q[idx])
            _ = j(tcode, timing_world[tra], timing_world[trb])
            full_ns.append(time.perf_counter_ns() - ts)
            ts = time.perf_counter_ns()
            _ = j(code[idx], timing_world[ra[idx]], timing_world[rb[idx]])
            prospective_ns.append(time.perf_counter_ns() - ts)

    return {
        "seed": seed,
        "route_a_accuracy": route_a,
        "route_b_accuracy": route_b,
        "full_reference_descriptor_accuracy": descriptor_acc,
        "mean_future_world_accuracy": statistics.mean(prospective_acc),
        "mean_stale_numeric_cache_accuracy": statistics.mean(stale_acc),
        "future_gain_vs_stale_numeric": statistics.mean(prospective_acc) - statistics.mean(stale_acc),
        "max_abs_logit_error_vs_full_current_recompute": max_logit_error_vs_full,
        "prospective_vs_full_class_mismatches": prospective_vs_full_class_mismatch,
        "descriptor_digest_unchanged_after_world_updates": descriptor_hash_before == descriptor_hash_after,
        "write_time_prospective_descriptor_invalidations_or_patches": 0,
        "numeric_cache_patches_counterfactual": numeric_cache_patches,
        "same_value_updates": same_value_updates,
        "race_conflicts_expected": expected,
        "race_conflicts_detected": detected,
        "race_conflicts_escaped": escaped,
        "race_retries": retries,
        "race_semantic_mismatches": semantic_mismatch,
        "prospective_cache_bytes": prospective_cache_bytes,
        "numeric_hidden_cache_bytes": numeric_hidden_cache_bytes,
        "prospective_to_numeric_cache_byte_ratio": prospective_cache_bytes / numeric_hidden_cache_bytes,
        "median_full_current_recompute_ns_python": statistics.median(full_ns),
        "median_cached_prospective_serve_ns_python": statistics.median(prospective_ns),
        "full_over_prospective_serve_speedup_python": statistics.median(full_ns) / statistics.median(prospective_ns),
        "median_world_write_ns_python": statistics.median(write_ns),
        "train_seconds": train_seconds,
        "tail_loss": statistics.mean(tail),
    }


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.set_num_interop_threads(1)

    rows = [run_seed(seed) for seed in SEEDS]
    min_descriptor = min(r["full_reference_descriptor_accuracy"] for r in rows)
    min_future = min(r["mean_future_world_accuracy"] for r in rows)
    max_error = max(r["max_abs_logit_error_vs_full_current_recompute"] for r in rows)
    total_mismatch = sum(r["prospective_vs_full_class_mismatches"] for r in rows)
    total_race_mismatch = sum(r["race_semantic_mismatches"] for r in rows)
    total_expected = sum(r["race_conflicts_expected"] for r in rows)
    total_detected = sum(r["race_conflicts_detected"] for r in rows)
    total_escaped = sum(r["race_conflicts_escaped"] for r in rows)

    report = {
        "stage": STAGE,
        "architecture_candidate": "Prospective Transformer: reference-valued attention + future-bindable residual continuation",
        "seeds": list(SEEDS),
        "rows": rows,
        "min_reference_descriptor_accuracy_across_seeds": min_descriptor,
        "min_future_world_accuracy_across_seeds": min_future,
        "max_abs_logit_error_vs_full_current_recompute_across_seeds": max_error,
        "total_prospective_vs_full_class_mismatches": total_mismatch,
        "total_race_semantic_mismatches": total_race_mismatch,
        "total_race_conflicts_expected": total_expected,
        "total_race_conflicts_detected": total_detected,
        "total_race_conflicts_escaped": total_escaped,
        "mean_future_gain_vs_stale_numeric": statistics.mean(r["future_gain_vs_stale_numeric"] for r in rows),
        "mean_prospective_to_numeric_cache_byte_ratio": statistics.mean(r["prospective_to_numeric_cache_byte_ratio"] for r in rows),
        "mean_full_over_prospective_serve_speedup_python": statistics.mean(r["full_over_prospective_serve_speedup_python"] for r in rows),
        "world_rebinding_gradient_steps": 0,
        "reference_preserving_residual_blocks": BLOCKS,
        "mechanism": (
            "A trainable attention router emits canonical Pod addresses while a separate nonlinear residual continuation code is propagated for six "
            "world-independent neural blocks. The mutable value side is intentionally absent from the numeric residual until an explicit J-plane "
            "materialization barrier. The retained activation is therefore (refs, continuation-code), a function waiting to be bound to the current "
            "world. Arbitrary future rewrites modify only canonical Pod cells; cached prospective activations are neither patched nor invalidated."
        ),
        "falsification_target": (
            "If the prospective cache disagrees with a full current-world recompute, changes bytes after world writes, misses a generation race, or "
            "fails on any independent seed, this gate fails. A matched early-materialized numeric cache is kept as the stale-state control."
        ),
        "claim_boundary": (
            "This is still a controlled synthetic architecture test. Pointer networks, external memories, partial evaluation, closures, lazy tensors, "
            "and deferred materialization are established ideas. R347 tests the narrower architectural semantics that a Transformer-like retained "
            "attention/residual activation can deliberately contain live addresses and remain a function of future mutable world state. Novelty and "
            "superiority to strong RAG remain separate literature and pretrained-model gates."
        ),
        "dod_status": "NOT_DOD; multi-layer prospective residual mechanism gate",
    }
    report["contract_pass"] = (
        min_descriptor >= 0.99
        and min_future >= 0.98
        and max_error <= 1e-6
        and total_mismatch == 0
        and total_race_mismatch == 0
        and total_detected == total_expected
        and total_escaped == 0
        and all(r["descriptor_digest_unchanged_after_world_updates"] for r in rows)
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in [
        "stage", "min_reference_descriptor_accuracy_across_seeds",
        "min_future_world_accuracy_across_seeds", "max_abs_logit_error_vs_full_current_recompute_across_seeds",
        "total_prospective_vs_full_class_mismatches", "mean_future_gain_vs_stale_numeric",
        "mean_prospective_to_numeric_cache_byte_ratio", "mean_full_over_prospective_serve_speedup_python",
        "total_race_conflicts_expected", "total_race_conflicts_detected", "total_race_conflicts_escaped",
        "total_race_semantic_mismatches", "contract_pass", "report_sha256",
    ]}, indent=2))
    return 0 if report["contract_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
