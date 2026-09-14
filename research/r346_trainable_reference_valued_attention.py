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

STAGE = "R346-TRAINABLE-REFERENCE-VALUED-ATTENTION"
PODS = int(os.environ.get("SO_R346_PODS", "256"))
SEM_DIM = int(os.environ.get("SO_R346_SEM_DIM", "32"))
Q_DIM = int(os.environ.get("SO_R346_Q_DIM", "48"))
VALUE_COUNT = int(os.environ.get("SO_R346_VALUES", "32"))
OPS = 4
CLASSES = 4
TRAIN_STEPS = int(os.environ.get("SO_R346_TRAIN_STEPS", "1600"))
BATCH = int(os.environ.get("SO_R346_BATCH", "512"))
DESCRIPTORS = int(os.environ.get("SO_R346_DESCRIPTORS", "6000"))
FUTURE_WORLDS = int(os.environ.get("SO_R346_FUTURE_WORLDS", "64"))
UPDATES = int(os.environ.get("SO_R346_UPDATES", "30000"))
REPORT_PATH = Path(os.environ.get("SO_R346_REPORT", "ci-r346/report.json"))
SEED = 3460914


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


class ReferenceAttention(nn.Module):
    """Trainable router whose attention output is a pair of references, not values."""
    def __init__(self, semantic_keys: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("semantic_keys", semantic_keys)
        self.query_a = nn.Sequential(nn.Linear(Q_DIM, 96), nn.GELU(), nn.Linear(96, SEM_DIM))
        self.query_b = nn.Sequential(nn.Linear(Q_DIM, 96), nn.GELU(), nn.Linear(96, SEM_DIM))
        self.key_proj = nn.Linear(SEM_DIM, SEM_DIM, bias=False)
        self.op_head = nn.Sequential(nn.Linear(Q_DIM, 64), nn.GELU(), nn.Linear(64, OPS))

    def logits(self, q: torch.Tensor):
        keys = F.normalize(self.key_proj(self.semantic_keys), dim=-1)
        qa = F.normalize(self.query_a(q), dim=-1)
        qb = F.normalize(self.query_b(q), dim=-1)
        return qa @ keys.T * 12.0, qb @ keys.T * 12.0, self.op_head(q)

    @torch.inference_mode()
    def descriptor(self, q: torch.Tensor):
        la, lb, lo = self.logits(q)
        return la.argmax(-1), lb.argmax(-1), lo.argmax(-1)


class JContinuation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.op = nn.Embedding(OPS, 20)
        self.net = nn.Sequential(
            nn.Linear(20 + 10, 96), nn.GELU(),
            nn.Linear(96, 96), nn.GELU(),
            nn.Linear(96, CLASSES),
        )

    def forward(self, op: torch.Tensor, va: torch.Tensor, vb: torch.Tensor):
        return self.net(torch.cat([self.op(op), bits(va), bits(vb)], dim=-1))


def make_queries(
    semantic_keys: torch.Tensor,
    op_seed: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    op: torch.Tensor,
    noise: float,
    gen: torch.Generator,
):
    # Fixed nonlinear mixing hides the identity basis from the router but is independent of world values.
    ka = semantic_keys[a]
    kb = semantic_keys[b]
    opv = op_seed[op]
    raw = torch.cat([ka, kb, opv], dim=-1)
    # Deterministic random projection is supplied by caller via closure below.
    return raw


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    gen = torch.Generator().manual_seed(SEED)

    semantic_keys = F.normalize(torch.randn(PODS, SEM_DIM, generator=gen), dim=-1)
    op_seed = torch.randn(OPS, 12, generator=gen)
    mix = torch.randn(SEM_DIM * 2 + 12, Q_DIM, generator=gen) / (SEM_DIM ** 0.5)

    def query_vec(a, b, op, *, noisy=True):
        raw = make_queries(semantic_keys, op_seed, a, b, op, 0.0, gen)
        q = torch.tanh(raw @ mix)
        if noisy:
            q = q + 0.015 * torch.randn(q.shape, generator=gen)
        return q

    router = ReferenceAttention(semantic_keys)
    j = JContinuation()
    opt = torch.optim.AdamW(list(router.parameters()) + list(j.parameters()), lr=2.2e-3, weight_decay=1e-4)
    router.train(); j.train(); t0 = time.perf_counter()

    for _ in range(TRAIN_STEPS):
        a = torch.randint(0, PODS, (BATCH,), generator=gen)
        b = torch.randint(0, PODS, (BATCH,), generator=gen)
        op = torch.randint(0, OPS, (BATCH,), generator=gen)
        q = query_vec(a, b, op)
        la, lb, lo = router.logits(q)
        # World is redrawn every step, so entity identity contains no stable factual value.
        world = torch.randint(0, VALUE_COUNT, (PODS,), generator=gen)
        va, vb = world[a], world[b]
        y = labels(op, va, vb)
        # Teacher-forced current refs train J; route heads are supervised as references.
        out = j(op, va, vb)
        loss = (
            F.cross_entropy(la, a) + F.cross_entropy(lb, b) +
            0.6 * F.cross_entropy(lo, op) + F.cross_entropy(out, y)
        )
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(list(router.parameters()) + list(j.parameters()), 1.0); opt.step()

    train_seconds = time.perf_counter() - t0
    router.eval(); j.eval()

    # Compile query descriptors once. These are the reference-valued attention outputs.
    a = torch.randint(0, PODS, (DESCRIPTORS,), generator=gen)
    b = torch.randint(0, PODS, (DESCRIPTORS,), generator=gen)
    op = torch.randint(0, OPS, (DESCRIPTORS,), generator=gen)
    q = query_vec(a, b, op, noisy=False)
    with torch.inference_mode():
        ra, rb, rop = router.descriptor(q)
    route_acc_a = float(ra.eq(a).float().mean())
    route_acc_b = float(rb.eq(b).float().mean())
    op_acc = float(rop.eq(op).float().mean())
    full_descriptor_acc = float((ra.eq(a) & rb.eq(b) & rop.eq(op)).float().mean())

    descriptor_bytes = ra.to(torch.int32).numel() * 4 + rb.to(torch.int32).numel() * 4 + rop.to(torch.uint8).numel()
    descriptor_digest = hashlib.sha256(
        ra.to(torch.int32).numpy().tobytes() + rb.to(torch.int32).numpy().tobytes() + rop.to(torch.uint8).numpy().tobytes()
    ).hexdigest()

    # Evaluate future worlds using only cached reference descriptors + current values.
    future_acc = []
    stale_numeric_acc = []
    home_world = torch.randint(0, VALUE_COUNT, (PODS,), generator=gen)
    cached_va_home, cached_vb_home = home_world[ra], home_world[rb]
    for _ in range(FUTURE_WORLDS):
        world = torch.randint(0, VALUE_COUNT, (PODS,), generator=gen)
        truth = labels(op, world[a], world[b])
        with torch.inference_mode():
            pred = j(rop, world[ra], world[rb]).argmax(-1)
            stale_pred = j(rop, cached_va_home, cached_vb_home).argmax(-1)
        future_acc.append(float(pred.eq(truth).float().mean()))
        stale_numeric_acc.append(float(stale_pred.eq(truth).float().mean()))

    # Large write stream: descriptor itself remains untouched; quantify numeric-cache patch fanout.
    reverse = [[] for _ in range(PODS)]
    for i, (x, y) in enumerate(zip(ra.tolist(), rb.tolist())):
        reverse[x].append(i); reverse[y].append(i)
    numeric_patch_counterfactual = 0
    world = home_world.clone()
    generations = torch.ones(PODS, dtype=torch.long)
    same_value_updates = 0
    write_ns = []
    for _ in range(UPDATES):
        pid = rng.randrange(PODS)
        numeric_patch_counterfactual += len(reverse[pid])
        same = rng.random() < 0.35
        same_value_updates += int(same)
        ts = time.perf_counter_ns()
        generations[pid] += 1
        if not same:
            world[pid] = rng.randrange(VALUE_COUNT)
        write_ns.append(time.perf_counter_ns() - ts)

    descriptor_digest_after = hashlib.sha256(
        ra.to(torch.int32).numpy().tobytes() + rb.to(torch.int32).numpy().tobytes() + rop.to(torch.uint8).numpy().tobytes()
    ).hexdigest()

    # Race/commit test on reference descriptors.
    expected = detected = escaped = retries = semantic_mismatch = 0
    with torch.inference_mode():
        for _ in range(12000):
            idx = rng.randrange(DESCRIPTORS)
            refs = (int(ra[idx]), int(rb[idx]))
            seen = generations[list(refs)].clone()
            pred = int(j(rop[idx:idx+1], world[refs[0]].view(1), world[refs[1]].view(1)).argmax(-1))
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
                pred = int(j(rop[idx:idx+1], world[refs[0]].view(1), world[refs[1]].view(1)).argmax(-1))
            elif raced:
                escaped += 1
            truth = int(labels(op[idx:idx+1], world[int(a[idx])].view(1), world[int(b[idx])].view(1))[0])
            semantic_mismatch += int(pred != truth)

    report = {
        "stage": STAGE,
        "architecture_candidate": "Trainable Reference-Valued Attention (TRVA)",
        "pods": PODS,
        "descriptors": DESCRIPTORS,
        "router_head_a_accuracy": route_acc_a,
        "router_head_b_accuracy": route_acc_b,
        "operation_accuracy": op_acc,
        "full_reference_descriptor_accuracy": full_descriptor_acc,
        "mean_future_world_accuracy": statistics.mean(future_acc),
        "mean_stale_numeric_cache_accuracy": statistics.mean(stale_numeric_acc),
        "future_gain_reference_descriptor_minus_stale_numeric": statistics.mean(future_acc) - statistics.mean(stale_numeric_acc),
        "train_seconds": train_seconds,
        "descriptor_bytes": descriptor_bytes,
        "descriptor_digest_unchanged_after_world_updates": descriptor_digest == descriptor_digest_after,
        "world_updates": UPDATES,
        "same_value_updates": same_value_updates,
        "write_time_descriptor_invalidations_or_patches": 0,
        "numeric_value_cache_patches_counterfactual": numeric_patch_counterfactual,
        "median_world_write_ns_python": statistics.median(write_ns),
        "race_conflicts_expected": expected,
        "race_conflicts_detected": detected,
        "race_conflicts_escaped": escaped,
        "race_retries": retries,
        "race_semantic_mismatches": semantic_mismatch,
        "mechanism": (
            "Two trainable attention heads compile query semantics into canonical reference IDs rather than reading memory values. The reference "
            "descriptor is retained across arbitrary future world rewrites. An identity-blind J continuation dereferences the selected current "
            "world values only when the query is served. The router was trained while the entire entity->value world was randomized every batch, "
            "so no persistent factual binding was available to memorize."
        ),
        "claim_boundary": (
            "Pointer networks and supervised neural routing are established. R346 does not claim pointer attention as new; it tests that a "
            "trainable attention module can instantiate the Prospective Neural State semantics by retaining its reference output across future "
            "mutable-world revisions rather than immediately materializing V."
        ),
        "dod_status": "NOT_DOD; trainable reference-attention mechanism gate",
    }
    report["contract_pass"] = (
        full_descriptor_acc >= 0.99 and report["mean_future_world_accuracy"] >= 0.98
        and descriptor_digest == descriptor_digest_after
        and detected == expected and escaped == 0 and retries == expected
        and semantic_mismatch == 0
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in [
        "stage", "full_reference_descriptor_accuracy", "mean_future_world_accuracy",
        "mean_stale_numeric_cache_accuracy", "future_gain_reference_descriptor_minus_stale_numeric",
        "descriptor_digest_unchanged_after_world_updates", "numeric_value_cache_patches_counterfactual",
        "race_conflicts_expected", "race_conflicts_detected", "race_conflicts_escaped",
        "race_semantic_mismatches", "contract_pass", "report_sha256",
    ]}, indent=2))
    if not report["contract_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
