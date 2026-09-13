from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

OUT = Path(os.environ.get("SO_R287_REPORT", "ci-r287/report.json"))
N_VALUES = 32
N_OPS = 6
NULL_LABEL = 44
N_LABELS = 45


def target(op: int, value: int | None) -> int:
    if value is None:
        return NULL_LABEL
    if op == 0:  # exact read
        return value
    if op == 1:  # parity
        return 32 + (value & 1)
    if op == 2:  # quartile
        return 34 + value // 8
    if op == 3:  # mod 4
        return 38 + value % 4
    if op == 4:  # low/high
        return 42 + int(value >= 16)
    if op == 5:  # xor-folded predicate
        return 42 + int(((value ^ (value >> 1)) & 1) == 1)
    raise ValueError(op)


@dataclass
class Lane:
    generation: int
    kind: str  # value | pointer
    payload: int


@dataclass
class Pod:
    lane0: Lane
    lane1: Lane
    active_lane: int
    revoked: bool = False


class TemporalPortFabric:
    """Reference semantics for generation-gated canonical Pods.

    The semantic identity -> Pod binding is external state, not learned model state.
    Both generation lanes remain physically materialized. Only the authority selector
    may choose a lane. Revocation makes neither lane readable.
    """

    def __init__(self):
        self.pods: dict[int, Pod] = {}
        self.alias_to_id: dict[str, int] = {}

    def add_value(self, pod_id: int, value: int, generation: int = 1):
        lane = Lane(generation, "value", value)
        stale = Lane(0, "value", 0)
        self.pods[pod_id] = Pod(lane, stale, 0, False)

    def add_pointer(self, pod_id: int, target_id: int, generation: int = 1):
        lane = Lane(generation, "pointer", target_id)
        stale = Lane(0, "value", 0)
        self.pods[pod_id] = Pod(lane, stale, 0, False)

    def bind_alias(self, alias: str, pod_id: int):
        self.alias_to_id[alias.casefold()] = pod_id

    def update_value(self, pod_id: int, value: int):
        p = self.pods[pod_id]
        old = p.lane0 if p.active_lane == 0 else p.lane1
        new_lane = 1 - p.active_lane
        lane = Lane(old.generation + 1, "value", value)
        if new_lane == 0:
            p.lane0 = lane
        else:
            p.lane1 = lane
        p.active_lane = new_lane
        p.revoked = False

    def update_pointer(self, pod_id: int, target_id: int):
        p = self.pods[pod_id]
        old = p.lane0 if p.active_lane == 0 else p.lane1
        new_lane = 1 - p.active_lane
        lane = Lane(old.generation + 1, "pointer", target_id)
        if new_lane == 0:
            p.lane0 = lane
        else:
            p.lane1 = lane
        p.active_lane = new_lane
        p.revoked = False

    def revoke(self, pod_id: int):
        self.pods[pod_id].revoked = True

    def mutate_stale(self, pod_id: int, rng: random.Random):
        p = self.pods[pod_id]
        stale_idx = 1 - p.active_lane
        lane = Lane(
            generation=rng.randrange(10_000, 1_000_000),
            kind=rng.choice(("value", "pointer")),
            payload=rng.randrange(0, max(N_VALUES, len(self.pods) + 1)),
        )
        if stale_idx == 0:
            p.lane0 = lane
        else:
            p.lane1 = lane

    def _active(self, pod_id: int) -> Lane | None:
        p = self.pods.get(pod_id)
        if p is None or p.revoked:
            return None
        return p.lane0 if p.active_lane == 0 else p.lane1

    def resolve_id(self, pod_id: int, max_hops: int = 8) -> tuple[int | None, list[tuple[int, int]]]:
        seen: set[int] = set()
        trace: list[tuple[int, int]] = []
        cur = pod_id
        for _ in range(max_hops):
            if cur in seen:
                return None, trace
            seen.add(cur)
            lane = self._active(cur)
            if lane is None:
                return None, trace
            trace.append((cur, lane.generation))
            if lane.kind == "value":
                if 0 <= lane.payload < N_VALUES:
                    return lane.payload, trace
                return None, trace
            cur = lane.payload
        return None, trace

    def resolve_alias(self, alias: str) -> tuple[int | None, list[tuple[int, int]]]:
        pod_id = self.alias_to_id.get(alias.casefold())
        if pod_id is None:
            return None, []
        return self.resolve_id(pod_id)


class PortReasoner(nn.Module):
    """Small neural consumer of typed port payloads.

    Entity identity never enters the reasoner. This forces the learned computation
    to factor identity/addressing from mutable value semantics.
    """

    def __init__(self, d: int = 96):
        super().__init__()
        self.value = nn.Embedding(N_VALUES + 1, d)
        self.op = nn.Embedding(N_OPS, d)
        self.net = nn.Sequential(
            nn.Linear(2 * d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, N_LABELS),
        )

    def forward(self, op: torch.Tensor, value: torch.Tensor, is_null: torch.Tensor):
        null_index = torch.full_like(value, N_VALUES)
        idx = torch.where(is_null, null_index, value)
        x = torch.cat([self.op(op), self.value(idx)], dim=-1)
        return self.net(x)


def train(seed: int) -> PortReasoner:
    torch.manual_seed(seed)
    random.seed(seed)
    model = PortReasoner()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed + 99)
    for step in range(950):
        b = 384
        op = torch.randint(0, N_OPS, (b,), generator=gen)
        value = torch.randint(0, N_VALUES, (b,), generator=gen)
        # Include revoked/null semantics during training but never entity identities.
        is_null = torch.rand((b,), generator=gen) < 0.08
        labels = torch.tensor(
            [target(int(o), None if bool(n) else int(v)) for o, v, n in zip(op, value, is_null)],
            dtype=torch.long,
        )
        logits = model(op, value, is_null)
        loss = nn.functional.cross_entropy(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.eval()


def infer(model: PortReasoner, op: int, value: int | None) -> int:
    with torch.inference_mode():
        o = torch.tensor([op], dtype=torch.long)
        v = torch.tensor([0 if value is None else value], dtype=torch.long)
        n = torch.tensor([value is None], dtype=torch.bool)
        return int(model(o, v, n).argmax(dim=-1).item())


def build_world(seed: int, n_train_entities: int = 256, n_new_entities: int = 128):
    rng = random.Random(seed)
    f = TemporalPortFabric()
    total = n_train_entities + n_new_entities
    values = {}
    for pod in range(total):
        v = rng.randrange(N_VALUES)
        f.add_value(pod, v)
        values[pod] = v
        f.bind_alias(f"canonical-{pod}", pod)
        f.bind_alias(f"alias-a-{pod}", pod)
    # Add pointer aliases in a disjoint namespace. Their targets may include entities
    # that did not exist during neural training, because identity is external state.
    pointer_start = total
    pointers = {}
    for i in range(128):
        pid = pointer_start + i
        target_id = rng.randrange(total)
        f.add_pointer(pid, target_id)
        f.bind_alias(f"pointer-{i}", pid)
        pointers[pid] = target_id
    return f, values, pointers, n_train_entities


def eval_seed(seed: int) -> dict:
    rng = random.Random(seed + 777)
    model = train(seed)
    fabric, values, pointers, train_cut = build_world(seed)

    # A) Completely unseen entity identities. The neural reasoner never consumes IDs,
    # so held-out entity/value bindings are data, not parameters.
    heldout = []
    for pod_id in range(train_cut, train_cut + 128):
        for op in range(N_OPS):
            v, trace = fabric.resolve_alias(f"canonical-{pod_id}")
            pred = infer(model, op, v)
            heldout.append(pred == target(op, values[pod_id]))

    # B) New aliases bound after neural training.
    alias_checks = []
    for pod_id in rng.sample(list(range(train_cut, train_cut + 128)), 64):
        alias = f"late-bound-natural-alias-{seed}-{pod_id}"
        fabric.bind_alias(alias, pod_id)
        v, _ = fabric.resolve_alias(alias)
        for op in range(N_OPS):
            alias_checks.append(infer(model, op, v) == target(op, values[pod_id]))

    # C) Pointer chains 1-4 hops over both seen and unseen identities.
    chain_checks = []
    next_id = max(fabric.pods) + 1
    for _ in range(160):
        base = rng.randrange(train_cut, train_cut + 128)
        cur = base
        depth = rng.randrange(1, 5)
        head = None
        for _hop in range(depth):
            pid = next_id
            next_id += 1
            fabric.add_pointer(pid, cur)
            cur = pid
            head = pid
        assert head is not None
        v, trace = fabric.resolve_id(head)
        for op in range(N_OPS):
            chain_checks.append(infer(model, op, v) == target(op, values[base]))

    # D) 1,000 online updates with zero optimizer steps. Old generation stays present.
    update_checks = []
    stale_invariance = []
    update_targets = rng.choices(list(range(train_cut, train_cut + 128)), k=1000)
    for pod_id in update_targets:
        new_v = rng.randrange(N_VALUES)
        fabric.update_value(pod_id, new_v)
        before = [infer(model, op, fabric.resolve_id(pod_id)[0]) for op in range(N_OPS)]
        fabric.mutate_stale(pod_id, rng)
        after = [infer(model, op, fabric.resolve_id(pod_id)[0]) for op in range(N_OPS)]
        stale_invariance.append(before == after)
        update_checks.extend(after[op] == target(op, new_v) for op in range(N_OPS))

    # E) Revocation: both materialized lanes remain, but neither may be served.
    revoke_checks = []
    revoked = rng.sample(list(range(train_cut, train_cut + 128)), 64)
    for pod_id in revoked:
        fabric.revoke(pod_id)
        for _ in range(3):
            fabric.mutate_stale(pod_id, rng)
            v, _ = fabric.resolve_id(pod_id)
            for op in range(N_OPS):
                revoke_checks.append(infer(model, op, v) == NULL_LABEL)

    # F) Update through pointer: pointer identity remains stable while target changes.
    pointer_update_checks = []
    for pid in rng.sample(list(pointers), 64):
        new_target = rng.randrange(train_cut, train_cut + 128)
        fabric.update_pointer(pid, new_target)
        v, trace = fabric.resolve_id(pid)
        for op in range(N_OPS):
            pointer_update_checks.append(infer(model, op, v) == target(op, fabric.resolve_id(new_target)[0]))

    metrics = {
        "heldout_entity_value_pair_accuracy": sum(heldout) / len(heldout),
        "late_bound_alias_accuracy": sum(alias_checks) / len(alias_checks),
        "pointer_chain_accuracy": sum(chain_checks) / len(chain_checks),
        "online_update_accuracy": sum(update_checks) / len(update_checks),
        "stale_generation_mutation_invariance": sum(stale_invariance) / len(stale_invariance),
        "revocation_accuracy": sum(revoke_checks) / len(revoke_checks),
        "pointer_update_accuracy": sum(pointer_update_checks) / len(pointer_update_checks),
        "online_updates": len(update_targets),
        "optimizer_steps_after_world_updates": 0,
        "new_alias_optimizer_steps": 0,
        "new_entity_optimizer_steps": 0,
    }
    return metrics


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    seeds = [11, 23, 47, 89, 131]
    started = time.perf_counter()
    results = {str(s): eval_seed(s) for s in seeds}
    elapsed = time.perf_counter() - started
    keys = [
        "heldout_entity_value_pair_accuracy",
        "late_bound_alias_accuracy",
        "pointer_chain_accuracy",
        "online_update_accuracy",
        "stale_generation_mutation_invariance",
        "revocation_accuracy",
        "pointer_update_accuracy",
    ]
    mean = {k: sum(results[str(s)][k] for s in seeds) / len(seeds) for k in keys}
    report = {
        "stage": "R287-TEMPORAL-CAPABILITY-PORT-SEMANTICS",
        "architecture_candidate": "Temporal Capability Port Fabric (TCPF)",
        "purpose": (
            "Kill the prior held-out identity/value linking failure by changing the architecture: "
            "identity-to-value binding is mutable port state, not a learned pair. The neural component "
            "learns reusable operations over typed payloads; a canonical Symlink table resolves identity."
        ),
        "seeds": seeds,
        "results": results,
        "mean": mean,
        "elapsed_seconds": elapsed,
        "guarantee_boundary": {
            "exact": [
                "only active generation lane is read",
                "revoked pod returns no payload",
                "stale lane can remain materialized without influence",
                "new aliases and entity/value bindings require no neural optimization",
            ],
            "learned": [
                "reasoning from typed value payload to answer label"
            ],
            "not_claimed": [
                "natural-language entity linking quality",
                "real LLM language quality",
                "RAG superiority",
                "novelty versus all prior art",
            ],
        },
        "next_gate": (
            "Integrate the same generation authority primitive into real transformer attention, then "
            "replace text-derived KV payloads by compact model-native port codes and benchmark against "
            "Knowledge Packs/TurboRAG/strong RAG."
        ),
        "dod_status": "NOT_DOD; architectural factorization gate",
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"mean": mean, "elapsed_seconds": elapsed}, indent=2))

    if any(mean[k] < 0.985 for k in keys):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
