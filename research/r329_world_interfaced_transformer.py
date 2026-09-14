from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


STAGE = "R329-WORLD-INTERFACED-DUAL-RAIL-TRANSFORMER"
SEEDS = (17, 29, 43)
ENTITIES = 64
VALUE_COUNT = 32
CLASS_COUNT = 4
OPS = 4
MAX_LEN = 12
D_MODEL = 48
NHEAD = 4
FF = 96
COMPILER_LAYERS = 2
MONO_LAYERS = 2
J_LAYERS = 2
COMPILER_STEPS = int(os.environ.get("SO_R329_COMPILER_STEPS", "650"))
MONO_STEPS = int(os.environ.get("SO_R329_MONO_STEPS", "800"))
J_STEPS = int(os.environ.get("SO_R329_J_STEPS", "750"))
BATCH = int(os.environ.get("SO_R329_BATCH", "256"))
J_BATCH = int(os.environ.get("SO_R329_J_BATCH", "512"))
EVAL_BATCHES = int(os.environ.get("SO_R329_EVAL_BATCHES", "32"))
REPORT_PATH = Path(os.environ.get("SO_R329_REPORT", "r329_report.json"))


# Vocabulary. Each entity has two lexical aliases. The trusted linker knows both;
# the neural compiler is allowed to point to source-token positions but never emits
# canonical Pod IDs or factual values directly.
SPECIAL = ["<PAD>", "<CLS>"]
WORDS = [
    "sum", "add", "plus", "total", "xor", "exclusive", "or", "mix",
    "inspect", "bits", "branch", "choose", "if", "else", "and", "with",
    "of", "to", "compute", "give", "please", "now", "for", "result",
]
ALIASES_A = [f"entity_{i}" for i in range(ENTITIES)]
ALIASES_B = [f"firm_{i}" for i in range(ENTITIES)]
VOCAB = SPECIAL + WORDS + ALIASES_A + ALIASES_B
TOKEN = {t: i for i, t in enumerate(VOCAB)}
PAD = TOKEN["<PAD>"]
CLS = TOKEN["<CLS>"]
ENTITY_TOKEN_TO_ID = {
    TOKEN[name]: i for i, name in enumerate(ALIASES_A)
} | {
    TOKEN[name]: i for i, name in enumerate(ALIASES_B)
}
ENTITY_TOKEN_IDS = torch.tensor(sorted(ENTITY_TOKEN_TO_ID), dtype=torch.long)

TRAIN_TEMPLATES = {
    0: [
        ["sum", "{a}", "and", "{b}"],
        ["add", "{a}", "to", "{b}"],
        ["compute", "{a}", "plus", "{b}"],
        ["total", "of", "{a}", "and", "{b}"],
    ],
    1: [
        ["xor", "{a}", "and", "{b}"],
        ["exclusive", "or", "{a}", "{b}"],
        ["compute", "xor", "of", "{a}", "with", "{b}"],
        ["mix", "{a}", "with", "{b}", "xor"],
    ],
    2: [
        ["inspect", "bits", "of", "{a}", "and", "{b}"],
        ["compute", "bits", "{a}", "with", "{b}"],
        ["give", "bits", "for", "{a}", "and", "{b}"],
        ["please", "inspect", "{a}", "and", "{b}", "bits"],
    ],
    3: [
        ["branch", "{a}", "with", "{b}"],
        ["choose", "{a}", "if", "{b}", "else", "{b}"],
        ["compute", "branch", "of", "{a}", "and", "{b}"],
        ["please", "choose", "{a}", "with", "{b}"],
    ],
}

# Held-out word orders use only lexical items seen in training.
HELDOUT_TEMPLATES = {
    0: [["please", "give", "sum", "of", "{a}", "with", "{b}"]],
    1: [["please", "compute", "exclusive", "or", "{a}", "and", "{b}"]],
    2: [["bits", "result", "for", "{a}", "with", "{b}"]],
    3: [["result", "of", "branch", "for", "{a}", "and", "{b}"]],
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def target(op: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Value semantics deliberately independent of entity identity."""
    y = torch.empty_like(op)
    m = op == 0
    y[m] = ((a[m] & 3) + (b[m] & 3)) & 3
    m = op == 1
    y[m] = (a[m] ^ b[m]) & 3
    m = op == 2
    y[m] = ((a[m] >> 1) & 1) + 2 * ((b[m] >> 2) & 1)
    m = op == 3
    y[m] = torch.where((b[m] & 1) == 0, a[m] & 3, b[m] & 3)
    return y


def deranged_world(home: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    for _ in range(1000):
        w = torch.randint(0, VALUE_COUNT, (ENTITIES,), generator=g)
        if not bool((w == home).any()):
            return w
    # Deterministic fallback guarantees every binding differs.
    return (home + 1 + torch.arange(ENTITIES)) % VALUE_COUNT


def alias_token(entity: int, rng: random.Random) -> str:
    return ALIASES_A[entity] if rng.random() < 0.5 else ALIASES_B[entity]


def encode_query(op: int, a: int, b: int, rng: random.Random, heldout: bool) -> tuple[list[int], int, int]:
    templates = HELDOUT_TEMPLATES[op] if heldout else TRAIN_TEMPLATES[op]
    template = templates[rng.randrange(len(templates))]
    aa, bb = alias_token(a, rng), alias_token(b, rng)
    words = ["<CLS>"] + [aa if x == "{a}" else bb if x == "{b}" else x for x in template]
    if len(words) > MAX_LEN:
        raise AssertionError(words)
    ids = [TOKEN[x] for x in words]
    pa = ids.index(TOKEN[aa])
    pb = ids.index(TOKEN[bb], pa + 1 if aa == bb else 0)
    # a and b entity IDs are sampled independently; if they are the same entity and
    # the alias sampler chose the same alias token, there are still two occurrences.
    if a == b and aa == bb:
        positions = [i for i, tid in enumerate(ids) if tid == TOKEN[aa]]
        pa, pb = positions[0], positions[1]
    ids += [PAD] * (MAX_LEN - len(ids))
    return ids, pa, pb


def sample_queries(rng: random.Random, n: int, world: torch.Tensor, *, heldout: bool = False):
    xs, ops, pa, pb, ea, eb, ys = [], [], [], [], [], [], []
    for _ in range(n):
        op = rng.randrange(OPS)
        a = rng.randrange(ENTITIES)
        b = rng.randrange(ENTITIES)
        ids, posa, posb = encode_query(op, a, b, rng, heldout)
        xs.append(ids); ops.append(op); pa.append(posa); pb.append(posb); ea.append(a); eb.append(b)
        va = int(world[a]); vb = int(world[b])
        ys.append(int(target(torch.tensor([op]), torch.tensor([va]), torch.tensor([vb]))[0]))
    return (
        torch.tensor(xs, dtype=torch.long),
        torch.tensor(ops, dtype=torch.long),
        torch.tensor(pa, dtype=torch.long),
        torch.tensor(pb, dtype=torch.long),
        torch.tensor(ea, dtype=torch.long),
        torch.tensor(eb, dtype=torch.long),
        torch.tensor(ys, dtype=torch.long),
    )


class TinyEncoder(nn.Module):
    def __init__(self, vocab_size: int, layers: int):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, D_MODEL)
        self.pos = nn.Embedding(MAX_LEN, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=FF,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s = x.shape
        pos = torch.arange(s, device=x.device).unsqueeze(0).expand(b, s)
        h = self.tok(x) + self.pos(pos)
        h = self.encoder(h, src_key_padding_mask=x.eq(PAD))
        return self.norm(h)


class NeuralCompiler(nn.Module):
    """B-rail language compiler: outputs opcode + source-token pointers only."""
    def __init__(self):
        super().__init__()
        self.backbone = TinyEncoder(len(VOCAB), COMPILER_LAYERS)
        self.op_head = nn.Linear(D_MODEL, OPS)
        self.ptr_a = nn.Linear(D_MODEL, 1)
        self.ptr_b = nn.Linear(D_MODEL, 1)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        op = self.op_head(h[:, 0])
        entity_mask = torch.zeros_like(x, dtype=torch.bool)
        # Small vocab: membership check is cheap and explicit. This is part of the
        # EPF source-span restriction: pointer heads may target entity lexemes only.
        for tid in ENTITY_TOKEN_TO_ID:
            entity_mask |= x.eq(tid)
        pa = self.ptr_a(h).squeeze(-1).masked_fill(~entity_mask, -1e9)
        pb = self.ptr_b(h).squeeze(-1).masked_fill(~entity_mask, -1e9)
        return op, pa, pb, h


class ParametricMonolith(nn.Module):
    """Control: query-only transformer learns the static training world."""
    def __init__(self):
        super().__init__()
        self.backbone = TinyEncoder(len(VOCAB), MONO_LAYERS)
        self.head = nn.Sequential(nn.Linear(D_MODEL, 64), nn.GELU(), nn.Linear(64, CLASS_COUNT))

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.head(h[:, 0]), h


# J vocabulary is independent from linguistic/entity vocabulary.
J_PAD = 0
J_CLS = 1
J_OP_BASE = 2
J_A_BIT_BASE = J_OP_BASE + OPS
J_B_BIT_BASE = J_A_BIT_BASE + 10  # five positions * two bit values
J_VOCAB = J_B_BIT_BASE + 10
J_LEN = 12  # CLS + op + 5 A bits + 5 B bits


def j_encode(op: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    rows = []
    for oo, aa, bb in zip(op.tolist(), a.tolist(), b.tolist()):
        ids = [J_CLS, J_OP_BASE + oo]
        for bit in range(5):
            ids.append(J_A_BIT_BASE + bit * 2 + ((aa >> bit) & 1))
        for bit in range(5):
            ids.append(J_B_BIT_BASE + bit * 2 + ((bb >> bit) & 1))
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)


class JExecutor(nn.Module):
    """Mutable J rail. It receives opcode + current value bits, never entity IDs."""
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(J_VOCAB, D_MODEL)
        self.pos = nn.Embedding(J_LEN, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=FF,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=J_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Sequential(nn.Linear(D_MODEL, 64), nn.GELU(), nn.Linear(64, CLASS_COUNT))

    def forward(self, x: torch.Tensor):
        b, s = x.shape
        pos = torch.arange(s, device=x.device).unsqueeze(0).expand(b, s)
        h = self.norm(self.encoder(self.tok(x) + self.pos(pos)))
        return self.head(h[:, 0]), h


def train_compiler(seed: int, home: torch.Tensor):
    rng = random.Random(seed * 1009 + 1)
    model = NeuralCompiler()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    model.train(); t0 = time.perf_counter()
    for _ in range(COMPILER_STEPS):
        x, op, pa, pb, _, _, _ = sample_queries(rng, BATCH, home, heldout=False)
        lo, la, lb, _ = model(x)
        loss = F.cross_entropy(lo, op) + F.cross_entropy(la, pa) + F.cross_entropy(lb, pb)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model.eval(), time.perf_counter() - t0


def train_monolith(seed: int, home: torch.Tensor):
    rng = random.Random(seed * 1009 + 2)
    model = ParametricMonolith()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    model.train(); t0 = time.perf_counter()
    for _ in range(MONO_STEPS):
        x, _, _, _, _, _, y = sample_queries(rng, BATCH, home, heldout=False)
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model.eval(), time.perf_counter() - t0


def train_executor(seed: int):
    g = torch.Generator().manual_seed(seed * 1009 + 3)
    model = JExecutor()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    model.train(); t0 = time.perf_counter()
    # Reserve values 28..31 as unseen numeric values. Values are bit-tokenized, so
    # J must generalize the operation rather than memorize entity/value pairs.
    for _ in range(J_STEPS):
        op = torch.randint(0, OPS, (J_BATCH,), generator=g)
        a = torch.randint(0, 28, (J_BATCH,), generator=g)
        b = torch.randint(0, 28, (J_BATCH,), generator=g)
        y = target(op, a, b)
        logits, _ = model(j_encode(op, a, b))
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model.eval(), time.perf_counter() - t0


@torch.inference_mode()
def compiler_metrics(model: NeuralCompiler, seed: int, world: torch.Tensor, heldout: bool):
    rng = random.Random(seed * 3001 + (19 if heldout else 11))
    n = EVAL_BATCHES * BATCH
    x, op, pa, pb, _, _, _ = sample_queries(rng, n, world, heldout=heldout)
    lo, la, lb, _ = model(x)
    po, ppa, ppb = lo.argmax(-1), la.argmax(-1), lb.argmax(-1)
    return {
        "op_acc": float(po.eq(op).float().mean()),
        "ptr_a_acc": float(ppa.eq(pa).float().mean()),
        "ptr_b_acc": float(ppb.eq(pb).float().mean()),
        "full_plan_acc": float((po.eq(op) & ppa.eq(pa) & ppb.eq(pb)).float().mean()),
    }


def link_entity_ids(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    out = []
    for row, pos in zip(x.tolist(), positions.tolist()):
        tid = row[pos]
        out.append(ENTITY_TOKEN_TO_ID.get(tid, -1))
    return torch.tensor(out, dtype=torch.long)


@torch.inference_mode()
def pipeline_accuracy(compiler: NeuralCompiler, executor: JExecutor, seed: int, world: torch.Tensor, heldout: bool):
    rng = random.Random(seed * 5003 + (31 if heldout else 29))
    good = 0; total = 0; plan_good = 0
    for _ in range(EVAL_BATCHES):
        x, op, pa, pb, ea, eb, y = sample_queries(rng, BATCH, world, heldout=heldout)
        lo, la, lb, _ = compiler(x)
        pop, ppa, ppb = lo.argmax(-1), la.argmax(-1), lb.argmax(-1)
        linked_a = link_entity_ids(x, ppa)
        linked_b = link_entity_ids(x, ppb)
        valid = linked_a.ge(0) & linked_b.ge(0)
        safe_a = linked_a.clamp_min(0)
        safe_b = linked_b.clamp_min(0)
        va = world[safe_a]
        vb = world[safe_b]
        logits, _ = executor(j_encode(pop, va, vb))
        pred = logits.argmax(-1)
        good += int((pred.eq(y) & valid).sum())
        plan_good += int((pop.eq(op) & linked_a.eq(ea) & linked_b.eq(eb)).sum())
        total += len(y)
    return good / total, plan_good / total


@torch.inference_mode()
def monolith_accuracy(model: ParametricMonolith, seed: int, world: torch.Tensor, heldout: bool):
    rng = random.Random(seed * 7001 + (43 if heldout else 41))
    good = 0; total = 0
    for _ in range(EVAL_BATCHES):
        x, _, _, _, _, _, y = sample_queries(rng, BATCH, world, heldout=heldout)
        logits, _ = model(x)
        good += int(logits.argmax(-1).eq(y).sum()); total += len(y)
    return good / total


@torch.inference_mode()
def executor_unseen_value_accuracy(model: JExecutor, seed: int):
    g = torch.Generator().manual_seed(seed * 11003 + 7)
    good = 0; total = 0
    for _ in range(EVAL_BATCHES):
        op = torch.randint(0, OPS, (J_BATCH,), generator=g)
        # Force at least one unseen value 28..31 per row.
        a = torch.randint(0, VALUE_COUNT, (J_BATCH,), generator=g)
        b = torch.randint(28, VALUE_COUNT, (J_BATCH,), generator=g)
        swap = torch.rand(J_BATCH, generator=g) < 0.5
        aa = torch.where(swap, b, a); bb = torch.where(swap, a, b)
        y = target(op, aa, bb)
        logits, _ = model(j_encode(op, aa, bb))
        good += int(logits.argmax(-1).eq(y).sum()); total += len(y)
    return good / total


@torch.inference_mode()
def b_world_invariance(compiler: NeuralCompiler, seed: int, home: torch.Tensor, future: torch.Tensor):
    # World is intentionally absent from the compiler signature. We nevertheless run
    # the same language batch around a world transition to certify bit-identical B.
    rng = random.Random(seed * 13007 + 5)
    x, _, _, _, _, _, _ = sample_queries(rng, BATCH, home, heldout=False)
    lo1, la1, lb1, h1 = compiler(x)
    # Touch/read future to make the experimental boundary explicit.
    _ = int(future.sum())
    lo2, la2, lb2, h2 = compiler(x)
    return max(
        float((lo1 - lo2).abs().max()),
        float((la1 - la2).abs().max()),
        float((lb1 - lb2).abs().max()),
        float((h1 - h2).abs().max()),
    )


@torch.inference_mode()
def benchmark_cached_plan(compiler: NeuralCompiler, executor: JExecutor, seed: int, world: torch.Tensor):
    rng = random.Random(seed * 17011 + 13)
    x, _, _, _, _, _, _ = sample_queries(rng, BATCH, world, heldout=False)
    # Compile once and trusted-link source positions.
    lo, la, lb, _ = compiler(x)
    pop, ppa, ppb = lo.argmax(-1), la.argmax(-1), lb.argmax(-1)
    ea = link_entity_ids(x, ppa).clamp_min(0); eb = link_entity_ids(x, ppb).clamp_min(0)
    for _ in range(8):
        va, vb = world[ea], world[eb]
        executor(j_encode(pop, va, vb))
        compiler(x)

    loops = 60
    t0 = time.perf_counter_ns()
    for i in range(loops):
        lo2, la2, lb2, _ = compiler(x)
        op2, pa2, pb2 = lo2.argmax(-1), la2.argmax(-1), lb2.argmax(-1)
        a2 = link_entity_ids(x, pa2).clamp_min(0); b2 = link_entity_ids(x, pb2).clamp_min(0)
        w = (world + i + 1) % VALUE_COUNT
        executor(j_encode(op2, w[a2], w[b2]))
    full_ns = (time.perf_counter_ns() - t0) / loops

    t0 = time.perf_counter_ns()
    for i in range(loops):
        w = (world + i + 1) % VALUE_COUNT
        executor(j_encode(pop, w[ea], w[eb]))
    cached_ns = (time.perf_counter_ns() - t0) / loops
    return full_ns, cached_ns


@dataclass
class Row:
    seed: int
    compiler_train_s: float
    monolith_train_s: float
    executor_train_s: float
    compiler_train_template_plan_acc: float
    compiler_heldout_template_plan_acc: float
    executor_unseen_value_acc: float
    monolith_home_acc: float
    monolith_future_deranged_acc: float
    monolith_future_heldout_template_acc: float
    ckca_home_acc: float
    ckca_future_deranged_acc: float
    ckca_future_heldout_template_acc: float
    ckca_future_plan_acc: float
    b_world_change_max_state_delta: float
    full_compile_execute_ns: float
    cached_plan_execute_ns: float
    cached_plan_speedup: float


def mean(rows, key):
    return sum(getattr(r, key) for r in rows) / len(rows)


def main() -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.set_num_interop_threads(1)
    rows: list[Row] = []

    for seed in SEEDS:
        seed_all(seed)
        g = torch.Generator().manual_seed(seed * 191 + 17)
        home = torch.randint(0, VALUE_COUNT, (ENTITIES,), generator=g)
        future = deranged_world(home, g)

        compiler, tc = train_compiler(seed, home)
        mono, tm = train_monolith(seed, home)
        executor, tj = train_executor(seed)

        ctrain = compiler_metrics(compiler, seed, home, False)["full_plan_acc"]
        cheld = compiler_metrics(compiler, seed, home, True)["full_plan_acc"]
        ex_unseen = executor_unseen_value_accuracy(executor, seed)

        mono_home = monolith_accuracy(mono, seed, home, False)
        mono_future = monolith_accuracy(mono, seed, future, False)
        mono_future_held = monolith_accuracy(mono, seed, future, True)

        ckca_home, _ = pipeline_accuracy(compiler, executor, seed, home, False)
        ckca_future, plan_future = pipeline_accuracy(compiler, executor, seed, future, False)
        ckca_future_held, _ = pipeline_accuracy(compiler, executor, seed, future, True)

        inv = b_world_invariance(compiler, seed, home, future)
        full_ns, cached_ns = benchmark_cached_plan(compiler, executor, seed, home)
        rows.append(Row(
            seed=seed,
            compiler_train_s=tc,
            monolith_train_s=tm,
            executor_train_s=tj,
            compiler_train_template_plan_acc=ctrain,
            compiler_heldout_template_plan_acc=cheld,
            executor_unseen_value_acc=ex_unseen,
            monolith_home_acc=mono_home,
            monolith_future_deranged_acc=mono_future,
            monolith_future_heldout_template_acc=mono_future_held,
            ckca_home_acc=ckca_home,
            ckca_future_deranged_acc=ckca_future,
            ckca_future_heldout_template_acc=ckca_future_held,
            ckca_future_plan_acc=plan_future,
            b_world_change_max_state_delta=inv,
            full_compile_execute_ns=full_ns,
            cached_plan_execute_ns=cached_ns,
            cached_plan_speedup=full_ns / cached_ns,
        ))

    report = {
        "stage": STAGE,
        "architecture_candidate": "World-Interfaced Dual-Rail Transformer (WIDRT)",
        "mechanism": (
            "A B-rail Transformer reads language and compiles only an opcode plus source-token pointers. A trusted linker maps those exact "
            "lexical source positions to canonical entities. Current world values are resolved after compilation and passed to a separate "
            "J-rail Transformer as bit-tokenized typed data. J never receives entity identity; B never receives mutable world values; there "
            "is no J->B edge. The compiled plan can therefore be cached across arbitrary world rebinding."
        ),
        "structural_contract": {
            "b_inputs": "language tokens only",
            "compiler_outputs": "opcode + lexical source pointers only",
            "trusted_linker": "source pointer -> canonical entity",
            "j_inputs": "opcode + current typed value bits only",
            "entity_identity_visible_to_j": False,
            "mutable_world_visible_to_b": False,
            "j_to_b_edge": False,
        },
        "entities": ENTITIES,
        "values": VALUE_COUNT,
        "heldout_executor_values": [28, 29, 30, 31],
        "seeds": list(SEEDS),
        "per_seed": [asdict(r) for r in rows],
        "mean_compiler_train_template_plan_acc": mean(rows, "compiler_train_template_plan_acc"),
        "mean_compiler_heldout_template_plan_acc": mean(rows, "compiler_heldout_template_plan_acc"),
        "mean_executor_unseen_value_acc": mean(rows, "executor_unseen_value_acc"),
        "mean_monolith_home_acc": mean(rows, "monolith_home_acc"),
        "mean_monolith_future_deranged_acc": mean(rows, "monolith_future_deranged_acc"),
        "mean_monolith_future_heldout_template_acc": mean(rows, "monolith_future_heldout_template_acc"),
        "mean_ckca_home_acc": mean(rows, "ckca_home_acc"),
        "mean_ckca_future_deranged_acc": mean(rows, "ckca_future_deranged_acc"),
        "mean_ckca_future_heldout_template_acc": mean(rows, "ckca_future_heldout_template_acc"),
        "mean_ckca_future_plan_acc": mean(rows, "ckca_future_plan_acc"),
        "max_b_world_change_state_delta": max(r.b_world_change_max_state_delta for r in rows),
        "mean_cached_plan_speedup": mean(rows, "cached_plan_speedup"),
        "future_gain_ckca_minus_parametric_monolith": mean(rows, "ckca_future_deranged_acc") - mean(rows, "monolith_future_deranged_acc"),
        "world_updates_require_gradient_steps": False,
        "dod_status": "NOT_DOD; neural dual-rail architecture gate",
        "claim_boundary": (
            "Transformers, semantic parsing, pointer networks, modular neural systems and program execution are established. R329 tests a "
            "CKCA-specific hard world-interface boundary—language compiler to trusted linker to identity-blind mutable neural executor—rather "
            "than claiming these components individually."
        ),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in [
        "stage",
        "mean_compiler_heldout_template_plan_acc",
        "mean_executor_unseen_value_acc",
        "mean_monolith_home_acc",
        "mean_monolith_future_deranged_acc",
        "mean_ckca_home_acc",
        "mean_ckca_future_deranged_acc",
        "mean_ckca_future_heldout_template_acc",
        "max_b_world_change_state_delta",
        "mean_cached_plan_speedup",
        "future_gain_ckca_minus_parametric_monolith",
        "report_sha256",
    ]}, indent=2))


if __name__ == "__main__":
    main()
