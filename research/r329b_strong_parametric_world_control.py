from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.r329_world_interfaced_transformer import (
    SEEDS, ENTITIES, VALUE_COUNT, BATCH, EVAL_BATCHES,
    train_compiler, train_executor, pipeline_accuracy,
    sample_queries, link_entity_ids, deranged_world,
    compiler_metrics, executor_unseen_value_accuracy, j_encode,
)

STAGE = "R329B-STRONG-PARAMETRIC-WORLD-CONTROL"
REPORT_PATH = Path(os.environ.get("SO_R329B_REPORT", "r329b_report.json"))
TABLE_STEPS = int(os.environ.get("SO_R329B_TABLE_STEPS", "300"))


class ParametricWorldTable(nn.Module):
    """Strong control: the entire static world lives directly in trainable parameters.

    This is intentionally much easier to fit than the R329 query-only monolith. It
    isolates the architectural question: with the same neural compiler and J executor,
    what happens when current world state lives in model parameters rather than the ABI?
    """
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(ENTITIES, VALUE_COUNT))

    def forward(self, entity_ids: torch.Tensor) -> torch.Tensor:
        return self.logits[entity_ids]

    @torch.inference_mode()
    def values(self, entity_ids: torch.Tensor) -> torch.Tensor:
        return self(entity_ids).argmax(-1)


def train_parametric_world(seed: int, home: torch.Tensor):
    torch.manual_seed(seed * 4001 + 17)
    model = ParametricWorldTable()
    opt = torch.optim.AdamW(model.parameters(), lr=0.08, weight_decay=0.0)
    ids = torch.arange(ENTITIES, dtype=torch.long)
    t0 = time.perf_counter()
    for _ in range(TABLE_STEPS):
        loss = F.cross_entropy(model(ids), home)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.inference_mode():
        fit = float(model.values(ids).eq(home).float().mean())
    return model.eval(), fit, time.perf_counter() - t0


@torch.inference_mode()
def parametric_pipeline_accuracy(compiler, executor, table, seed: int, label_world: torch.Tensor, heldout: bool):
    rng = random.Random(seed * 8009 + (71 if heldout else 67))
    good = 0
    plan_good = 0
    total = 0
    for _ in range(EVAL_BATCHES):
        x, op, _, _, ea, eb, y = sample_queries(rng, BATCH, label_world, heldout=heldout)
        lo, la, lb, _ = compiler(x)
        pop, ppa, ppb = lo.argmax(-1), la.argmax(-1), lb.argmax(-1)
        linked_a = link_entity_ids(x, ppa)
        linked_b = link_entity_ids(x, ppb)
        valid = linked_a.ge(0) & linked_b.ge(0)
        safe_a = linked_a.clamp_min(0)
        safe_b = linked_b.clamp_min(0)
        # Difference from CKCA: values are read from frozen model parameters, not
        # from the current external generation-scoped world.
        va = table.values(safe_a)
        vb = table.values(safe_b)
        logits, _ = executor(j_encode(pop, va, vb))
        good += int((logits.argmax(-1).eq(y) & valid).sum())
        plan_good += int((pop.eq(op) & linked_a.eq(ea) & linked_b.eq(eb)).sum())
        total += len(y)
    return good / total, plan_good / total


@dataclass
class Row:
    seed: int
    parametric_table_fit_acc: float
    parametric_table_train_s: float
    compiler_heldout_plan_acc: float
    executor_unseen_value_acc: float
    ckca_home_acc: float
    ckca_future_acc: float
    ckca_future_heldout_acc: float
    parametric_home_acc: float
    parametric_future_acc: float
    parametric_future_heldout_acc: float
    shared_future_plan_acc_ckca: float
    shared_future_plan_acc_parametric: float


def mean(rows, key):
    return sum(getattr(r, key) for r in rows) / len(rows)


def main() -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.set_num_interop_threads(1)
    rows = []

    for seed in SEEDS:
        g = torch.Generator().manual_seed(seed * 191 + 17)
        home = torch.randint(0, VALUE_COUNT, (ENTITIES,), generator=g)
        future = deranged_world(home, g)

        compiler, _ = train_compiler(seed, home)
        executor, _ = train_executor(seed)
        table, table_fit, table_s = train_parametric_world(seed, home)

        cheld = compiler_metrics(compiler, seed, home, True)["full_plan_acc"]
        ex_unseen = executor_unseen_value_accuracy(executor, seed)

        ckca_home, _ = pipeline_accuracy(compiler, executor, seed, home, False)
        ckca_future, ckca_plan = pipeline_accuracy(compiler, executor, seed, future, False)
        ckca_future_held, _ = pipeline_accuracy(compiler, executor, seed, future, True)

        param_home, _ = parametric_pipeline_accuracy(compiler, executor, table, seed, home, False)
        param_future, param_plan = parametric_pipeline_accuracy(compiler, executor, table, seed, future, False)
        param_future_held, _ = parametric_pipeline_accuracy(compiler, executor, table, seed, future, True)

        rows.append(Row(
            seed=seed,
            parametric_table_fit_acc=table_fit,
            parametric_table_train_s=table_s,
            compiler_heldout_plan_acc=cheld,
            executor_unseen_value_acc=ex_unseen,
            ckca_home_acc=ckca_home,
            ckca_future_acc=ckca_future,
            ckca_future_heldout_acc=ckca_future_held,
            parametric_home_acc=param_home,
            parametric_future_acc=param_future,
            parametric_future_heldout_acc=param_future_held,
            shared_future_plan_acc_ckca=ckca_plan,
            shared_future_plan_acc_parametric=param_plan,
        ))

    report = {
        "stage": STAGE,
        "architecture_candidate": "World-Interfaced Dual-Rail Transformer vs strong parametric-world control",
        "control_design": (
            "Both paths use the same trained neural language compiler, trusted source-pointer linker and identity-blind J Transformer. "
            "The only architectural difference is the mutable value source: CKCA reads the supplied current external world; the control "
            "reads a parameter table trained to 100% on the home entity->value bindings and receives no parameter update after rebinding."
        ),
        "seeds": list(SEEDS),
        "per_seed": [asdict(r) for r in rows],
        "mean_parametric_table_fit_acc": mean(rows, "parametric_table_fit_acc"),
        "mean_compiler_heldout_plan_acc": mean(rows, "compiler_heldout_plan_acc"),
        "mean_executor_unseen_value_acc": mean(rows, "executor_unseen_value_acc"),
        "mean_ckca_home_acc": mean(rows, "ckca_home_acc"),
        "mean_ckca_future_acc": mean(rows, "ckca_future_acc"),
        "mean_ckca_future_heldout_acc": mean(rows, "ckca_future_heldout_acc"),
        "mean_parametric_home_acc": mean(rows, "parametric_home_acc"),
        "mean_parametric_future_acc": mean(rows, "parametric_future_acc"),
        "mean_parametric_future_heldout_acc": mean(rows, "parametric_future_heldout_acc"),
        "future_gain_ckca_minus_strong_parametric_control": mean(rows, "ckca_future_acc") - mean(rows, "parametric_future_acc"),
        "world_updates_applied_to_ckca_via_runtime_data": True,
        "world_updates_applied_to_parametric_control": False,
        "gradient_steps_required_by_ckca_world_rebinding": 0,
        "same_value_generation_identity_representable_by_value_only_parametric_table": False,
        "dod_status": "NOT_DOD; strong parametric-world counterfactual gate",
        "claim_boundary": (
            "This is a controlled synthetic architecture experiment. It does not establish that external world state is universally superior "
            "to model editing, nor does it compare against strong RAG or modern editable-memory methods. It isolates future rebinding when the "
            "language compiler and neural executor are held common."
        ),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in [
        "stage", "mean_parametric_table_fit_acc", "mean_compiler_heldout_plan_acc",
        "mean_executor_unseen_value_acc", "mean_ckca_home_acc", "mean_ckca_future_acc",
        "mean_ckca_future_heldout_acc", "mean_parametric_home_acc", "mean_parametric_future_acc",
        "mean_parametric_future_heldout_acc", "future_gain_ckca_minus_strong_parametric_control",
        "report_sha256",
    ]}, indent=2))


if __name__ == "__main__":
    main()
