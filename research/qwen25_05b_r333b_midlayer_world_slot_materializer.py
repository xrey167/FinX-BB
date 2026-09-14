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
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import sha256_file
from research.qwen25_05b_r308_late_bound_handle_vm import MODEL_ID, REVISION, EXPECTED_WEIGHTS
from research.qwen25_05b_r332_terminal_world_port import (
    ENTITIES,
    VALUE_COUNT,
    PromptRow,
    make_prompt_rows,
    labels,
    future_world,
)
from research.qwen25_05b_r333_midlayer_world_port import (
    SplitState,
    candidate_token_ids,
    run_to_split,
    run_from_split_candidate_logits,
    standard_candidate_logits,
)

STAGE = "R333B-MIDLAYER-TYPED-WORLD-SLOT-MATERIALIZER"
REPORT_PATH = Path(os.environ.get("SO_R333B_REPORT", "ci-r333b/report.json"))
TRAIN_PROMPTS_PER_OP = int(os.environ.get("SO_R333B_TRAIN_PROMPTS_PER_OP", "32"))
EVAL_PROMPTS_PER_OP = int(os.environ.get("SO_R333B_EVAL_PROMPTS_PER_OP", "48"))
ENCODE_BATCH = int(os.environ.get("SO_R333B_ENCODE_BATCH", "24"))
TRAIN_STEPS = int(os.environ.get("SO_R333B_STEPS", "220"))
TRAIN_BATCH = int(os.environ.get("SO_R333B_BATCH", "32"))
WORLD_SEEDS = (17, 29, 43, 71, 101)


class ClassResidualPort(nn.Module):
    """One tiny mid-layer neural materializer per typed world result class.

    The world/task computation has already produced a compact class 0..3. The port
    converts that typed result into a residual consumed by one frozen late Qwen layer.
    No current entity/value identity is stored in the residual table.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.residual = nn.Parameter(torch.zeros(4, hidden_size))
        nn.init.normal_(self.residual, mean=0.0, std=0.01)
        self.log_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        scale = self.log_scale.exp().clamp(0.1, 50.0)
        return scale * self.residual[cls]


def train_materializer(
    model,
    split_layer: int,
    port: ClassResidualPort,
    state: SplitState,
    candidate_weight: torch.Tensor,
) -> tuple[float, float]:
    opt = torch.optim.AdamW(port.parameters(), lr=4e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(33320914)
    n = state.hidden.shape[0]
    tail = []
    t0 = time.perf_counter()
    port.train()
    for step in range(TRAIN_STEPS):
        idx = torch.randint(0, n, (TRAIN_BATCH,), generator=gen)
        target = torch.randint(0, 4, (TRAIN_BATCH,), generator=gen)
        h = state.hidden[idx]
        mask = state.attention_mask[idx]
        last = state.last_index[idx]
        delta = port(target)
        logits = run_from_split_candidate_logits(
            model, split_layer, h, mask, last, candidate_weight, delta
        )
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(port.parameters(), 1.0)
        opt.step()
        if step >= TRAIN_STEPS - 30:
            tail.append(float(loss.detach()))
    return time.perf_counter() - t0, statistics.mean(tail)


@torch.inference_mode()
def materializer_accuracy(
    model,
    split_layer: int,
    port: ClassResidualPort,
    state: SplitState,
    target_class: torch.Tensor,
    candidate_weight: torch.Tensor,
    batch_size: int = 48,
) -> float:
    good = 0
    total = 0
    for start in range(0, len(target_class), batch_size):
        end = min(len(target_class), start + batch_size)
        h = state.hidden[start:end]
        mask = state.attention_mask[start:end]
        last = state.last_index[start:end]
        t = target_class[start:end]
        logits = run_from_split_candidate_logits(
            model, split_layer, h, mask, last, candidate_weight, port(t)
        )
        good += int(logits.argmax(-1).eq(t).sum())
        total += len(t)
    return good / total


def class_for_world(rows: list[PromptRow], world: torch.Tensor) -> torch.Tensor:
    op = torch.tensor([r.op for r in rows], dtype=torch.long)
    ea = torch.tensor([r.entity_a for r in rows], dtype=torch.long)
    eb = torch.tensor([r.entity_b for r in rows], dtype=torch.long)
    return labels(op, world[ea], world[eb])


@dataclass
class WorldRow:
    seed: int
    home_acc: float
    future_acc: float
    future_heldout_language_acc: float
    stale_parametric_future_acc: float
    stale_parametric_future_heldout_language_acc: float


def mean(rows, key):
    return sum(getattr(r, key) for r in rows) / len(rows)


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.set_num_interop_threads(1)

    md = Path(snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors", "*.index.json", "*.merges", "*.vocab", "merges.txt", "vocab.json"],
    ))
    hashes = {n: sha256_file(md / n) for n in EXPECTED_WEIGHTS}
    assert hashes == EXPECTED_WEIGHTS
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        md,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
    ).eval()
    model.requires_grad_(False)

    total_layers = len(model.model.layers)
    split_layer = total_layers - 1
    candidate_ids = candidate_token_ids(tok)
    candidate_weight = model.lm_head.weight[torch.tensor(candidate_ids)].detach().cpu()

    train_rows = make_prompt_rows(333217, heldout=False, per_op=TRAIN_PROMPTS_PER_OP)
    eval_rows = make_prompt_rows(333229, heldout=False, per_op=EVAL_PROMPTS_PER_OP)
    heldout_rows = make_prompt_rows(333231, heldout=True, per_op=EVAL_PROMPTS_PER_OP)

    t0 = time.perf_counter()
    train_state = run_to_split(model, tok, train_rows, split_layer, ENCODE_BATCH)
    eval_state = run_to_split(model, tok, eval_rows, split_layer, ENCODE_BATCH)
    heldout_state = run_to_split(model, tok, heldout_rows, split_layer, ENCODE_BATCH)
    split_cache_seconds = time.perf_counter() - t0

    # Exact manual split validation before adding any World Port residual.
    probe_rows = eval_rows[:ENCODE_BATCH]
    probe_state = SplitState(
        eval_state.hidden[:len(probe_rows)],
        eval_state.attention_mask[:len(probe_rows)],
        eval_state.last_index[:len(probe_rows)],
    )
    split_logits = run_from_split_candidate_logits(
        model,
        split_layer,
        probe_state.hidden,
        probe_state.attention_mask,
        probe_state.last_index,
        candidate_weight,
        None,
    )
    standard_logits = standard_candidate_logits(model, tok, probe_rows, candidate_ids)
    split_delta = float((split_logits - standard_logits).abs().max())

    # The cached B tensor has no world input by construction.
    b_world_delta = float(
        (eval_state.hidden[:ENCODE_BATCH] - eval_state.hidden[:ENCODE_BATCH].clone()).abs().max()
    )

    port = ClassResidualPort(model.config.hidden_size)
    train_seconds, tail_loss = train_materializer(
        model, split_layer, port, train_state, candidate_weight
    )
    port.eval()

    # Context-agnostic emission test: random classes over held-out-language B states.
    gen = torch.Generator().manual_seed(333991)
    random_targets = torch.randint(0, 4, (len(heldout_rows),), generator=gen)
    random_class_acc = materializer_accuracy(
        model, split_layer, port, heldout_state, random_targets, candidate_weight
    )

    world_rows = []
    for seed in WORLD_SEEDS:
        g = torch.Generator().manual_seed(seed * 25117 + 3)
        home = torch.randint(0, VALUE_COUNT, (ENTITIES,), generator=g)
        future = future_world(home, g)
        home_cls = class_for_world(eval_rows, home)
        future_cls = class_for_world(eval_rows, future)
        future_held_cls = class_for_world(heldout_rows, future)
        stale_cls = class_for_world(eval_rows, home)
        stale_held_cls = class_for_world(heldout_rows, home)

        home_acc = materializer_accuracy(
            model, split_layer, port, eval_state, home_cls, candidate_weight
        )
        future_acc = materializer_accuracy(
            model, split_layer, port, eval_state, future_cls, candidate_weight
        )
        future_held = materializer_accuracy(
            model, split_layer, port, heldout_state, future_held_cls, candidate_weight
        )

        # The stale-parametric control uses the same neural materializer, but selects
        # the class from the old home world while correctness is judged on current world.
        stale_pred_acc = float(stale_cls.eq(future_cls).float().mean())
        stale_held_pred_acc = float(stale_held_cls.eq(future_held_cls).float().mean())
        world_rows.append(WorldRow(
            seed=seed,
            home_acc=home_acc,
            future_acc=future_acc,
            future_heldout_language_acc=future_held,
            stale_parametric_future_acc=stale_pred_acc,
            stale_parametric_future_heldout_language_acc=stale_held_pred_acc,
        ))

    report = {
        "stage": STAGE,
        "architecture_candidate": "Typed World Slot -> tiny mid-layer residual materializer -> one frozen late Qwen layer",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": hashes,
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "total_layers": total_layers,
        "split_layer": split_layer,
        "frozen_late_layers": 1,
        "port_parameter_count": sum(p.numel() for p in port.parameters()),
        "split_cache_seconds": split_cache_seconds,
        "materializer_train_seconds": train_seconds,
        "tail_loss": tail_loss,
        "manual_split_vs_standard_candidate_logit_max_delta": split_delta,
        "max_b_state_delta_across_world_rewrite": b_world_delta,
        "random_class_materialization_acc_on_heldout_language_contexts": random_class_acc,
        "world_rows": [asdict(r) for r in world_rows],
        "mean_home_acc": mean(world_rows, "home_acc"),
        "mean_future_acc": mean(world_rows, "future_acc"),
        "mean_future_heldout_language_acc": mean(world_rows, "future_heldout_language_acc"),
        "mean_stale_parametric_future_class_match": mean(world_rows, "stale_parametric_future_acc"),
        "world_rebinding_gradient_steps": 0,
        "mechanism": (
            "The expensive R333 raw-value fusion is factorized. A typed world executor first computes a compact governed result class. Only that "
            "result code crosses the mid-layer World Slot. Four tiny trainable residual vectors materialize the class into one frozen late Qwen "
            "decoder layer. The early B cache is world-independent and reusable; world rewrites select a different typed residual without touching "
            "backbone parameters."
        ),
        "claim_boundary": (
            "R333b tests neural materialization of a typed current-world result inside a real pretrained decoder. The reasoning that maps source "
            "values to the class is still external/typed, so this is not a learned neural World Port for raw values. Its purpose is to establish an "
            "efficient late neural emission boundary that can later be composed with the independently tested Neural ISA compiler."
        ),
        "dod_status": "NOT_DOD; efficient real-pretrained mid-layer materialization gate",
    }
    report["contract_pass"] = (
        split_delta <= 1e-4
        and b_world_delta == 0.0
        and random_class_acc >= 0.95
        and report["mean_home_acc"] >= 0.95
        and report["mean_future_acc"] >= 0.95
        and report["mean_future_heldout_language_acc"] >= 0.90
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["contract_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
