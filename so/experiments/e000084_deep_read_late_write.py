"""E-000084 -- address deep, write late: which half of the symlink reader needs depth?

E-000083 (GitHub Actions run 33966506365, seed 2, three consistency weights) put the
whole reader -- routing query, dereference chain and payload write -- on GPT-2's
final block and lost the strict gate by a wide margin: held-out candidate mean
0.669-0.686 and worst template 0.41-0.49, against 0.984 / 0.96 for the identical
recipe at read layers (8, 10) (E-000081, seed 2, consistency 0.15).  Fixed final-
block late binding therefore does NOT preserve capability, and the programme's
"mandatory strong baseline" is not available as stated.

That failure confounds two placements.  The ADDRESS -- the routing query, taken
from the last-token residual of the read block -- and the WRITE -- the resolved
payload added to the residual -- were moved together.  E-000084 moves only the
write.  Arm C keeps the routing query and the dereference chain at blocks 8 and 10
exactly as E-000081 does, injects nothing there, and writes the summed read once
after block 11 (``AdapterConfig.write_layer=11``).  By construction every K/V
tensor the frozen model persists is then a function of the prompt alone, which
the experiment verifies on real prompts rather than assumes.  Arm A is the
E-000081 configuration (write in place at 8 and 10), run on the same seeds as the
capability anchor.

Two outcomes were fixed before the run, and the second is what happened:

* C passes the unchanged strict gate on every seed A passes: E-000083's failure
  was addressing depth alone; "read deep / write late" is the corrected cache-pure
  baseline, and no persistent neural state depends on a pod.  This is an
  engineering result (kNN-LM-shaped), not a novelty claim.
* C fails on a seed where A passes: the frozen blocks after the write must PROCESS
  the payload for the reader to work; capability requires in-model participation
  and is in genuine tension with cache purity.  Then the depth/participation
  frontier is a live mechanism question rather than a tautology.

Run 33970654975 gave the second: arm C reached held-out candidate means of
0.664 / 0.645 / 0.621 on seeds 0 / 1 / 2 with K/V exposure exactly 0.0, while arm A
reached 0.955 / 0.990 on seeds 0 / 1.  Something has to ride through the frozen
blocks; a memory that leaves no trace in them is not read back.

Arms D and E follow from that.

* D (write after block 10) answers a confound in C: C removes the payload from every
  block that could process it AND removes the block-8 write from the input of the
  block-10 read.  D keeps the second change and restores one block of processing.
* E asks whether what rides has to be the KNOWLEDGE.  Each row gets a fixed random
  handle; the read layers inject the routing-weighted handle in place, so it takes
  part in the frozen computation exactly as a payload write would, and the value is
  bound to the handle only after the last cache-writing block.  Handles are a
  function of row position, so a payload UPDATE, an alias RELINK and a SHRED leave
  every persisted tensor bit-identical while still changing the answer.  If E holds
  the capability gate, participation and revocability stop trading off.

Declared by construction (pipeline rows, not claim rows): C's persisted K/V and
every block input are bit-identical to the no-memory forward (exposure 0.0) and
A's are not; E's lifecycle exposure is 0.0 for UPDATE, RELINK and SHRED while its
exposure against the no-memory forward is not.  Claim rows: the four held-out
templates, candidate-set and full-vocabulary, per seed, under the unchanged
>= 0.95 bars.

This is a capability screen.  It makes no novelty claim and changes no CAVI
semantics, threshold or attack battery.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from so.data import bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments import e000081_symlink_consistency_reader as E81
from so.llm_adapter import AdapterConfig, transformer_blocks

ARMS = {
    "A": dict(read_layers=(8, 10), write_layer=None),   # E-000081: address and write at (8, 10)
    "C": dict(read_layers=(8, 10), write_layer=11),     # address at (8, 10), one write after block 11
    "D": dict(read_layers=(8, 10), write_layer=10),     # address at (8, 10), one write after block 10
    "E": dict(read_layers=(8, 10), write_layer=11, reference_carrier=True),  # handles ride, value binds late
}

# Arm D exists because arm C changes two things at once, and the audit of run 33970654975 said so:
# it removes the payload from every block that could process it AND it removes the block-8 write from
# the input of the block-10 read, so the second read's routing sees a different residual.  D holds the
# second change fixed -- it also has no in-place write at block 8, so its block-10 read sees exactly
# the residual C's does -- and restores exactly one block of downstream processing.  Reading C, D and A
# together therefore separates "the frozen blocks after the write must process the payload" from "the
# first write's feedback into the second read is what capability needs".
#
# The participation depth d = (n_blocks - 1) - write_layer is the size of the K/V light cone the write
# creates, so the three arms are also the first interior points of the exposure/capability frontier:
# C has d = 0 and exposure exactly 0.0, D has d = 1, A writes in place at 8 and 10.


def _kv_tensors(pkv) -> List[torch.Tensor]:
    """Every persisted K and V tensor, whichever cache class this transformers version returns."""
    out = []
    if hasattr(pkv, "layers"):
        for layer in pkv.layers:
            out += [layer.keys, layer.values]
    elif hasattr(pkv, "key_cache"):
        for k, v in zip(pkv.key_cache, pkv.value_cache):
            out += [k, v]
    else:
        for k, v in pkv:
            out += [k, v]
    return out


@torch.no_grad()
def exposure(gk: E8.GPT2Knowledge, bank, texts: List[str]) -> Dict[str, float]:
    """Max-abs difference between the memory-bearing and the no-memory forward on what the model persists.

    Runs the frozen core through the adapter's hooks directly so that ``past_key_values`` and every
    block input are visible.  ``kv_maxabs`` and ``block_input_maxabs`` are the exposure; ``last_logit_maxabs``
    shows the memory was material on the same prompts.
    """
    m = gk.model
    m.eval()
    ids, am, last = E8.encode_texts(gk.tok, texts)
    ar = torch.arange(ids.shape[0])

    def run(with_bank: bool):
        m._ctx = m.make_ctx(bank, last) if with_bank else None
        o = m.lm(input_ids=ids, attention_mask=am, output_hidden_states=True, use_cache=True)
        m._ctx = None
        return [t.clone() for t in _kv_tensors(o.past_key_values)], [h.clone() for h in o.hidden_states], o.logits.clone()

    kv_m, hs_m, lg_m = run(True)
    kv_0, hs_0, lg_0 = run(False)
    n_blocks = len(transformer_blocks(m.lm))
    kv_maxabs = max(float((a - b).abs().max()) for a, b in zip(kv_m, kv_0))
    # hidden_states[i] for i < n_blocks is the INPUT of block i; hidden_states[n_blocks] is the final
    # (ln_f-normalised) output and is the only one a final-block write is allowed to move.
    block_input_maxabs = max(float((hs_m[i] - hs_0[i]).abs().max()) for i in range(n_blocks))
    return {
        "kv_maxabs": kv_maxabs,
        "block_input_maxabs": block_input_maxabs,
        "last_logit_maxabs": float((lg_m[ar, last] - lg_0[ar, last]).abs().max()),
        "n_prompts": int(ids.shape[0]),
        "prompt_tokens_max": int(ids.shape[1]),
    }


@torch.no_grad()
def lifecycle_exposure(gk: E8.GPT2Knowledge, bank, texts: List[str]) -> Dict[str, float]:
    """Does a lifecycle operation move anything the model persists?

    The question arm E exists for. A payload UPDATE, an alias RELINK and a SHRED of every marker are
    applied to the bank, and the persisted K/V is compared with the pre-operation cache on the same
    prompts. Zero means no cached state has to be invalidated, recomputed or lineage-tracked when the
    knowledge changes; the answer is still required to move, otherwise the memory is simply inert.
    """
    m = gk.model
    ids, am, last = E8.encode_texts(gk.tok, texts)
    ar = torch.arange(ids.shape[0])

    def run_bank(b):
        m._ctx = m.make_ctx(b, last)
        o = m.lm(input_ids=ids, attention_mask=am, use_cache=True)
        m._ctx = None
        return [t.clone() for t in _kv_tensors(o.past_key_values)], o.logits[ar, last].clone()

    kv0, lg0 = run_bank(bank)
    out: Dict[str, float] = {}
    n = int(bank["obj"].shape[0])
    ops = {
        "update_payload": lambda b: b.__setitem__("obj", (b["obj"] + 1) % int(gk.n_entities)),
        "relink": lambda b: b.__setitem__("resolved_idx", torch.roll(b["resolved_idx"], max(1, n // 7))),
        "shred_markers": lambda b: b.__setitem__("marker", torch.zeros_like(b["marker"])),
    }
    for name, mutate in ops.items():
        mutated = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in bank.items()}
        mutate(mutated)
        kv1, lg1 = run_bank(mutated)
        out[f"{name}_kv_maxabs"] = max(float((a - b).abs().max()) for a, b in zip(kv0, kv1))
        out[f"{name}_logit_maxabs"] = float((lg0 - lg1).abs().max())
    return out


def run(arm: str, seed: int, steps: int, consistency: float, alt_supervision: float,
        n_groups: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    placement = ARMS[arm]
    cfg = AdapterConfig(
        read_layers=placement["read_layers"],
        write_layer=placement["write_layer"],
        reference_carrier=placement.get("reference_carrier", False),
        status_gated=True,
        use_links=True,
        n_deref=E20.N_DEREF,
    )
    gk = E8.GPT2Knowledge(cfg)
    blocks = transformer_blocks(gk.model.lm)
    if len(blocks) != 12:
        raise RuntimeError(f"E-000084 is specified for 12-block GPT-2 small; got {len(blocks)} blocks")
    if arm == "C" and cfg.write_layer != len(blocks) - 1:
        raise RuntimeError("arm C must write after the final block")

    trained = E81.train_symlink_consistent(
        gk, seed, steps, consistency=consistency, alt_supervision=alt_supervision,
        n_groups=max(24, n_groups), verbose=True,
    )
    centre = np.asarray(trained["centre"])

    # Identical independent-world construction family as E-000081 / E-000083.
    rng = np.random.default_rng(70000 + seed)
    world, spec = E15.sample_alias_world(rng, 180, n_groups, 2, gk.n_entities, 4, E20.N_TRAIN_TEMPLATES)
    store, _kids = E15.load_arm(world, spec, centre, 71000 + seed, symlink=True)
    bank = bank_from_store(store)

    per_template = {
        str(t): E81._evaluate_template(gk, bank, spec.alias_keys, world, t)
        for t in range(E20.N_TRAIN_TEMPLATES, E20.N_TRAIN_TEMPLATES + E17.N_HELDOUT)
    }
    candidate_rates = [per_template[str(t)]["candidate_correct"] for t in range(8, 12)]
    full_rates = [per_template[str(t)]["full_vocab_top1_correct"] for t in range(8, 12)]
    template9 = per_template["9"]["candidate_correct"]

    # Exposure on the first 16 held-out alias prompts of template 9 (the historically strict form).
    texts = [E17.TEMPLATES12[r][9].format(s=gk.names[s]) for (s, r) in list(spec.alias_keys)[:16]]
    expo = exposure(gk, bank.tensors(), texts)
    life = lifecycle_exposure(gk, bank.tensors(), texts)

    checks = {
        "strict_template9_real_symlink_gate": template9 >= 0.95,
        "heldout_paraphrase_mean_ge_095": float(np.mean(candidate_rates)) >= 0.95,
        "heldout_every_template_ge_095": float(np.min(candidate_rates)) >= 0.95,
        "heldout_full_vocab_mean_ge_095": float(np.mean(full_rates)) >= 0.95,
        "heldout_full_vocab_every_template_ge_095": float(np.min(full_rates)) >= 0.95,
    }
    participation_depth = None if cfg.write_layer is None else (len(blocks) - 1) - int(cfg.write_layer)
    by_construction = {
        # Declared before the run: only a write after the LAST block is cache-pure.  Every other
        # placement leaves a light cone of participation_depth blocks, so its exposure must be nonzero.
        # Both must be material.
        "participation_depth": participation_depth,
        "expected_kv_exposure_zero": participation_depth == 0 and not cfg.reference_carrier,
        "kv_exposure_is_zero": expo["kv_maxabs"] == 0.0 and expo["block_input_maxabs"] == 0.0,
        "memory_is_material": expo["last_logit_maxabs"] > 0.0,
        # Arm E's defining property, and the one thing that separates it from every other arm: the
        # persisted state must not move when the KNOWLEDGE changes, while the answer must.
        "expected_lifecycle_exposure_zero": cfg.reference_carrier,
        "lifecycle_exposure_is_zero": all(
            life[f"{op}_kv_maxabs"] == 0.0 for op in ("update_payload", "relink", "shred_markers")),
        "lifecycle_changes_the_answer": all(
            life[f"{op}_logit_maxabs"] > 0.0 for op in ("update_payload", "relink", "shred_markers")),
    }
    by_construction["pipeline_ok"] = (
        by_construction["kv_exposure_is_zero"] == by_construction["expected_kv_exposure_zero"]
        and by_construction["memory_is_material"]
        and (by_construction["lifecycle_exposure_is_zero"]
             == by_construction["expected_lifecycle_exposure_zero"]
             or not cfg.reference_carrier)
        and (by_construction["lifecycle_changes_the_answer"] or not cfg.reference_carrier)
    )
    return {
        "arm": arm,
        "seed": seed,
        "steps": steps,
        "consistency": consistency,
        "alt_supervision": alt_supervision,
        "read_layers": list(cfg.read_layers),
        "write_layer": cfg.write_layer,
        "n_decoder_blocks": len(blocks),
        "bos_enabled": E8.bos_enabled(),
        "groups": n_groups,
        "n_alias_eval": len(spec.alias_keys),
        "per_heldout_template": per_template,
        "template9_candidate_correct": template9,
        "heldout_candidate_mean": float(np.mean(candidate_rates)),
        "heldout_candidate_min": float(np.min(candidate_rates)),
        "heldout_full_vocab_mean": float(np.mean(full_rates)),
        "heldout_full_vocab_min": float(np.min(full_rates)),
        "exposure": expo,
        "lifecycle_exposure": life,
        "reference_carrier": cfg.reference_carrier,
        "checks": checks,
        "by_construction": by_construction,
        "strict_pass": all(checks.values()),
        "train_seconds": float(trained["train_seconds"]),
        "last_training_record": trained["history"][-1] if trained["history"] else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), default="C")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--groups", type=int, default=100)
    ap.add_argument("--consistency", type=float, default=0.15)
    ap.add_argument("--alt-supervision", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--results-dir", default="so/results")
    a = ap.parse_args()
    os.environ.setdefault("SO_BOS", "1")
    torch.set_num_threads(a.threads)
    rows = [run(a.arm, s, a.steps, a.consistency, a.alt_supervision, a.groups) for s in a.seeds]
    rec = {
        "experiment": "E-000084",
        "arm": a.arm,
        "placement": {"read_layers": list(ARMS[a.arm]["read_layers"]), "write_layer": ARMS[a.arm]["write_layer"]},
        "rows": rows,
        "all_strict_pass": all(bool(r["strict_pass"]) for r in rows),
        "all_pipeline_ok": all(bool(r["by_construction"]["pipeline_ok"]) for r in rows),
        "gate_unchanged": (
            "candidate and full-vocabulary held-out >= 0.95 on every template and every seed; "
            "same trainer, worlds and templates as E-000081 / E-000083"
        ),
        "not_claimed": (
            "late binding, decoupled read/write placement, cache purity, symlink routing or capability "
            "training as novelty; this is a capability screen that decides whether E-000083's failure "
            "was addressing depth or payload processing"
        ),
    }
    out_dir = Path(a.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = "-".join(str(s) for s in a.seeds)
    out = out_dir / f"e000084_arm{a.arm}_s{seeds}_c{a.consistency:g}_a{a.alt_supervision:g}.json"
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    if not rec["all_pipeline_ok"]:
        raise SystemExit(3)
    if not rec["all_strict_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
