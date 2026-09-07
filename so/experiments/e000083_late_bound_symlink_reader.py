"""E-000083 -- real-symlink capability screen at the cache-pure final block.

E-000082 established a structural baseline on two public backbones: a residual
memory contribution injected after the final transformer block changes logits
while leaving the persistent self-attention KV byte-identical to no-memory KV.
That is useful only if a *trained real-symlink reader* can still read accurately
at that placement.

This experiment therefore changes exactly one architectural variable relative
to E-000081: the memory read/dereference is attached only to GPT-2 block 11,
after the last cache-writing transformer computation.  Training and held-out
real-symlink evaluation are otherwise reused from E-000081.

This is a SCREEN, not qualification and not a novelty claim.  A passing seed-2
screen must be repeated with one fixed training configuration across >=3 seeds
before any CAVI attack result is interpretable.  A failure is informative: it
would show that downstream transformer computation after memory access is
needed by this reader and would justify testing learned/fixed revocation-local
persistent layouts rather than assuming late binding is free.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from so.data import bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments import e000081_symlink_consistency_reader as E81
from so.llm_adapter import AdapterConfig, transformer_blocks


def run(seed: int, steps: int, consistency: float, alt_supervision: float,
        n_groups: int) -> dict[str, object]:
    torch.manual_seed(seed)
    # GPT-2 has 12 blocks; verify rather than silently assuming the topology.
    cfg = AdapterConfig(
        read_layers=(11,),
        status_gated=True,
        use_links=True,
        n_deref=E20.N_DEREF,
    )
    gk = E8.GPT2Knowledge(cfg)
    blocks = transformer_blocks(gk.model.lm)
    if len(blocks) != 12 or cfg.read_layers != (len(blocks) - 1,):
        raise RuntimeError(
            f"E-000083 requires final-only GPT-2 read: blocks={len(blocks)} read_layers={cfg.read_layers}"
        )

    trained = E81.train_symlink_consistent(
        gk,
        seed,
        steps,
        consistency=consistency,
        alt_supervision=alt_supervision,
        n_groups=max(24, n_groups),
        verbose=True,
    )
    centre = np.asarray(trained["centre"])

    rng = np.random.default_rng(70000 + seed)
    world, spec = E15.sample_alias_world(
        rng, 180, n_groups, 2, gk.n_entities, 4, E20.N_TRAIN_TEMPLATES
    )
    store, _kids = E15.load_arm(world, spec, centre, 71000 + seed, symlink=True)
    bank = bank_from_store(store)

    per_template = {
        str(t): E81._evaluate_template(gk, bank, spec.alias_keys, world, t)
        for t in range(E20.N_TRAIN_TEMPLATES, E20.N_TRAIN_TEMPLATES + E17.N_HELDOUT)
    }
    candidate_rates = [per_template[str(t)]["candidate_correct"] for t in range(8, 12)]
    full_rates = [per_template[str(t)]["full_vocab_top1_correct"] for t in range(8, 12)]
    template9 = per_template["9"]["candidate_correct"]
    checks = {
        "final_block_only": cfg.read_layers == (11,),
        "strict_template9_real_symlink_gate": template9 >= 0.95,
        "heldout_paraphrase_mean_ge_095": float(np.mean(candidate_rates)) >= 0.95,
        "heldout_every_template_ge_095": float(np.min(candidate_rates)) >= 0.95,
        "heldout_full_vocab_mean_ge_095": float(np.mean(full_rates)) >= 0.95,
        "heldout_full_vocab_every_template_ge_095": float(np.min(full_rates)) >= 0.95,
    }
    return {
        "seed": seed,
        "steps": steps,
        "consistency": consistency,
        "alt_supervision": alt_supervision,
        "read_layers": list(cfg.read_layers),
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
        "checks": checks,
        "strict_pass": all(checks.values()),
        "train_seconds": float(trained["train_seconds"]),
        "last_training_record": trained["history"][-1] if trained["history"] else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--groups", type=int, default=100)
    ap.add_argument("--consistency", type=float, default=0.15)
    ap.add_argument("--alt-supervision", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--results-dir", default="so/results")
    a = ap.parse_args()
    os.environ.setdefault("SO_BOS", "1")
    torch.set_num_threads(a.threads)
    rows = [run(s, a.steps, a.consistency, a.alt_supervision, a.groups) for s in a.seeds]
    rec = {
        "experiment": "E-000083",
        "candidate_only": True,
        "architectural_screen": "final-block-only real-symlink reader",
        "rows": rows,
        "all_strict_pass": all(bool(r["strict_pass"]) for r in rows),
        "gate_unchanged": (
            "candidate and full-vocabulary held-out >=0.95; seed2 screen only; "
            "requires fixed-config >=3-seed rerun before CAVI interpretation"
        ),
        "not_claimed": (
            "late binding, final-layer adapters, cache purity, symlink routing, paraphrase consistency, "
            "or capability training as novelty; E-000082 controlled cache mechanics remain a separate result"
        ),
    }
    out_dir = Path(a.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"e000083_late_bound_symlink_c{a.consistency:g}_a{a.alt_supervision:g}.json"
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    if not rec["all_strict_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
