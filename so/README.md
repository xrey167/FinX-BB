# SO — experimental research code

Code behind the SO documents in `docs/`:

- [Project state, vision and architecture](../docs/so-modular-neural-os.md)
- [Experiment and evidence ledger](../docs/so-experiment-ledger.md)
- [Session results 2026-09-02](../docs/so-results-2026-09-02.md) — what this code has actually demonstrated, with evidence levels

## What is implemented

| Module | Content |
|---|---|
| `so/world.py` | Synthetic functional knowledge worlds `(subject, relation) → object` with exact ground truth, paraphrase surface forms, alternative-path structures, derivable shortcuts |
| `so/mvcc.py` | The mutable knowledge layer: versioned cells with WRITE / UPDATE / REVOKE / RESTORE / ROLLBACK / DELETE, marker SHRED / RESIGN, SWAP / REPLACE interventions, operation log, deterministic replay |
| `so/reference.py` | Mechanical reference resolver with provenance traces (E-000001-A) |
| `so/data.py` | Training banks with random lifecycle states, query encoding, re-sampled-world batches |
| `so/model.py` | Mini-Transformer neural core with a routed knowledge interface: attention over cell keys, null cell, learned marker gate, multi-hop composition, routing distribution = provenance |
| `so/train.py` | Re-sampled-world training with answer loss and routing loss |
| `so/evaluation.py` | Test families against the reference: direct, 2-hop, 3-hop, broken paths, provenance, reverse, lifecycle, locality, alternative paths, replay determinism, noise sweep |
| `so/attacks.py` | Reconstruction attacks: forced choice, logit rank, linear representation probe, routing-mass activation probe |
| `so/interventions.py` | Causal interventions: disable mask, routed-cell identification |
| `so/ledger.py` | Result recording (JSON + Markdown) with evidence levels E0–E7 and deletion levels F0–F5 |
| `so/experiments/` | E-000001-A, E-000001-B, E-000002 … E-000007, `run_all.py` |
| `so/tests/` | Unit tests |

## Running

```bash
pip install -r so/requirements.txt     # numpy, torch (CPU is enough), pytest
python -m pytest so/tests -q
python -m so.experiments.e000001a_reference
python -m so.experiments.e000001b_mini_transformer      # trains 5 models (~4 min each on 4 CPU cores)
python -m so.experiments.e000002_memorization_control
python -m so.experiments.e000003_retention_generalization
python -m so.experiments.e000004_reconstruction_attacks
python -m so.experiments.e000005_causal_interventions
python -m so.experiments.e000006_ablations
python -m so.experiments.e000007_biomarker
python -m so.experiments.run_all --quick                 # reduced smoke run of the whole chain
```

Every experiment writes `so/results/<name>.json` (complete record: config, per-seed numbers, aggregate, claim, what is *not* claimed, evidence level) and `so/results/<name>.md` (summary tables). Trained models are cached under `so/results/checkpoints/` and are not committed; delete a checkpoint (or pass `--force`) to retrain.

## Design in one paragraph

The neural core never stores facts. Every training step samples a fresh world, so the only stable signal is *how to read* the knowledge layer: build a query from the tokens, attend over cell keys (subject + relation), take the gated value (object), and feed it into the next hop. Knowledge therefore lives in cells that have an identity (`kid`), versions, a status and a marker, and the control plane can WRITE / UPDATE / REVOKE / RESTORE / ROLLBACK / SHRED them without touching the weights. The experiments then test whether the *learned* computation respects those lifecycle semantics (E-000001-B), whether the guarantee survives when facts *can* be copied into weights (E-000002), whether deletion generalises and retention holds (E-000003), whether the deleted object can be reconstructed (E-000004), whether the routing signature is causal (E-000005), which components are necessary (E-000006), and whether internal signals separate suppression from deletion (E-000007).
