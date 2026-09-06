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
| `so/audit.py` | The deletion certificate: exhaustive payload sweeps, the interface certificate that generalises over every query, autograd reachability, `check_absence` and `certify_store_absence` for a row that is gone, `check_retention` for what the store still holds, and `certify_fact`, which composes a record-level certificate with a store-level closure |
| `so/closure.py` | The store-side half of an erasure guarantee, with no model involved: how many records must go before a key stops answering (`deletion_closure`) and before no query in a declared workload yields an object (`fact_closure`, also exported as `resilience`), with a certified lower bound; `pod_keys` and `value_keys` for the two ways of individuating a fact |
| `so/ledger.py` | Result recording (JSON + Markdown) with evidence levels E0–E7 and deletion levels F0–F5 |
| `so/llm_adapter.py` | The knowledge layer attached to a frozen pretrained GPT-2 as a symlink adapter (E-000008) |
| `so/report.py` | Assembles `docs/so-results-2026-09-02.md` from the recorded results |
| `so/experiments/` | E-000001-A, E-000001-B, E-000002 … E-000034, `run_all.py` |
| `so/tests/` | Unit tests |

## Running it on an Ubuntu server

Nothing here needs a GPU. Every recorded number was produced on a four-core CPU box.

```bash
git clone <this repository> && cd FinX-BB
./setup.sh                 # apt packages, a virtualenv, CPU-only PyTorch, then the unit tests
make test                  # 45 unit tests, about 10 seconds
make smoke                 # a reduced version of the whole synthetic chain, about 15 minutes
```

`setup.sh --system` skips the virtualenv. If you keep the virtualenv but do not activate it, pass it
along: `make smoke PY=.venv/bin/python`.

| target | what it runs | measured cost on 4 cores |
|---|---|---|
| `make test` | unit tests | ~10 s |
| `make smoke` | the synthetic chain at one seed and 800 steps, trained from scratch in a `-quick` namespace | ~35 min |
| `make synthetic` | the recorded synthetic chain: E-000001-A through E-000010, plus 10k cells, symlink cells, alias chains, the fresh-seed chance test and the gate error rates | ~3 h |
| `make gpt2` | the frozen-GPT-2 chain: E-000008, E-000011, E-000012, E-000013, E-000017, E-000020 | ~20 h, downloads GPT-2 once (~550 MB) |
| `make demo` | the deletion claim as a transcript on one fact in a frozen GPT-2: the model answers it, one operation removes it, four attacks come back at chance. Needs a checkpoint from `make gpt2` | ~3 min |
| `make report` | rebuilds `docs/so-results-2026-09-02.md` from whatever is in `so/results/` | seconds |
| `make env` | prints interpreter, versions, thread count and free disk | instant |

Useful knobs: `make gpt2 SEEDS="0"` for one seed instead of three, `make synthetic THREADS=8` on a
bigger machine, and `python -m so.experiments.run_all --only e000015 e000016` to run part of a chain.

The per-model costs below are the `train_seconds` field of the recorded results, so they are what
this code actually took rather than an estimate: mini transformer 2.2 min, 10k-cell bank 19.7 min,
symlink cells 7.7 min, GPT-2 adapter 20 min, GPT-2 v2 47 min, prior conflict 66 min, symlink in
GPT-2 89 min. Three seeds each unless you change `SEEDS`.

Disk: the GPT-2 chain writes about 500 MB of cached adapters into `so/results/checkpoints/`, which
is not committed. Re-running an experiment reuses those checkpoints and only re-evaluates; pass
`--force` to retrain.

Network: only the frozen-GPT-2 experiments need it, and only once, to fetch `gpt2` from the Hugging
Face hub. Set `HF_HOME` to move the cache. The synthetic chain runs fully offline.

## Running individual experiments

```bash
pip install -r so/requirements.txt     # numpy, torch (CPU is enough), pytest, transformers (E-000008)
python -m pytest so/tests -q
python -m so.experiments.e000001a_reference
python -m so.experiments.e000001b_mini_transformer      # trains 5 models (2-3.5 min each on this 4-core box; 'train_seconds' is in the record)
python -m so.experiments.e000002_memorization_control
python -m so.experiments.e000003_retention_generalization
python -m so.experiments.e000004_reconstruction_attacks
python -m so.experiments.e000005_causal_interventions
python -m so.experiments.e000006_ablations
python -m so.experiments.e000007_biomarker
python -m so.experiments.e000009_verification_gate                                           # E-000009: verification loss
python -m so.experiments.e000009_verification_gate --gate-weight 5 --balanced \
    --name e000010_balanced_gate --experiment E-000010                                       # E-000010: class-balanced
python -m so.experiments.e000008_gpt2_adapter             # frozen GPT-2 small + adapter (needs transformers; ~20 min per seed on CPU)
python -m so.experiments.e000030_deletion_certificate --with-gpt2   # the certificate; sweeps the whole payload domain
python -m so.experiments.e000032_deletion_closure        # the store-side half, composed with it (needs the E-000015 checkpoints)
python -m so.experiments.e000033_retrieval_closure       # the same measurement in a chunked vector index
python -m so.experiments.e000034_pointer_separability --phase diagnose   # what the store gives away about a pointer
python -m so.report                                      # regenerate docs/so-results-2026-09-02.md
python -m so.experiments.run_all --quick                 # reduced smoke run of the synthetic chain (E-000001 … E-000007, E-000009, E-000010; results get a -quick suffix)
```

Every experiment writes `so/results/<name>.json` (complete record: config, per-seed numbers, aggregate, claim, what is *not* claimed, evidence level) and `so/results/<name>.md` (summary tables). Trained models are cached under `so/results/checkpoints/` and are not committed; delete a checkpoint (or pass `--force`) to retrain.

## Design in one paragraph

The neural core never stores facts. Every training step samples a fresh world, so the only stable signal is *how to read* the knowledge layer: build a query from the tokens, attend over cell keys (subject + relation), take the gated value (object), and feed it into the next hop. Knowledge therefore lives in cells that have an identity (`kid`), versions, a status and a marker, and the control plane can WRITE / UPDATE / REVOKE / RESTORE / ROLLBACK / SHRED them without touching the weights. The experiments then test whether the *learned* computation respects those lifecycle semantics (E-000001-B), whether the guarantee survives when facts *can* be copied into weights (E-000002), whether deletion generalises and retention holds (E-000003), whether the deleted object can be reconstructed (E-000004), whether the routing signature is causal (E-000005), which components are necessary (E-000006), and whether internal signals separate suppression from deletion (E-000007), and whether the same layer works as an adapter on a frozen pretrained GPT-2 with natural-language queries (E-000008).

## Smoke runs

A reduced run must never overwrite a recorded result. Set `SO_RESULT_SUFFIX` (for example
`SO_RESULT_SUFFIX=-smoke`) so `so/results/<name><suffix>.json|md` is written instead, and delete
those files afterwards; the report generator only reads the unsuffixed records.
