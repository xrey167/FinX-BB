# E-000063 on freshly trained checkpoints — a reproduction, kept apart from the record

Date: 2026-09-06. Three seeds, 16 pods, the BOS-trained symlink adapter, `--n-groups 16`.

These files are **not** the record. `so/results/e000063/` is. They were produced by re-running
E-000063 unchanged against checkpoints trained fresh in this session, because
`so/results/checkpoints/` is git-ignored and no checkpoint from the recorded run survives — the
condition ledger §31.48 records as *"a checkpoint SHA cannot tie a new number to an old record"*.

They are kept in a separate directory because a first attempt wrote them **over** the recorded files.
That was caught by the diff before any commit, the recorded files were restored with
`git checkout --`, and this directory exists so the two can never again be confused.

## What reproduces, and what does not

| row | recorded s0 / s1 / s2 | retrain s0 / s1 / s2 | bar |
|---|---|---|---|
| `active_alias_correct` | 0.8795 / 0.9732 / 0.9821 | 0.9688 / 0.9777 / 0.9821 | ≥ 0.80 |
| `shred_alias_unknown` | 1.0000 / 0.9955 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | ≥ 0.90 |
| `bystander_top1_agree` | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | ≥ 0.98 |
| `finalprobe(ACTIVE − NEVER)` | 0.9018 / 0.8750 / 0.8616 | 0.8125 / 0.8438 / 0.8616 | ≥ 0.30 |
| **`jprobe(ACTIVE − NEVER)`** | **−0.0134 / −0.0045 / −0.0134** | **−0.0089 / −0.0045 / −0.0134** | ≥ 0.30 — **FAIL on every seed, both times** |
| `jprobe(SHRED − NEVER)` | −0.0179 / −0.0045 / −0.0134 | −0.0089 / −0.0045 / −0.0134 | ≤ 0.05 — the headline, passes both times |
| `finalprobe(SHRED − NEVER)` | **0.0536** / — / — | **0.0000 / −0.0446 / −0.0625** | ≤ 0.05 |

**The finding reproduces, and on this retrain it is stronger.** The certificate's headline row —
*no workspace trace after deletion* — passes on all three seeds while its own pre-registered validity
row fails on all three, with the memory now read at **0.9688 / 0.9777 / 0.9821** rather than the
record's 0.8795 on seed 0. The instrument's blindness is therefore not a weak-reading artefact on any
seed.

**One recorded failure does not reproduce, and it is reported rather than folded away.**
`finalprobe(SHRED − NEVER)` failed its ≤ 0.05 bar at 0.0536 on the recorded seed 0 and
`docs/novelty/audit-siting-claim.md` reports it as *"a residual trace survives SHRED in the final
state"*. On the retrain it is 0.0000, −0.0446 and −0.0625 — passing on every seed. Since no checkpoint
ties the two runs, the honest reading is **retrain variance at the bar**, and the recorded row should
be read as one that a second training does not support.

**Seed 2 is bit-identical between the two runs** (the recorded and retrained JSONs differ only in the
checkpoint path string), so the recipe is deterministic under a fixed seed; seeds 0 and 1 moved, which
means the recorded seed-0 and seed-1 checkpoints were not produced by exactly this recipe. That is a
fact about the record's provenance, not about the finding.

## Reproduce

```bash
SO_BOS=1 SO_CKPT_SUFFIX=_bos python -m so.experiments.e000052_symlink_bos_train --seeds 0 1 2
for s in 0 1 2; do SO_BOS=1 python -m so.experiments.e000063_workspace_pod_certificate --seed $s \
    --threads 4 --n-groups 16 --checkpoint so/results/checkpoints/e000020_gpt2_bos_seed$s.pt \
    --results-dir so/results/e000063_retrain; done
```
