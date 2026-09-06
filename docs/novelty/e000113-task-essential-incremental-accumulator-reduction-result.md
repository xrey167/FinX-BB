# E-000113 Result — Task-Essential Incremental Accumulator Reduction

## Decision

**KILL_TASK_ESSENTIAL_SPARSE_INCREMENTAL_ACCUMULATOR_AS_STANDALONE_LIFECYCLE_NOVELTY**

This is a decisive scoped falsification, not a major-invention promotion and not a universal impossibility claim.

## Registered evidence

- Registered source/workflow head: `801cb75e6554e97ee14f737262beb8a82a0687a0`
- GitHub Actions run: `34028199572`
- CI conclusion: `success`
- Focused regressions: `4 passed`
- Artifact: `e000113-task-essential-incremental-accumulator-reduction`, ID `9987737374`
- Artifact digest: `sha256:d335a30dbf750723b82c1b307e938924d3e2bb0876175a158ed23edd636f9471`
- Result JSON SHA-256: `e2578bbc1154bf595c13f2ca5d18bcd23ffd31f8e9394bd3e19777207a527198`
- Preregistration SHA-256: `2fd29cea7ccc48094fb66cdde3c0448a263f966a76297ac62cc12a3501838392`
- Experiment SHA-256: `ae2171e2d841a1003b9559aa016d16d1199aecee16d919a1f4612edf3cd7d030`
- Test SHA-256: `8a66147db9f670e30e664ac1b2ab91548d6f6a87ad661b175e384d3a49034fd4`

The exact integer assay executed 16 deterministic seeds × widths `8,16,32` × two architecture arms × 64 cached sessions, each with sequential `UPDATE -> DELETE -> RESTORE -> ABA` lifecycle mutation. The retained accumulator is the actual first hidden task state consumed by two dense nonlinear suffix layers; it is not an external lineage tag.

| Metric | Result |
|---|---:|
| Registered cells | 96 |
| Exact lifecycle cases | 24,576 |
| Material final-state changes | 24,576 / 24,576 |
| Candidate -> fresh accumulator mismatches | 0 |
| Generic -> fresh accumulator mismatches | 0 |
| Candidate/generic accumulator mismatches | 0 |
| Candidate -> fresh final mismatches after suffix | 0 |
| Generic -> fresh final mismatches after suffix | 0 |
| Candidate mutation additions | 688,128 |
| Generic mutation additions | 688,128 |
| Full first-layer refresh additions | 7,684,096 |
| Nonlinear suffix multiplications still required | 22,020,096 |
| Candidate exact-ready normalized work | 22,708,224 |
| Generic exact-ready normalized work | 22,708,224 |
| Fresh exact-ready normalized work | 29,704,192 |
| First-layer refresh / incremental accumulator ratio | **11.1667x** |
| Fresh exact-ready / incremental exact-ready ratio | **1.30808x** |
| Global arm max exact deltas per lifecycle transition | **1** |
| Context-indexed arm min exact deltas per transition | **8** |
| Wrong one-global-receipt failures in contextual arm | **10,752** |

## What the candidate achieved

The candidate demonstrates that an architecture can make mutation-enabling state genuinely task-essential rather than attaching a passive lifecycle sidecar. With sparse immutable context features `C_s`, mutable Pod state `p`, and bucket `b_s`, the first hidden accumulator is

`a_s(p) = bias + sum_{i in C_s} W_i + U[b_s,p]`.

A Pod transition removes the old feature column and adds the new feature column. In the global arm, all 64 sessions share exactly one accumulator delta for a canonical lifecycle transition. Incremental first-layer maintenance is exact and reduces first-layer update additions by 11.17x versus rebuilding the first layer from all active features.

The context-indexed arm is stronger than a single global translation: the Pod feature interacts with a session bucket, giving eight distinct exact deltas for every lifecycle transition. A deliberately incorrect single global receipt failed in 10,752 contextual lifecycle cases, confirming that the assay really contains session-dependent mutation effects.

## Why it is still killed

An independently implemented generic sparse-feature accumulator was handed the identical retained first-hidden state and identical removed/added feature columns but no Symlink, Pod, J-space, knowledge, or lifecycle semantics. It reproduced every fresh accumulator exactly with the same 688,128 additions. Therefore the entire incremental first-layer advantage belongs to ordinary sparse feature accumulation.

More importantly for the FinX contract, making the first hidden accumulator exact does not make the already-materialized final neural state exact. The changed accumulator still has to traverse the two dense nonlinear suffix layers for every cached session. Candidate and generic baseline therefore both spent the same 22,020,096 nonlinear suffix multiplications and had identical exact-ready work. The structural system's large 11.17x first-layer advantage shrank to only 1.308x versus fresh end-to-end ready-state work once the nonlinear suffix required by the exact-state contract was included.

Thus an architecture whose editability claim is only `sparse mutable feature -> incrementally maintained task-essential accumulator` is not the missing FinX invention. It is useful neural engineering, but neither the update law nor the exact-ready systems advantage is FinX-specific.

## Fresh prior-art boundary checked 2026-09-06

The search makes this family particularly unsafe as a novelty anchor:

- Yu Nasu's 2018 NNUE implementation explicitly describes itself as an `Efficiently Updatable Neural-Network-based evaluation function` and is the origin implementation for computer shogi: https://github.com/ynasu87/nnue
- Current Stockfish NNUE documentation describes retaining the first-layer accumulator as position state, subtracting a weight-matrix column when a sparse feature disappears, and adding the column when a feature appears. With quantized arithmetic this incremental implementation is consistent: https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html
- Acar et al.'s self-adjusting computation work already establishes dynamic dependence tracking and memoized change propagation as generic exact incremental-computation machinery: https://doi.org/10.1145/1133255.1133993
- Nakandala, Kumar & Papakonstantinou, SIGMOD 2019, cast repeated neural inference under small input changes as incremental view maintenance and report up to 5x speedup for exact CNN inference: https://doi.org/10.1145/3299869.3319874
- NVIDIA `DE102021132980A1` / related family covers caching and reuse of data generated by neural-network layers (priority 2020-12-15): https://patents.google.com/patent/DE102021132980A1/en
- Huawei `WO2026051507A1`, published 2026-03-12, covers running incremental neural-model inference alongside full inference on the same compute unit: https://patents.google.com/patent/WO2026051507A1/en

These references do not establish the narrow FinX lifecycle guarantees and are not patentability/infringement advice. They do decisively remove broad novelty credit from `task-essential neural accumulator + sparse feature delta + cached incremental inference`.

The fresh targeted search again did not surface one canonical knowledge edit producing a bounded reusable correction that transforms heterogeneous already-materialized transformer states to exact fresh post-mutation state while beating a guarantee-matched generic engine. Current editable/composable KV work remains behavioral/approximate rather than fresh-state exact; ordinary cache systems generally reuse only content-identical/matching-prefix state; and exact incremental neural inference prior art works by exploiting ordinary locality/materialized intermediates rather than a new lifecycle algebra.

## Scope boundary / next invention pressure

E-000113 does **not** kill architectures designed for editability in general. It kills the most obvious way to answer E-000112's challenge by saying that the mutation-enabling sufficient state is simply the task's retained sparse first-layer accumulator.

The remaining candidate must cross the nonlinear boundary itself. Useful computation must produce a task-essential state for which the canonical Pod mutation updates **already-materialized post-nonlinear state** substantially more cheaply than exact suffix recomputation, while the same representation does not hand the entire advantage to generic sparse accumulation, self-adjusting computation, incremental view maintenance, associative recomposition, AD, group actions, Koopman lifting, or another already-known dynamic-computation engine.

No real-model promotion gates are changed. A future candidate still needs the full DistilGPT-2/Pythia-70M, >=3 seeds, real LINK->Pod reader, leakage/UNKNOWN, stale-state/lifecycle/race/key attacks, independent J-space/J-lens, <=5% steady-state overhead, matched memory, and material fleet-level exact mutation-to-ready advantage.