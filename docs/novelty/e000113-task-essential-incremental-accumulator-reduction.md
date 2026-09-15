# E-000113 — Task-Essential Incremental Accumulator Reduction

Status: **PREREGISTERED STRUCTURAL KILL SCREEN**

## Question

After E-000112, the surviving pressure is no longer another update algebra but a neural architecture whose ordinary useful computation itself produces and maintains mutation-enabling sufficient state. The strongest obvious construction is an efficiently updatable sparse first-layer accumulator: the state is not an external lineage sidecar; it is the actual task-essential hidden preactivation consumed by the nonlinear suffix.

Can a Symlink–Pod architecture obtain standalone major-invention credit by representing the mutable Pod as one or a few sparse neural features, retaining the first hidden accumulator for every session, and applying exact feature-column add/subtract updates when the canonical Pod changes?

## Candidate

For session `s`, let immutable sparse context features be `C_s`, context bucket be `b_s`, and mutable Pod state be `p`. The first task-essential accumulator is

`a_s(p) = bias + sum_{i in C_s} W_i + U[b_s, p]`.

Two arms are registered:

1. **global Pod feature** — `U[p]` is context independent, so one canonical Pod mutation has one shared exact accumulator delta;
2. **context-indexed Pod feature** — `U[b_s,p]` is a sparse conjunction feature, so the exact delta depends on the session bucket while remaining cheaply incrementally updatable.

The actual task state then passes through two dense piecewise-linear nonlinear layers. The accumulator is therefore causally required for normal inference and is not merely passive metadata.

Lifecycle sequence per cached session:

`UPDATE 0->1 -> DELETE 1->None -> RESTORE None->1 -> ABA 1->0`.

All arithmetic is exact integer arithmetic.

## Candidate update

Given the retained accumulator, remove the old Pod feature column if present and add the new Pod feature column if present. Then recompute the nonlinear suffix to make the cached final state exactly equal to fresh inference.

## Strong generic baseline

An independently implemented generic sparse-feature accumulator receives the identical feature columns and cached accumulator but no Symlink, Pod, J-space, knowledge, or lifecycle semantics. It performs ordinary removed-feature subtraction and added-feature addition, followed by the identical nonlinear suffix required for an exact ready final state.

The candidate receives zero invention credit if the generic accumulator has identical exact state, memory representation, and mutation work.

## Registered controls

- 16 deterministic seeds.
- hidden widths `8, 16, 32`.
- 64 cached sessions per seed/width/arm.
- 8 context buckets.
- 16 immutable active sparse features per session drawn from 128 static features.
- two nonlinear dense suffix layers.
- both global and context-indexed Pod feature arms.
- exact fresh accumulator and exact fresh final-state recomputation after every lifecycle transition.
- a deliberately wrong single global receipt applied to the context-indexed arm, to verify that context-conditioned sparse features really require session/bucket information.

## Preregistered kill rule

Return

`KILL_TASK_ESSENTIAL_SPARSE_INCREMENTAL_ACCUMULATOR_AS_STANDALONE_LIFECYCLE_NOVELTY`

if all of the following hold:

1. every lifecycle transition materially changes the final task state;
2. candidate accumulator equals fresh accumulator in every case;
3. generic sparse-feature accumulator equals fresh accumulator in every case;
4. after the necessary nonlinear suffix evaluation, candidate and generic final states equal fresh final state in every case;
5. candidate and generic incremental mutation work are identical;
6. the global arm genuinely has one reusable accumulator delta per lifecycle transition;
7. the context-indexed arm genuinely needs multiple distinct exact deltas and the wrong single-global-receipt control fails materially;
8. the candidate's first-layer refresh saving does not translate into a FinX-specific advantage because the generic accumulator inherits it exactly, while exact ready-state construction still performs the nonlinear suffix per session.

This is a scoped family kill, not a universal impossibility theorem. It does not kill a future architecture whose task-essential mutable representation supports exact post-nonlinear fleet transport unavailable to a generic algorithm given the same representation.

## Prior-art boundary to verify before closure

The closure decision must explicitly compare the candidate with:

- Yu Nasu's 2018 NNUE / efficiently updatable neural-network accumulator pattern;
- current Stockfish NNUE accumulator documentation, where removed sparse features subtract first-layer columns and added features add columns;
- generic self-adjusting/incremental computation;
- exact incremental neural inference / cached-layer reuse literature and patents;
- the project's existing exact suffix-recompute/KV-reconstruction baseline.

No real-model promotion is permitted from this structural screen alone.