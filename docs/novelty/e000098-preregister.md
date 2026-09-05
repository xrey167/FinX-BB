# E-000098 — clean pre-memory identity channel

Date: 2026-09-05
Status: preregistered before execution
Classification: architecture recovery baseline. **No novelty credit** for separate encoders, retrieval heads, exact routing, pointers, curriculum, or hard boundaries.

## Trigger

E-000097A seed 2 fails the preregistered prerequisite: held-out template 9 candidate/full-vocab accuracy is 0.93 and immutable alias-row argmax accuracy is 0.775, with exact bypass 0.0. Since E97A required all three seeds to qualify, Stage B distillation/calibration is blocked regardless of remaining seeds. Do not execute E97B as evidence.

Combined with E91/E92/E94/E95, the current reader family shows that (a) dense memory attention preserves capability but causes bystander neural contamination, while (b) exactifying the same memory-coupled address path restores exact locality but destroys or severely harms capability.

## Hypothesis

The semantic address query in the current adapter is produced *inside the same residual stream that receives memory writes*. Later read queries can therefore be influenced by prior mutable memory injections. We will test a separate **clean identity channel** computed from a frozen pre-memory backbone state before any mutable payload is injected.

The purpose is not to claim a new retrieval architecture. It is to establish a capable exact mutable-read primitive without allowing mutable values to participate in semantic identity formation.

## Architecture

1. Choose a fixed pre-memory layer `L_id` earlier than the first payload injection layer.
2. Capture the frozen backbone hidden state `h_id` for the query before any mutable Bank read/write.
3. Compute semantic identity query `q_id = Q_id(LN(h_id))` in a separate address head.
4. Compare `q_id` only against immutable row-address keys derived from stable subject/relation identity features. Mutable payloads, current target object, generation, status and pointer destination are excluded from the address score.
5. Train this address head directly against exact immutable alias-row identity on training templates 0..7 only, with paired-paraphrase consistency across those training templates.
6. At execution, select the immutable row by exact argmax/one-hot.
7. Only after exact row selection, resolve its current symlink target and `(pod_id, incarnation, generation)` and read exactly one current mutable payload.
8. Inject that payload at later layer(s); the clean identity channel is not recomputed from a payload-contaminated state.

No held-out template 8..11 may be used for training or model selection.

## Three-seed capability gate

For seeds 0,1,2 on fresh independent real-symlink worlds:

- candidate correctness >=0.95 on **every** held-out template 8..11;
- full-vocabulary top-1 correctness >=0.95 on every held-out template;
- immutable alias-row argmax correctness >=0.95 on every held-out template;
- exact no-memory bypass maxabs == 0.0;
- exactly one selected immutable row and one current mutable payload dependency per executed read.

No averages may substitute for per-template/per-seed gates.

## Lifecycle locality cells

Only after all three seeds pass capability:

- unrelated canonical Pod B UPDATE;
- unrelated alias B RELINK;
- relevant target A UPDATE;
- queried alias A RELINK;
- relevant REVOKE;
- relevant SHRED;
- rollback;
- dangling/missing target.

For unrelated mutations, if A's selected immutable identity is unchanged, routing/hidden/KV/full logits/stale-KV continuation must be byte-identical under deterministic CPU execution.

For relevant UPDATE/RELINK, fresh execution must follow the current generation/target with >=0.95 correctness. For REVOKE/SHRED, old-object leakage <=0.02 and missing/closed read must become UNKNOWN according to existing semantics.

## Required controls

Compare:

A. dense E81/E88-style memory-coupled reader;
B. E92 post-hoc exactification;
C. E95 joint soft-address/exact-payload baseline;
D. E98 clean pre-memory identity channel.

The clean channel receives no novelty credit simply for being separate or exact.

## Failure conditions

E98 is falsified for this architecture if any occurs:

- any seed/template misses 0.95 candidate/full-vocab/identity accuracy;
- clean identity requires mutable payload/value/target-generation features;
- bystander mutation changes derived neural state despite unchanged selected identity;
- relevant UPDATE/RELINK fails fresh-current reading;
- exact bypass breaks;
- generated-history contamination later defeats the full lifecycle battery.

If E98 fails, stop trying to obtain exact lifecycle locality by one-row internal retrieval in this adapter family. Move the hard mutable boundary outside the residual-stream reader or change the backbone-level architecture.

## Novelty boundary

Even E98 success is **not** a breakthrough. Separate semantic encoders, retrieval heads, hard selection, pointers, versioned objects and dependency tracking are prior art. The later research-level claim remains blocked until a multi-read LLM mechanism *learns/structures mutable-knowledge causal frontiers so that future lifecycle mutations affect materially less neural state*, producing a measured mutation-to-ready advantage over strongest correct incremental/recompute baselines while passing full CAVI/J-lens/security/scaling gates.

Breakthrough = false by construction for E98.
