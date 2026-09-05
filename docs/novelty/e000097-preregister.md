# E-000097 — teacher-distilled immutable identity compiler

Date: 2026-09-05
Status: preregistered before execution
Classification: capability/architecture recovery baseline. **No novelty credit** for distillation, curriculum learning, semantic addressing, hard routing, pointers, decoupled storage, or frozen teachers.

## Motivation from falsification

E-000091 established that the qualified dense Softmax reader has effectively Bank-wide mutable causal support under bystander mutations. E-000092 showed that post-hoc exact routing can make bystander hidden/KV/logits/continuation byte-identical but loses held-out capability. E-000094 showed that straight-through exact routing from the start collapses capability across all three seeds. E-000095B further showed on completed seeds 0 and 1 that a jointly trained `soft semantic mixture -> exact row identity -> pointer -> exact payload` architecture has exact bystander invariance and exact mutable support, but zero held-out candidate/full-vocabulary capability and no fresh-current reaction to relevant UPDATE/RELINK.

Therefore the next question is not whether exact support is desirable. It is whether semantic alias recognition must be learned under a capable dense teacher *before* introducing the exact mutable read boundary.

E-000096 remains blocked and must not be interpreted because its E-000095B prerequisite failed.

## Architecture under test

Two-stage curriculum on the same frozen public backbone and synthetic real-symlink world family:

### Stage A — capable semantic teacher

Train the existing dense E-000081/E-000088-style reader to the unchanged real-symlink capability regime. The retained teacher must itself satisfy >=0.95 candidate and full-vocabulary correctness on every held-out template before it can supervise a student.

The teacher is used only to provide immutable semantic-address information. Mutable payload mixtures from the teacher are never accepted as lifecycle-safe state.

### Stage B — immutable identity compiler

Freeze the pretrained LM/backbone and the teacher semantic query representation used for alias recognition. Train a separate address compiler whose output space is **stable row identity**, not mutable payload value and not mutable target identity.

The compiler receives the frozen semantic query representation and immutable row-address features. It is supervised by the exact semantic row identity from the training world and may additionally distill teacher address logits/representation. The final executed selection is exact/discrete.

After exact row identity is selected:

1. current alias->target pointer resolution is deterministic control-plane state;
2. current target `(pod identity, incarnation, generation)` is recorded;
3. exactly one current payload is read;
4. no nonselected mutable payload participates in the executed forward path.

Dense teacher attention is not reused during student inference.

## Critical distinction from E-000095

E-000095 tried to learn the semantic mixture and exact identity boundary jointly from scratch. E-000097 tests whether a capable semantic representation can be *compiled* into an exact immutable identity boundary after semantic competence is already learned.

The experiment is a baseline, not a novelty claim. DKME already decouples semantic addressing and storage; teacher/student distillation and hard retrieval are established methods.

## Three-seed validity gate

Use independent training seeds 0,1,2. A seed is interpretable only if the retained Stage-A teacher satisfies on templates 8..11:

- candidate correctness >=0.95 on every template;
- full-vocabulary top-1 correctness >=0.95 on every template;
- exact no-memory bypass on the preregistered bypass set.

If fewer than three teachers qualify, stop; do not interpret student lifecycle attacks.

## Student capability gate

For every qualified seed, the final exact-support student must satisfy:

- candidate correctness >=0.95 on every held-out template;
- full-vocabulary top-1 correctness >=0.95 on every held-out template;
- <=0.02 absolute per-template degradation relative to its retained dense teacher;
- exact no-memory bypass;
- exactly one positive real mutable payload dependency per executed read.

No averaging across seeds/templates can replace these gates.

## Lifecycle cells

Using fresh independent real-symlink worlds, run the same prompt before/after:

1. unrelated canonical Pod B UPDATE;
2. unrelated alias B RELINK;
3. relevant target A UPDATE;
4. queried alias A RELINK;
5. relevant REVOKE;
6. relevant SHRED;
7. rollback after UPDATE;
8. missing/dangling target.

For cells 1-2, row identity selected for A must remain the same and routing/hidden/KV/full logits/stale-KV continuation must be byte-identical where deterministic kernels permit it.

For cells 3-4, the same semantic row identity may remain selected, but pointer/generation resolution must change and the answer must follow the fresh current target/payload with >=0.95 correctness across retained cases.

For REVOKE/SHRED, old-object leakage must remain <=0.02 and the read must become UNKNOWN/closed according to existing CAVI semantics; no stale teacher or student state may supply the old payload.

## Required controls

Compare at minimum:

A. qualified dense teacher;
B. E-000095-style jointly trained exact-boundary student;
C. post-hoc hard routing from E-000092;
D. E-000097 curriculum/distilled identity compiler.

All models use matched backbone/data/evaluation. Extra teacher compute is reported separately and cannot be hidden in inference or mutation-time claims.

## Failure conditions

E-000097 is falsified if any occurs:

- exact student support remains incompatible with >=0.95 held-out capability across all three seeds;
- relevant UPDATE/RELINK does not change fresh-current answer while row identity stays stable;
- unrelated UPDATE/RELINK changes neural state despite unchanged selected immutable identity;
- student needs mutable payload information inside its semantic address compiler;
- student requires keeping teacher dense mutable attention active at inference;
- generated-history contamination defeats exact reuse under the later full battery.

If E-000097 fails, do not keep tuning exact one-row routing in this reader family. Move to a different representation of semantic identity or accept that exact selective reuse requires an external retrieval boundary rather than an in-model neural reader.

## Novelty boundary

Even success is **not** a breakthrough. Relevant exclusions include:

- DKME (ACL Findings 2026): decoupled semantic addressing and partitioned storage;
- ASMem (2026): sparse memory/prototype routing/partitioning;
- knowledge distillation, teacher-student curricula and exact retrieval;
- pointers/symlinks, aliases, canonical IDs, MVCC, generations and dependency tracking.

E-000097 exists only to establish a capable exact-support primitive for a later lifecycle-frontier experiment. A research-level novelty claim remains blocked until a later learned multi-read mutable-knowledge frontier demonstrates a material lifecycle mutation-to-ready advantage over strongest ordinary incremental/recompute baselines while passing the full CAVI/J-lens/security/utility battery.

Breakthrough = false by construction for E-000097.
