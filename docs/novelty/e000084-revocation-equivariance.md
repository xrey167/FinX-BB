# E-000084 — revocation-equivariant neural state

Date: 2026-09-05

Status: **hypothesis / falsification screen only. Not a breakthrough.**

## Why this lane exists

The broad mutable-neural-memory direction is no longer a defensible novelty target:

- IBM `US20260119893A1` directly discloses a trainable KV-cache network layer in an LLM with real-time insertion, modification, and deletion of knowledge data points.
- ReCache (`arXiv:2608.19662`) makes composition-invariant, resource-local KV with selective resource visibility a strong fixed-isolation baseline.
- `Models Take Notes at Prefill` (`arXiv:2606.17107`) causally demonstrates that changing a source field while reusing downstream KV can preserve the old conclusion because downstream cache positions memoize field-conditioned notes; it also develops efficient edit/composition mechanisms.
- MUNKEY (`arXiv:2603.15033`), Memory Adapters (MemFM @ ICML 2026), Forgetful Attention (`arXiv:2607.12204`), Knowledge Externalization (ICLR 2026), and SHINE (`arXiv:2602.06358`) further eliminate broad claims around unlearning-by-design, source-local memories, exact deletion in specialized memories, externalized editable memory tokens, and context-to-adapter generation.

Therefore the project should not claim novelty for a mutable pod, a removable adapter, source isolation, a reversible layer, or selective cache repair by itself.

## Narrow hypothesis

A possible neural-specific mechanism is **revocation-equivariant computation**.

Let a mutable pod induce an invertible lifecycle action `T_p` on a dedicated neural state. If downstream persistent computation `F` is constrained so that

`F(T_p h) = T_p F(h)`,

then after the pod is revoked the old deep state can be repaired by applying `T_p^-1` to the already-computed state rather than replaying the whole suffix. For multiple independently mutable pods, their actions must either commute or be isolated in a representation where exact removal order is defined.

The proposed systems benefit is not "security by tagging". It is a potential reduction in neural recomputation cost: **deep memory-conditioned computation whose lifecycle mutation can be undone exactly at the state level**.

## What would make it interesting

The candidate only survives if a real LLM implementation demonstrates all of the following at matched capability:

1. Deep memory-conditioned computation materially improves over final-only late binding on a task that actually requires downstream neural processing.
2. Removing/updating one pod from already-computed neural state matches a clean current-state recomputation within a pre-registered numerical/logit tolerance.
3. Unaffected state remains reusable and the lifecycle repair cost is materially lower than full recomputation.
4. The result beats ReCache-style fixed source isolation at a task where useful cross-pod composition matters.
5. The result is not reproduced by an ordinary removable adapter, a generic reversible network, ordinary cache invalidation, or replay.
6. Real-symlink fresh and held-out correctness is >=0.95 across >=3 genuine training seeds before any stale-state attack is interpreted.
7. REVOKE and SHRED are each >=0.95, deleted-object leakage <=0.02, missing-key UNKNOWN >=0.90, generic KL <=0.05 or exact bypass, and the full replay/race/rollback battery passes.
8. Independent J-space/J-lens remains audit-only and passes NEVER-memory controls.
9. Generated-token/history contamination is accounted for. Exact repair of hidden/KV state does not erase information that was already emitted and then re-ingested.
10. A final 2025-2026 paper/standards/patent search finds no direct lifecycle-unlearning use of an equivariant group action with exact post-computation inverse repair in LLM neural state.

## Immediate structural falsifier

`so/experiments/e000084_revocation_equivariance_screen.py` tests the mathematical mechanism before spending on LLM training.

- Knowledge pods are represented as commuting per-channel phase rotations.
- A nonlinear radial neural lane is exactly equivariant to those rotations.
- Single deletion, multi-deletion, and rollback are compared against fresh counterfactual recomputation.
- A generic nonlinear network receives the same lifecycle actions as a negative control; inverse action at its output should not repair it.

Passing this screen only establishes that the mechanism is numerically coherent. It does **not** establish language capability, novelty, usefulness, or any CAVI security gate.

## Promotion boundary

A future positive claim would have to be phrased around a measured capability/cost frontier, for example:

> A trained LLM reader preserves useful deep cross-pod computation while an authorized pod lifecycle change can be applied to already-computed neural state by an exact bounded repair operation, avoiding suffix recomputation and preserving unrelated state, under the full symlink/revocation/locality/audit contract.

Even that wording is provisional until direct prior art and real-model experiments are complete.
