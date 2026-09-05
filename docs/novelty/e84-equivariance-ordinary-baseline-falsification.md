# EQ-AUDIT-001: the E84 equivariance screen has an exact ordinary equivalent

Date: 2026-09-05. **Decisive falsification of the standalone novelty/utility interpretation for the pinned numerical screen; not a breakthrough or a universal impossibility result.**

## Scope and provenance

This concerns `so/experiments/e000084_revocation_equivariance_screen.py` at `7689a2aabb6f551d0f3b757e10bd5bb02d93a3ce`, branch `research/e000084-revocation-equivariance`, run `33969542375`. Source fidelity is enforced by Git blob `8b82f36128b33eab19cbfbb8f5b5021e3651833e`. Original artifact `9970496593` was downloaded and matched SHA-256 `6986d3a93a3d947f9b0642a8b11bcff74f3904a9cd9e7936ed9d0986fd11e059`.

The original source/results remain unchanged. All five original numerical-seed controls reproduced. Their algebra is valid; their interpretation as a distinct deep-computation repair advantage does not survive the missing baseline.

A DIFFERENT E-000084 exists in PR #4: `e000084_deep_read_late_write.py`, branch `claude/vision-technical-novelty-cfj0ei`, run `33970654975`. Do not conflate them. Its five capability jobs were in progress when inspected. This audit does not assume their result or modify that branch.

## Exact ordinary baseline

Let x be a fixed prompt-derived state, T_g the channel-wise phase action for the aggregate pod phase, and F the E84 radial depth stack. Its required identity is:

    F(T_g x) = T_g F(x).

Therefore an ordinary implementation can cache H=F(x), independently of pod phases, and materialize T_g H. Deleting p gives:

    T_(g-p) H = F(T_(g-p) x) = T_-p F(T_g x).

Both implementations perform one phase action per update and zero deep-layer replay. They have identical mathematical outputs, not merely comparable accuracy. An identical downstream decoder receives the same state. An invertible basis change does not avoid this reduction.

Conversely, exact same-inverse repair on every singleton, T_-p F(T_p x)=F(x), already implies equivariance on those states. Renaming the property does not escape the baseline.

This is elementary algebra, NOT a new theorem. It is scoped to fixed x, available cheap phase actions and the specified computation. It is not an impossibility claim for all lifecycle-aware models, state-dependent repair, or changing prompts.

## Why depth was misleading

The nonlinear radial gates depend on squared magnitudes. Pod values change only phases, so those gates and all amplitudes are independent of the pod-phase values. The phase effect at the output is material (largest observed change 6.07458), and pairwise finite differences can be nonzero (up to 0.265346). But the one late action reproduces those interactions: neither observation establishes knowledge-dependent processing inside the expensive depth stack.

Adding phase-sensitive content to the gates breaks the simple inverse rule in three checked synthetic seeds. A nonlinear suffix/decoder that does useful knowledge processing must be included in the repair contract and timing, rather than excluded as a readout.

## Executed local evidence

Five synthetic parameter/intervention seeds at depths 1,8,24,96; 64 channels, 32 pods. These are NOT trained-reader seeds. There were 1,280 subset comparisons and 2,560 sequential numerical replacements/zeroings/restorations. All 39 unit tests passed. Original tolerance remains 1e-10.

| Quantity | Maximum absolute error |
|---|---:|
| Eager output vs exact ordinary late bind | 4.57218e-15 |
| Subset eager vs late bind | 7.02167e-15 |
| Late sequential updates vs fresh sum/recomputation | 7.70786e-15 |
| Inverse sequential updates vs fresh sum/recomputation | 9.49354e-15 |
| Radial gate change under phase intervention | 2.22045e-16 |

The tracing implementation matched the upstream output exactly. Distinct operation orders are only numerically equal, not claimed bitwise identical.

Paired CPU timing used identical 96-event streams, seven order-rotated rounds, and charged initial setup. At depth 24, local medians across five seeds were: full replay/inverse 24.69x; full replay/ordinary late bind 23.91x; ordinary late bind/inverse 1.0233x. A roughly 2% microbenchmark difference is not a material neural systems advance. The large replay-relative gain belongs to the ordinary equivalent too. Raw timing records are emitted by the runner; no statistical significance or public-model speedup is claimed.

Each representation's persistent vector plus aggregate is 1,536 bytes at 64 channels, excluding shared model/registry, transient output and scratch. This is not measured process RSS. Authorization, alias fan-out, generated-history rollback and production I/O are outside this numerical timing.

## Capability remains separate

The three E83 artifacts (run 33966506365; artifact IDs 9970561259, 9970463653, 9970451768) were separately downloaded/hash-verified. They are three training settings on seed 2, not three training seeds. Held-out candidate means are 0.66875, 0.68625, 0.675; full-vocabulary means are 0.59875, 0.635, 0.57875. None passes 0.95.

Moving that real reader to the final block failed capability. This neither invalidates the exact same-model transformation above nor makes our numerical baseline a capable LLM. The deep-address/late-write lane remains independently evaluated.

## Current primary-source exclusions

- LieLAC, ICLR 2025, arXiv:2410.02698v2: existing canonicalization/equivariance mechanisms, including pretrained models. https://arxiv.org/abs/2410.02698
- Equivariance by Local Canonicalization, 2025: efficient representations of equivariant computation. https://arxiv.org/abs/2509.26499
- ReCache, agent-resource version, 2026: independent composition-invariant resource KV caching. https://arxiv.org/abs/2608.19662
- Models Take Notes at Prefill, 2026: downstream stale KV influence and cache editing/composition. https://arxiv.org/abs/2606.17107
- IBM US20260119893A1, published 2026-04-30: in-model KV knowledge insertion/modification/deletion with specific GMM/twin-distribution machinery. https://patents.google.com/patent/US20260119893A1/en

These are broad-claim exclusions, not an exhaustive patentability/infringement opinion or claims that the references implement the entire CAVI target. The collapse proof comes directly from the inspected code and identity.

## Decision and gates

Demote pinned E84-equivariance to an ordinary-equivalent baseline. Preserve the original positive identity and negative nonlinear control. Do not allocate additional training budget to it as a standalone novelty mechanism without a specific advantage surviving the exact baseline.

For subsequent candidates, compare the SAME model against its strongest algebraic/graph simplification and require useful task dependence in expensive persistent computation. Dense replay alone is an insufficient comparison. This evaluation correction is not a renamed invention.

No real-symlink, CAVI attack, SHRED, UNKNOWN, generic KL or independent J-lens claim is supported by these tests. Before CAVI interpretation, require >=0.95 fresh real-symlink reading across >=3 genuine training seeds. Retain >=0.95 held-out reading and separate REVOKE/SHRED; <=0.02 deleted-object leakage; >=0.90 missing-key UNKNOWN; <=0.05 nats generic KL or exact out-of-scope bypass; every stale Bank/router/route/payload/hidden/KV race/replay attack; and independent J-space/J-lens audit. Public-backbone replication, fan-out/mutation cost, inference overhead, footprint and rollback latency remain required.

E78 is not promoted beyond ordinary snapshot isolation. No original experiment or main-branch file is modified. No new scientific success is asserted.

Reproduce:

    python -m pytest -q research/equivariance_audit/tests
    python research/equivariance_audit/src/equivariance_audit.py
