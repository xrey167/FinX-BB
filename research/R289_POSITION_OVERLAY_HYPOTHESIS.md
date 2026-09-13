# R289 — Position-Overlay Generation Ports (POGP)

Status: architecture hypothesis + real-model gate in progress. **Not DoD.**

## Problem

A double-buffered neural knowledge port normally has an awkward interaction with positional encoding: if revision `g+1` is appended after revision `g`, the model sees the new fact at a later RoPE coordinate. Keeping stale generations resident therefore changes relative positions and can perturb answers even when stale slots are masked.

That is a storage-layout artifact, not a semantic requirement. A Pod revision is a replacement of the same logical memory location, not a later linguistic token.

## Core proposal

Separate **physical generation lanes** from **logical model position**.

For Pod `p`, every materialized revision is compiled at the same logical position interval `P_p = [s, s+w)`. The runtime may physically keep two or more revision lanes:

`KV_phys(p) = [KV(p,g_old@P_p), KV(p,g_new@P_p), ...]`

but authority masking exposes exactly one lane to attention:

`M(p,g) = 0 if g = authoritative_generation(p), else -inf`.

The query continues at `s+w`, not at `s + (#physical_generations)*w`.

Thus revision count does not consume logical context length.

## Why the logits should be exactly invariant

For a fixed query and one authoritative generation, the unmasked set of `(K,V)` pairs is identical to the single-generation execution. The active generation has identical RoPE coordinates, and the query has the identical logical position. Masked stale entries contribute `exp(-inf)=0` before softmax normalization.

Therefore, modulo numerical/kernel implementation details:

`logits(single active generation) = logits(overlaid physical generations + authority mask)`.

A stronger causal certificate follows:

> Arbitrarily mutate every K/V byte of every non-authoritative generation while holding authority state fixed. Full-vocabulary logits must not change beyond numerical tolerance.

For revocation, all generation lanes are masked. The same mutation-invariance test applies, and the result should equal a logical empty-port execution at the reserved port position.

## What this buys

1. **No positional drift across edits.** A fact can be rewritten repeatedly without changing its logical memory coordinate.
2. **Atomic double-buffer updates.** Compile into inactive physical lane, verify, then flip one authority bit.
3. **Anti-resurrection inside attention.** Invalid generations may remain allocated but have zero causal influence.
4. **Stable downstream caches.** The logical length of the Port plane is revision-independent; downstream positional coordinates do not move when a Pod is updated.
5. **Revocation without reprefill.** Mask both lanes; compaction becomes a storage optimization rather than a correctness step.

## Relation to prior work

Position-independent KV reuse, RoPE relocation, and cached-prefix injection are active areas; they are not novelty by themselves. Likewise, exact/verified deletion exists in specialized memory mechanisms. POGP is only interesting as part of the larger TCPF joint mechanism: canonical Symlink identity, temporal authority, model-native compact Ports, pre-softmax generation validity, anti-resurrection barriers across caches/restore/replicas, independent lifecycle verification, and reusable neural reasoning over mutable world state.

## Required tests

R289 tests the mechanism on a frozen, revision/hash-pinned Qwen2.5-0.5B:

- new generation physically co-resident with old generation;
- both generations RoPE-compiled at the same logical coordinate;
- forward and reverse physical lane order;
- full-vocabulary equivalence to the single active generation;
- random stale K/V mutation invariance;
- revoked stale-mutation invariance;
- equivalence of revoked execution to a logical empty Port;
- direct timing of single-generation versus dual-physical-lane query execution.

The same gate must subsequently pass on a larger real model and inside an explicit Port-Attention implementation rather than only reused standard attention.

## Next representation step

POGP solves lifecycle/position semantics, not payload size. The complementary R288 Shared Latent Port Codec tests whether exact per-fact capsules lie in a shared low-dimensional subspace. The intended end state is:

`canonical Pod -> compact code c_p^g -> shared layer/head decoder -> position-overlay Port K/V -> authority mask -> sparse Port Attention`.

A future write performs one bounded compile/projection and no per-fact optimizer loop. A future edit writes a new code into the inactive generation lane and atomically flips authority. The old code can remain physically present while being causally dead.
