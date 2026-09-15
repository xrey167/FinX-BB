# E-000082 — late-bound cache purity as a strong non-novel baseline

Date: 2026-09-05

Status: **structural result passed; not a novelty claim; not capability-qualified.**

## Why this experiment exists

E-000080 showed that object-scoped lineage can selectively invalidate stale KV
while preserving unrelated cache state. That is useful engineering, but it is a
composition of ordinary dependency/version tracking, cache invalidation and
recomputation. E-000082 therefore asks a stronger falsification question:

> Can the architecture avoid putting mutable external-memory information into
> persistent self-attention KV in the first place?

If yes, a large part of the downstream lineage problem can be removed by a
strong fixed architectural baseline rather than by increasingly elaborate
revocation wrappers.

## Pre-registered mechanism

On frozen public causal LMs, inject a controlled residual payload either:

1. after the penultimate transformer block, where the following block can write
   memory-dependent K/V; or
2. after the final transformer block, after the last cache-writing attention
   computation.

After an old -> new memory lifecycle change, compare old-cache reuse against a
fresh current recomputation under a fixed teacher-forced continuation. Also
compare the token-history path when old and current memory select different
committed tokens.

Models: `distilgpt2` and `EleutherAI/pythia-70m-deduped`.
Seeds: 0, 1, 2.
Prompt lengths: 16, 64, 256 tokens.
Payload RMS: 8.0.
Workflow run: GitHub Actions `33966368607`.

## Result

Both backbone jobs passed all structural checks for all 3 seeds and all 3 prompt
lengths.

### Final-block late binding

For every tested cell on both backbones:

- the memory payload materially changed output logits;
- `late_old_cache_vs_base_maxabs == 0.0`;
- `late_new_cache_vs_base_maxabs == 0.0`;
- reusing the old cache with the new/current payload matched a fresh-current
  cache exactly under the fixed continuation (`maxabs == 0.0`).

Thus, for this controlled intervention, the final-block memory contribution is
absent from persistent self-attention KV even though it changes the neural
output distribution.

### Penultimate-block control

For every tested cell:

- injecting one block earlier changed stored KV;
- old and new memory produced different KV;
- reusing the old contaminated cache after the lifecycle change differed from
  current recomputation.

This confirms that the late-bound result is caused by the placement relative to
the cache-writing attention operation, not by the payload being ineffective.

### Repair-cost screen

At 256 prompt tokens, the measured ratio of contaminated full repair
(prefill + one decode) to late-bound cache reuse (one decode) was approximately:

- `distilgpt2`: 10.84x–10.97x across seeds;
- `pythia-70m-deduped`: 11.26x–12.56x across seeds.

These are single hosted-CPU measurements, useful only as an engineering screen;
they are not a production performance claim.

### Discrete-token boundary

In every tested cell, old and current memory selected different top-1 tokens.
After the lifecycle update, continuing from the already-selected old token still
differed materially from the current counterfactual path. Therefore cache purity
does **not** solve committed memory-dependent token history.

The resulting execution rule is sharper:

> Mutable memory can be kept out of reusable internal KV by sufficiently late
> binding, but once memory has influenced a committed token or external effect,
> rollback/restart from a clean boundary is required if exact counterfactual
> revocation is demanded.

Rollback/effect-boundary semantics are treated as prior art / ordinary systems
semantics, not novelty.

## What E-000082 does not establish

- It uses controlled residual payloads, not a trained real-symlink reader.
- It does not meet the >=0.95 fresh-reader prerequisite.
- It does not establish held-out paraphrase capability, REVOKE/SHRED, UNKNOWN,
  exact active-bank BYPASS or <=0.05 nats locality.
- It does not prove deletion safety for serialized Bank/router/payload/hidden
  state.
- It does not establish novelty.
- The timing measurements are not statistically benchmarked.

## Prior-art tightening after the result

Fresh 2025–2026 search materially strengthens the ordinary baseline:

- **ProphetKV (2026)**: query-driven selective KV recomputation for RAG cache
  reuse; selective recomputation is not new.
- **KVEraser (2026)**: learned local KV steering/repair after context erasure;
  learned cache repair is not new.
- **KV Packet (2026)**: trainable soft-token adapters for recomputation-free,
  context-independent cached-document reuse; trainable cache-purity/reuse ideas
  are nearby prior art.
- **LFD (2025/2026)**: layer-specific external-knowledge exploitation and
  representation fusion; choosing/fusing a layer for external knowledge is not
  itself novel.
- **LOKA (ACL 2026)**: adaptive external knowledge memory split into memory
  units with a learned router; adaptive modular knowledge memory is not new.
- **LoKiFormer (2026)**: explicit parametric key-value Knowledge Memory Module
  decoupling sequence-external knowledge storage from computation; decoupling
  knowledge storage from transformer computation is not new.
- **MemoryLACE (2026-09-02)**: explicit lifecycle relations for textual evidence;
  lifecycle-aware memory organization is not new.
- Existing PAMSPEC/MemTX/generational-handle/cache-invalidation results remain
  prior-art constraints from the earlier CAVI audit.

### Direct patent collision: learned side-channel knowledge intervention

A targeted patent search found **US20260105279A1**, priority 2024-10-15,
published 2026-04-16, assigned to ETRI. It is pending; this note is not a legal
or patentability opinion.

The disclosure is materially close to any broad claim of learning *where* or
*whether* updated external knowledge should intervene inside a frozen residual
foundation model:

- a knowledge-control unit is separated from the neural network;
- an updated-knowledge adapter coexists with frozen primitive-model weights;
- a side-channel controls combination/non-combination and reflection strength;
- the combination can return the primitive model output unchanged when false;
- adapters/combination units can exist across residual blocks or a selected
  subset;
- residual-block locations are represented as an array of reflection degrees;
- the representation-combination strength can itself be learned;
- token-level intervention can change during autoregressive generation.

Relevant claims include claim 1 (separate knowledge-control unit + side-channel
combination with an updated-knowledge adapter), claims 6–8 (learned combination
strength and exact primitive-model output on non-combination), and claim 10
(iterative layer-wise selective combination during inference).

Source: https://patents.google.com/patent/US20260105279A1/en

The patent text does **not** appear to mention cache/KV state, deletion,
revocation, lineage or lifecycle-repair cost. Therefore it does not by itself
falsify the still-narrow candidate below, but it kills a broader framing such as
“learned layer placement/side-channel intervention for mutable knowledge.”

Primary references searched:

- https://arxiv.org/abs/2602.02579
- https://arxiv.org/abs/2606.17034
- https://arxiv.org/abs/2604.13226
- https://arxiv.org/abs/2508.19614
- https://aclanthology.org/2026.acl-long.760/
- https://arxiv.org/abs/2608.12419
- https://arxiv.org/abs/2609.03201
- https://patents.google.com/patent/US20260105279A1/en

## Consequence for the novelty target

**Fixed late binding is now a mandatory strong baseline.** A proposed learned
`revocation-local persistent neural execution` mechanism gets no credit merely
for reducing cache repair if final-only/fixed compartmentalized memory can
achieve the same capability and lifecycle guarantees.

The ETRI patent adds a second mandatory exclusion: **learning layer-wise
knowledge intervention or a side-channel combination policy is not itself the
novelty target.** Any surviving contribution must be specifically tied to the
lifecycle of persistent neural state and must beat equivalent fixed and learned
intervention baselines.

The live question is now empirical:

1. Can a real trained symlink reader retain >=0.95 held-out/full-vocabulary
   capability when moved to the final cache-pure block?
2. If not, what minimum earlier/deeper computation is required for capability?
3. Can an execution policy trained explicitly against *measured future
   lifecycle repair cost* produce less contaminated persistent state at matched
   capability/safety than (a) fixed final/compartment placement and (b) a
   learned side-channel/layer intervention policy with no lifecycle objective?

Only question 3 remains a plausible research-novelty candidate, and it must also
survive a further targeted prior-art/patent search for lifecycle-cost-aware
neural state placement.

E-000083 directly tests question 1 on the historically failing seed before any
3-seed qualification run.
