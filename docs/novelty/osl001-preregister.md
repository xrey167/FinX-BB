# OSL-001 — mixed-context saturation boundary for object-scoped neural lineage

Registered before numerical execution on 2026-09-05. Parent commit: `e5292a7f082ff55aea198c592e339d528ea6f493` on `research/e000051-clean-bystanders`. Status at registration: UNRUN. This is a falsification/correctness experiment, not a novelty claim.

## Question

E-000080 demonstrated useful locality with one independent cache per pod. Does its monolithic `LineagedState` still satisfy the stronger target — invalidate all and only neural-derived state depending on one knowledge generation — when a single autoregressive cache contains multiple knowledge objects and a late pod read?

The current implementation validates one `DerivedLineage` for the entire cached payload. If that payload contains dependencies on many pod generations, one stale witness rejects the whole object. This may collapse object-scoped locality to session/global-like invalidation inside mixed contexts even though causal attention guarantees that prefix K/V created before a later memory read cannot depend on that future read.

## Fixed tests

1. **Actual implementation test.** Build one `DerivedLineage` containing m current witnesses for m=1,2,4,8,16,32. Update exactly one pod. Confirm the entire lineage becomes stale even though all other witnesses remain current. This tests implementation semantics, not neural causality.
2. **Causal segmentation arithmetic.** For sequence length L=4096 and memory-read positions spread across the sequence, compare monolithic invalidation (recompute L token states) with the strongest ordinary causal-prefix/suffix baseline: reuse all positions strictly before the earliest token that can causally depend on the edited memory read and recompute only the suffix. Include an adversarial late-read case at position L-1. No novelty credit for segmentation, prefix caching, dependency graphs or self-adjusting computation.
3. **Transformer mechanics control.** On a tiny randomly initialized GPT-2 architecture in eval mode, inject old/new controlled payloads at one late token after an intermediate block. Compare full `past_key_values` layer-by-layer and position-by-position. Require exact equality for K/V positions causally before the injected token in downstream layers; require at least one changed downstream coordinate at/after the read. This establishes unnecessary collateral of whole-cache rejection under actual causal-attention mechanics, not pretrained-model capability.
4. **Mixed-pod saturation.** For m pod reads, compute the fraction of a monolithic cache invalidated by one random touched pod versus the ordinary exact suffix baseline. Report worst/mean collateral and lineage metadata growth. Do not report this as measured LLM throughput.

## Fixed interpretation rules

- A valid counterexample where monolithic lineage rejects prefix K/V that is byte-identical under fresh rebuild falsifies the claim that E-000080's current whole-cache object is already an "all and only" neural derivation closure.
- This does **not** falsify canonical pod authority, alias fan-out, generation-safe validation, or object-scoped lineage in general. It changes the required granularity: sub-cache/token/layer regions or another exact representation must be used.
- If the repair becomes ordinary causal prefix/suffix invalidation, dependency segmentation, selective KV recomputation or self-adjusting computation, it gets zero novelty credit and is a mandatory baseline.
- No >=95% real-symlink capability, UNKNOWN, leakage, J-space, generic-divergence, complete-memory, <=5% overhead or >=10x strongest-baseline system gate is established here.
- Preserve negative results and implementation corrections. No historical evidence is rewritten.
