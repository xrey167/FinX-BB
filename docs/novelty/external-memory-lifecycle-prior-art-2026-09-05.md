# External-memory lifecycle prior-art collision — 2026-09-05

Status: **decisive narrowing / direction change; not a breakthrough claim**

This note records a prior-art result that materially changes the remaining research target for CAVI/NDSR. It is not a patentability opinion.

## Executive result

The broad target

> externalize factual knowledge into mutable, versioned knowledge objects/pods; let many linguistic aliases reach the same object; update/delete that object without retraining; audit whether the deleted fact remains accessible

is no longer defensible as a research novelty target.

The collision is substantially stronger than ordinary RAG/editable-database prior art because **Limited Memory Language Models (LMLMs)** train the model specifically to externalize factual knowledge to a database and perform targeted lookups, and a 2026 forgetting audit evaluates **12,228 alias-closure deletions** under adversarial Alias/Noise/Collision topologies and multiple prompt formulations.

The remaining live research seam, if any, is not external-memory editability itself. It is the **cross-layer consistency problem created after a valid external-memory read has already entered retained neural execution state**.

## Direct collisions

### 1. LMLM: editable factual knowledge is deliberately externalized during pretraining

Zhao et al., *Pre-training Limited Memory Language Models with Internal and External Knowledge* (arXiv:2505.15962; first submitted 2025-05-21) train language models to externalize factual knowledge to an external database instead of memorizing it in weights. Factual values retrieved from the external store are masked from the training loss so that the model learns targeted lookup. The stated benefits include an explicit, editable and verifiable knowledge base.

Source: https://arxiv.org/abs/2505.15962

Consequence: **external factual memory + model-native lookup + fast database edit/delete is prior art**. A CAVI claim cannot be promoted merely because the store is canonical, mutable or nonparametric.

### 2. Co-LMLM: the external-memory query can itself be generated from hidden state

Feldman et al., *Co-LMLM: Continuous-Query Limited Memory Language Models* (arXiv:2607.07707; submitted 2026-07-08) extend LMLM with continuous vector queries generated during inference from model state, retrieving human-readable knowledge values from an external index.

Source: https://arxiv.org/abs/2607.07707

Consequence: **learned neural addressing of an editable external memory is also prior art**. Continuous/neural routing into a mutable store is not sufficient novelty.

### 3. LMLM forgetting audit: alias-closure deletion is already explicitly tested

Raeesi & Roed, *Auditing Forgetting in Limited Memory Language Models* (arXiv:2607.00605; submitted 2026-07-01) hold the model fixed while varying the external database under FULL, DEL-ON and DEL-OFF interventions. They evaluate **12,228 alias-closure deletions across thirteen databases**, including Base, Alias, Noise and Collision adversarial topologies in three domains and six prompt formulations.

They report near-zero parametric leakage in the audited variants, while the surviving post-delete residual is reconstructed through the retrieval graph and ranges from 0.7% on the released database to 13.6% on the most adversarial variant.

Source: https://arxiv.org/abs/2607.00605

Consequence: **many surface forms / aliases, deletion of a fact from an external memory, and causal post-delete auditing are directly occupied prior art territory**. The symlink/pod abstraction may still be useful engineering, but it is not itself the research novelty.

### 4. Direct neural KV knowledge delivery is independently crowded

Nearby work further rules out moving the novelty claim from “external database” to “neural KV memory”:

- Ju et al., *Knowledge Capsules: Structured Nonparametric Memory Units for LLMs* (arXiv:2604.20487) compile structured nonparametric knowledge into attention-compatible key/value representations for direct model attention.
- Pustovit, *Knowledge Packs: Zero-Token Knowledge Delivery via KV Cache Injection* (arXiv:2604.03270) precomputes and injects KV knowledge packs.
- US20260119893A1 / WO2026087278A1, *Direct knowledge injection into large language models using a key-value cache network layer*, has a claimed priority date of 2024-10-24 and explicitly describes addition/modification/deletion of knowledge data points in a KV-cache network layer.

Sources:
- https://arxiv.org/abs/2604.20487
- https://arxiv.org/abs/2604.03270
- https://patents.google.com/patent/US20260119893A1/en

Consequence: **knowledge pods/capsules compiled into neural KV, and real-time add/modify/delete in a KV-like neural layer, are not available broad novelty claims**.

### 5. Generic rollback consistency and selective cache repair are already occupied

Zhang & Yang, *Aborted but Not Forgotten: KV-Cache Retention Breaks Rollback Consistency in Language Agents* (arXiv:2608.15939; submitted 2026-08-16), show that application-level rollback can leave stale attended KV state. Their same-token/different-cache audit finds a protected-effect flip in 25/63 cells across seven open-weight model families, and transaction-local cache restoration closes every tested cell.

Source: https://arxiv.org/abs/2608.15939

ReCache (arXiv:2608.19662) independently demonstrates resource-wise attention that removes cross-resource interactions and produces independently reusable composition-invariant KV blocks.

Source: https://arxiv.org/abs/2608.19662

Consequence: **generic rollback integrity, cache restoration, version/invalidation semantics, and source-local/composable KV are not novelty by themselves**.

## What remains open

The searches above did **not** identify a work that directly tests this exact lifecycle:

1. a model performs a valid neural lookup from an authoritative, mutable external knowledge object;
2. that knowledge causally enters retained neural-derived state such as routing state, selected route, payload, hidden state, residual state, KV, activation caches or generated internal history;
3. the authoritative knowledge object is then UPDATEd, RELINKed, REVOKEd, DELETEd, SHREDded or ROLLBACKed while the inference/session state remains live;
4. later execution attempts to reuse the old neural-derived state;
5. the system must behave as if invalid knowledge can no longer authorize or resurrect itself, while unrelated cached computation remains reusable and current-generation knowledge remains available.

This is provisionally named **external-memory lifecycle consistency (EMLC)** for experiment bookkeeping only. The name is not a novelty claim.

The distinguishing question is not whether a cache can carry a version tag. The strongest ordinary systems baseline is expected to do exactly that. The research question is whether the neural execution graph creates a correctness/performance frontier that is not closed adequately by full reset, ordinary object-version invalidation, resource-isolated KV, or transaction-local cache restore.

## Promotion boundary

EMLC must be considered **falsified as novelty** if the strongest ordinary baseline provides the same correctness and comparable utility:

- authoritative object/version/incarnation identity;
- complete dependency sets for every reusable derived object;
- integrity binding between metadata and neural material;
- lazy invalidation at reuse;
- full or selective recomputation;
- transaction-local cache restore / truncate-to-clean-prefix;
- resource-isolated KV where applicable;
- clean re-prefill after a lifecycle change;
- rollback to the first affected generated-history boundary.

A positive result is research-worthy only if a neural-specific mechanism produces a material practical advantage under the **same correctness gates** rather than weakening them.

## Required next falsification: E-000083 candidate

A valid next experiment should use a **real external-memory reader**, not a controlled residual payload.

### Capability prerequisite

Before interpreting stale-state attacks:

- fresh current external-memory / real-symlink correctness >= 0.95 on every one of >=3 seeds;
- held-out paraphrase reading >= 0.95;
- REVOKE and SHRED >= 0.95;
- deleted-object leakage <= 0.02;
- missing-key UNKNOWN >= 0.90;
- generic active-memory KL <= 0.05 nats or exact BYPASS;
- preferably >1 public backbone.

The current E-000077/E-000081 line has **not yet met the >=0.95 every-seed/every-required-template prerequisite**, so positive CAVI attack interpretations remain blocked.

### Same-token / different-neural-state lifecycle audit

For a retained candidate that clears the capability gate:

1. Resolve alias A -> current pod P and produce a live inference prefix/state.
2. Capture each reusable state class independently: Bank material, routing distribution, selected route, resolved payload, hidden/residual state, KV/activation cache and generated-history checkpoint.
3. Mutate authoritative state under UPDATE, RELINK, REVOKE, SHRED and rollback/recreate/ABA cases.
4. Hold subsequent input tokens identical where possible and compare:
   - stale retained state;
   - fresh full recompute from current authority;
   - version/dependency-tag selective invalidation;
   - transaction-local clean-prefix restore;
   - resource-isolated/composition-invariant cache baseline;
   - any proposed neural-specific repair.
5. Require stale state to be causally capable of changing behavior before claiming a problem beyond metadata bookkeeping.
6. Require the proposed mechanism to match current-authority recomputation while preserving unaffected state.
7. Measure fan-out scaling, mutation cost, invalidation/recompute fraction, latency, memory overhead and rollback frontier length.
8. Run independent J-space/J-lens audit only as an audit, never as a training or authorization signal.

## Research decision

**Direction change:** stop treating “symlinked editable external knowledge” or “revocable KV memory” as the novelty target. Those broad targets are collided by current prior art.

Continue only on the narrower cross-layer question:

> Can a live LLM preserve useful neural computation across authoritative knowledge lifecycle changes while provably preventing every stale neural-derived carrier of the old knowledge from being reused, at materially lower cost than the strongest ordinary reset/version/dependency baselines?

Until that survives the capability gate, adversarial lifecycle tests, scaling, >1 backbone where feasible, and a final direct prior-art search, it is a hypothesis rather than a breakthrough.
