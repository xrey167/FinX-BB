# R287 Architecture Hypothesis — Temporal Capability Port Fabric (TCPF)

Status: research candidate, **not DoD**.

## Non-negotiable target

Build a live neural knowledge substrate in which one canonical Symlink/Pod identity can be updated or revoked once; every linguistic access path follows the same lifecycle; no cached, copied, latent, or previously materialized generation can restore an invalid value; the causal workspace independently verifies the lifecycle state; ordinary inference is not materially degraded; and the resulting system must ultimately beat a strong RAG stack on the joint frontier of answer quality, write/update/delete cost, TTFT, context use, and lifecycle correctness.

This target is deliberately stronger than “editable RAG”, “KV cache reuse”, or “model editing”.

## Why the architecture changes now

The previous learned linking route failed on held-out entity/value pairs. Treating a world-state binding as something the network must learn is the wrong abstraction for a system that must accept arbitrary future world changes cheaply. A new binding is data, not a new skill.

TCPF therefore factorizes four concerns:

1. **Symlink identity plane** — language aliases resolve to one stable `pod_id`. The alias set can change without changing model weights.
2. **Temporal authority plane** — each Pod has an append-only generation history and an authoritative generation selector. Revocation is an authority transition, not a request to “forget harder”.
3. **Latent payload plane** — the active generation carries a model-native payload. R284 starts with exact text-derived KV as a bootstrap mechanism; this is not the final compact representation.
4. **Reasoning plane** — the neural model learns reusable computations over payload types, relations, and operators. It must not memorize identity/value pairs.

The intended long-term transformer is a **bi-plane model**: ordinary token self-attention for linguistic/algorithmic skill plus a sparse Port Attention path for mutable world state.

## Causal lifecycle primitive

For Pod `p` with materialized generations `g0`, `g1`, define an authority variable `a_p` and a validity bit `v_p`.

The model-visible mask is

`m(p,g) = 1[g = a_p] * v_p`.

The mask is applied **before attention normalization**. An invalid generation therefore has attention logit `-inf` and exact attention weight zero. Its keys and values may remain physically allocated, but their contents cannot affect the attention output.

This is stronger than value-only stale-cache repair: setting stale values to zero still leaves stale keys in the softmax denominator. TCPF’s target invariant is stronger:

> Randomize every byte of every invalid generation while holding authority state fixed; full-vocabulary logits must remain invariant within numerical tolerance.

That mutation-invariance test is the causal anti-resurrection certificate.

## Double-buffer generation protocol

Each hot Pod initially owns two fixed generation lanes.

1. Read active lane `a`.
2. Compile/write new payload into inactive lane `1-a`.
3. Verify payload digest, model revision, and generation predecessor.
4. Atomically flip authority `a <- 1-a`.
5. Keep old lane materialized but mask it causally.
6. Asynchronous compaction may reclaim it later; correctness never depends on compaction.

Revocation sets both lane masks to zero while retaining the minimum lifecycle record needed to stop re-ingestion.

## Anti-resurrection boundary

Attention masking only protects the model-visible execution surface. The full contract also needs a tombstone barrier on every write/import/restore path:

`(pod_id, generation, payload_digest, source_lineage, model_revision)` is checked against the lifecycle ledger before a payload can become authoritative.

Caches are keyed by `(pod_id, generation, model_revision)`, never `pod_id` alone. A restored old cache therefore cannot become current merely because its bytes are valid.

## Causal verifier

The generative path and the verifier must not share the same mutable decision object.

The verifier recomputes the admissible generation from the immutable ledger entry and checks:

- canonical `pod_id`;
- monotonic generation;
- payload digest;
- model-revision binding;
- authority selector;
- tombstone/revocation state.

A mismatch fails closed before Port Attention activation.

## Compact-port research direction

R284 uses exact text-derived KV only to validate lifecycle mechanics on a real frozen model. That mechanism overlaps prior work on precomputed KV delivery and is **not** the novelty claim.

The next representation target is a compact, position-light or positionless Port code with these properties:

- query-independent write;
- no per-fact gradient optimization at serving time;
- one/few slots rather than fact-length token caches;
- sparse layer residency;
- composable across multiple Pods;
- model revision migration path;
- no degradation when no Pod is active.

Candidate implementations to kill-test:

1. layer-sparse K/V capsules;
2. low-rank per-head port factors;
3. shared decoder from compact Pod code -> layer K/V;
4. explicit positionless Port Attention sublayer trained during pretraining;
5. quantized generation capsules with error-bounded reconstruction.

## Prior-art kill matrix

The candidate must not claim novelty for components already shown elsewhere:

- **Knowledge Packs (arXiv:2604.03270):** exact zero-token knowledge delivery by precomputed KV-prefix equivalence; therefore “fact text -> KV and reuse” is prior art.
- **TurboRAG / FusionRAG / SpecCache / ProphetKV / CoinRAG:** efficient RAG through offline KV reuse, selective recomputation, and fine-grained cache assembly; therefore speed from cache reuse alone is not enough.
- **DKME and external-memory model editing:** address/storage decoupling is not new by itself.
- **TEPA (arXiv:2608.07429):** explicit lifecycle revocation of retrieved evidence is prior art at the retrieval-memory layer.
- **Forgetful Attention (arXiv:2607.12204):** certified selection and exact deletion exist for a support-vector memory; therefore “attention memory that can forget exactly” is not enough.
- **LineageKV:** stale KV value repair with lineage/history overlaps stale-cache repair; value-zeroing is not our novelty.
- **Memory^3, MEMORYLLM, Larimar, Prometheus Mind, TF-Engram, TransMem:** explicit/latent external memory and hidden-state injection are established directions.

The research claim must therefore live in the **joint mechanism**, not one ingredient: canonical mutable neural ports + generation authority integrated into attention + anti-resurrection across runtime state + mutation-invariance causal certification + future-world bindings that are data rather than learned pairs + compact model-native payloads competitive with strong retrieval systems.

## Required breakthrough gates

No breakthrough claim before all of the following pass:

1. **Real-model lifecycle gate:** real open pretrained LM; frozen or explicitly bounded adapter; update/revoke with stale generations still resident; mutation-invariance of invalid generations.
2. **Held-out binding gate:** new entities, new entity/value bindings, and new aliases after model training without neural optimization.
3. **Composition gate:** multi-Pod and multi-hop reasoning, including updates to intermediate nodes.
4. **Compactness gate:** materially smaller active payload than text-equivalent full KV; no hidden on-the-fly full re-encoding counted as “storage saving”.
5. **Speed gate:** TTFT and steady decode benchmarked against strong RAG and a correct Knowledge-Pack/KV baseline at equal quality.
6. **Quality gate:** HotpotQA/MuSiQue/2WikiMultiHopQA plus dynamic-update benchmarks; not synthetic-only.
7. **Lifecycle gate:** update, revoke, restart, replica replay, stale cache, restore, and alias-path attacks; zero invalid-generation admissions.
8. **Normal-inference gate:** no-Pod workloads stay within the agreed degradation budget.
9. **Novelty gate:** systematic literature/code/patent audit against the closest mechanisms, with claims narrowed to what remains unsupported by prior art.

R284, R285, R286, and R287 are parallel mechanism probes toward these gates; none is itself the final architecture or DoD.
