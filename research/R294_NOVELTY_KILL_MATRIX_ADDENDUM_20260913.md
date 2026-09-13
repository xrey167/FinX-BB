# R294 Novelty Kill Matrix — Addendum after dedicated-memory search

This addendum further narrows the claim after searching explicitly for frozen-backbone memory sidecars and separate attention channels.

## TokenMem is very close to the read-path shape

**TokenMem: Faithful Knowledge Injection for Frozen LLMs — arXiv:2607.22625** reports a dedicated cross-attention knowledge channel attached to frozen LLMs, with a small trained gating adapter. It is explicitly motivated as avoiding conflicts that arise when retrieved knowledge and parametric knowledge share the ordinary self-attention/residual pathway.

Therefore the following are **not** So novelty:

- a frozen LLM plus a separate knowledge-attention channel;
- a small trained reader/gating adapter;
- bypassing ordinary text-context competition;
- claiming better knowledge compliance than vanilla RAG from a dedicated reader.

R295 is consequently classified only as an engineering/mechanism gate for causal separation, never as a novelty result.

## Persistent frozen-backbone memory adapters are occupied

**Trained Persistent Memory for Frozen Decoder-Only LLMs — arXiv:2603.22329** studies multiple persistent-memory read interfaces for a frozen decoder-only model, including parallel cross-attention, KV extension, Hebbian memory, context-gated branches, and slot-based sparse writes.

Therefore generic "persistent latent memory attached to a frozen decoder" and generic parallel-memory branches are occupied.

## Hidden-state memory injection is occupied

**TransMem — arXiv:2607.29032** transforms sparse historical hidden states from a frozen backbone into reusable memory representations with a lightweight gating network, with evaluations including LoCoMo and HotpotQA.

Therefore reusable historical hidden-state memory and lightweight latent intervention are occupied.

## Compact latent personal memory is occupied

**Latent Personal Memory — arXiv:2606.20911** uses persistent compact latent slots and a shared cross-attention projection network to create dynamic soft prompts for a frozen LLM, with large KV-use reductions.

Therefore compact latent slots + shared decoder/projection + frozen backbone is occupied as a generic mechanism.

## External addressable memory + lightweight reader is occupied

**Cross-Model Memory Transfer via Target-Side Reader Adaptation — arXiv:2608.17050** studies a frozen, addressable Engram-style external memory table attached to other frozen backbones with a lightweight target-side reader.

Therefore reusable external neural artifacts plus trained readers, and the address/storage/reader factorization by itself, are occupied.

## Stronger surviving research question

The So candidate must now be tested on properties these memory-reader papers do not establish merely by having a separate memory channel:

1. **Canonical lifecycle identity** — one `pod_id`, many aliases, monotonic generations, single authority.
2. **Exact future-world rebinding** — arbitrary post-training entity/value changes are data writes, not learned memory entries requiring per-record optimization.
3. **Authority capability** — only an independently verified current generation can be materialized into the neural read path.
4. **Transitive lifetime closure** — any derived J-Space/cache/assistant/tool state carries the exact union of source generation lifetimes and becomes inadmissible when any source expires.
5. **Anti-resurrection across physical stale state** — stale pages/caches/replicas may remain byte-for-byte present yet cannot affect current logits or regain authority through restore/restart paths.
6. **Base-cache non-contamination** — mutable world state must not force invalidation of the reusable linguistic/skill KV plane.
7. **Compact model-native state** — competitive memory footprint without turning the system back into text retrieval.
8. **Independent causal verification** — lifecycle admissibility is recomputed by a separate control path rather than trusted from the reader that requested the memory.
9. **Joint serving frontier** — demonstrate quality, TTFT, update/delete latency, context cost and lifecycle correctness against TokenMem-like memory, strong RAG, Knowledge-Packs/KV and full-recompute baselines.

If prior work is found that already combines these lifecycle properties, the claim must narrow again. The project is not allowed to promote R295/R296 success alone to a breakthrough claim.
