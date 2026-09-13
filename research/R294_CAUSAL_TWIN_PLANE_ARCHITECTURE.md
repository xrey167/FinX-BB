# R294 — Causal Twin-Plane Transformer (CTPT)

Status: architecture candidate derived from R287–R293 and 2026 prior-art pressure. **Not DoD and not yet a novelty claim.**

## 1. Why the architecture must move beyond editable standard KV

R292 is motivated by a structural problem: once mutable world state enters ordinary causal self-attention, later cached token states can become implicit carriers of that state. Merely replacing or deleting the source token's K/V is therefore insufficient. This is consistent with the causal observation in *Models Take Notes at Prefill* (arXiv:2606.17107) and with rollback-consistency failures reported for retained KV state (arXiv:2608.15939).

The problem is deeper than an implementation bug.

### Dense contamination proposition

For an unmasked causal self-attention row with finite scores,

`alpha_ij = exp(s_ij) / sum_{k<=i} exp(s_ik)`

and therefore `alpha_ij > 0` for every visible predecessor `j <= i`.

If a mutable Port generation `g` influences a visible value vector `V_j`, then generically

`d Attn_i / d V_j = alpha_ij I != 0`.

So every downstream token that can attend to that Port becomes causally dependent on `g`, except in degenerate exact-cancellation cases. Subsequent layers project those dependent hidden states into new K/V states, propagating the dependency further.

**Consequence:** under an exact anti-resurrection guarantee, a mutable fact placed in the ordinary token stream tends to taint the entire downstream suffix. An exact lifecycle implementation then has only three options:

1. replay/recompute the dependent suffix;
2. retain exact transitive dependency metadata and invalidate the dependent suffix;
3. structurally prevent mutable world-state influence from being written into the persistent token-KV plane.

The target architecture chooses option 3 for the normal path, while retaining option 2 for intentionally durable outputs that depend on mutable Pods.

## 2. Twin-plane model

CTPT separates persistent linguistic computation from mutable world state.

### Plane B — Base / Skill Plane

- ordinary tokenizer and causal Transformer;
- language, algorithms, general reasoning skill, grammar, tool syntax;
- persistent prefix KV is allowed;
- **mutable Pods are not injected as ordinary B-plane tokens**;
- B-plane cache is therefore not invalidated when a Pod value changes.

### Plane P — Temporal Port Plane

- canonical `pod_id` resolves through the Symlink plane;
- a Causal Page Table selects exactly one authoritative generation;
- payload is a compact model-native Port code/capsule;
- previous generations may remain physically allocated;
- only current authority is materialized into Port Attention;
- revoke materializes no page.

R290 provides the current compact bootstrap candidate: shared-anchor residual K3/V3 capsules preserved the exact-cache top choice on its held-out read-value probe while using 21.875% of the BF16 per-fact bytes. This is a mechanism result, not the final representation.

### Plane J — Ephemeral Causal Workspace / J-Space

J-Space is the mutable reasoning workspace.

It may read:

- current B-plane hidden state;
- authoritative P-plane Port states;
- other live J-Space states.

But J-Space state is **not written into persistent B-plane K/V**.

A minimal layer form is:

`B_{l+1} = BaseLayer_l(B_l)`

`P_l = PortRead_l(B_l, Authority(pod_ids))`

`J_{l+1} = Workspace_l(J_l, B_l, P_l)`

`Y_l = OutputBridge_l(B_l, J_l)`

Only `B` is allowed to populate the reusable ordinary KV cache. `P` and `J` have explicit lifecycle semantics.

This is the central architectural break from treating mutable knowledge as context tokens.

## 3. Lifetime factors in J-Space

Every J-Space state carries an exact dependency set:

`deps(J_i) = {(pod_id, generation), ...}`.

Validity is:

`valid(J_i) = AND_{(p,g) in deps(J_i)} [authority[p] = g AND live[p]]`.

When a Pod changes, dependent J-Space states disappear from the causal workspace immediately. Their bytes may remain allocated but are not admissible.

Unlike ordinary self-attention, the Port dependency graph is explicit and sparse because Port reads are explicit operations rather than dense token-prefix visibility.

This recovers the user's original "shared lifetime" intuition in a concrete mechanism: derived neural state does not merely remember provenance; its **right to participate in future computation is conjunctively linked to the lifetime of every source generation it used**.

## 4. Generated text and the transcript problem

Even if P/J state never contaminates persistent B KV, generated token IDs can encode an obsolete fact in the visible transcript. Exact lifecycle semantics therefore distinguish:

- **immutable user/system text** — reusable in B;
- **assistant/tool segments with no mutable dependencies** — reusable in B;
- **segments derived from mutable Pods** — tagged with their Pod-generation read-set.

When any dependency expires, that segment's cached neural state is invalid. The raw human-visible transcript may still be retained for audit, but it is not silently treated as current authoritative evidence. The model-facing representation is either:

1. masked as historical/stale;
2. regenerated from current authority;
3. admitted only as explicitly quoted historical evidence.

This prevents a deleted Pod from resurrecting through the assistant's own old answer.

## 5. Independent causal verifier

The generative Port read and the verifier do not share a mutable decision object.

For each materialization, verifier V independently checks:

- canonical `pod_id`;
- generation monotonicity;
- authority generation;
- payload digest;
- model/codec revision;
- tombstone state;
- J-Space dependency validity;
- segment dependency validity.

A mismatch fails closed before Port/J-Space admission.

The verifier result is a small capability:

`Capability(pod_id, generation, payload_digest, model_revision, expiry/revoke state)`.

Port Attention consumes capabilities, not unverified page references.

## 6. Why this can attack the old 73% lifecycle overhead

The old architecture put lifecycle bookkeeping directly on the hot inference path and/or allowed stale generations to remain in the attention sequence. CTPT changes the scaling law:

- historical generations live in the external page store, not the attention sequence;
- normal B-plane KV remains reusable across Pod updates;
- Port Attention scales with **active selected Pods**, not revision history;
- invalidation touches the authority record and sparse J/segment reverse indexes;
- compaction is asynchronous and not required for correctness.

The desired hot-path overhead is therefore approximately:

`O(resolve aliases + verify active capabilities + PortAttention(active_ports))`

instead of:

`O(replay downstream prefix + scan stale generations + rebuild global cache)`.

R291 directly tests the active-only page-table part on a real frozen model. System-wide <5% remains unproved until GPU serving benchmarks exist.

## 7. Why this is not merely RAG

RAG performs query-time retrieval of textual/external evidence and places retrieved material into the model's context or equivalent generation path.

CTPT's intended endpoint differs:

- canonical mutable identities exist independently of a query;
- live generation authority is a model execution invariant, not retrieval freshness metadata;
- Port payloads are model-native compact states rather than retrieved source text;
- derived neural workspace carries source lifetimes and becomes causally inadmissible when a source generation expires;
- no stale generation may participate even if its bytes survive in caches, replicas, restart images, or prior workspace states;
- mutable world state is an architectural plane of the model rather than a document-injection layer.

Nevertheless, until quality/speed benchmarks beat strong RAG at matched correctness, this remains an architectural hypothesis rather than a superiority claim.

## 8. Prior-art kill boundaries added in R294

The claim must explicitly exclude the following as standalone novelty:

- paged KV storage / block tables;
- encoder-decoder or generic cross-attention memory;
- prefix/KV composition and RoPE relocation;
- append-only errata for stale prefill notes;
- policy-directed KV span removal/replacement (Leyline);
- rollback-consistent cache restoration;
- provenance-based semantic cache invalidation;
- low-bit or mixed-precision KV quantization;
- external memory or editable-memory modules in general.

A defensible candidate contribution, if experiments support it, is the **joint lifecycle contract and architecture**:

> a causally separated mutable neural Port plane with canonical identities, generation-authority capabilities, active-only materialization, compact model-native payloads, lifetime-carrying ephemeral reasoning state, transitive anti-resurrection of derived neural state, and an independent verifier — while preserving a reusable knowledge-independent base KV plane.

This claim must still survive a full paper/code/patent audit.

## 9. Immediate experimental gates

### R291 — Causal Page Table

Prove on a real frozen LM that thousands of old pages can remain allocated while only the independently verified current page reaches attention; mutate stale pages and require full-logit invariance.

### R292 — Transitive Lifetime Closure

Construct downstream cached neural notes under generation `g`, edit only the source Port to `g+1`, and measure stale-cache divergence from full recompute. Then lifetime-mask the `g`-dependent notes and require mutation invariance.

### R293 — Functional compactness

Use one query-independent K3/V3 compressed Port for unseen holdout values across multiple different operations. No per-query or per-fact optimization is allowed.

### R295 — Twin-plane real-model sidecar gate

Freeze the base LM and train only a small Port/J-Space bridge. The same B-plane query KV must be reused unchanged while thousands of Pod updates change answers through P/J. New entity/value bindings and aliases must require zero neural optimization. With Port/J disabled, base logits must be bit-identical to the untouched LM.

### R296 — Multi-Pod J-Space composition

Two-to-eight active Pods, relational chains, conjunctions, updates to intermediate nodes, deletion, and stale derived-state attacks. Track exact read-sets and invalidate only dependent J states.

### R297 — serving comparison

GPU/vLLM-like benchmark against:

- strong text RAG;
- Knowledge-Packs-style KV baseline;
- programmable-KV/erratum-style edit baseline where applicable;
- full reprefill oracle.

Measure quality, TTFT, ITL, write/update/revoke latency, bytes per fact, active-context cost, normal-inference degradation, and invalid-generation admissions.

No breakthrough/DoD claim before these converge.
