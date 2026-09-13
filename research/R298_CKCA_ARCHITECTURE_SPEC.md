# Causal Knowledge Coherence Architecture (CKCA)

Status: research architecture candidate. **Not a breakthrough or novelty claim yet.**

## Non-negotiable objective

The target is not “better RAG” as a retrieval implementation. The target is a new knowledge substrate for neural systems in which mutable world state has the lifecycle semantics of authoritative state rather than the accidental semantics of tokens, weights, or long-lived caches.

One canonical knowledge identity must be writable, replaceable and revocable once. Every alias, neural read, cached derivation, intermediate reasoning state, tool state, replica and restart path must follow the same authoritative lifecycle. Old generations may remain physically present, but may never regain causal influence. The immutable language/skill path must remain reusable and normal inference must not materially degrade.

The complete architecture must eventually beat strong RAG on the joint frontier of answer quality, TTFT, context cost, online write/update/delete cost and lifecycle correctness.

## Architectural split

CKCA separates four concerns that ordinary context injection conflates.

### B — immutable base plane

The base transformer contains language, reusable skills and general reasoning operators. Mutable world-state values do not become persistent members of its reusable cache.

Invariant:

`B(query)` is independent of the current generation of every mutable Pod unless a verified Port is explicitly read through the knowledge plane.

A world-state edit therefore does not require a base-model weight update or global base-cache invalidation.

### P — canonical temporal Port plane

Each mutable knowledge identity owns one stable `pod_id`. Natural-language aliases resolve to that identity; aliases are not separate copies of the fact.

A Pod has monotonic generations:

`p:g1 -> p:g2 -> ...`

A future entity/value binding is data written to a Port, not a pair the neural network must learn.

The current bootstrap payload is a compact model-native capsule. R288 kills naïve global PCA. R290 keeps coordinate-preserving low-bit residual compression as the current engineering representation, not as the novelty claim.

### A — authority plane / Causal Page Table

Historical pages are external physical storage. The attention path never receives all revisions and then tries to ignore stale ones. Instead an independent verifier materializes exactly one current page.

For capability `c=(pod,g,page,digest,model_revision)`:

`Admit(c) = live[pod] AND authority_generation[pod]=g AND authority_page[pod]=page AND digest(page)=digest AND model_revision=current_model_revision`

If `Admit(c)=false`, the page is not a member of neural attention at all.

Update protocol:

1. compile/write a new immutable page;
2. verify digest, model revision and monotonic predecessor;
3. atomically switch authority to the new generation;
4. invalidate the old generation lifetime leaf;
5. leave old page bytes physically present until asynchronous reclamation.

Revocation clears admissible authority and invalidates the previous generation leaf.

This removes revision-count growth from the neural sequence and makes position-overlay generation tricks unnecessary for the normal serving path.

### J — ephemeral causal workspace

Any state computed from mutable knowledge is not ordinary timeless cache. It is a **lifetime-carrying derivation**.

If state `x` read source generations `R(x)`:

`Lifetime(x) = AND_(p,g in R(x)) Lifetime(p,g)`

A derived J state may itself be an input to another state. Lifetime composition is transitive:

`Lifetime(z = f(x,y)) = Lifetime(x) AND Lifetime(y)`

The runtime represents these conjunctions with the Causal Lifetime Factor DAG (CLFD). Each `(pod,generation)` is an immutable lifetime leaf. A derived state references a factor node. When a source generation expires, reverse edges invalidate exactly the descendants that depend on it.

Hot-path admission becomes one bit lookup:

`Serve(x) = valid[factor_id(x)]`

New generations allocate new leaves. A false old factor is monotonic-dead. Therefore same-value rewrites, ABA cycles and revoke-then-revive cannot resurrect an old neural state merely because semantic bytes happen to match again.

## Core coherence law

CKCA treats mutable neural knowledge as a cache-coherence problem with semantic authority rather than as prompt replacement.

For every neural artifact `n` that can influence output:

`causal_sources(n)` must be either:

1. immutable B-plane state, or
2. a set of verified generation capabilities with a live lifetime factor.

There is no third class of untracked mutable-derived state.

This includes:

- Port pages;
- J-Space hidden states;
- assistant prefix/KV segments;
- tool-call result caches;
- speculative branches;
- distilled intermediate notes;
- cross-request caches;
- replica-local copies;
- restored persisted artifacts.

If mutable knowledge influenced an artifact and that influence is not represented in its lifetime factor, the implementation violates CKCA.

## Anti-resurrection theorem target

Assume:

1. generations are strictly monotonic per Pod;
2. old lifetime leaves never transition from false back to true;
3. neural admission requires current authority capability plus a live lifetime factor;
4. every mutable-derived artifact carries the transitive union of its causal source lifetimes;
5. restart/replica recovery reconstructs authority from an append-only authenticated ledger and fails closed on uncertainty.

Then an artifact derived from generation `g` cannot become admissible after generation `g` has been replaced or revoked, even if:

- its bytes remain in memory;
- a stale replica restores it;
- a cache key collides semantically;
- the same fact value is later written again;
- an alias resolves to the same `pod_id`;
- an intermediate neural note still contains information about the old generation.

The research program must turn this target from an architectural argument into a formal model plus crash/replay and real-model tests.

## Evidence already supporting the split

- R287: held-out entity/value binding, late aliases, pointer chains, online updates and revocation reached 100% in the synthetic typed-Port reasoner without post-world-update optimization.
- R288: global low-rank PCA failed badly on future values; delete it as the primary codec hypothesis.
- R290: coordinate-preserving residual K3/V3 compression preserved the narrow held-out read decision at 21.875% of BF16 per-fact bytes.
- R292: source-only KV replacement left measurable downstream stale influence in 9/12 real-model cases; transitive lifetime masking produced zero full-vocabulary delta against the fresh masked oracle and zero delta under arbitrary stale-byte randomization.
- R297: CLFD achieved 0 mismatches in 50,000 exactness audits, 3.19x faster validity checks than explicit dependency scans, and touched ~0.00666% of the graph per update on average.
- R298 v1: integrated real frozen Qwen + two Ports + multi-hop pointer world + lifetime factors reached 98.78% held-out two-Port accuracy, 98.59% initial multi-hop accuracy and 98.05% after 1,000 post-training world changes, with 100% stale-capability rejection, zero factor-vs-explicit mismatches, ABA/revoke-revive protection and zero invalid J-state admissions. Its B-plane repeatability check used mismatched batch shapes and is being rerun with an identical standalone probe.

## Explicit kill decisions

The project must stop treating the following as candidate breakthroughs:

- cached fact text or precomputed KV by itself;
- a generic frozen-backbone cross-attention memory sidecar;
- generic latent memory slots;
- address/storage decoupling by itself;
- value-only stale-cache repair;
- source-only KV replacement;
- global low-rank PCA of Port K/V;
- position-overlay multiple revisions in the attention sequence when active-only page materialization can avoid co-resident stale revisions entirely.

These may be implementation tools, but they are not the contribution.

## Surviving invention candidate

The potentially new object is the **coherence contract over neural knowledge**, implemented jointly as:

`Canonical Symlink Identity -> Temporal Pod Generations -> Independently Verified Authority Capability -> Compact Neural Port -> Ephemeral J-Space -> Transitive Lifetime Factor -> Causal Admission`

The candidate contribution is not “memory”. It is a runtime/architecture in which mutable knowledge and all neural state derived from it obey exact generation-lifetime coherence while immutable neural skill state remains reusable.

## Required breakthrough gates

No major-contribution claim before all of these are satisfied:

1. Real open LMs at multiple scales and architectures.
2. New entities, aliases, values and relations after training with no per-record optimizer loop.
3. Multi-Pod and multi-hop reasoning under online edits to leaves and intermediate pointers.
4. Free-form generation, not only classification/read probes.
5. Exact lifecycle across downstream KV, hidden/J state, tools and cross-request caches.
6. Crash/restart, snapshot restore, out-of-order replica replay and stale-handle attacks with zero invalid admissions under the declared consistency model.
7. Compact payload competitive with state-of-the-art retrieval/memory systems.
8. End-to-end no-Pod inference degradation within the agreed <5% budget.
9. Quality/TTFT/context/update/delete frontier against strong RAG, Knowledge-Pack/KV baselines and dedicated-memory readers.
10. Systematic literature/code/patent audit showing the **joint coherence mechanism** is not already disclosed.

Until these pass, CKCA is a research architecture candidate, not a claimed breakthrough.
