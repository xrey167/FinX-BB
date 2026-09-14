# CKCA Architecture V3 — Causal Neural World ABI

Date: 2026-09-14
Status: **architecture candidate under active falsification; not yet a breakthrough or novelty claim**

## 0. The new abstraction boundary

The central proposal is no longer “give an LLM a better memory.” It is to give a neural system an explicit **world ABI**.

A conventional LLM blurs three kinds of state:

1. reusable language/reasoning skill;
2. mutable world facts;
3. neural execution artifacts that have already absorbed those facts.

CKCA V3 separates them by contract.

> **Skill is reusable computation. Mutable knowledge is generation-scoped world state. Neural artifacts that consume world state are temporary borrows whose validity is derived from the exact generations they read.**

The ABI is the boundary between the reusable neural machine and the changing world.

This creates a design target analogous to a processor architecture, but for mutable knowledge:

- natural language compiles into operations;
- Symlinks are logical addresses;
- Pods are canonical typed world cells;
- generations are temporal identities;
- capabilities are verified read handles;
- CKVM is the world instruction set;
- J-space is the mutable neural register/workspace;
- NLTS is a temporal type system;
- CKT is the publication transaction;
- CLFD is the transitive lifetime fabric;
- GCRD is the incremental repair mechanism.

The goal is not to emulate a database inside an LLM. The goal is to define when mutable information is legally allowed to enter neural computation and when every derivative of that information ceases to be admissible.

---

# 1. Six architectural laws

## Law 1 — Address/value separation

Identity is not value.

A canonical entity can be named by many linguistic forms, but they resolve to one stable address:

```text
"Acme"
"ACME Corp"
"the supplier from Lyon"
          │
          ▼
      Symlink resolve
          │
          ▼
       pod:71c9
```

The current value is read only after resolution:

```text
pod:71c9 @ generation 184 -> current typed payload
```

No reusable query plan is allowed to copy the current value merely because it knows the address.

---

## Law 2 — Binding equivariance

After address resolution, a value operator must not be able to recover mutable facts from identity shortcuts.

For a world `W`, canonical addresses `a,b`, and operation `op`:

```text
Exec(op, a, b, W) = R(op, W[a], W[b])
```

where `R` does not receive `a` or `b` unless identity itself is semantically required by the operation.

This turns future rebinding from a learned generalization problem into an architectural symmetry.

If the world changes from:

```text
W1[a] = v7
```

to:

```text
W2[a] = v42
```

then the reusable operation remains the same. Only runtime data changes.

R315 showed that random world-swapping during training did not reliably teach this separation. R317 showed that removing the identity->fact shortcut by construction changed deranged future-world accuracy from roughly 29% for a monolithic shortcut model to roughly 99.6% for the binding-equivariant path on the synthetic gate.

The architectural decision is therefore stronger than a training curriculum:

> **Do not merely penalize an illegal information path. Remove it.**

---

## Law 3 — One-way mutable information flow

CKCA V3 uses a causal dual rail:

```text
        reusable skill / plan
              B rail
                │
                │  B -> J allowed
                ▼
        mutable world execution
              J rail
                │
                X  J -> B forbidden
```

The B rail may provide language interpretation, reusable reasoning operators, schema knowledge, and control state to J.

The J rail may consume current world values and B control.

But mutable world state is structurally forbidden from flowing backward into reusable B state.

Conceptually:

```text
∂ B_cache / ∂ mutable_world = 0
```

by architecture, not by regularization.

R318b demonstrated the mechanism on a synthetic neural gate: B-state delta under world changes was exactly zero, cached-B execution was numerically identical to full B execution, future-world quality remained within about 0.21 percentage points of the shared-rail control, and cached mutable execution was about 3.17x faster than recomputing both rails.

---

## Law 4 — Temporal borrow and seal

A mutable generation is never an ordinary timeless tensor.

Inside CKVM, every value belongs to a lifetime class:

```text
Static<T>
Plan<T, plan_generation>
Borrowed<T, {pod:g ...}>
Sealed<T, lifetime_factor>
```

A world read creates a `Borrowed` value.

A borrowed value can participate in J-space computation, but it cannot legally enter a reusable B cache and cannot escape the knowledge transaction as a retained artifact.

Before retention, its exact generation dependencies are sealed into a lifetime factor.

The compiler contract is:

```text
Borrowed<T, G> -> B-cache          ILLEGAL
Borrowed<T, G> -> retained J       ILLEGAL
Borrowed<T, G> -> publish          ILLEGAL without commit(G)
Borrowed<T, G> -> Seal(G) -> J     LEGAL
```

R316 tested a CKVM-specific temporal borrow checker over 250,000 valid programs and 250,000 lifecycle-mutated programs with zero false rejects and zero false accepts in that bounded gate.

This is important because lifecycle safety should not depend only on runtime vigilance. Several classes of stale-state bugs should be **unrepresentable in a well-typed CKVM program**.

---

## Law 5 — Transactional publication

A neural computation may be internally speculative. Publication may not be.

A Causal Knowledge Transaction records the exact generation read-set actually used:

```text
ReadSet = {(pod_i, generation_i), ...}
```

Immediately before an answer or retained J state becomes externally visible, CKCA executes the Neural Commit Barrier:

```text
commit allowed iff
for every (pod, generation) in ReadSet:
    generation == authority.current_generation(pod)
```

If any generation changed:

```text
abort J
reuse B
refresh capabilities
retry mutable execution
```

A same-value rewrite is still a conflict:

```text
value(g17) == value(g18)
```

does **not** imply:

```text
g17 == g18
```

R313 found 7,664 such same-value generation conflicts in its concurrency gate and correctly rejected them.

After commit, the artifact receives a CLFD factor over the exact committed generations. That factor protects against later writes.

Thus CKT has two coherence moments:

1. **commit-time coherence** — do not publish a world that was never jointly current;
2. **post-commit coherence** — do not serve a once-correct artifact after one of its source generations expired.

R314 showed both are necessary: removing the commit barrier admitted mixed generation transactions; removing post-commit lifetime admitted stale retained artifacts.

---

## Law 6 — Causal repair instead of weak invalidation

Strict invalidation should not require throwing away everything.

Each generation records where it first entered an autoregressive/neural trajectory.

When a source expires:

```text
changed generation
       │
       ▼
reverse lifetime index
       │
       ▼
earliest affected neural checkpoint
       │
       ├── safe prefix: reuse
       │
       └── dependent suffix: regenerate
```

This is Generational Causal Rewind Decoding (GCRD).

R319 tested partial replay against full current-world recomputation over 120,000 update trials. The partial replay matched both output bytes and generation read-sets in every tested case. When mutable binding began late in the decode, affected updates required regenerating about 15.8% of the trace on average rather than the complete answer in the synthetic gate.

This gives CKCA a useful performance principle:

> **Never relax generation correctness to save latency. Reduce the causal repair surface instead.**

---

# 2. The Causal Neural World ABI

The ABI exposed to neural execution is intentionally narrow.

A conceptual interface is:

```text
resolve(alias) -> PodRef
acquire(PodRef) -> Capability<pod,generation,schema>
read(Capability, field) -> Borrowed<T, pod:g>
call(opcode, operands...) -> Borrowed<U, union(lifetimes)>
commit(read_set) -> CommitToken | Conflict
seal(value, CommitToken) -> Sealed<U, factor>
materialize(Sealed<U>) -> J/KV/token state
publish(Sealed<U>) -> externally visible result
```

Notably absent is:

```text
retrieve arbitrary factual text -> concatenate into prompt -> hope model obeys it
```

Text retrieval may still exist for discovery, but it is not the authority path for governed facts.

---

# 3. Discovery is not truth

Semantic search is useful for discovering **which canonical objects a query refers to**. It should not be the authoritative carrier of their mutable payload.

CKCA therefore separates:

## Discovery plane

Potentially approximate and eventually consistent:

- lexical search;
- embeddings;
- vector indexes;
- graph search;
- learned entity resolution;
- aliases.

Output:

```text
candidate PodRefs / relation handles
```

## Truth plane

Strictly current and generation-scoped:

```text
PodRef -> current authority capability -> current typed value
```

A stale discovery index may at worst return an old candidate address. It is not allowed to return an old factual payload as truth. The authority plane will either resolve the candidate to its current generation or reject/revoke it.

This changes the role of RAG in the architecture.

RAG becomes an optional **address-discovery compiler**, not the knowledge truth substrate.

For already compiled aliases/relations, no vector retrieval is needed at all.

---

# 4. Microarchitecture

## B0 — language encoder

Interprets natural language and establishes reusable semantic structure.

No governed current value is permitted here.

## B1 — plan compiler

Compiles intent into typed CKVM operations and canonical resolution requests.

Example:

```text
%a = RESOLVE("Acme")
%b = RESOLVE("Beta")
%x = CURRENT(%a, price)
%y = CURRENT(%b, price)
%z = SUB(%x, %y)
RETURN(%z)
```

The compiled plan depends on interpretation/schema/alias-binding lifetimes, not ordinary fact values.

## W — world register file

Pods expose typed current data through generation capabilities.

The world register file can be local, distributed, persistent, or replicated. Physical residence does not imply authority.

## J — mutable execution rail

Receives:

- B-plan control;
- current capabilities;
- current typed values.

Every J register inherits the union of causal generation lifetimes.

## C — commit barrier

Validates the actual read-set before publication.

## M — materializer

Converts committed typed results into the smallest neural form needed by the next operation:

1. typed scalar / compact ID;
2. structured relation handle;
3. token sequence;
4. compact latent Port;
5. KV/hidden state only when required.

## D — decoder

Uses late materialization and GCRD checkpoints so mutable state enters as late as possible and can be repaired with minimal rewind.

---

# 5. Why the generation is part of identity

CKCA treats knowledge identity as at least:

```text
(pod_id, generation)
```

not merely:

```text
hash(payload)
```

Two writes may contain identical bytes but represent different lifecycle events.

This prevents ABA resurrection:

```text
g17 = "42"
revoke/update
g18 = "42"
```

An artifact derived from `g17` cannot become valid again just because `g18` has identical bytes.

The old lifetime leaf stays dead permanently. `g18` receives a fresh leaf.

---

# 6. Cache hierarchy

The architecture intentionally creates several cache classes with different invalidation domains.

## B-plan cache

Depends on:

- query interpretation;
- schema/compiler revision;
- alias-binding generation where captured.

Does not depend on ordinary current values.

## B-prefix/KV cache

Legal only before mutable materialization.

Its world dependency set is empty by construction.

## J-state cache

Depends on exact generations actually read.

## Decode suffix cache

Depends on J state plus any generations read later during generation.

## Discovery cache

Can be approximate but may contain only candidate identities for governed knowledge, not authoritative stale payloads.

This cache hierarchy is what allows an update to be cheap without pretending that stale state is safe.

---

# 7. Training contract

The architecture should be trained around the ABI rather than trained to memorize the world.

Desired objectives include:

## Operation competence

Learn reusable transformations over typed values and relations.

## Compiler competence

Map language into CKVM programs and canonical resolution requests.

## Binding-equivariant execution

The post-resolution operator path must not have an identity-to-value shortcut.

## Null/revocation competence

The model must handle `REVOKED`, `UNKNOWN`, and `NO_CURRENT_VALUE` as first-class typed states instead of falling back to parametric memory.

## Causal read minimization

Prefer plans that read only the generations actually needed. This reduces lifetime fanout.

## Late-materialization pressure

Prefer symbolic/typed operations until neural representation is necessary.

Crucially, the hardest invariants are not left to the loss function:

- B cannot read mutable values;
- post-resolution value execution cannot secretly read identity unless explicitly typed;
- borrowed state cannot escape without seal;
- publication cannot bypass commit.

Those are architecture/compiler rules.

---

# 8. Deletion and revocation semantics

A revocation is not “please forget this text.”

It is a transition in authority:

```text
pod:g17 live -> pod:g18 revoked/null
```

Effects:

1. old capability fails verification;
2. old CLFD generation leaf becomes permanently false;
3. every transitive J/KV/cache descendant becomes inadmissible;
4. in-flight transactions that read g17 fail their commit barrier;
5. retained outputs whose lifetime included g17 cannot be served;
6. GCRD finds the earliest affected checkpoint for active decoding;
7. governed namespace firewall forbids fallback to pretrained lexical recall.

The bytes of g17 may still physically exist in an audit archive. Possession does not confer authority.

---

# 9. Persistence and replicas

A node starting from a stale snapshot cannot declare itself current merely because its local tensors are internally consistent.

Recovery is:

```text
restore neural/lifetime snapshot at watermark S
recover authenticated authority to tip T
replay S<T authority transitions into CLFD
invalidate dead generations
only then admit retained neural artifacts
```

Replica acceptance requires current authority, not historical signature validity alone.

---

# 10. Concurrency

CKCA V3 intentionally avoids global locks around neural reasoning.

Execution is optimistic:

1. acquire generation capabilities lazily;
2. compute speculatively;
3. record exact dynamic read-set;
4. validate at Neural Commit Barrier;
5. on conflict, use the earliest causal checkpoint and retry the dependent J suffix.

Thus the architecture combines:

```text
optimistic world reads
+ exact commit validation
+ post-commit lifetime
+ minimal causal rewind
```

instead of pessimistically freezing the world for an entire LLM generation.

---

# 11. Formal invariants targeted by V3

## I1 — no mutable contamination of reusable B

For any two worlds `W1`, `W2` with the same query/schema interpretation:

```text
B(q, schema, W1) = B(q, schema, W2)
```

because `W` is not an input to B.

## I2 — binding equivariance

For operations whose semantics are value-only after address resolution, renaming canonical identities while preserving the resolved value tuple must not alter the result.

## I3 — generation monotonicity

Once `(pod,g)` is invalid, it never becomes valid again.

## I4 — no unsealed escape

Every retained mutable-derived neural artifact has a non-empty valid lifetime factor or is proven world-independent.

## I5 — publication coherence

Every published mutable-derived result corresponds to a read-set that was current at its commit barrier.

## I6 — post-publication admission coherence

Every served retained artifact has a live transitive factor at the instant of admission.

## I7 — full-recompute equivalence

Any optimized execution path—cache reuse, late splice, causal rewind—must match the current-world full-recompute oracle at the defined boundary.

## I8 — same-value ABA safety

Equal payload bytes across distinct generations never revive old lifetimes.

---

# 12. What this architecture is trying to replace

Not every use of RAG should disappear.

But for governed mutable knowledge, the desired long-term path is:

```text
OLD:
query -> vector retrieval -> stale/current text chunks -> prompt -> model

CKCA V3:
query -> compile/resolve -> canonical handles -> current typed values
      -> transactional neural execution -> validated output
```

Semantic retrieval remains useful for open-ended corpus discovery, unstructured evidence collection, and finding unknown entities. Once governed knowledge has a canonical identity, repeated value transport through retrieved text is unnecessary overhead and weakens lifecycle semantics.

---

# 13. Evidence accumulated so far

Important mechanism gates currently include:

- **R312b:** frozen real Qwen2.5-0.5B; full recompute vs late splice full-vocabulary max logit delta 0.0; old factors die on source update including same-value rewrites; ~12.46x CPU wall-time speedup on the reduced gate.
- **R313:** 150,000 mutable transactions; 32,922 stale/mixed naive commits rejected; zero undetected stale commits; same-value generation conflicts detected.
- **R314:** bounded exhaustive transaction+lifetime composition; commit barrier alone and post-lifetime alone each have counterexamples; composed protocol had zero mixed commits and zero stale serves in the explored state space.
- **R315:** counterfactual world-swap training alone failed; this killed the curriculum-only solution.
- **R316:** temporal borrow/type rules caught all 250,000 injected lifecycle mutations in the bounded gate with no false reject of 250,000 valid programs.
- **R317:** structural binding equivariance gave ~99.61% accuracy on deranged unseen bindings vs ~28.76% for a monolithic shortcut control.
- **R318:** first dual-rail network preserved exact B isolation but failed quality (~58%); rejected.
- **R318b:** repaired directional-gated dual rail restored ~99.69% deranged-world accuracy with exact B invariance and ~3.17x cached mutable-path speedup.
- **R319:** generational causal rewind matched full recomputation in output and lifetime read-set across 120,000 update trials; late binding reduced affected suffix recomputation substantially in the synthetic gate.

These are mechanism results, not yet the full DoD.

---

# 14. Hard DoD — unchanged and strengthened

CKCA V3 is **not** a breakthrough until one integrated implementation demonstrates all of the following:

1. **Real pretrained models** — multiple model families, not one toy network.
2. **Open-ended language** — free multi-token answers, not only fixed synthetic labels.
3. **Future binding generalization** — unseen identities, unseen values, unseen combinations, and online rewrites without gradient steps.
4. **Exact lifecycle** — edit, revoke, same-value rewrite, restart, replica, stale-cache, and stale-KV attacks produce zero resurrection.
5. **Concurrency** — in-flight updates cannot yield mixed-generation publication.
6. **Causal repair** — optimized retry/rewind is equivalent to current-world full recomputation.
7. **Low overhead** — ordinary non-governed inference remains essentially untouched and governed execution reaches the <5% production-serving overhead target excluding unavoidable task work.
8. **Quality** — normal language/reasoning does not materially regress.
9. **Strong baselines** — compare against strong RAG, cached RAG, graph RAG where relevant, and editable neural memory/model-editing systems.
10. **Economics** — measure update cost, query cost, memory footprint, index maintenance, and scaling.
11. **Novelty audit** — literature and patent review against the **complete composition**, not cherry-picked components.
12. **Independent reproduction** — one-command artifact, pinned model revisions, checksums, machine-readable reports.

---

# 15. The architectural thesis in one sentence

> **A future neural system should not continuously rewrite its reusable intelligence every time the world changes; it should execute reusable intelligence over a generation-safe, typed, transactional world interface whose neural derivatives have explicit causal lifetimes.**

That is the V3 research target.
