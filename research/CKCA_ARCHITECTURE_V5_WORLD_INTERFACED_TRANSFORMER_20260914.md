# CKCA Architecture V5 — World-Interfaced Transformer

Date: 2026-09-14
Status: **research architecture candidate under active falsification; no breakthrough/novelty claim yet**

## 0. The architectural leap

V5 separates three things that ordinary LLM stacks usually collapse into one token/parameter stream:

1. **reusable intelligence** — language, operators, plans, skills;
2. **mutable world state** — current facts, relations, aliases, generations;
3. **derived neural/semantic state** — KV, hidden state, conclusions, cached answers and conversation objects that have consumed mutable world state.

The proposed model is a **World-Interfaced Transformer (WIT)** inside a **Causal Neural Machine (CNM)**.

The Transformer is no longer expected to own all current facts. It compiles natural language into verified operations over a model-independent world address space. Mutable values cross a typed World Port only after canonical resolution and authority validation. Every neural or semantic derivative inherits an explicit generation lifetime.

The goal is not “RAG without vectors.” The goal is a different execution model:

> **language/skill is computation; current knowledge is versioned data; derived neural state is a causal borrow.**

---

# 1. Machine overview

```text
Natural language
      │
      ▼
┌─────────────────────┐
│ B-rail Transformer  │  reusable language / plan / skill
│ neural compiler     │  NO mutable value input
└─────────┬───────────┘
          │ opcode + source-span pointers
          ▼
┌─────────────────────┐
│ Epistemic Plan      │
│ Firewall + Linker   │  model may not invent facts/PodRefs
└─────────┬───────────┘
          │ verified plan + Versioned Symlink capabilities
          ▼
┌───────────────────────────────────────────┐
│ Causal Neural World ABI                   │
│ Symlink:g -> Pod:g -> Predicate:g -> data │
└──────────────┬────────────────────────────┘
               │ current capabilities
               ▼
┌─────────────────────┐
│ World Port / ISWA   │  stable routing keys
│ current value pages │  value lifetime != routing lifetime
└─────────┬───────────┘
          │ typed current values
          ▼
┌─────────────────────┐
│ J-rail Transformer  │  mutable neural execution
│ identity-blind      │  B -> J allowed, J -> B forbidden
└─────────┬───────────┘
          │ exact dynamic read-set
          ▼
┌─────────────────────┐
│ Neural Commit       │
│ Barrier             │
└─────────┬───────────┘
          │ committed result
          ▼
┌─────────────────────┐
│ CLFD + Neural MMU   │  generation lifetime + GANPT
└─────────┬───────────┘
          │ admitted live neural state
          ▼
┌──────────────────────────────────────┐
│ Semantic Airlock / World Slots       │
│ CSOG / Live Views / Semantic Weave   │
└──────────────────────────────────────┘
```

---

# 2. B rail — compile meaning, not current facts

The B rail receives language and immutable/schema-level context. It produces a plan, not a factual answer.

A plan can contain:

- CKVM opcode;
- source-token/source-span pointers;
- schema fields;
- control flow;
- typed user/schema constants.

A plan cannot contain:

- model-origin governed factual literals;
- model-generated Pod addresses;
- current mutable values;
- hidden “recover from model memory” operations.

The Epistemic Plan Firewall verifies this boundary and a trusted linker resolves exact source spans to canonical Versioned Symlinks.

The B state is reusable across ordinary value updates. Conceptually:

```text
∂B_cache / ∂ current_world_value = 0
```

R329 instantiates this with an actual Transformer compiler. Across complete world rebinding, B hidden/logit state was bit-identical in the tested gate while the CKCA pipeline retained ~97% task accuracy. A stronger parametric-world control is being tested in R329b before interpreting the baseline gap.

---

# 3. Versioned Symlinks — the linguistic address is temporal state

The linguistic-to-canonical mapping is itself mutable.

```text
alias -> Symlink(binding_generation) -> Pod
```

A plan captures only the Symlink generations it actually resolved. Rebinding one alias invalidates only plans that depend on that binding.

R328 showed why a global namespace epoch is unacceptable at scale: 20,000 rebinds over 250,000 plans would imply five billion global plan invalidations, while per-Symlink generations required only 71,822 plan invalidations in the gate, a ~99.9986% reduction with zero stale accepts.

Same-target rebinding still creates a new generation. Temporal identity is not payload equality.

---

# 4. Binding Equivariance — identities select data, not factual shortcuts

After canonical resolution, value-semantic execution should obey:

```text
Exec(op, a, b, W) = R(op, W[a], W[b])
```

where `R` cannot see `a` or `b` unless identity itself is semantically required.

This is a hard information-flow rule rather than a preference learned from data.

R315 killed the weaker curriculum-only approach. R317 then showed the structural version: on fully deranged future bindings, the monolithic shortcut model fell to ~28.8%, while the identity-blind execution path reached ~99.6% in the synthetic gate.

R329 carries the same rule into a Transformer executor: J receives opcode + current typed value bits, not entity identity.

---

# 5. Indirection-Stable World Attention — routing lifetime != value lifetime

A second shortcut appears when the current mutable value changes the key used to retrieve that value.

If routing keys depend on mutable payloads, an update to a previously unselected item can alter future top-k selection. A cache that validates only previously selected value generations can then look valid while its fresh route has changed.

V5 introduces **Indirection-Stable World Attention (ISWA)**:

```text
query
  -> immutable/stable semantic routing key
  -> selected PodRefs
  -> current generation capabilities
  -> mutable value pages
```

Ordinary value updates cannot modify routing keys. Routing-key/schema/Symlink changes are distinct temporal events with their own generations.

R330 produced 3,210 hidden false-route accepts in an entangled-key control, but zero routing/output false serves in the stable-key architecture. Local selected-generation invalidation reduced the counterfactual global-predicate invalidation surface by ~99.8%.

---

# 6. J rail — mutable neural execution

The J rail is the only neural path allowed to consume current governed values.

It may consume:

- B-plan/control state;
- current typed value pages;
- committed parent semantic state.

It may not write mutable state back into reusable B state.

The topology is directional:

```text
B -> J     allowed
J -> B     forbidden
```

R318 rejected an underpowered J parameterization despite perfect isolation. R318b repaired it with directional gating, recovering ~99.69% deranged-world accuracy with exact B invariance and ~3.17x cached mutable-path speedup in that gate.

V5 therefore treats quality and isolation as a joint requirement. Architectural purity is not useful if it destroys model capability.

---

# 7. Neural Lifetime Type System

World reads produce temporal borrows:

```text
Static<T>
Plan<T, plan_lifetime>
Borrowed<T, {generation dependencies}>
Sealed<T, lifetime_factor>
Historical<T, explicit temporal reference>
```

Rules include:

```text
Borrowed -> reusable B cache        illegal
Borrowed -> retained J/KV           illegal before seal
Borrowed -> public result           illegal before commit
Sealed(dead factor) -> serve        illegal
```

R316 detected every injected lifecycle violation in 250,000 mutated CKVM programs while accepting 250,000 valid programs in the bounded test.

This makes several lifecycle failures type errors instead of production incidents.

---

# 8. Causal Knowledge Transaction

A J execution records the exact generations actually read.

```text
ReadSet = {(pod,g), (predicate,g), ...}
```

Before publication, the Neural Commit Barrier verifies that every generation is still current.

On conflict:

```text
abort J
reuse B plan
refresh capabilities
retry mutable execution
```

After successful commit, all retained neural/semantic derivatives receive a CLFD factor over their exact generation dependencies.

R313 and R314 established the need for both phases in bounded gates:

- commit-time validation prevents mixed-world publication;
- post-commit lifetime prevents a once-correct artifact from surviving a later update.

Same-value rewrites remain conflicts because generation is temporal identity.

---

# 9. Predicate Generations — relational absence/presence is also state

A graph query can go stale without any previously-read member value changing.

Example:

```text
max risk among suppliers(company X)
```

A later supplier-edge insertion creates a new candidate the old read-set never saw.

V5 therefore adds relation/predicate generations:

```text
query lifetime = predicate/set generation
               + exact generations of enumerated values
```

R327 found 16,961 lifetime false accepts and 2,036 semantic false serves when only member value generations were tracked. Adding the predicate generation reduced both to zero in the gate.

This is the neural analogue of phantom-read protection.

---

# 10. Epistemic MMU — physical bytes are not semantic authority

The Neural MMU combines:

- generation capabilities;
- CLFD lifetime factors;
- reverse invalidation;
- GANPT page admission;
- restart/replica watermarks.

The critical invariant is:

```text
physical possession != logical neural membership != semantic authority
```

R321 showed that simply masking stale in-sequence KV positions was not exact: top-1 remained stable but full-vocabulary logits moved by up to 0.125. That primitive was rejected.

R321b instead used a Generation-Aware Neural Page Table. Stale physical KV pages were retained and deliberately corrupted, yet because they were not admitted into the logical attention working set, the direct-current path, page-table path and full-current recompute remained full-vocabulary identical (`max delta = 0.0`) on frozen Qwen2.5-0.5B.

Production CKCA should push generation/factor admission into a paged-attention kernel so revocation is a page-table/lifetime operation and physical memory reclamation can occur asynchronously.

---

# 11. Causal repair — regenerate only the dependent region

Exact invalidation should not imply global recomputation.

GCRD records the earliest neural checkpoint where a mutable generation became causally relevant.

```text
expired generation
  -> reverse dependency
  -> earliest affected checkpoint
  -> preserve safe prefix
  -> regenerate dependent suffix
```

R319 matched both output and generation read-set against full current-world recomputation across 120,000 update trials. Late binding reduced the affected recompute fraction from ~51% to ~15.8% in the synthetic trace gate.

The performance law is:

> **Never weaken freshness to save latency; shrink the causal region that freshness governs.**

---

# 12. Causal Semantic Airlock — parametric stale memory does not get the final vote

A pretrained model may still contain a revoked fact in its parameters. External metadata cannot erase those weights instantly.

The governed rendering path therefore uses an airlock:

```text
lexical compiler
  -> verified typed IR / opaque PodRef only
  -> fresh governed renderer
  -> runtime-owned Typed World Slot
```

Compiler KV/hidden state does not cross. EPF prevents factual literals or forged addresses from crossing through the plan. The final governed literal is materialized from the committed World ABI result; revocation materializes `UNKNOWN` instead of asking the model to “remember correctly.”

R323 showed zero lexical leaks into the renderer, zero renderer-logit change from compiler-side latent/cache attacks, and zero live/revoked slot errors on frozen Qwen2.5-0.5B. The stale-value log-probability also dropped strongly after the lexical entity address was removed, but the safety claim rests on the structural channel separation, not on that probability shift.

---

# 13. Semantic interaction state — conversation is not flat authoritative text

A stale assistant sentence can resurrect a revoked generation even after KV and memory caches are fixed.

V5 therefore stores governed interaction state as semantic objects:

```text
Text(static language)
CurrentSlot(PodRef, field)
HistoricalRef(PodRef, generation)
LiveView(plan, sealed_result, lifetime)
DerivedClause(parent lifetimes, direct world lifetimes)
```

R324 demonstrated that stale archival transcript bytes can remain physically present and even be corrupted without altering the active model-context render. R326 showed that persistent governed answers can behave like lazy live materialized views: world updates invalidate factors, while actual message recomputation occurs only on access.

---

# 14. Lifetime-Carrying Semantic Weave — free-form language gets explicit causal reset boundaries

A monolithic autoregressive answer conservatively makes every later token depend on an early mutable read.

V5 introduces independently regenerable semantic clauses. Each clause is generated from:

- immutable language/plan context;
- explicit current world inputs;
- explicit parent semantic clauses.

Its lifetime is the transitive union of those dependencies.

```text
Clause A(dep Pod1)
Clause B(dep Pod2)
Clause C(dep A, Pod3)
Clause D(static)
```

An update to Pod1 invalidates A and C, not necessarily B or D.

R331 compared this semantic weave against a conservative monolithic autoregressive suffix. The weave produced zero oracle mismatches/stale lifetime survivors while reducing the repair surface by ~71.5% versus monolithic suffix regeneration and ~86.6% versus whole-document regeneration in the systems gate.

This is intended to bring exact lifecycle semantics to long-form explanations rather than only to literal factual slots.

---

# 15. Model-independent world state

The world exists once. Model-specific neural materialization happens after authority resolution.

R322 consumed the same canonical generation log with two frozen model families, Qwen2.5-0.5B and SmolLM2-360M-Instruct, without model-specific fact updates or reindexing. Both matched their full-recompute oracles at the tested late-binding boundary.

A multi-model company system should therefore be able to edit/revoke a governed fact once and have all planners, coding models, agents and renderers consume the same current generation.

---

# 16. The World-Interfaced Transformer contract

The target WIT architecture can be summarized as:

```text
B = Transformer(language, schema)
plan = EPF(B.plan_proposal)
addresses = TrustedLinker(plan.source_spans)
routes = ISWA(plan.query, stable_address_keys)
capabilities = WorldABI.acquire(routes)
J = Transformer_J(B.control, current_typed_values(capabilities))
commit = validate(J.dynamic_readset)
sealed = CLFD.seal(J, commit)
output = SemanticWeave.render(sealed)
```

Hard constraints:

```text
current values are not B inputs
entity identity is not a J value shortcut
model-origin factual literals are not legal plan data
uncommitted borrowed state cannot escape
expired pages are not admitted by the neural MMU
archival transcript bytes are not current truth
```

---

# 17. Why this is different from ordinary RAG

RAG remains useful for discovering unknown documents/entities. V5 does not remove open-ended retrieval.

But once governed knowledge has a canonical identity, retrieval should become address discovery rather than repeated factual-text transport.

```text
RAG truth path:
query -> retrieve chunks -> inject text -> model arbitrates currentness

CKCA/WIT governed path:
query -> compile/resolve -> stable addresses -> current capabilities
      -> typed neural execution -> committed semantic objects
```

A vector index may help find a PodRef. It does not own the Pod's current value, generation or downstream neural lifetime.

The architecture must still beat strong RAG experimentally on realistic quality/latency/cost workloads before superiority can be claimed.

---

# 18. Falsification ledger

V5 keeps failed branches as first-class evidence:

- R315: training-time world randomization alone failed future binding.
- R318: first dual rail preserved isolation but lost too much quality.
- R321: masked stale KV overlay preserved top-1 but changed full logits; rejected.
- R329: neural WIT shape succeeded, but the query-only monolithic baseline underfit; R329b was created to test a stronger parametric-world control.

A mechanism is promoted only if its failure mode is understood and the repaired variant passes a stronger gate.

---

# 19. Hard DoD

No breakthrough claim until one integrated artifact demonstrates all of the following:

1. multiple real pretrained model families and a real World Port implementation;
2. free multi-token long-context output with semantic-weave lifetimes;
3. future identities, values, bindings and relations without gradient updates;
4. edit/revoke/same-value ABA/restart/replica/stale-KV/stale-hidden/stale-transcript attacks with zero resurrection;
5. concurrent world changes with zero mixed-generation publication;
6. EPF compiler non-interference and CSA/TWS parametric-memory containment;
7. predicate generations for relational/graph absence/presence changes;
8. exact optimized paths versus a full current-world recompute oracle;
9. production neural-MMU/page-table implementation rather than Python tensor gathering;
10. <5% coherence/control overhead excluding unavoidable governed task work;
11. no material degradation of ordinary language/reasoning quality;
12. strong RAG, cached-RAG, GraphRAG, editable-memory and model-editing baselines;
13. update/query/token/VRAM/index economics at realistic scale;
14. literature and patent audit of the complete composition;
15. one-command independent reproduction with pinned revisions and checksums.

---

# 20. V5 thesis

> **A Transformer should not need to rewrite its reusable intelligence whenever reality changes. It should compile language into verified operations over a temporal world address space, consume current values through typed World Ports, and attach explicit lifetimes to every neural and semantic derivative of those values.**

The intended “new stone” is therefore not a memory plugin. It is a candidate **neural machine model for mutable reality**: compiler, World ABI, temporal type system, transaction protocol, neural MMU, causal repair and semantic output substrate as one architecture.
