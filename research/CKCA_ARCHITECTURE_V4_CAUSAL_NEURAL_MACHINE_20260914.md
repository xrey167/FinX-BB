# CKCA Architecture V4 — Causal Neural Machine

Date: 2026-09-14
Status: **architecture candidate under falsification; not yet a breakthrough or novelty claim**

## 0. From “LLM + memory” to a machine model

CKCA V4 takes a stronger position than “build a better memory system.”

A conventional LLM application usually has one dominant abstraction: a token stream enters a neural model, while current facts are either expected to already exist in parameters or are injected as additional text.

That abstraction is the root of several lifecycle problems:

- identity and value can collapse into the same learned representation;
- old facts can survive in weights, KV caches, hidden states, retrieved text and conversation history;
- a factual update has no universal meaning across those representations;
- physical possession of old neural state can accidentally become semantic authority;
- the model itself is often asked to decide whether an old or current fact should win.

V4 changes the machine boundary.

> **The model is a compute core over a mutable world, not the owner of the mutable world.**

The proposed machine therefore has an explicit **Causal Neural World ABI**, a temporal type system, an epistemic compiler membrane, a generation-aware neural MMU and a semantic-output substrate.

The architectural target is closer to a processor + virtual memory + coherence protocol than to a vector database bolted onto an LLM.

---

# 1. The machine

```text
                       NATURAL LANGUAGE
                              │
                              ▼
                    ┌──────────────────┐
                    │ Neural Compiler  │
                    │ language/intent  │
                    └────────┬─────────┘
                             │ untrusted plan proposal
                             ▼
                    ┌──────────────────┐
                    │ Epistemic Plan   │
                    │ Firewall + Linker│
                    └────────┬─────────┘
                             │ verified CKVM plan
                             ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                     CAUSAL NEURAL MACHINE                   │
 │                                                             │
 │  ┌────────────────┐             ┌────────────────────────┐  │
 │  │ B rail         │ ──────────► │ J rail                 │  │
 │  │ reusable skill │             │ current world execution│  │
 │  │ immutable plan │      X ◄────│ mutable derivatives    │  │
 │  └───────┬────────┘  NO J→B     └──────────┬─────────────┘  │
 │          │                                  │                │
 │          │                         Neural Commit Barrier     │
 │          │                                  │                │
 │          ▼                                  ▼                │
 │  ┌────────────────┐             ┌────────────────────────┐  │
 │  │ Neural MMU     │             │ Semantic Object Graph  │  │
 │  │ GANPT / CLFD   │             │ live views / slots     │  │
 │  └───────┬────────┘             └──────────┬─────────────┘  │
 └──────────┼─────────────────────────────────┼─────────────────┘
            │                                 │
            ▼                                 ▼
 ┌──────────────────────┐          ┌──────────────────────────┐
 │ Causal World ABI     │          │ UI / model context      │
 │ Symlink→Pod→generation│          │ current semantic render │
 │ capability authority │          │ historical audit plane  │
 └──────────────────────┘          └──────────────────────────┘
```

The pieces are not independent products. They form one execution contract.

---

# 2. Architectural law A — identity is an address, never a mutable value

A governed entity may have many linguistic names, but those names resolve to one canonical logical address.

```text
"Acme"
"ACME Corp"
"the supplier in Lyon"
        │
        ▼
 trusted resolution
        │
        ▼
   PodRef 0x71c9
```

The current fact is another operation:

```text
acquire(PodRef 0x71c9)
        -> Capability<0x71c9, generation=184>
        -> READ(field)
```

A reusable neural plan may know the address or the way to resolve it. It may not copy the current fact into immutable plan state.

This creates a stable semantic pointer across edits and revocations.

---

# 3. Architectural law B — binding equivariance is structural

The old research branch repeatedly encountered a failure mode: a learned network could appear to perform well while memorizing identity/value pairs and then collapse on held-out future bindings.

V4 no longer treats this as primarily a training problem.

For value-semantic operations:

```text
Exec(op, a, b, W) = R(op, W[a], W[b])
```

After address resolution, `R` does not receive canonical identities `a` or `b` unless identity itself is part of the requested semantics.

This means an arbitrary future permutation of entity/value bindings changes runtime inputs but not the operator.

R315 killed the weaker hypothesis that random world-swapping during training alone was sufficient. R317 then showed the structural version on a synthetic gate: the monolithic shortcut control fell to ~28.8% on fully deranged future bindings, while the identity-blind post-resolution path reached ~99.6%.

The machine rule is therefore:

> **Illegal shortcut channels are removed, not merely discouraged by loss functions.**

---

# 4. Architectural law C — mutable information flows one way

CKCA V4 uses a causal dual rail.

```text
      B rail
 reusable skill
 language / plan
      │
      │ allowed
      ▼
      J rail
 current world
 mutable state
      │
      X forbidden
      └────────────► B rail
```

B can influence J.

J cannot modify, contaminate or become a dependency of reusable B cache state.

Conceptually:

```text
∂ B_cache / ∂ mutable_world = 0
```

as a topology constraint.

R318 showed that the first parameterization could satisfy isolation while destroying quality; it was rejected. R318b repaired the J rail with directional gating and recovered ~99.69% deranged-world accuracy while preserving exact B invariance and gaining ~3.17x cached mutable-path speedup in that synthetic gate.

The important invention target is not “a mixture of experts.” It is the **directional lifetime contract**: mutable world data may consume reusable intelligence but cannot flow backward into the state that is supposed to survive a world update.

---

# 5. Architectural law D — neural values are temporal borrows

Inside CKVM, a current world value is not a timeless tensor. Its type includes the generation that authorized it.

Conceptual types:

```text
Static<T>
Plan<T, plan_generation>
Borrowed<T, {pod:g ...}>
Sealed<T, lifetime_factor>
Historical<T, pod:g>
```

A read produces `Borrowed` state.

Borrowed state may participate in the mutable J transaction, but it may not enter a reusable B cache and may not escape as a retained neural artifact.

Before retention it must be sealed against the exact generation set that causally produced it.

```text
Borrowed<T,G> -> B cache               illegal
Borrowed<T,G> -> retained J/KV         illegal
Borrowed<T,G> -> public answer         illegal without commit(G)
Borrowed<T,G> -> commit(G) -> seal(G)  legal
```

R316 instantiated this as a temporal borrow checker and detected all 250,000 injected lifecycle mutations in its bounded test while accepting all 250,000 valid programs.

The compiler should therefore eliminate whole classes of lifecycle bugs before inference begins.

---

# 6. Architectural law E — the neural compiler is untrusted

The Semantic Airlock introduced another problem: discarding compiler KV state is not enough if the compiler can smuggle a stale fact through the plan itself.

For example, a model asked:

```text
What is the capital of France?
```

must not be allowed to compile:

```text
RETURN("Paris")
```

or:

```text
READ(PodRef_of_Paris)
```

based on latent parametric memory.

V4 therefore adds the **Epistemic Plan Firewall (EPF)**.

The neural compiler may propose only:

- source-span references into the user's actual bytes;
- allowlisted schema fields and CKVM opcodes;
- constants proven to originate from user spans or trusted schema constants;
- control/dataflow over those values.

It may not propose:

- direct Pod addresses;
- new factual literals;
- ungrounded entity strings;
- hidden “recover from model memory” opcodes;
- provenance overwrites.

A non-neural trusted linker converts exact source spans into PodRefs.

Thus:

```text
user bytes
   │
   ▼
model proposes span [17:23]
   │
   ▼
trusted linker verifies "France"
   │
   ▼
PodRef 0x...
```

The model cannot invent the address.

R325 stress-tested this membrane with 250,000 valid plans and 250,000 adversarial plan mutants spanning nine smuggling classes; the bounded verifier produced zero false accepts and zero false rejects.

---

# 7. Architectural law F — publication is a causal transaction

A neural reasoning process may be speculative. Its publication cannot be.

Every world read adds an exact generation to the dynamic read-set:

```text
ReadSet = {(pod_1,g_1), ..., (pod_n,g_n)}
```

Immediately before an answer or retained neural artifact becomes externally visible:

```text
for every (pod,g) in ReadSet:
    require authority.current_generation(pod) == g
```

If any source changed:

```text
abort only J
reuse B
refresh capabilities
retry mutable suffix
```

A same-value rewrite remains a conflict. Generation is temporal identity, not a payload checksum.

After a successful commit, the derived artifact is sealed into CLFD so later writes invalidate it transitively.

This gives two distinct coherence moments:

1. **transaction-time coherence** — prevent publication of a world that was never jointly current;
2. **post-transaction admission coherence** — prevent later reuse of a once-valid neural artifact after any source expired.

R313 and R314 supply the current bounded evidence that both are required.

---

# 8. Architectural law G — physical neural memory is not authority

R321 exposed an important distinction.

Keeping stale K/V positions physically present but masked changed exact logits because the logical cache geometry itself changed. A soft attention mask was therefore rejected as the exact coherence primitive.

R321b replaces it with a **Generation-Aware Neural Page Table (GANPT)**.

```text
physical KV backing store
  ├── current pages
  ├── old pages
  ├── revoked pages
  └── pages awaiting GC
          │
          ▼
 generation/lifetime authority
          │
          ▼
 GANPT logical page table
          │
          ▼
 exact logical attention working set
```

The machine invariant becomes:

```text
physical_slot != logical_position != semantic_authority
```

On frozen Qwen2.5-0.5B, R321b retained and aggressively corrupted stale physical pages while excluding them from the logical page table. Direct current splice, page-table execution and full-current recomputation remained full-vocabulary identical (`max delta = 0.0`) across the tested transitions.

This lets authority change first and memory reclamation happen later.

A production implementation should push factor/generation admission into a paged-attention kernel rather than materializing admitted ranges in Python.

This subsystem is the **neural MMU** of the architecture.

---

# 9. Architectural law H — lexical knowledge and governed rendering are separated by an airlock

Pretrained parameters can already contain an old fact. External lifetime metadata cannot erase those weights.

Therefore governed rendering must not depend on the model choosing to ignore its parametric memory.

V4 uses the **Causal Semantic Airlock (CSA)**:

```text
lexical compiler
(sees natural entity names)
       │
       │ only verified typed IR / opaque PodRef
       ▼
semantic airlock
       │
       ▼
fresh governed renderer
(no compiler KV / no entity lexical key)
       │
       ▼
Typed World Slot
(runtime-owned literal)
```

R323 used 12 common facts on frozen Qwen2.5-0.5B. The airlocked renderer contained zero entity/stale-value lexical leaks, compiler-side latent/cache attacks changed renderer logits by exactly `0.0`, and live/revoked slot materialization had zero errors. The stale-value log-probability also fell strongly when the lexical entity key was removed, but that probability shift is descriptive rather than the safety proof.

The safety argument is structural:

- compiler neural state does not cross;
- the compiler cannot smuggle factual constants through EPF;
- the renderer lacks the governed entity's lexical factual address;
- the final governed literal is owned by the committed World ABI result;
- revocation maps to a typed `UNKNOWN`, not to “try to remember.”

---

# 10. Architectural law I — persisted conversation is not flat authoritative text

A lifecycle-safe runtime can still fail if an old assistant message is copied verbatim into a future prompt.

Example:

```text
Turn 4 assistant: "Supplier status = APPROVED"
world update: status -> REVOKED
Turn 20 prompt includes Turn 4 text again
```

The transcript itself has become a stale factual address.

V4 therefore introduces the **Causal Semantic Object Graph (CSOG)**.

A persisted message is a graph/sequence of typed semantic segments:

```text
Text("Supplier status = ")
CurrentSlot(PodRef, field=status)
Text(".")
```

not necessarily the flattened string that the UI showed when the message was first produced.

Historical audit is separate:

```text
HistoricalRef(PodRef, generation=g17)
```

The archival plane can preserve old rendered bytes for audit, but active model context receives an opaque temporal reference unless an explicit historical operation is authorized.

R324 exercised 50,000 semantic messages with 200,000 current slots and 12,000 world updates. Incremental active rendering matched full rerender with zero mismatches, stale archival byte corruption changed the archive digest but did not change active model context, and the average update touched only ~0.040% of messages.

Thus the transcript can stop being a second uncontrolled knowledge store.

---

# 11. Architectural law J — answers can be live semantic views

V4 pushes CSOG one step further.

Instead of eagerly rewriting every old message when the world changes, a governed result can persist as:

```text
LiveView {
    immutable_plan,
    sealed_cached_result,
    lifetime_factor
}
```

On access:

```text
if factor is live:
    reuse cached result
else:
    execute immutable plan against current capabilities
    commit
    reseal
    render current result
```

World updates therefore invalidate lifetimes but need not eagerly rewrite conversation history.

This turns long-lived assistant output into generation-aware live materialized views.

The intended semantics are important:

- a UI may show an archival “what was said then” representation;
- the active reasoning context uses current/live semantic objects;
- historical queries are explicit temporal operations, not accidental reuse of stale text.

This is designed to close the “previously materialized textual state resurrects an invalid generation” hole without making every write proportional to the size of the user's conversation history.

---

# 12. Architectural law K — repair the causal suffix, never weaken freshness

When a generation changes during or after decoding, there are two bad extremes:

1. keep stale neural state for speed;
2. throw away the entire answer/context for safety.

V4 chooses neither.

**Generational Causal Rewind Decoding (GCRD)** records the earliest neural checkpoint at which each generation entered the causal trajectory.

```text
source generation expires
        │
        ▼
reverse lifetime index
        │
        ▼
earliest affected checkpoint
        │
        ├── safe prefix survives
        └── dependent suffix regenerates
```

R319 compared partial replay with full current-world recomputation over 120,000 update trials. In the bounded transducer, both output and generation read-set matched exactly. Moving mutable binding late reduced the mean recomputed fraction on affected updates by ~35.17 percentage points relative to early binding.

The performance rule becomes:

> **Do not make freshness approximate to save latency. Shrink the causal region that freshness governs.**

---

# 13. Model-independent world state

R322 extends the machine boundary across model families.

One canonical Pod generation log was consumed by frozen Qwen2.5-0.5B and SmolLM2-360M without model-specific fact updates or reindexing. Each model owned its own immutable neural materializer, while the world lifecycle existed once.

The architectural consequence is:

> **A model is a consumer/compiler of the World ABI, not the owner of the current fact.**

This is required for a future multi-model agent system. A company fact should be edited once, not separately unlearned from every planner, coding model, voice model and background agent.

---

# 14. CKVM instruction model

A minimal conceptual ISA is:

```text
RESOLVE_SPAN(user_span)          -> PodRef
ACQUIRE(PodRef)                  -> Capability<pod,g>
READ(Capability, field)          -> Borrowed<T,{pod:g}>
CALL(opcode, operands...)        -> Borrowed<U,union(G)>
BRANCH(condition, blocks...)     -> lazy dynamic read-set
COMMIT(read_set)                 -> CommitToken | Conflict
SEAL(value, CommitToken)         -> Sealed<U,factor>
MATERIALIZE(Sealed<U>)           -> current neural representation
PUBLISH(Sealed<U>)               -> semantic object / world slot
HISTORICAL(PodRef,g)             -> Historical<T,pod:g>
```

The following are intentionally absent from the governed path:

```text
READ_MODEL_MEMORY(entity)
DIRECT_POD(model_generated_id)
UNSCOPED_FACT_LITERAL(model_generated_text)
PUBLISH_BORROWED_WITHOUT_COMMIT
REUSE_DEAD_KV_PAGE
```

---

# 15. Epistemic MMU

The combination of capability resolution, CLFD and GANPT is best treated as a single subsystem: the **Epistemic MMU**.

Responsibilities:

- map logical Pod generations to admissible neural pages;
- ensure old physical pages cannot become live merely by existing;
- expose O(1)/small-factor liveness tests on hot paths;
- maintain reverse dependencies for invalidation and GCRD;
- separate authority revocation from asynchronous physical GC;
- prevent same-value ABA resurrection;
- validate snapshot/restart watermarks before re-admitting persisted neural state.

This is a central direction for the production implementation because it moves lifecycle control below prompt orchestration and into the neural memory-management layer.

---

# 16. Three state spaces, not one

V4 explicitly distinguishes:

## 16.1 World state

Canonical, durable, generation-scoped, model independent.

## 16.2 Neural execution state

Ephemeral/derived, model specific, lifetime-carrying.

Examples:

- hidden states;
- KV pages;
- J registers;
- decode suffixes;
- speculative branches.

## 16.3 Semantic interaction state

Persistent conversation/output objects that may outlive an inference call.

Examples:

- current world slots;
- live derived views;
- historical references;
- static explanatory prose.

The old architecture treated interaction state mainly as text. V4 treats it as typed semantic state with its own lifecycle rules.

---

# 17. A complete governed turn

A governed request conceptually executes as:

```text
1. User tokens enter language compiler.
2. Compiler proposes source spans + schema operations.
3. EPF validates provenance and forbids model-origin factual constants.
4. Trusted linker resolves source spans -> PodRefs.
5. B rail compiles/reuses immutable plan state.
6. CKVM lazily acquires current generation capabilities.
7. J rail executes over current typed values.
8. Dynamic generation read-set is recorded.
9. Neural Commit Barrier validates the read-set.
10. Result is sealed into CLFD.
11. Epistemic MMU admits only live neural pages.
12. CSA starts a fresh governed rendering context when lexical parametric recall is dangerous.
13. Typed World Slots own governed literal spans.
14. CSOG stores the answer as semantic objects, not only flattened text.
15. Later world writes invalidate factors/pages/views.
16. GCRD repairs only the earliest dependent neural suffix on active executions.
```

At no point is “the model should prefer the new fact” the coherence mechanism.

---

# 18. Why this may beat RAG for governed knowledge

RAG remains valuable for open-ended document discovery. V4 does not pretend otherwise.

But once a fact has a canonical governed identity, repeatedly transporting it as retrieved text has structural disadvantages:

- retrieval is approximate;
- stale chunks can coexist with current chunks;
- every query pays retrieval/token costs;
- retrieved text has no intrinsic transitive lifetime over downstream KV/hidden states;
- old assistant text can later reintroduce the stale fact;
- multiple models may build separate indexes over the same world.

The target CKCA path is:

```text
query
 -> compile/resolve
 -> canonical Pod handles
 -> current typed capabilities
 -> transactional neural execution
 -> current semantic objects
```

Vector search is demoted from **truth transport** to optional **address discovery**.

A fair benchmark must still prove whether this is actually faster/better on realistic workloads; the architecture alone does not establish superiority.

---

# 19. Current falsification record

V4 intentionally preserves failed experiments.

## Killed / rejected

- **R315:** counterfactual world-swap training alone did not create future binding generalization.
- **R318:** first dual-rail parameterization preserved isolation but quality collapsed to ~58%.
- **R321:** masked in-sequence stale KV overlay kept top-1 but changed full logits by up to 0.125; rejected as exact cache primitive.

## Surviving mechanism gates

- **R312b:** real frozen Qwen late-splice/current-lifetime integration, full-vocab delta 0.0, same-value generation rewrite safety, ~12.46x reduced-gate speedup.
- **R313/R314:** commit-time + post-commit coherence are both required.
- **R316:** temporal lifetime type system catches bounded lifecycle leaks.
- **R317:** structural binding equivariance fixes the held-out binding failure in the synthetic gate.
- **R318b:** one-way B→J rail retains near-control quality and exact B invariance.
- **R319:** causal rewind matches full recomputation in the bounded trace model.
- **R320:** one executable World ABI reference runtime composes canonical identity, generations, commit and lifetime over 200k transactions.
- **R321b:** generation-aware KV page admission is full-logit exact on the real-model gate even with corrupted stale backing pages.
- **R322:** one canonical generation log is consumable by two frozen model families.
- **R323:** semantic airlock removes compiler-state/lexical channels into governed rendering; typed world slots own final literals.
- **R324:** semantic transcript virtualization prevents stale archival bytes from becoming active factual context in the systems gate.
- **R325:** epistemic plan firewall blocks model-latent factual smuggling through compiler IR in the bounded gate.

None of these alone is a novelty or breakthrough claim.

---

# 20. V4 hard DoD

Do not declare the target reached until one integrated implementation demonstrates:

1. multiple real pretrained model families;
2. free multi-token language and long-context interaction;
3. future identities, future values and unseen combinations without gradient updates;
4. edits, revocations, same-value rewrites, restarts, stale replicas, stale transcript bytes, stale KV pages and stale hidden-state attacks with zero resurrection;
5. concurrent updates during reasoning with zero mixed-generation publication;
6. EPF non-interference so compiler parametric memory cannot smuggle factual values/addresses into the plan;
7. CSA/TWS protection so governed factual rendering does not fall back to pretrained stale recall;
8. CSOG/live-view persistence so old conversation text cannot become a second authority path;
9. GCRD or equivalent exact causal repair versus a full-current recomputation oracle;
10. production-serving overhead below the target (<5% control/coherence overhead excluding unavoidable governed task work);
11. normal language/reasoning quality not materially degraded;
12. strong RAG, cached-RAG, GraphRAG, editable-memory and model-editing baselines;
13. economics: update cost, query cost, VRAM/RAM, index maintenance and multi-model scaling;
14. literature + patent audit of the **complete composition**;
15. one-command independent reproduction with pinned revisions/checksums and machine-readable evidence.

---

# 21. The V4 thesis

> **A long-lived neural system should treat mutable knowledge the way a coherent computer treats mutable memory: through explicit addresses, temporal versions, capabilities, typed borrows, transactions, page admission and invalidation—not by repeatedly hoping a statistical model remembers which copy of reality is current.**

And one additional consequence now matters just as much:

> **The lifecycle contract must continue through neural caches and through the conversation/output layer. A fact is not truly revoked if an old KV page or an old assistant sentence can still become an uncontrolled address to it.**

That is the CKCA V4 research target.
