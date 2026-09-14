# Prospective Neural State — Novelty Audit V2

Date: 2026-09-14
Status: **candidate scope narrowed further; no novelty or breakthrough claim.**

## Candidate under audit

The project must not claim novelty for external memory, pointers, dereferencing, late binding, partial evaluation, incremental computation, editable KV, or symbolic/neural hybrid execution.

The only remaining architectural candidate worth testing is narrower:

> A Transformer retained activation is intentionally represented as a **future-world-parameterized neural state** rather than a historical numeric world snapshot: its reusable component contains neural continuation state plus unresolved authoritative references; mutable payload does not enter the decoder-observable retained state until an explicit materialization barrier; after materialization, exact generation dependencies become part of the artifact lifetime and publication is transactionally validated.

In notation:

`kappa_q = (h_q, refs_q)`

with semantics

`[[kappa_q]](W_current) -> H_current`.

The important claim candidate is about **activation semantics inside a Transformer execution/cache boundary**, not about the existence of references.

---

## Prior art that kills broader interpretations

### Neural Random-Access Machines — 2016

Kurach, Andrychowicz, Sutskever introduced a neural architecture that manipulates and dereferences pointers into external variable-sized memory. Corresponding patent family includes EP3362951B1.

**Killed claim:** neural pointer manipulation/dereferencing of external memory is not new.

### Pointer Networks — 2015

Pointer Networks train attention to emit positions/addresses instead of vocabulary classes.

**Killed claim:** attention producing a reference/pointer is not new.

### Pointer-Augmented Neural Memory (PANM) — ICLR/TMLR line

PANM gives memory slots explicit addresses, a learned Pointer Unit, pointer manipulation and dereference modes.

**Killed claim:** explicit neural memory addresses + learned pointer manipulation + dereference are not new.

### Global-to-Local Memory Pointer Networks / patent US11514915B2

Dialogue encoders generate memory pointers and later fill sketch responses from an external knowledge base.

**Killed claim:** carrying a pointer through language generation and filling from external knowledge later is not new broadly.

### Neural Turing Machines / Differentiable Neural Computers

Learned external-memory read/write heads and differentiable addressing are established.

**Killed claim:** a neural model with mutable external memory is not new.

### Self-adjusting / incremental computation

Acar et al. and subsequent work track dependencies, cache intermediate results, and change-propagate after mutable input updates while preserving equivalence to full recomputation.

**Killed claim:** exact dependency-aware incremental repair after input changes is not new.

### Binding-time analysis / partial evaluation

Static/dynamic input separation, residual-program generation and delayed computation have decades of PL literature. ML-specific systems such as StageML further apply partial evaluation to model inference.

**Killed claim:** separating known/query-time computation from future/runtime inputs and producing a residual program is not new.

### Fast weights / external memory / modular neural memory

Fast Weight Programmers, Larimar, Knowledge Externalization, DKME and related systems already separate some mutable knowledge from core model parameters and allow rapid edits.

**Killed claim:** model-independent or rapidly editable mutable knowledge is not new broadly.

### Editable/composable KV and deferred state

Recent work shows KV state can be edited/composed, and DeltaLog-style systems defer folding updates into dense recurrent state.

**Killed claim:** neural-cache editing or deferred materialization itself is not new.

### Residual-stream alternatives

Residual Matrix Transformers treat the residual stream explicitly as a memory bus and replace it with a larger structured memory matrix. Recent KV-Direct work argues residual state can reconstruct KV exactly.

**Killed claim:** changing the representation/role of the Transformer residual stream or treating it as the fundamental state object is not new.

### General late binding / virtual memory / relocation

Systems literature and patents have decades of late-bound symbols, pointers, relocation tables, virtual memory and indirection.

**Killed claim:** late binding, indirection or physical/logical address separation are not new.

---

# What still appears different enough to test

The candidate distinction is the conjunction of the following **inside neural retained state**:

1. **Retrospective numeric state is deliberately avoided** for governed mutable payload.
2. **References remain first-class operands across neural computation**, instead of being immediately dereferenced into the value side of attention.
3. The retained state is semantically a **function of a future authoritative world**, not merely a compressed record of a historical world.
4. A named **materialization barrier** changes the activation's lifetime class: before the barrier ordinary value generations are unresolved; after it the numeric descendant carries the exact generation read-set.
5. Current-world publication is protected by a **generation commit barrier**, including same-value ABA transitions.
6. The same prospective state may remain byte-identical through arbitrary ordinary payload writes while its later numeric denotation changes.
7. The architecture is intended to be a **native Transformer state/cache type**, not merely an external program calling a database.

None of these phrases alone proves novelty. The open question is whether the complete activation semantics already exists under another name in neural-memory, staged-computation, differentiable-programming or architecture literature.

---

# R351 normal-form pressure

The project's Prospective Quotient Theorem gives a useful falsification criterion, not a novelty claim.

If a cache builder `K(q,w_old)` and serve function `S` satisfy exact Universal Future Freshness

`S(K(q,w_old), w) = F(q,w)`

for all historical and current worlds without cache rewriting, then all historical-world cache variants for fixed `q` must be observationally equivalent under all future worlds.

Thus any **decoder-observable** old-world component is incompatible with universal exact freshness unless it is repaired. The useful quotient cache can be represented without historical payload dependence.

This is conceptually close to partial evaluation, memoization and self-adjusting computation; the potential contribution is applying the normal form to neural activation/cache design.

---

# Current evidence relevant to the candidate

## R346

Trainable reference-valued attention produced stable reference descriptors and current-world execution under arbitrary rewrites.

## R347

Across three seeds, the retained `(refs, continuation-code)` state survived six nonlinear world-independent residual blocks. Results:

- descriptor accuracy 1.0;
- future-world accuracy 1.0;
- full-current recompute logit delta 0.0;
- stale numeric control lost ~0.733 accuracy;
- prospective cache ~0.636x matched numeric-cache bytes;
- ~4.99x controlled CPU serve advantage;
- 3162/3162 generation races detected; zero escaped/mismatched.

This is still synthetic.

## R349/R350

The decisive follow-up is underway: insert the materialization barrier into a pinned frozen Qwen residual path, then learn a current-world World Port and test language output.

---

# Novelty kill criteria

The candidate should be killed or renamed as an application/composition if direct prior art is found that already provides all or nearly all of:

- Transformer hidden/residual state carrying unresolved external references as a native activation type;
- references surviving multiple neural blocks without value dereference;
- retained activation explicitly parameterized by future mutable memory contents;
- explicit later neural materialization into current numeric hidden state;
- update-stable cache bytes across external value revisions;
- version/generation-scoped validity for the materialized descendants.

Finding ordinary pointer networks, memory tokens, retrieval, tool calls, external memory, staged computation or cache editing is **not** sufficient to establish this exact prior art—but it does prevent broad claims.

---

# Current verdict

**No breakthrough claim. No established novelty claim.**

The candidate has, however, become technically specific enough to be falsifiable:

> **Prospective Neural State (PNS): a future-world-parameterized Transformer activation/cache type whose retained decoder-observable state excludes historical mutable payload and whose explicit materialization transition acquires exact temporal knowledge lifetimes.**

The decisive remaining novelty work is a direct scholarly + patent search using that semantic definition, combined with R349/R350 pretrained-model evidence and strong RAG/editable-memory controls.
