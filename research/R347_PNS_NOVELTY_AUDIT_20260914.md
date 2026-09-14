# R347 novelty audit — Prospective Neural State / Prospective Transformer

Date: 2026-09-14
Status: **candidate novelty survives this audit; not yet an established novelty claim**.

## Candidate claim under audit

The claim is intentionally narrow:

> A Transformer-style attention/residual activation may be retained as a **Prospective Neural State (PNS)** — a first-class neural object containing live canonical references plus a learned continuation/operator state, whose denotation is a function of the future authoritative world `kappa(W_current) -> H_current`. Mutable `V` payload is not copied into the residual/KV state until an explicit materialization barrier. The materialized numeric state is generation-scoped and transactionally validated; the prospective state itself can survive arbitrary payload rewrites without value-derived patching or invalidation.

This is stronger and narrower than “an LLM with external memory”, “pointer attention”, “editable memory”, “lazy evaluation”, or “KV-cache editing”.

## Prior art that kills broader claims

### Neural Turing Machines / Differentiable Neural Computers

Memory-augmented neural networks already learn to address mutable external memory with read/write heads. Therefore **external mutable neural memory and learned addressing are not novel**.

Relevant lineage: Graves et al., Neural Turing Machines / Differentiable Neural Computers.

### Pointer Networks and pointer-like attention

Attention can already be trained to emit positions/addresses rather than ordinary class values. Therefore **attention producing a pointer/reference is not novel by itself**.

### Fast Weight Programmers / linear attention

Fast-weight work establishes neural controllers that dynamically write and read key/value associations. Therefore **dynamic neural key/value memory and learned memory programming are not novel**.

Schlag, Irie, Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers*, ICML 2021.

### Larimar

Larimar provides a distributed episodic memory for LLMs with one-shot updates and selective forgetting. Therefore **editable external neural memory, fast fact updates and selective forgetting are not novel**.

Das et al., *Larimar: Large Language Models with Episodic Memory Control*, ICML 2024.

### Knowledge Externalization

ICLR 2026 work externalizes knowledge into memory tokens and supports reversible unlearning, dynamic editing and composition. Therefore **externalizing knowledge into modular editable neural memory tokens is not novel**.

Li et al., *Knowledge Externalization: Reversible Unlearning and Modular Retrieval in Multimodal Large Language Models*, ICLR 2026.

### DKME

DKME explicitly argues for decoupling semantic addressing from partitioned storage for lifelong model editing. Therefore **decoupling routing/addressing from stored mutable knowledge is not novel on its own**.

Zheng et al., *DKME: Rethinking Coupled Knowledge Memory for Lifelong Model Editing of Large Language Models*, Findings of ACL 2026.

### Programmable / editable KV cache

Li 2026 demonstrates that existing LLM KV caches can be edited and composed, including append-only errata and portable precompiled KV notes. Therefore **editable/composable KV caches or avoiding a full prefill after an edit are not novel**.

Bojie Li, *Models Take Notes at Prefill: KV Cache Can Be Editable and Composable*, arXiv:2606.17107.

Important difference from PNS: programmable-KV repairs or supplements an already materialized historical KV notebook. PNS tries to avoid binding mutable world payload into the retained activation in the first place.

### DeltaLog

DeltaLog represents a recurrent linear-attention state as a dense base plus deferred compact updates. Therefore **deferred materialization of neural/recurrent state is not novel as a systems idea**.

Lin, Sun, Sun, *DeltaLog: Deferred Materialization of Recurrent States for Linear Attention Decoding*, arXiv:2608.15533.

Important difference from PNS: DeltaLog defers physically folding known historical state updates into a dense tensor. It does not, as described in the paper, retain a neural activation whose semantic value is intentionally late-bound to future revisions of an external authoritative world.

### Exact deletion / support-vector memory

Recent work provides addressable language-model memories with exact deletion and shows that exact deletion depends on memory representation. Therefore **addressable influence, exact record deletion and generation-independent deletion certificates are not broad novelty claims for this project**.

Ramesh, *Forgetful Attention: A Trainable Support-Vector Memory with Certified Selection and Exact Unlearning*, arXiv:2607.12204.

Ramesh, *Subtract or Replay? Exact Deletion from Language-Model Memory*, arXiv:2607.27539.

## What remains different after this audit

The surviving candidate is not “memory”. It is a proposed **neural activation semantics**.

Ordinary numeric hidden state:

`H_t = F(q, W_t)`

is retrospective. Once cached, it denotes a computation already bound to world snapshot `W_t`.

PNS instead retains:

`kappa_q = (refs, operator/continuation state, authority contract)`

with denotation:

`[[kappa_q]](W) = F_bound(q, deref(refs, W))`.

The object itself is therefore intentionally **world-parameterized**. A later world update `W_t -> W_t+1` does not edit the PNS object. The same object acquires a different numeric denotation when materialized under the new authoritative world.

The strongest architectural distinction is:

1. attention can return live references on its value side rather than immediately returning `sum alpha_i V_i`;
2. those references remain first-class operands across multiple neural blocks;
3. world-independent query/skill computation continues while mutable payload remains unresolved;
4. an explicit materialization barrier converts prospective state into numeric state;
5. the materialized state inherits exact `(reference, generation)` lifetimes and must pass a commit check before publication;
6. cached prospective state is reusable across arbitrary payload rewrites with zero value-derived cache maintenance.

This resembles partial evaluation, closures, query plans and lazy tensors at a computer-science level. **Those abstractions are prior art.** The unresolved novelty question is whether their integration as a native Transformer attention/residual activation type over authoritative mutable knowledge has direct prior art or would be considered an obvious composition.

## Evidence accumulated in R340–R346

The following are mechanism gates, not novelty proof:

- R340: affine live-reference closure; exact current-world equivalence; zero derived-cache patching on writes.
- R341: reference-valued attention/residual stream; exact current-world equivalence; zero write fanout, but poor cache-size efficiency in the dense descriptor.
- R342: nonlinear polynomial referential state; exact nonlinear equivalence but severe storage/runtime growth, therefore not practical.
- R343: compact referential continuation; descriptor smaller than a BF16 numeric hidden cache and faster than full recompile on the controlled test.
- R345: bounded temporal model check; generation commit prevents mixed/stale publication including same-value ABA cases.
- R346: **trainable** reference-valued attention. Across the executed gate, full descriptor accuracy and future-world accuracy were 1.0; a stale numeric cache averaged ~0.268 accuracy; 1,070/1,070 injected races were detected with 0 escaped and 0 semantic mismatches. The descriptor digest stayed unchanged after world updates.

R347 now tests the missing architectural bridge: a retained reference-valued state surviving multiple nonlinear **world-independent residual blocks**, compared directly with a matched early-materialized numeric cache.

## Falsifiers still required before any breakthrough claim

A breakthrough/major-contribution claim is **not** justified until all of the following survive:

1. **Direct-prior-art audit:** scholarly + patent search aimed specifically at function-valued / reference-valued hidden activations, not just memory editing.
2. **Pretrained Transformer intervention:** at least one open pretrained LLM where a real attention/value path is changed to emit/reference live world cells rather than only a synthetic network.
3. **Language generalization:** held-out paraphrases and entity aliases; no exact-template shortcut.
4. **Reasoning:** multi-hop tasks whose intermediate conclusions depend on mutable cells and remain current after edits/deletions.
5. **Deletion:** record-omitted reference equality for the externalized governed knowledge path, including derived reasoning state.
6. **No bypass:** probes must show the answer cannot be recovered from stale parametric/internal state when the governed channel is revoked.
7. **Performance:** prospective cache must beat full recomputation and a strong RAG/GraphRAG-style current-world baseline under realistic update/query ratios; synthetic Python timing is insufficient.
8. **Utility:** normal language inference must not materially degrade.
9. **Scale:** millions of knowledge cells / large query caches, not only hundreds of Pods.
10. **Independent seeds and adversarial schedules:** race/restart/ABA/crash/revocation tests with durable authority state.

## Current verdict

**Do not claim “breakthrough” yet.**

However, this audit changes the research position: the project now has a considerably narrower candidate than the earlier Symlink/Pod/RAG framing. The most defensible research object is **Prospective Neural State / reference-valued Transformer activations**, not generic editable memory.

The next decisive experiments are R347 (matched prospective-vs-numeric residual state), then a pretrained-model R348/R349 in which the same semantic object is inserted into an actual Transformer path and compared against strong RAG and programmable-KV controls.
