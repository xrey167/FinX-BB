# R341 Result — Reference-Valued Attention + Referential Residual Stream

Status: **reference-valued neural activation gate passed; this is a stronger novelty track than the earlier lifecycle-wrapper work, but practicality is not yet good enough because the first compiled representation is memory-heavy. No novelty claim yet.**

## Primitive

Standard attention computes numeric `softmax(QKᵀ)V` immediately. R341 keeps the selected value side as live canonical references plus stable coefficients instead of immediately collapsing mutable world values into a numeric hidden vector.

The resulting referential residual stream is propagated through eight B-conditioned, world-independent residual blocks. Current world values are dereferenced only at a materialization barrier. The cached object contains neither mutable value bytes nor last-seen generations.

## Result

- Pods: **4,096**
- cached queries: **2,400**
- attention heads: **4**
- top-k/head: **3**
- world-value dimension: **8**
- hidden dimension: **16**
- reference-preserving residual blocks: **8**
- world updates: **50,000**
- same-value generation updates: **17,598**
- transactional serves: **100,000**

Correctness:

- semantic mismatches vs full numeric network: **0**
- explicit reference-sum mismatches: **0**
- max absolute error vs full numeric network: **7.994e-15**
- expected injected generation races: **7,995**
- detected: **7,995 / 7,995**
- escaped: **0**
- same-value race conflicts: **3,636**

Update behavior:

- derived cache invalidations/patches on ordinary world writes: **0**
- numeric-cache invalidations that would otherwise have been required: **351,145**
- write-fanout elimination: **100%**

Reference Python timing:

- median canonical write: **1.714 µs**
- median referential-stream materialization: **40.156 µs**
- median full numeric network: **51.908 µs**
- full/reference materialization ratio: **1.293x**

Important negative result:

- counterfactual numeric cache size: **307,200 bytes**
- first compiled referential cache: **5,568,000 bytes**
- referential/numeric storage ratio: **18.125x**

Report SHA256: `ad75234ab80e734d1cdcdce3debf847fe0de3ca89d729e2001437a6750a3bfa2`.
Artifact ZIP SHA256: `7c9dadcbfa7ff490f0edb52d6e660ac569e2c34beae6f9f72f9fd37b8f734c13`.

## Decision

Promote the **activation semantics**, reject the first dense compiled encoding as the final representation.

The important result is that an attention-derived hidden state can remain an exact neural function of *future* mutable world values across multiple residual blocks while incurring zero write-time maintenance. But storing a dense query-specific post-transform defeats the practical memory target.

The next architecture must keep the referential state factorized: sparse `(PodRef, coefficient)` support plus a compact continuation/operator code that reuses shared network weights, rather than expanding the continuation into a dense matrix per query.

The direct prior-art question is now much narrower than earlier CKCA work: **is there prior neural architecture where attention values are live mutable references, remain undereferenced as first-class hidden-state operands across layers, and cached hidden state is therefore a function of future external world state?** The searches so far found attention-weight propagation, external memories, symbolic tensors and ordinary pointer/reference mechanisms, but not this exact state semantics. That is not yet sufficient for a novelty claim.
