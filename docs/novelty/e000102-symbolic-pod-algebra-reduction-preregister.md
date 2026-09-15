# E-000102 — Symbolic Pod Algebra / Generic Incremental-Computation Reduction

Date: 2026-09-06
Status: **PREREGISTERED STRUCTURAL KILL SCREEN — not a novelty claim**

## Trigger

E-000101 closed ordinary operator-local exact delta propagation on frozen DistilGPT-2 and Pythia-70M: even granting all exact work before the first downstream MLP-down for free, the post-nonlinearity correction was effectively dense and full calibration-row-rank. The remaining tempting architectural escape is to redesign the nonlinear computation so mutable Pod variables remain symbolically separable through exact addition/multiplication rather than trying to recover compactness after a normal transformer has destroyed it.

This screen asks whether **symbolic lifecycle algebra by itself** creates a neural-specific invention, or merely instantiates generic provenance / incremental computation.

## Candidate class under test

A lifecycle-native polynomial lift represents a network state as an exact polynomial in mutable Pod variables `x_1..x_n`. Static learned/context coefficients are ordinary scalars. Addition and multiplication in the network propagate the polynomial exactly. A Pod UPDATE/REVOKE changes one variable value and the cached polynomial state is re-evaluated rather than rerunning the original network.

The registered finite families are:

1. additive: `1 + sum_i a_i x_i`;
2. pairwise: additive terms plus all `a_i a_j x_i x_j / 2` interactions;
3. full multilinear product: `prod_i (1 + a_i x_i)`, which contains every interaction subset while remaining exactly factorable as an arithmetic circuit.

All arithmetic uses Python `Fraction`; no floating tolerance is involved.

## Strong baselines

### B1 — generic provenance polynomial

Independently construct the exact coefficient map of the same function as a generic multivariate provenance polynomial. The candidate receives no credit merely for storing Pod dependence inside a polynomial/tensor algebra if the generic provenance representation is identical.

### B2 — generic incremental computation on the compact circuit

Expanded coefficients can be avoided by retaining a factorized arithmetic DAG. Therefore the candidate is also compared against ordinary dependency-directed incremental recomputation of the same compact factor tree. The candidate survives only if its exact update work differs materially from the generic dirty-descendant update using the same circuit.

This comparison is intentionally stronger than dense replay. Generic incremental computation is the correct baseline when a compact exact circuit exists.

## Registered exact checks

For deterministic seeds and Pod counts 4, 6, 8, 10 and 12:

- construct the candidate lift and the independently constructed provenance baseline;
- require exact coefficient-map equality;
- test three one-Pod old→new edits per size (first, middle, last Pod);
- require candidate old state == direct fresh old computation;
- require candidate new state == direct fresh new computation;
- require provenance baseline new state == direct fresh new computation;
- require old state + exact sum of only polynomial terms containing the edited Pod == fresh new state;
- report total symbolic monomials and exact edit-dependent terms;
- for the full-product family, require `2^n` expanded monomials and `2^(n-1)` edit-dependent monomials;
- compare a specialized lifecycle factor-tree update against a generic dependency product DAG, requiring exact output equality and identical multiplication counts.

## Kill rule

Kill **symbolic Pod algebra / coefficient-carrying polynomial lifecycle computation by itself** as a major-invention seam if all registered checks pass and:

1. the expanded candidate state is exactly the same object/function as a generic provenance polynomial;
2. the candidate and provenance baseline require the same edit-dependent expanded terms;
3. high-order interaction causes the registered full-product state to expand as `2^n`, with a one-Pod edit touching `2^(n-1)` terms;
4. retaining the compact factorization avoids that expansion but the specialized lifecycle update is exactly work-equivalent to generic dependency-directed incremental recomputation.

Passing this screen means the representation location and Pod naming earn zero novelty credit.

## Escape condition

E-000102 does **not** rule out a new compact exact representation of useful nonlinear Pod interactions. A successor is interesting only if it demonstrates an exact neural-specific quotient/interaction representation that:

- is materially smaller or cheaper to update than both generic provenance expansion and generic incremental recomputation of the same computation;
- preserves useful nonlinear cross-Pod reasoning rather than avoiding interaction;
- cannot be reduced to ordinary arithmetic-circuit factorization, semiring provenance, dependency graphs, scans/segment trees, low-rank approximation, late binding, passive tags, or selective invalidation;
- then survives the full real-reader, multi-backbone, lifecycle, leakage, UNKNOWN, race/replay, J-space/J-lens and systems gates.

## Prior-art boundary checked before execution

The broad ingredients are already occupied and receive no novelty credit:

- ULD-Net, ICLR 2026, trains fully polynomial networks at ViT/ImageNet scale; polynomial-only neural computation is not new. https://proceedings.iclr.cc/paper_files/paper/2026/hash/0feb70f08090d165e6bbfa9c85d05e87-Abstract-Conference.html
- ProvSQL update provenance (ProvenanceWeek 2025) extends semiring provenance to DELETE/INSERT/UPDATE and undo/history; algebraic update provenance is not new. https://doi.org/10.1145/3736229.3736253
- Zaiser et al., arXiv:2606.05348 / PLDI 2026, compositionally incrementalizes expressive functional programs with correctness arguments; generic incremental computation is a required baseline. https://arxiv.org/abs/2606.05348
- Weighted Rewriting (FSCD 2025) explicitly develops semiring provenance for arbitrary reduction systems. https://doi.org/10.4230/LIPIcs.FSCD.2025.6
- GrapNet (arXiv:2606.18923) makes neural architecture/program structure directly editable; editable neural graphs are not a broad novelty target. https://arxiv.org/abs/2606.18923

The patent search did not identify a single 2025–2026 claim set for this exact Pod-polynomial reduction. That is a search limitation, not evidence of patentability or freedom to operate.

## Non-claims

This is not an impossibility theorem for all lifecycle-native architectures. It does not test language capability, pretrained transformers, real LINK→Pod reading, deletion leakage, or fleet latency. It is a scoped reduction intended to prevent the programme from renaming generic provenance or incremental arithmetic circuits as the invention.
