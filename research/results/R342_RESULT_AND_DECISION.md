# R342 Result — Causal Referential Polynomial Tensor (CRPT)

Status: **nonlinear referential-neural-state gate passed; exact but currently too expensive in storage/materialization to be a practical final design. No novelty claim yet.**

## Why R342

R340/R341 were exact because the mutable-world influence stayed affine. That is not enough for general reasoning. R342 tests whether a retained neural state can stay referential even after genuine cross-value multiplicative interactions.

The hidden result is compiled into a polynomial whose variables are live canonical world references. It stores constant, linear and quadratic neural coefficients but no current world values/generations. Values are dereferenced only at materialization and validated transactionally.

## Result

- Pods: **4,096**
- query programs: **2,500**
- live refs/query: **6**
- hidden dimension: **10**
- output dimension: **8**
- explicit quadratic neural terms: **6**
- world updates: **50,000**
- same-value updates: **17,442**
- transactional serves: **100,000**

Correctness:

- semantic mismatches vs full current nonlinear network: **0**
- max absolute error: **1.776e-15**
- expected generation races: **7,935**
- detected: **7,935 / 7,935**
- escaped: **0**

Nonlinearity check:

The best fitted affine control for the same query/network had held-out RMSE **0.1618** and max error **2.0772**, so this gate is not reducible to the affine R340 representation.

Update behavior:

- derived cache invalidations/patches on world writes: **0**
- conventional numeric-cache invalidations counterfactual: **182,895**
- write-fanout elimination: **100%**

Important negative results:

- polynomial materialization median: **69.888 µs**
- full nonlinear numeric network median: **25.330 µs**
- materialization/full ratio is therefore unfavorable (~2.76x slower)
- numeric cache counterfactual: **160,000 bytes**
- referential polynomial cache: **6,940,000 bytes**
- storage ratio: **43.375x**

Report SHA256: `cb337b665a5e0c4355e715c5761b77c950af6d6b739c597c217c3497127fa5e5`.
Artifact ZIP SHA256: `ca074364ecf0ead44785570ffcb803d4ba6293d3b20d8a1d6054eebf4c135a43`.

## Decision

CRPT demonstrates that the referential-state concept can extend beyond affine world influence, but **expanded polynomial coefficients are not the practical solution**. Degree/support growth is exactly the failure mode we need to avoid.

The next design should keep nonlinear referential state in a **factorized operator graph / compact continuation form**, sharing global neural weights and carrying only sparse live references plus a small query-conditioned code. The semantic breakthrough target remains: retain a neural state that is a function of future world state without paying expanded symbolic-expression storage.

Polynomial neural networks and symbolic polynomial evaluation are established. The possible novelty is not polynomial algebra itself; it is the use of live generation-scoped world references as variables of a retained neural activation. This still requires a stronger literature/patent audit and a practical compact representation before any claim.
