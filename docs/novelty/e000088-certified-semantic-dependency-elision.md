# E-000088 — Certified Semantic Dependency Elision

Status: preregistered falsification experiment; **not a novelty claim**.
Date: 2026-09-05

## Why this experiment exists

E-000086R killed the previous heterogeneous-coherence seam: an explicit alias-generation + Pod-generation witness makes the same reuse/invalidate decisions as an ordinary generic dependency snapshot for Bank, router, resolved payload, post-read hidden state and KV. Therefore explicit knowledge-object freshness tags are not a neural-specific invention.

The remaining question is stronger than dependency tracking:

> Can a neural artifact be syntactically dependent on a Pod, while being *semantically independent* of the Pod over the complete declared mutation domain, so that a generic dependency engine must invalidate it but a neural certificate can prove exact reuse safe?

If no useful population of such states exists, this direction is killed.

## Candidate mechanism under test

For a derived neural artifact `A` and a canonical Pod `P`, define a declared finite mutation domain `D(P)` (for the first experiment: the complete synthetic payload domain used by the model).

A **Non-Influence Certificate** for `(A,P)` is valid only if

`A(P := v) == A(P := v0)` for every `v in D(P)`

under the same query, other Pods, lifecycle state and model weights.

The first implementation is intentionally expensive: exhaustive enumeration of the declared domain. It is a correctness oracle, not the proposed production algorithm.

A generic dependency baseline records that `A` read/was downstream of `P`; any generation change of `P` invalidates `A`.

The semantic baseline may elide that dependency **only** when the exhaustive certificate passes. No tolerance-based equality is allowed for the primary exact synthetic arm; floating-point diagnostics are reported separately.

## Why this is not already claimed as an invention

Semantic dependence / non-interference, incremental computation, dependency graphs, memoization, abstract interpretation, program slicing and from-scratch-consistent incremental recomputation are established fields. The repository's own E-000030 deletion certificate already proves payload independence in a narrower deletion setting. Therefore:

- semantic dependency analysis is not novel;
- exhaustive payload sweeps are not novel;
- cache reuse after proving non-influence is not claimed as novel;
- J-space / J-lens is not novel and is not used to make the certificate pass.

A future technical novelty is possible only if a **neural-specific, materially cheaper certificate** can replace the exhaustive oracle and beat the strongest guarantee-matched dependency/recompute baseline at useful scale.

## Experiment A — existence / utility kill screen

Use the trained synthetic Symlink reader because its payload domain is finite and the complete mutation domain can be enumerated.

For multiple seeds, Pods, query classes and artifact classes:

1. capture the original artifact;
2. mark the target Pod as a syntactic dependency whenever it is in the consumed Bank / routing computation;
3. enumerate every allowed object payload for that Pod without changing any other input;
4. recompute the artifact for each value;
5. classify the `(artifact, Pod)` pair as semantically independent only if all values are identical under the preregistered equality rule;
6. separately include positive controls in which the queried Pod is known to determine the answer, so a vacuous certificate cannot pass;
7. compare invalidation counts against a generic versioned dependency baseline and full recomputation.

Artifact classes, in order of priority:

- routing decision/distribution;
- resolved payload;
- post-read hidden state;
- logits / answer;
- later KV when a public-backbone integration is available.

## Preregistered validity controls

V1. **Real dependency control:** at least 95% of positive-control queried-Pod cases must change under some payload value. Otherwise the mutation instrument is void.

V2. **Full-domain control:** every payload in the declared synthetic payload domain is enumerated; sampling does not count as a certificate.

V3. **No audit optimization:** no J-space/J-lens metric is used in training, routing, equality, or certificate construction.

V4. **Counterfactual reference:** every reuse decision is checked against direct recomputation from the mutated Bank.

V5. **Lifecycle separation:** ACTIVE, REVOKED/SHRED and absent-Pod cases are reported separately. A hard-gated absent payload cannot be used to claim that an ACTIVE payload is irrelevant.

## Decision rule

### KILL

Kill this seam if either:

- zero or a negligible preregistered fraction (< 5%) of syntactically dependent ACTIVE-state artifact/Pod pairs are exactly semantically independent; or
- semantic elision does not reduce exact recomputation work by at least 2x on the registered workload; or
- any certified reuse differs from direct recomputation.

A KILL means the project should stop pursuing semantic dependency elision as the invention and return to revision-native representation / affected-cone reconstruction.

### SURVIVE AS RESEARCH DIRECTION

The seam survives only if all validity controls pass, **zero false reuse** is observed, and semantic certificates cut exact invalidations/recomputations by >= 2x over generic dependency tracking on the registered workload.

This still does **not** establish novelty. It only establishes that there is a useful neural phenomenon worth replacing the exhaustive oracle with a production certificate.

## Experiment B — only if A survives

Design a cheap certificate that predicts/proves the exhaustive result without enumeration. Candidate families to test against one another:

1. analytical nullspace / structural certificate;
2. interval or Lipschitz bound over the declared payload set;
3. local activation-region certificate with exact fallback;
4. source-scoped Jacobian/JVP bound **with a finite-edit validity guard**;
5. J-space only as an independent post-hoc audit, never as the certificate target.

Required production-level target before any major-technology language:

- zero false reuse against exhaustive/full-rebuild reference on the certified domain;
- >= 10x lower mutation-to-ready cost than the strongest guarantee-matched recomputation baseline at scale;
- <= 5% normal inference overhead;
- >= 3 seeds and >1 public backbone where feasible;
- real Symlink reading/lifecycle gates retained;
- stale Bank/router/payload/hidden/KV replay attacks retained;
- independent J-space audit retained.

## Stronger invention question if this survives

The potentially distinct question is not "can we track a Pod dependency?" It is:

> Can a live neural memory attach compact, source-scoped **counterfactual non-influence certificates** to persistent internal computation, allowing exact reuse across knowledge revisions that would force ordinary dependency systems to invalidate, while falling back to recomputation whenever the certificate cannot prove invariance?

That question remains unproven and must be attacked against semantic incremental computation / program-analysis prior art before any novelty claim.
