# E-000107 — State-Only Exact Causal-Lineage Identifiability Reduction

Date: 2026-09-06
Status: **preregistered scoped impossibility/kill screen; not a novelty claim**

## Trigger

After E-000106, one parallel surviving programme seam is neural causal-lineage discovery/certification not supplied as ordinary provenance metadata. Before spending trained-model budget on a decoder that tries to recover exact source lineage from mixed neural state, test the information-theoretic identifiability requirement.

## Question

Can an exact lifecycle certifier recover the source lineage needed for deletion/repair from the **current neural state itself**, even when it knows the model and the complete source catalogue, but receives no history/provenance record and performs no counterfactual replay/intervention?

## Candidate class

A state-only certifier is a deterministic function

`C(model, source_catalogue, observed_state, lifecycle_target)`

that must return enough exact lineage information to determine the fresh post-lifecycle state (equivalently, whether/how the target source contributed) for every admissible history in the registered domain.

The certifier may know all source identities and their fixed content/contributions. It does **not** receive which sources were actually admitted, an execution trace, a dependency graph, a generation log, or a provenance certificate.

## Identifiability reduction

If two admissible histories `h1 != h2` satisfy

`State(model, catalogue, h1) == State(model, catalogue, h2)`

while the same lifecycle request produces different exact fresh states,

`FreshAfterDelete(target, h1) != FreshAfterDelete(target, h2)`,

then the certifier receives exactly the same input in both worlds but is required to return two different correct answers. No deterministic state-only certifier can be exact on both.

This is elementary indistinguishability, not a new theorem.

## Registered exact families

Use Python `Fraction` arithmetic and exhaustively enumerate all source-presence subsets.

1. **Additive mixed state** — fixed scalar source contributions are summed. The catalogue intentionally contains subset-sum collisions.
2. **Uniform associative/attention-style average** — active source values are averaged exactly; different active subsets can have the same mixed state.
3. **ReLU-saturated mixed state** — apply `max(0, sum(contributions))`; nonlinear many-to-one state creates additional collisions.
4. **Low-dimensional projected vector state** — fixed rational source vectors are summed then projected to fewer observable coordinates, producing exact source-history collisions.

For every family, enumerate every target source and every pair of histories sharing the same observable state. Record a decisive witness whenever target deletion has different fresh outcomes in the two histories.

## Kill rule

Kill **state-only exact neural causal-lineage recovery/certification as a general lifecycle guarantee** if:

- every registered family contains at least one exact indistinguishable-history witness;
- at least four distinct target sources across the assay have witnesses;
- all witness pairs use an identical model, identical complete source catalogue, identical observed neural state and identical lifecycle target, but require different exact post-delete states;
- an oracle lineage/history bit resolves every registered ambiguity, confirming that the failure is missing information rather than arithmetic inability.

## Strong baseline consequence

A future exact causal-lineage mechanism must obtain the missing information from at least one of:

- explicit provenance/dependency/history state;
- an injective/provenance-preserving neural representation whose additional state must be charged to memory/overhead and compared with generic provenance;
- active counterfactual intervention/replay over retained source state;
- an external authenticated execution trace/certificate.

Calling an added lineage field a neural biomarker, causal code or latent certificate does not avoid the baseline comparison if it is simply stored provenance.

## What is not killed

This does not kill:

- approximate influence ranking or mechanistic attribution;
- active intervention-based causal discovery;
- exact lineage under structural assumptions that make history identifiable;
- a provenance-preserving architecture that beats generic provenance/dependency tracking in a measured systems dimension;
- causal certificates generated during computation and cryptographically/authentically bound, provided the claim is not merely that metadata exists.

## Programme gates unchanged

No major invention may be promoted without the real LINK->Pod reader >=0.95 on every held-out template, >=3 genuine seeds, >=2 backbone families, <=2% old/deleted leakage, >=90% scoped UNKNOWN, exact bypass or <=0.05 nats generic divergence, complete stale-state/lifecycle/race/reconstruction battery, independent J-space/J-lens audit, matched memory, <=5% steady-state inference overhead, and material fleet-level mutation-to-ready advantage over the strongest guarantee-matched baseline.
