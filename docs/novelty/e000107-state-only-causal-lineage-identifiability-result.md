# E-000107 result — state-only exact causal lineage is non-identifiable in the registered class

Date: 2026-09-06
Status: **DECISIVE SCOPED GUARANTEE KILL / no invention promoted**

## Decision

E-000107 kills **state-only exact neural causal-lineage recovery/certification as a general lifecycle guarantee** when the certifier receives only `(model, complete source catalogue, observed neural state, lifecycle target)` and no execution history/provenance, no counterfactual replay/intervention, and no additional injective lineage state.

The reduction is elementary: if two admissible source histories give exactly the same observable state under the same model and catalogue, yet deleting the same target requires different exact fresh states, a deterministic state-only certifier receives identical inputs but is required to produce two different correct answers.

## CI evidence

Registered source commit: `32b353a3d418d4d0936b34611d092ba5f280e139`.
GitHub Actions run: `34012157290`.
Job conclusion: success.
Focused regressions: **4 passed**.
Artifact ID: `9982791984`.
Uploaded artifact ZIP SHA-256 reported by Actions: `e249779304595bd1089aaf21422f03dbda9fcb161b2748b04da999d670732afb`.

Exact Python `Fraction` enumeration covered four distinct mixed-state families, each with a fixed complete source catalogue and all 32 presence/absence histories over five sources:

1. additive subset-sum state;
2. uniform associative/attention-style average;
3. ReLU-saturated sum;
4. low-dimensional projected vector state.

Result:

- **4 / 4 families** contain exact indistinguishable-history witnesses;
- **20 target/family witness slots** were found (all five targets in every family);
- oracle history/provenance resolves every registered ambiguity: **0 oracle-history failures**;
- kill screen: **PASS**;
- decision: `KILL_STATE_ONLY_EXACT_CAUSAL_LINEAGE_AS_GENERAL_GUARANTEE`.

Example additive witness with the same complete catalogue and target source 0:

- history 1: only source 4 active;
- history 2: sources 0, 1 and 3 active;
- observed state in both worlds: exactly `5`;
- after deleting target 0: fresh state is `5` in world 1 but `4` in world 2.

The state-only certifier's entire registered input is identical in the two worlds, so exact lineage/deletion repair is not identifiable from that input.

The other families independently provide the same type of witness, including nonlinear ReLU saturation and a low-dimensional vector projection.

## Consequence

Do not allocate major-invention budget to a decoder that promises an **exact** source-set/lifecycle certificate from already-mixed neural state alone unless the architecture first establishes an identifiability condition.

A future exact lineage mechanism must obtain the missing information through at least one of:

- explicit provenance/dependency/history state;
- an injective or provenance-preserving neural representation;
- active counterfactual intervention/replay over retained source state;
- an authenticated execution trace/certificate.

If the mechanism simply stores source identity/lineage in an added latent field, it must be compared against ordinary provenance carrying the same sufficient information. Calling it a biomarker, causal code or neural certificate does not by itself create a stronger guarantee.

## External boundary

2026 Mechanistic Data Attribution (`arXiv:2601.21996`) traces LLM units toward training samples using influence functions and validates influence via interventions, but it does not provide the state-only exact source-set lifecycle certificate ruled out here.

Recent identifiability work also warns against uniqueness assumptions in neural explanations: `Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable?` (`arXiv:2502.20914`) exhibits multiple valid mechanistic explanations, while 2026 work on steering-vector identifiability shows behaviorally equivalent internal interventions can be non-unique without additional structural assumptions. These references support the boundary but are not the proof; the E-000107 kill follows from the registered exact indistinguishability construction.

## What is not killed

E-000107 does not kill:

- approximate influence ranking;
- active intervention-based causal discovery;
- exact lineage under a proven injectivity/identifiability condition;
- provenance-preserving architectures that demonstrate a measured advantage over generic provenance/dependency tracking;
- authenticated lineage certificates emitted during computation when their system/security advantage is independently established.

## Surviving programme frontier

After E-000106 and E-000107, the two broad late-stage escape hatches are narrower:

1. **Transport:** mutation-time compact exact nonlinear sufficient state derived from old session state + Pod edit, without fresh-target precomputation, region enumeration or suffix replay, and not reducible to generic incremental/piecewise-affine evaluation.
2. **Lineage:** exact causal lineage requires new information/structure; state-only post-hoc recovery is not a general guarantee. A candidate must establish and pay for an identifiability/provenance invariant, then beat the guarantee-matched generic provenance baseline.

## Major-break status

**No major invention is promoted.** All full reader, lifecycle, leakage, UNKNOWN, stale-state/race/reconstruction, independent J-space/J-lens, matched-memory, <=5% steady-state overhead and fleet-level mutation-to-ready gates remain required.
