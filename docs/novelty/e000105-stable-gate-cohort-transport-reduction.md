# E-000105 — Stable-Gate Cohort Exact-Transport Reduction

Date: 2026-09-06
Status: **PREREGISTERED STRUCTURAL KILL SCREEN — not a novelty claim**

## Trigger

Accepted evidence through E-000104 has closed passive freshness, late-binding/noninterference, learned cross-context receipts, associative recomposition, shared linear correction spans, ordinary sparse-coordinate repair, ordinary-transformer layerwise delta propagation, symbolic provenance/incremental arithmetic, low-rank resolvent updates, and TT/MPS local-core updates as standalone major-invention seams.

The next plausible architectural escape is to train a lifecycle-native piecewise-affine network so many sessions share one of a small number of **stable activation signatures**. Inside one fixed activation region, the network is exactly affine in the mutable Pod. A canonical Pod revision could then be transported once per activation cohort rather than replayed independently per session.

E-000105 asks whether that exact cohort transport has any computational content beyond generic exact specialization / forward-mode sensitivity of the same piecewise-affine network.

## Registered candidate class

For each context cohort `z`, the activation masks of all lifecycle-sensitive piecewise-affine layers are invariant across the registered Pod lifecycle values. With context/session input `c_s` and mutable Pod vector `p`, the exact network restricted to that activation region has the form

`y_s(p) = a_s + J_z p`.

The candidate caches the old session output and one exact Pod-to-output sensitivity `J_z` per activation cohort. A canonical mutation `p_old -> p_new` computes one cohort correction

`delta_z = J_z (p_new - p_old)`

and applies that exact correction to every cached session in cohort `z`.

The old cached output is genuinely Pod-dependent. This is not the E-000094 static-cache/noninterference class.

## Strong generic baseline

The baseline receives the **same network, the same stable activation masks/signatures, the same cohort partition, and the same cached old outputs**. It independently obtains the exact Pod sensitivity by ordinary region specialization / forward-mode propagation through the fixed affine maps, caches one sensitivity per cohort, and applies the same lifecycle delta.

The candidate receives zero credit for ReLU/piecewise-affine regions, activation-pattern caching, Jacobians, forward-mode automatic differentiation, partial evaluation, cohorting identical activation patterns, or exact affine specialization.

## Registered exact assay

Use Python `Fraction` arithmetic only; no tolerance-based scientific comparison and no model download.

- 16 deterministic seeds;
- hidden widths 4, 6, 8;
- depths 2, 4, 6;
- 4 stable activation cohorts per network;
- 32 distinct context sessions per network cell, balanced across cohorts;
- Pod dimension 3 and output dimension 3;
- lifecycle states UPDATE, DELETE, RESTORE, ROLLBACK and ABA;
- fresh full network evaluation under the same registered stable activation signature is gold;
- candidate must equal fresh gold exactly;
- generic specialization baseline must equal fresh gold exactly;
- candidate and generic baseline must equal each other exactly;
- mutation-path multiplication counts must match exactly.

The registered stable-gate class is deliberately narrow: if a Pod mutation changes the activation region/signature, E-000105 does **not** cover that transition.

## Non-vacuity / interaction controls

For every network cell:

1. every UPDATE must materially change every registered session output;
2. the Pod-to-output sensitivity must not be identical across all four activation cohorts;
3. at least 95% of registered cross-cohort comparisons must show a different exact correction for the same Pod edit;
4. each network has at least two piecewise-affine nonlinear gating stages;
5. cached old outputs are generated with the old Pod and therefore contain mixed Pod/context state.

A cheap cohort update with no context-dependent Pod interaction is VOID.

## Work comparison

Mutation work is measured after sensitivities/signatures have been cached for both candidate and baseline, because both are allowed identical persistent state. The candidate and generic baseline each pay one `J_z * delta_p` multiplication per cohort plus per-session output application. Fresh-replay multiplication work is separately counted as a descriptive ceiling.

No systems speed claim is inferred from scalar-operation counts.

## Kill rule

Kill stable-gate cohort transport as a **standalone major-invention seam** if:

- all validity controls pass;
- candidate and generic baseline both reproduce every fresh lifecycle output exactly;
- candidate and generic mutation work is identical.

A large fleet-level saving versus independent fresh replay does not rescue novelty if the generic exact specialization baseline obtains the same saving from the identical representation/signature cache.

## Escape condition

E-000105 does not kill Pod-dependent **region transitions**, a representation whose exact update crosses activation boundaries without replay, or a learned structural quotient whose state/update complexity is materially better than the strongest generic exact algorithm given the same state. Such a successor must still beat last-read-site patch + minimal suffix recomputation / exact KV reconstruction and later survive the full real-reader, lifecycle, leakage, UNKNOWN, attack, audit, memory, and <=5% overhead gates.

## Prior-art boundary

Piecewise-affine neural regions, exact affine restriction under fixed activation patterns, Jacobian/sensitivity propagation, partial evaluation, and incremental neural inference are prior-art baseline territory. Current 2026 work continues to study exact affine-region enumeration and stable activation regimes, while existing incremental-computation work already exploits unchanged local structure. No novelty is claimed for those ingredients.

## Non-claims

No real LINK->Pod reader, pretrained-LLM deletion guarantee, J-space authorization role, matched-memory systems benchmark, inference-overhead result, or patentability/infringement conclusion is claimed.