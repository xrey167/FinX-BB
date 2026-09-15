# E-000103 — Low-Rank Resolvent Lifecycle Transport / Generic Woodbury Reduction

Date: 2026-09-06
Status: **PREREGISTERED STRUCTURAL KILL SCREEN — not a novelty claim**

## Trigger

E-000101 showed that ordinary transformer delta propagation becomes dense and session-specific by the first downstream nonlinear MLP. E-000102 then showed that carrying Pod variables symbolically either expands into generic provenance or reduces to generic incremental computation when kept factorized.

A different architecture-level escape remains mathematically plausible: make useful cross-Pod computation live in the solution of a shared linear system whose coefficient operator contains low-rank Pod contributions. Then a canonical Pod revision changes the shared operator by low rank, and cached session solutions might be transported exactly by a Sherman–Morrison/Woodbury update rather than recomputed session-by-session.

This screen asks two questions before any trained-model work:

1. Does the mechanism really provide exact, state-dependent, nonlinear cross-Pod lifecycle transport with favorable fleet scaling?
2. If yes, does the mechanism nevertheless collapse exactly to ordinary low-rank inverse-update / recursive-least-squares machinery and therefore fail the major-invention novelty boundary?

## Registered architecture family

For a shared invertible operator

`M(P) = A + sum_i u_i(P_i) v_i(P_i)^T`,

session `s` carries context vector `b_s` and mixed neural state

`x_s(P) = M(P)^(-1) b_s`.

A Pod revision `p_old -> p_new` replaces one rank-one contribution. The operator delta is rank at most two:

`Delta M = -u_old v_old^T + u_new v_new^T`.

This state is genuinely interaction-bearing: because of the inverse, the contribution of one Pod generally depends on the other Pod contributions and on the session context. It is not the E-000094 static+late-bound factorization.

## Candidate transport

The candidate performs two exact rank-one lifecycle actions:

1. remove the old Pod contribution using Sherman–Morrison;
2. insert the new Pod contribution using Sherman–Morrison.

It updates the shared inverse once and updates every cached session solution from its already-materialized old solution. No per-session fresh linear solve is allowed in the candidate arm.

All arithmetic in the registered assay uses `fractions.Fraction`; equality is exact rational equality, not tolerance-based.

## Strong generic baseline

The guarantee-matched baseline is ordinary dynamic linear algebra. It receives the identical old inverse, old session states and low-rank operator delta and applies a direct rank-2 Woodbury update. A second baseline performs fresh exact Gaussian elimination/solve from the revised operator.

Sherman–Morrison, Woodbury, recursive least squares, pseudoinverse memory updates, associative memory and low-rank model editing receive zero novelty credit.

## Registered domain

- deterministic seeds: 16;
- dimensions: 4, 6, 8, 10;
- Pods per system: 4;
- cached sessions per system: 24;
- revisions: each Pod receives one distinct old->new rank-one replacement;
- lifecycle controls: old state, UPDATE, DELETE old contribution, RESTORE new contribution, and ABA restore to the exact old contribution;
- all generated operators and all intermediate delete/restore operators must be exactly invertible or the deterministic generator retries before the seed is admitted.

## Validity requirements

V1. Every candidate transported session state equals fresh exact recomputation entry-for-entry as `Fraction`.

V2. The independently implemented direct rank-2 Woodbury baseline also equals fresh exact recomputation entry-for-entry.

V3. Candidate and generic Woodbury shared inverse states are exactly equal after the revision.

V4. UPDATE is materially state-changing: at least 95% of registered session states differ from their old state.

V5. Cross-Pod interaction is material: removing a second Pod changes the finite response to the target Pod in at least 95% of admitted systems. This prevents a vacuous additive/noninteraction pass.

V6. DELETE -> RESTORE(new) -> ABA(old) returns exactly to the corresponding fresh operators/states.

## Decision rule

### KILL AS A MAJOR-INVENTION SEAM

Kill low-rank resolvent lifecycle transport **as a standalone FinX-BB invention mechanism** if all exact lifecycle checks pass but the independent generic Woodbury/RLS baseline reproduces the same shared inverse and every transported session state from the same sufficient state.

A successful exact transport result is then a useful architecture/system substrate but not a defensible major technical novelty: the fleet advantage belongs to classical dynamic low-rank linear algebra.

Also kill immediately if fresh 2025–2026 literature plus older direct associative-memory prior art already places Sherman–Morrison/pseudoinverse add-delete memory updates inside neural/LLM memory systems.

### SURVIVE

Survive only if the candidate requires a neural-specific exact sufficient object or update law not available to the strongest generic Woodbury/RLS/pseudoinverse baseline, while retaining the exact fleet update advantage.

## Prior-art boundary to verify this turn

The search must explicitly include:

- Sherman–Morrison / Woodbury neural or associative memory;
- exact add/delete pseudoinverse associative memory;
- recursive least-squares neural memory;
- 2025–2026 online model editing using Sherman–Morrison;
- 2025–2026 transformer/linear-attention memory using Sherman–Morrison;
- patents using low-rank inverse updates for dynamic model/memory state.

Known leads before execution include:

- Hui, Lillo & Zak (1996), *Learning and Forgetting in Generalized Brain-state-in-a-box Neural Associative Memories*: online add/delete memory patterns by updating the pseudoinverse rather than recomputing it;
- Fei, Li & Li (IEEE TNNLS 2025), selective-memory recursive least squares in RBF neural networks;
- Variational Linear Attention (arXiv:2605.11196, 2026), adaptive transformer associative memory with Sherman–Morrison inverse tracking;
- M-ORE (arXiv:2605.20273, 2026), online recursive multimodal model editing with Sherman–Morrison recursion;
- ScopeEdit (arXiv:2607.01978, 2026), scope-aware online editing with branch-wise preconditioners maintained by Sherman–Morrison.

These references are not patent-clearance conclusions. They are baseline/novelty exclusions.

## Major-programme boundary

Passing the algebraic assay cannot promote a major invention. Any successor still needs real LINK->Pod reading >=0.95 on every held-out template, >=3 genuine seeds, >=2 backbone families, <=2% old/deleted leakage, >=90% UNKNOWN in the declared missing-key scope, exact bypass or <=0.05 nats generic divergence, stale Bank/router/resolved-payload/Hidden/KV attacks, UPDATE/RELINK/REVOKE/SHRED/DELETE/RESTORE/ABA/rollback/TOCTOU, key/reconstruction attacks, independent J-space/J-lens audit only, <=5% steady-state inference overhead, matched memory, and a material fleet-level mutation-to-ready advantage over the strongest guarantee-matched baseline.
