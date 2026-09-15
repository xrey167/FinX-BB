# E-000112 Result — Finite Koopman Lift Reusable Revision Receipt Reduction

## Decision

**KILL_FINITE_EXACT_KOOPMAN_CARLEMAN_LIFT_AS_STANDALONE_LIFECYCLE_TRANSPORT_NOVELTY**

This is a decisive scoped falsification, not a major-invention promotion and not a universal impossibility claim.

## Registered evidence

- Registered source/workflow head: `9d02db10d4b5ae035a0894c233a58273c50af79a`
- GitHub Actions run: `34025749155`
- CI conclusion: `success`
- Artifact: `e000112-finite-koopman-receipt-reduction`
- Artifact digest: `sha256:5c8aedc0b54deca33dfe0282852c8c7f1553a14bef874b70f879d260271d8cb2`
- Result JSON SHA-256: `63424d877b8cac0bd33b14f7a4a4645155ee96569167b8b9f3bdf2ef48dfcee5`

The registered exact `Fraction` assay executed 16 deterministic seeds × six nonlinear depths (`1,2,4,8,16,32`) × 64 cached sessions, with sequential `UPDATE -> DELETE -> RESTORE -> ABA` lifecycle mutation.

| Metric | Result |
|---|---:|
| Seed/depth cells | 96 |
| Exact lifecycle cases | 24,576 |
| Candidate → fresh mismatches | 0 |
| Generic Koopman → fresh mismatches | 0 |
| Candidate/generic receipt mismatches | 0 |
| Candidate/generic state mismatches | 0 |
| Candidate/generic normalized-work mismatches | 0 |
| Material hidden-state changes | 24,576 / 24,576 |
| State-dependent edit cells | 384 / 384 |
| Candidate normalized work | 113,664 |
| Generic normalized work | 113,664 |
| Fresh suffix-replay normalized work | 1,806,336 |
| Aggregate replay/candidate advantage | **15.8918918919×** |
| Minimum cell replay advantage | **3.1111111111×** |
| Maximum cell replay advantage | **22.4×** |
| Candidate lifted-state slots | 36,864 |
| Generic lifted-state slots | 36,864 |

## What the candidate achieved

The original-coordinate dynamics were deliberately nonlinear:

\[
x' = \lambda x,\qquad c'=c,
\]

\[
y'=\mu y+\nu xc+\rho x^2+b+\eta c.
\]

Yet the bounded observable state

\[
\phi=(1,x,c,xc,x^2,y)
\]

is exactly closed under a six-dimensional linear Koopman operator. A canonical Pod edit therefore factors at the read site as

\[
\Delta\phi_0=A_0+cB_0,
\]

and one edit-level compilation produces

\[
(A_d,B_d)=(K^dA_0,K^dB_0).
\]

Every cached session is then repaired exactly by

\[
\phi'_s=\phi_s+A_d+c_sB_d.
\]

Because `B_d[y] != 0`, the hidden correction is genuinely session/context dependent; this is not merely a global translation vector. In the registered structural system the mechanism is also materially useful: it beats full original-coordinate suffix replay by 15.89× aggregate normalized work.

## Why it is still killed

An independently implemented generic sparse-linear/Koopman engine was handed the identical six observables and exact operator `K`, but no Pod, Symlink, J-space, neural-memory, or lifecycle semantics. It independently propagated the two receipt basis vectors and obtained the same receipt, same exact fresh states, same memory footprint, and same normalized mutation work in every registered case.

Therefore the fleet-level advantage is real, but it belongs to the disclosed finite invariant lift and ordinary linear-operator algebra. The lifecycle interpretation does not create a technical advantage unavailable to a guarantee-matched generic engine.

The programme should therefore assign zero standalone major-invention credit to:

- exact finite Koopman lifts used only as mutable-state transport coordinates;
- finite Carleman closures where the disclosed closure is itself the sufficient state;
- learned finite invariant embeddings whose mutation benefit is completely reproduced after handing the embedding/operator to a generic linear state engine.

## Current literature / patent boundary checked 2026-09-06

The fresh search strengthens rather than weakens the kill:

- Shang, Haseli, Cortés & Zheng, arXiv:2602.14537 (2026), gives necessary and sufficient conditions for exact finite-dimensional Koopman linear embeddings of controlled nonlinear systems and characterizes a special control-affine-preserved structure plus a finite invariant autonomous subsystem.
- Iacob, Tóth & Schoukens, arXiv:2507.15093 (2025), constructs exact finite Koopman embeddings for block-oriented polynomial systems.
- Kvalheim & Arathoon, *Selecta Mathematica* 32:38 (published 2026-04-07), studies necessary/sufficient conditions for global linearizability of flows by finite-dimensional embeddings.
- `US20240119265A1` (priority 2022-09-28) covers a Koopman neural forecaster that learns a linear Koopman space and operators using deep neural networks. This does not by itself anticipate FinX lifecycle guarantees, but it removes any broad novelty claim around “neural network + learned Koopman latent operator”.
- Li, arXiv:2606.17107 (2026), demonstrates editable/composable KV-cache interventions with strong decision/latency results, but reports logit-cosine/decision equivalence rather than exact equality to a fresh post-mutation neural state under the FinX contract.
- Ramesh, arXiv:2607.27539 (2026), provides a directly relevant representation boundary: addressable influence can be algebraically removed, whereas entangled recurrent writes require rewind/replay to match the record-omitted state exactly.
- `WO2026087278A1` (IBM, published 2026-04-30) covers real-time insertion/modification/deletion through an external KV-cache network layer; `US20260080217A1` (Intel, published 2026-03-19) covers gauge/rank-r KV-cache transformation/compression and claims priority to “COMPOSABLE EXACT KEY-VALUE CACHE COMPRESSION”. These crowd generic mutable/cache transformations but do not establish the missing FinX exact reusable cross-session mutation receipt.

The search did not surface a mechanism that provides one canonical knowledge edit -> one bounded reusable correction -> exact fresh neural state across many already-cached heterogeneous sessions while also beating a guarantee-matched generic engine on the same sufficient representation.

## Scope boundary / next invention pressure

E-000112 does **not** pass the real FinX promotion gates: the toy state is six-dimensional versus three original coordinates and has not been tested on DistilGPT-2/Pythia-70M, real LINK->Pod readers, <=5% steady-state overhead, leakage/UNKNOWN, stale-state attacks, full lifecycle adversarial sequences, or independent J-space/J-lens audit.

The useful conclusion is narrower: another exact nonlinear fleet-transport representation can work spectacularly and still have zero invention credit when the same sufficient representation can simply be handed to a generic algorithm.

The surviving pressure therefore moves from inventing another update algebra toward the **production and maintenance of the sufficient state during useful neural computation**. A successor only becomes interesting if its neural architecture produces/maintains mutation-enabling state under the real overhead and utility constraints in a way that a guarantee-matched generic provenance, incremental, AD, group-action, tensor-network, Koopman, or replay engine cannot inherit merely by receiving the same representation.
