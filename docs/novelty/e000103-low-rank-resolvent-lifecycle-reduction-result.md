# E-000103 result — exact resolvent lifecycle transport works, but is generic Woodbury/RLS

Date: 2026-09-06
Status: **DECISIVE DIRECTION CHANGE / scoped family kill**

## Decision

**Kill low-rank resolvent / inverse-state lifecycle transport as a standalone FinX-BB major-invention seam.**

The registered architecture demonstrates a genuinely useful systems pattern: a shared low-rank Pod mutation can update an interaction-bearing global operator once and transport many cached session states exactly without fresh per-session solves. However, an independently implemented ordinary Woodbury baseline reproduces the exact same revised inverse and every revised session state from the same sufficient state.

The lifecycle advantage therefore belongs to classical dynamic low-rank linear algebra, not to Pod naming, neural placement, Symlink semantics, or J-space.

This is a scoped novelty/baseline kill, not an impossibility theorem for every lifecycle-native neural architecture.

## Execution provenance

GitHub Actions run: `34006041327`
Head SHA: `4176f13bcd5031b70e7558ebdadf62caec70de3b`
Workflow: `.github/workflows/e000103-low-rank-resolvent-reduction.yml`
Artifact: `e000103-low-rank-resolvent-lifecycle-reduction`
Artifact ID: `9980951945`
Artifact ZIP SHA-256 reported by Actions: `cdf504d6331b8c0806197a5fb6bdaf2464db6cc8ffb6ff0d95b22cee1e63dcad`

Focused regression suite: **4 passed in 0.43 s**.
Registered exact assay: **success**.

All registered scientific arithmetic used Python `fractions.Fraction`; the equality claims below are exact rational equality, not floating-point tolerance claims.

## Registered mechanism

For four mutable Pods, the architecture used

`M(P) = A + sum_i u_i(P_i) v_i(P_i)^T`

and each cached session `s` stored the mixed state

`x_s(P) = M(P)^(-1) b_s`.

A target Pod revision replaced one rank-one contribution:

`Delta M = -u_old v_old^T + u_new v_new^T`.

The candidate removed the old contribution and inserted the new contribution through two exact Sherman–Morrison updates. The generic baseline independently applied one rank-2 Woodbury update. A fresh exact Gaussian inverse/solve was the gold state.

Because the inverse couples all Pod contributions, the mechanism is not additive late binding. The finite response to a target Pod revision changes when a second Pod is removed; all registered interaction controls witnessed this dependence.

## Exact result

Registered domain:

- deterministic seeds: **16**;
- dimensions: **4, 6, 8, 10**;
- Pods per system: **4**;
- sessions per system: **24**;
- target revision cells: **256**;
- cached-session revision cases: **6,144**.

Results:

| Measurement | Result |
|---|---:|
| Candidate vs fresh revised session-state mismatches | **0 / 6,144** |
| Generic rank-2 Woodbury vs fresh revised session-state mismatches | **0 / 6,144** |
| Candidate revised inverse vs generic Woodbury inverse mismatches | **0 / 256** |
| DELETE mismatches | **0** |
| RESTORE mismatches | **0** |
| ABA mismatches | **0** |
| Materially changed sessions | **6,144 / 6,144** |
| Cross-Pod interaction cells material | **256 / 256** |
| Maximum observed Fraction numerator size | **63 bits** |
| Maximum observed Fraction denominator size | **67 bits** |

The registered decision is therefore:

`KILL_RESOLVENT_TRANSPORT_AS_STANDALONE_NOVELTY_SEAM`.

## Why this result matters

Unlike many earlier failed candidates, E-000103 does **not** fail technically. It demonstrates all three properties that made the lane attractive:

1. **exactness** — every transported state equals fresh recomputation exactly;
2. **state-dependent nonlinear interaction** — a Pod's effect depends on the other Pods through the inverse operator;
3. **fleet asymmetry** — a low-rank operator revision permits shared inverse-update work and cheap per-session state transport rather than an independent fresh solve for every session.

That combination is useful architecture evidence. But it is not a defensible FinX-BB invention because the same representation and same lifecycle update are ordinary Sherman–Morrison/Woodbury dynamic linear algebra.

The strongest baseline is not required to throw away the inverse or recompute from scratch. Once the candidate exposes `M^-1` and the low-rank operator delta, the baseline is entitled to use the exact same sufficient state and the best classical low-rank update algorithm. E-000103 confirms exact equivalence rather than merely asymptotic similarity.

## Prior-art boundary — decisive

The surrounding neural/memory literature already occupies the broad mechanism strongly enough that no major novelty credit should be spent on this lane:

- Hui, Lillo & Zak, *Learning and Forgetting in Generalized Brain-state-in-a-box (BSB) Neural Associative Memories* (Neural Networks, 1996) explicitly proposes online add/delete memory patterns by updating a pseudoinverse instead of recomputing the synaptic solution from scratch. This is particularly damaging prior art for broad claims around exact add/delete neural associative memory through inverse updates.
- Fei, Li & Li, *Selective Memory Recursive Least Squares: Recast Forgetting Into Memory in RBF Neural Network-Based Real-Time Learning* (IEEE TNNLS 2025) uses recursive least squares as a neural memory/forgetting mechanism.
- Pandey & Singh, *Variational Linear Attention: Stable Associative Memory for Long-Context Transformers* (`arXiv:2605.11196`, 2026) maintains an adaptive inverse penalty matrix inside transformer associative memory using Sherman–Morrison rank-1 updates.
- Li et al., *Modality-Decoupled Online Recursive Editing* / M-ORE (`arXiv:2605.20273`, 2026) derives online model editing with a Sherman–Morrison recursion and constant per-edit overhead.
- Li et al., *Multimodal Knowledge Edit-Scoped Generalization for Online Recursive MLLM Editing* / ScopeEdit (`arXiv:2607.01978`, 2026) again maintains edit preconditioners through Sherman–Morrison recursions.
- Zhao et al., *WIN-U: Woodbury-Informed Newton-Unlearning* (`arXiv:2604.13438`, 2026) uses Woodbury structure for efficient machine unlearning, although its neural retraining target is approximate rather than the exact E-000103 state contract.

Additional 2025–2026 context further crowds low-rank contextual patches:

- Goldwaser et al., *Equivalence of Context and Parameter Updates in Modern Transformer Blocks* (`arXiv:2511.17864`, ICML 2026) provides exact analytical context-to-parameter patch constructions, including rank-1 MLP patches in modern transformer blocks.
- Mazzawi et al., *Transmuting prompts into weights* (`arXiv:2510.08734`, 2025; revised 2026) derives reusable thought vectors/matrices from prompt information.

These references do not all implement the complete E-000103 Symlink/Pod lifecycle contract. They do make the primitive ingredients — inverse updates, recursive neural memory, low-rank online editing and context-derived parameter patches — prior art/baselines rather than a defensible major-invention core.

Targeted patent searches this turn did not surface a patent claim set that exactly matches the full E-000103 construction. That limited result is not patentability, infringement or freedom-to-operate advice and cannot rescue the lane because the literature/generic-algorithm reduction is already decisive.

Primary source URLs inspected this turn:

- https://pubmed.ncbi.nlm.nih.gov/12662567/
- https://pubmed.ncbi.nlm.nih.gov/38619955/
- https://arxiv.org/abs/2605.11196
- https://arxiv.org/abs/2605.20273
- https://arxiv.org/abs/2607.01978
- https://arxiv.org/abs/2604.13438
- https://arxiv.org/abs/2511.17864
- https://arxiv.org/abs/2510.08734

## Consequence for the invention frontier

E-000103 provides a useful positive design lesson while closing the claim:

> A lifecycle-native architecture can achieve exact, nonlinear, fleet-efficient mutation if its computation admits a compact dynamic sufficient state. But if that sufficient state is an ordinary inverse/factorization and its update is an ordinary low-rank identity, the major systems advantage is baseline-owned.

Do not spend future major-invention budget on:

- Sherman–Morrison/Woodbury Pod memories;
- recursive-least-squares Pod stores;
- pseudoinverse add/delete associative memories;
- low-rank resolvent memories;
- exact lifecycle claims whose speedup comes solely from classical rank-k inverse updates;
- context-to-low-rank-weight-patch mechanisms by themselves.

A successor must retain the useful E-000103 fleet property — one canonical mutation producing substantially cheaper exact updates across many already-materialized sessions — but derive that advantage from a **neural-specific compact nonlinear sufficient state/update law not reproduced by generic dynamic linear algebra, provenance, incremental computation, sidecars or late binding**.

No real LINK->Pod reader, stale Bank/router/payload/Hidden/KV attack battery, J-space/J-lens audit, normal-inference overhead or public-backbone capability claim is made from E-000103. The full major-break gates remain unchanged.
