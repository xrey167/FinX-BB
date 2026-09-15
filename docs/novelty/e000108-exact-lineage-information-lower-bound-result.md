# E-000108 — Exact Lifecycle Lineage Information Lower Bound — Result

Date: 2026-09-06  
Decision: **KILL_COMPACT_EXACT_CENTRAL_LINEAGE_CAPSULE_AS_GENERAL_ADVANTAGE**  
Major-invention claim: **NO**

## Evidence identity

- Branch: `research/e000051-clean-bystanders`
- Preregistered source commit: `c10b821c8557c663136d471250a4c3c9122c2bb5`
- GitHub Actions run: `34012895248`
- Workflow: `E000108 exact lineage information lower bound`
- Reduction job conclusion: **success**
- Registered exact reduction step: **success**
- Focused regressions: **success**
- Evidence artifact: `e000108-exact-lineage-information-lower-bound`
- Artifact digest: `sha256:7a882e094a758f550de201c5a9b5c9ce67c6e319e78ae279d639be4116c48761`

## Registered result

The exhaustive exact reduction passed its preregistered kill rule in all four cells.

| q | Pods n | Histories | Histories per visible state | Distinct arbitrary-target DELETE profiles per visible state | Required auxiliary certificate states | Generic ledger states | Generic failures |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 12 | 4,096 | 2,048 | 2,048 | 2,048 | 2,048 | 0 |
| 3 | 8 | 6,561 | 2,187 | 2,187 | 2,187 | 2,187 | 0 |
| 5 | 6 | 15,625 | 3,125 | 3,125 | 3,125 | 3,125 | 0 |
| 7 | 5 | 16,807 | 2,401 | 2,401 | 2,401 | 2,401 | 0 |

Totals:

- exact histories exhaustively enumerated: **43,089**
- exact DELETE comparisons: **279,425**
- exact UPDATE comparisons: **279,425**
- exact ABA restoration comparisons: **279,425**
- generic baseline failures across DELETE/UPDATE/ABA: **0**

## Decisive reduction

For `v_i in Z_q`, the task-visible mixed state is only a bijective encoding of

`T = sum_i v_i mod q`.

Fixing `T` leaves exactly `q^(n-1)` admissible source-value histories.

For target `i`, exact deletion must produce

`DELETE_i = psi(T - v_i mod q)`.

Because `psi` is bijective, the vector of all target-specific DELETE outcomes recovers every `v_i`. Exhaustive enumeration confirmed that **every one of the `q^(n-1)` histories inside each visible-state bucket has a distinct exact deletion profile**.

Therefore a central capsule `C` paired with the visible state and required to answer arbitrary target deletion must distinguish at least

`q^(n-1)`

states:

`log2 |C| >= (n-1) log2 q`.

The strongest generic ledger attains this lower bound exactly: store any `n-1` Pod values and reconstruct the final value from the visible aggregate. It then performs exact DELETE, UPDATE and ABA with no additional lineage information.

The binary `q=2, n=12` cell is especially direct: the current visible state retains one parity bit, while an exact arbitrary-target lineage capsule needs **2,048 states = 11 additional bits**. A generic packed membership ledger using the visible parity also needs exactly **11 additional bits**. There is no central information-compression advantage left for a differently named neural lineage code.

## Scope of the kill

Close as standalone major-invention routes:

- centralized learned lineage embeddings whose claimed advantage is exact compression of arbitrary source membership/contribution history;
- neural source-set hashes or latent causal codes that must support exact arbitrary-target deletion;
- causal "biomarkers" whose guarantee requires preserving the same sufficient source information as an ordinary ledger;
- any central lineage capsule that does not exploit additional restrictions beyond the registered arbitrary-contribution family.

This result does **not** reopen or rely on E-000086R metadata novelty. The generic ledger is the baseline that saturates the exact information lower bound.

## What remains alive

The reduction does not kill:

1. **structured/compressible lifecycle domains** where admissible Pod histories occupy far less than the full Cartesian product and the structure is itself useful/non-generic;
2. **ticketed/distributed lifecycle information**, where exact deletion information is moved to the Pod/user rather than centrally retained — but the baseline must be a generic ticketed scheme with the same information;
3. architectures whose normal task computation already requires and exposes an injective per-Pod state, so lineage storage is not an extra capsule;
4. active intervention/replay;
5. approximate attribution;
6. a lineage mechanism providing an independently useful certified capability beyond preserving the exact source ledger.

## External-evidence boundary

Cherapanamjeri et al., *The Space Complexity of Learning-Unlearning Algorithms* (2025), independently show that exact future deletion can require `Omega(n)` retained bits even for some low-complexity hypothesis classes, and that moving information into a ticketed-memory model can fundamentally change the storage tradeoff. E-000108 is narrower and constructive: for the registered finite-group lifecycle family, it gives an exact state-count lower bound and an ordinary generic ledger that meets the bound with equality.

Fresh 2026 cache-edit work still does not supply an exact cross-session active transport escape. KVEraser explicitly treats exact localized erasure as suffix recomputation and gains speed by approximate learned steering; Leyline supplies serving-side KV edit/splice directives and position correction but does not establish equality to a fresh post-knowledge-mutation state; recent position-independent cache repair such as Kamera targets reuse/reconstruction under relocation rather than canonical knowledge lifecycle deletion.

## Programme consequence

The causal-lineage branch is now substantially narrowed:

- E-000107: exact lineage cannot in general be decoded from already-collapsed state alone.
- E-000108: preserving the missing information in a **central exact capsule** cannot, on an unrestricted arbitrary-contribution family, be more information-compact than a generic optimal ledger.

Accordingly, do not allocate major-invention search budget to a centralized exact lineage code unless it first demonstrates a structural restriction or a second useful capability that changes the matched-memory baseline.

The main surviving invention pressure returns to **active lifecycle-conditioned computation** or to a genuinely structured/ticketed lineage architecture that beats the strongest generic counterpart on a measured systems dimension.

All major-break system gates remain unchanged.
