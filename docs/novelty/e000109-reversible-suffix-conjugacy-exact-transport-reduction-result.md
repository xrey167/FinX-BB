# E-000109 — Reversible Suffix Conjugacy Exact Transport Reduction — Result

Date: 2026-09-06  
Decision: **KILL_REVERSIBILITY_AND_CONJUGACY_AS_STANDALONE_EXACT_TRANSPORT_ADVANTAGE**  
Major-invention claim: **NO**

## Evidence identity

- Branch: `research/e000051-clean-bystanders`
- Preregistered source/workflow commit: `9629e95579354ef5d7c439194239c742629cdc01`
- GitHub Actions run: `34017387683`
- Workflow: `E000109 reversible suffix conjugacy exact transport reduction`
- Reduction job conclusion: **success**
- Focused regressions: **3 passed**
- Registered exact reduction step: **success**
- Evidence artifact: `e000109-reversible-suffix-conjugacy-exact-transport-reduction`
- Artifact ID: `9984328353`
- Artifact ZIP digest: `sha256:514b68179853a693462a0e2cae76e545722f9732f64cdfadc82a439688818198`

## Registered result

The exact nonlinear reversible-state assay passed every preregistered kill condition.

| Quantity | Result |
|---|---:|
| deterministic reversible suffix cells | **192** |
| exact state-transport cases | **111,168** |
| materially changed cases | **111,168 / 111,168 (100%)** |
| candidate → fresh mismatches | **0** |
| independent generic map → fresh mismatches | **0** |
| candidate conjugate map → generic map mismatches | **0** |
| ABA failures | **0** |
| candidate reversible-layer evaluations | **1,445,184** |
| last-read suffix-recompute layer evaluations | **722,592** |
| candidate / baseline layer-evaluation ratio | **2.0×** |
| materialized candidate/generic transport-table entries audited | **111,168** |

The candidate is therefore **perfectly exact**. For stale final state `y = F(x)`, it transports a Pod edit by

`F(tau_delta(F^-1(y)))`.

But exactness does not create the required fleet-level advantage. With the exact old last-read state retained, the guarantee-matched baseline applies the same Pod edit and runs the suffix only once. The reversible state-only candidate must first traverse the inverse suffix and then traverse the forward suffix: `2d` registered layer evaluations versus `d` for exact suffix recomputation at every tested depth `d in {2,4,8,12}`.

If the last-read state is *not* retained, `F^-1(y)` is simply generic exact reconstruction of that state before replay. Reversibility changes what can be reconstructed; by itself it does not reduce the exact affected computation.

## Precompiled one-edit conjugate reduction

The obvious escape is to precompile the edit-keyed operator

`T_delta = F o tau_delta o F^-1`

once and apply it directly to every cached final state.

E-000109 independently materialized that operator in two ways:

1. the candidate evaluated `F o tau_delta o F^-1` over the complete final-state domain;
2. a generic compiler enumerated read-site states and independently recorded `F(x) -> F(tau_delta(x))`, without using the candidate inverse-transport routine.

Across all **111,168 registered table entries**, the two maps were identical, and both matched fresh recomputation exactly. Once `T_delta` itself is materialized, it is simply an ordinary exact function/table; a generic evaluator given the same representation has the same mutation-time behavior and memory footprint.

Therefore precompiling the conjugate does not rescue a neural-specific invention claim unless a **new compact/evaluable representation of that conjugate** supplies an advantage that a generic compiler using the identical representation cannot obtain.

## Reversibility does not imply a compact conjugate family

A second exact control tests whether invertibility alone forces `T_delta` into a small reusable operator class.

For a state domain of size `n`, fix the Pod edit to one full `n`-cycle `tau`, enumerate **every** suffix bijection `F`, and count distinct exact transports `F tau F^-1`.

| n | suffix bijections exhaustively enumerated | distinct exact transports observed | exact expected conjugacy class `(n-1)!` |
|---:|---:|---:|---:|
| 4 | 24 | **6** | 6 |
| 5 | 120 | **24** | 24 |
| 6 | 720 | **120** | 120 |
| 7 | 5,040 | **720** | 720 |
| 8 | 40,320 | **5,040** | 5,040 |

Every cell matched the complete conjugacy class exactly. This is a finite exact witness that **bijectivity/reversibility alone places essentially no useful compactness constraint on the transported edit operator**. It is not a lower bound for a particular trained transformer and does not rule out specially structured normalizers/equivariant architectures; it rules out claiming that compact one-edit transport follows merely from invertibility.

## Scope of the kill

Close as standalone major-invention routes:

- “make the post-Pod suffix reversible, then transport stale final state by inversion + edit + forward”;
- reversible coupling/flow architecture **when reversibility itself** is the claimed mutation-to-ready advantage;
- `F o tau o F^-1` notation as an invention claim without an independently new compact exact representation/evaluation law;
- a fully materialized per-edit conjugate lookup/function when the same function can be handed to a generic evaluator;
- claims that invertibility by itself guarantees a compact family of cross-session exact edit receipts.

This result does **not** kill a specially structured lifecycle-equivariant/normalizer architecture whose edit action provably remains in a compact cheap operator family. But such a successor must earn all credit from that additional structure, and its strongest baseline must receive the same group/action representation and generic compiler. Reversibility receives zero standalone credit.

## Fresh external-evidence boundary

The August 13, 2026 v2 of Vishwajith Ramesh, *Subtract, Transport, or Replay? Auditable Deletion from Language-Model Memory* (`arXiv:2607.27539`) is unusually aligned with this frontier. In native Kimi Delta Attention it reports that the raw recurrent contribution of a record changes by **12–49%** with the suffix, remains **8–49%** after a decay-ledger correction, and that native omission changes later transition/write terms and other active caches. A frozen-input full-affine transport control succeeds numerically, but native omission lies outside the tested fixed-receipt interfaces; checkpoint replay reaches the audited omitted state with zero residual across final logits and all 80 audited KDA arrays. This independently strengthens the programme's distinction between transport under a fixed computation and edits that change subsequent computation.

The constructive counterpoint is Ramesh, *Forgetful Attention* (`arXiv:2607.12204`): a specially addressable support-vector memory uses maintained reversible incremental/decremental optimization state for verified deletion. That is evidence that architecture/representation can make exact deletion cheap, but the mechanism's advantage comes from addressable maintained structure, not from generic invertibility of an already-entangled nonlinear suffix.

Fresh cache-reuse work remains below the FinX exact lifecycle bar: KVBoost (`arXiv:2608.21362`) uses deviation-guided selective recomputation and reports task accuracy, while Kamera (`arXiv:2606.23581`) uses exact position rerotation plus low-rank conditioning patches and reconstructs re-prefill KV to within bf16 rounding rather than exact knowledge-mutation equality.

Patent search remains crowded around generic mutable/composable cache mechanisms, including IBM `WO2026087278A1` (direct knowledge injection through a KV-cache network layer), Intel `US20260080217A1` (gauge/rank-r KV transformation/compression), and Huawei `WO2026086089A1` (KV segment recomputation). No patent-clearance claim is made.

## Programme consequence

E-000109 removes another tempting retrofit path. A successor cannot obtain major-invention credit merely by making neural computation invertible. The surviving active-transport target is narrower:

`one canonical Pod edit -> one compact exact shared action -> fresh state across many sessions`

where the **cheap shared action is an architectural invariant**, survives meaningful nonlinear Pod/context interaction, and cannot be reduced to a generic group action, classical maintained optimizer state, table/function materialization, inverse+replay, provenance, or late binding.

A plausible next kill screen is therefore **lifecycle equivariance / edit-group normalizers**: deliberately co-design a nonlinear architecture so Pod UPDATE/DELETE actions remain in a bounded-size exact transformation group at every layer, then hand the identical group representation to the strongest generic equivariant/group-action baseline. If the generic baseline inherits the same mutation advantage, that route also loses standalone invention credit; if a neural-specific representation materially beats it, that is the first place to consider escalation to real readers.

All major-break system gates remain unchanged.
