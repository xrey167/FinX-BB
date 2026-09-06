# E-000114 Result — Exact Softmax Sufficient-Statistic Revision Receipt Reduction

## Decision

**KILL_EXACT_SOFTMAX_SUFFICIENT_STATISTIC_RECEIPT_AS_STANDALONE_LIFECYCLE_NOVELTY**

This is a decisive scoped falsification, not a major-invention promotion and not a universal impossibility claim.

## Registered evidence

- Registered source/workflow head: `9f40f914b57de809b5fff839a51ab2db3d8634be`
- GitHub Actions run: `34030738615`
- CI conclusion: `success`
- Artifact: `e000114-exact-softmax-sufficient-statistic-revision-receipt-reduction`, ID `9988516765`
- Artifact digest: `sha256:619eb58e0895415d752131fca1bda6e7c1107ae3d8a61ac62c4884efff6066ec`
- Result JSON SHA-256: `c6e93b371bed418a610b8df5181224944058a1ff85ba84bcb9e1e64986ed34e2`
- Preregistration SHA-256: `3f4fe26551b2ffc1a5722648d0d5ff45c0d86288fa6f7a42f26d3edc33704354`
- Experiment SHA-256: `fa40a5c952d564d2a0b72b6d98a6c14cfd08a6e235562418a183b7c2e8b05020`
- Test SHA-256: `fbe11a37ed24169e904a94a730eb1ecf500ba4e28ca06845d5e1d1d47477fdef`

The CI job successfully compiled the experiment, passed the focused regression suite, completed the registered exact-rational assay, hashed the evidence, and uploaded the artifact.

## Exact registered result

The assay executed 16 deterministic seeds × widths `4,8,16` × 32 heterogeneous cached sessions with a canonical shared Pod record and sequential `UPDATE -> DELETE -> RESTORE -> ABA`.

Attention was a true scaled-softmax special case:

`weight_i = exp(log(2) * q_s * k_i) = 2^(q_s*k_i)`,

with integer `q_s*k_i`. Consequently all attention weights, normalizers, outputs, and downstream task states were represented exactly with `fractions.Fraction`; no floating-point tolerance was used.

| Metric | Result |
|---|---:|
| Registered cells | 48 |
| Exact lifecycle cases | 6,144 |
| Material final-state changes | 6,144 / 6,144 |
| Candidate -> fresh post-softmax mismatches | 0 |
| Generic -> fresh post-softmax mismatches | 0 |
| Candidate/generic post-softmax mismatches | 0 |
| Candidate -> fresh final-state mismatches | 0 |
| Generic -> fresh final-state mismatches | 0 |
| Candidate local attention-update work | 209,920 |
| Generic local attention-update work | 209,920 |
| Fresh attention-recompute work | 4,168,192 |
| Required nonlinear suffix recompute work | 1,376,256 |
| Candidate exact-ready work | 1,586,176 |
| Generic exact-ready work | 1,586,176 |
| Fresh exact-ready work | 5,544,448 |
| Fresh/candidate exact-ready ratio | **3.49548x** |
| Oracle exact-read-patch + suffix work | 1,376,256 |
| Candidate/oracle ratio | **1.15253x** |
| Minimum distinct exact output deltas per transition/cell | **32** |
| Wrong one-global-output-delta failures | **5,952** |

## What the candidate achieved

This candidate crosses a boundary E-000113 did not.

The mutable Pod record is already inside a **normalized exponential attention operation**. The cached object is not merely a pre-activation sparse accumulator. Each session retains its real attention output `A_s`, the scalar softmax normalizer `Z_s`, and its query `q_s`.

Because

`N_s = Z_s * A_s`,

the unnormalized attention numerator is exactly recoverable from the materialized post-softmax output. A lifecycle edit can therefore remove the old record contribution and add the new record contribution without rescanning the other 64 memory records:

`Z'_s = Z_s - w_old(q_s) + w_new(q_s)`

`N'_s = Z_s A_s - w_old(q_s) v_old + w_new(q_s) v_new`

`A'_s = N'_s / Z'_s`.

This produced exact fresh post-softmax state in all 6,144 lifecycle cases.

The edit is genuinely state/context dependent. Every cell had 32 distinct exact attention-output deltas for every lifecycle transition; a deliberately incorrect fleet-global output correction failed 5,952 comparisons. Thus E-000114 is not reviving the E-000095 global-translation claim.

Locally, the mechanism looks highly attractive: updating the post-softmax read costs 209,920 normalized operations across the registered fleet instead of 4,168,192 for fresh attention recomputation. After the required task suffix, the structural fresh/candidate exact-ready ratio remains 3.50x.

## Why it is still killed

### 1. The exact nonlinear update is generic normalized aggregation

An independently written generic dynamic normalized-weighted-average engine was handed the identical session query, normalizer, aggregate output, and removed/added records, but no Symlink, Pod, J-space, neural-memory, or lifecycle semantics.

It reproduced every fresh attention state exactly and used exactly the same **209,920** local update operations. Candidate and generic exact-ready work were therefore identical at **1,586,176**.

The local exactness comes from an ordinary sufficient statistic for a normalized weighted sum, not from a FinX-specific lifecycle algebra.

### 2. Exact post-softmax repair still does not transport the later nonlinear state

The user-required strong baseline starts at the last real memory-read site: patch the exact read-site delta and perform only the minimal suffix recomputation needed to regenerate downstream state.

E-000114's two downstream nonlinear task layers still have to run for every cached session. They account for **1,376,256** operations, and both candidate and generic baseline pay them identically.

An oracle baseline handed the exact attention-read patch spends only those 1,376,256 suffix operations. The candidate, which additionally computes the local softmax correction, is **1.15253x more work** than that strong boundary. Therefore the attractive 3.50x advantage over full fresh attention does not constitute the required fleet-level advantage over guarantee-matched exact read-site patch + suffix repair.

This is the decisive kill. Exact softmax repair solves the wrong boundary: it updates the read itself but not the already-materialized entangled suffix state.

## Fresh literature and patent boundary checked 2026-09-06

The targeted search for one-edit reusable neural correction receipts did not surface a mechanism that changes this conclusion.

- Milakov & Gimelshein, *Online normalizer calculation for softmax* (2018), already establishes exact online maintenance of the classical softmax normalizer with reduced memory access: https://arxiv.org/abs/1805.02867
- Dao et al., *FlashAttention* (NeurIPS 2022), makes exact attention computation explicitly rely on compact running softmax statistics and output accumulation. This is a strong generic-prior-art boundary around `normalizer + weighted-output sufficient state`: https://arxiv.org/abs/2205.14135
- Li, *Models Take Notes at Prefill: KV Cache Can Be Editable and Composable* (2026), demonstrates highly useful editable/composable KV behavior and large latency gains, but reports decision identity / logit-cosine equivalence rather than FinX fresh-hidden-state identity; the reported composition cosine is `0.90-0.999`: https://arxiv.org/abs/2606.17107
- Ramesh, *Subtract or Replay? Exact Deletion from Language-Model Memory* (2026), remains the strongest nearby exact-state boundary found: addressable contribution can be decremented, whereas suffix-dependent recurrent influence is restored by checkpoint rewind/replay; the paper reports bit-for-bit recovery of the audited recurrent state/logits in its deterministic path: https://arxiv.org/abs/2607.27539
- Ramesh, *Forgetful Attention* (2026), provides a constructive trainable addressable-memory route via support-vector/KKT state, but its efficiency and guarantee arise from that specialized addressable representation rather than a cross-session correction receipt for arbitrary already-materialized transformer state: https://arxiv.org/abs/2607.12204
- Intel `US20260080217A1` covers exact-function-preserving/gauge-based KV-cache transformation territory, so broad exact KV transformation is already crowded: https://patents.google.com/patent/US20260080217A1/en
- `EP4738200A1` and related 2025/2026 KV-cache patent material explicitly discusses cached K/V tensors being updated and then participating in full softmax attention, further removing broad novelty from mutable-KV attention plumbing: https://patents.google.com/patent/EP4738200A1/en
- `CN118227535B` discusses online-softmax / FlashAttention-style accumulation in an accelerator context, showing that compact softmax state is also present in patent-side implementation prior art: https://patents.google.com/patent/CN118227535B/en

These references do not establish the narrow FinX lifecycle guarantees and are not patentability or infringement advice. They do make `cache/update exact softmax sufficient statistics around a mutable KV record` an unsafe major-novelty anchor.

The fresh search also surfaced 2026 work such as DeltaLog, which preserves exact recurrent-model semantics while deferring dense state materialization via a bounded update log. That is useful systems prior art for exact state-maintenance engineering, but it does not provide the missing canonical knowledge-mutation correction across heterogeneous already-entangled sessions.

## Closed seam

Assign zero standalone invention credit to the family:

> **retain query + softmax normalizer + materialized attention output, then subtract/add one edited Pod K/V contribution to repair an already-composed softmax read exactly.**

This remains true even when:

- the edit is canonical fleet-wide,
- every session has a different exact correction,
- the read-site repair is exact,
- UPDATE/DELETE/RESTORE/ABA all work,
- and the local update is much cheaper than rescanning attention memory.

The local mechanism is generic normalized aggregation, and the downstream exact-state obligation remains the same guarantee-matched suffix-recompute problem.

## Frontier after E-000114

The remaining transport candidate must cross **more than the memory-read nonlinearity**. It must update a state that has already passed through later context-dependent nonlinear transformations while obtaining a material fleet-level mutation-to-ready advantage over the exact read-site-patch + suffix/KV-repair baseline.

A future candidate is not interesting merely because it can maintain an exact sufficient statistic at one neural operator. The sufficient state must survive or co-evolve through the downstream computation in a way that:

1. remains exact under canonical Pod mutation,
2. stays compact enough for <=5% steady-state overhead,
3. avoids simply becoming generic incremental computation/provenance/AD/group/Koopman/tensor/dynamic-aggregate state,
4. beats matched-memory per-session suffix/KV repair on mutation-to-ready cost, and
5. later passes the unchanged real DistilGPT-2/Pythia-70M lifecycle, leakage, UNKNOWN, stale-state/race/key, and independent J-space/J-lens gates.

No real-model promotion gate is changed.

**Major-break status remains negative. E-000114 is a decisive falsification.**
