# E-000095 — State-Dependent Exact Lifecycle Transport

Date: 2026-09-05
Status: **PREREGISTERED SUCCESSOR KILL SCREEN / not a novelty claim**

## Trigger

E-000093 killed passive in-band freshness when reuse factors through current authority plus a statistic recoverable from cached neural state. E-000094 then killed exact static/mutable noninterference by itself: if the reusable persistent state is invariant to Pod state, the computation factors into cached static state plus a late-bound mutable argument.

The remaining active-computation escape is therefore narrower: an already-materialized **mixed** neural state depends on the old Pod lifecycle state, and a lifecycle transition applies a bounded state-dependent transform that converts that mixed state to the exact fresh-current state without replaying the full affected suffix.

## Candidate contract

For context `x`, Pod state `p`, and a chosen persistent neural state representation `H(x,p)`, define an authorized lifecycle transition `p_old -> p_new` and candidate transport

`Phi(H_old, p_old, p_new, a) -> H_transport`

where `a` is only the minimum current authority needed to identify the authorized transition.

The candidate is interesting only if:

1. `H_old = H(x,p_old)` is genuinely mixed with Pod information;
2. `H_transport` matches a fresh recomputation `H(x,p_new)` under the registered exactness contract;
3. `Phi` materially depends on the actual joint state `H_old` rather than reducing to a Pod-local action or late-bound mutable branch;
4. transport work/memory is materially smaller than the strongest exact guarantee-matched baseline;
5. unrelated Pod state remains reusable;
6. no passive freshness marker, external mask, version sidecar, deletion, or global invalidation supplies the correctness result.

## Exactness

Primary exactness is byte identity when deterministic execution makes that meaningful. Otherwise the preregistered numerical fallback is both:

- max absolute state error <= `1e-10`, and
- downstream logits KL <= `1e-12` with identical greedy top-1,

with full-recompute repeatability measured first. Approximate task recovery, logit cosine similarity, steering quality, or "near full recompute" is not a pass.

## Strong baselines

The candidate must be compared against the strongest applicable implementation, not a dense-replay strawman:

A. full affected-suffix recomputation;
B. residual-stream checkpoint recomputation / KV-Direct-style exact reconstruction where applicable;
C. exact field/span recomputation when the edit boundary is explicit;
D. ordinary dependency/change propagation with matched materialized dependency state;
E. programmable-KV / erratum or note-edit mechanisms when they achieve the task contract;
F. KVEraser / AgentKVShift / related 2026 learned KV-edit methods as approximate systems references, clearly separated from the exact guarantee comparison;
G. co-located sidecar/forwarder for authorization only;
H. strongest algebraic simplification of the same candidate, including factoring any Pod-local transform outside expensive depth.

A result receives no novelty credit merely because it edits KV instead of residual state. 2026 residual-stream evidence makes KV a deterministic projection of a smaller exact checkpoint for standard transformer families, so the systems comparison must include that representation where valid.

## Phase A — structural transport screen

Before a real Symlink reader is interpreted, test whether the chosen transport family actually escapes previous reductions.

Required negative reductions:

1. **Late-bind reduction:** if the supposedly mixed persistent state is invariant to Pod state, E-000094 applies and the arm is killed.
2. **Pod-local action reduction:** if `Phi(H,p_old,p_new)=A(p_old,p_new) H` or an equivalent cheap action can be pulled through the expensive computation and applied once at the boundary, compare to that exact ordinary equivalent; kill if equal-cost/equal-output.
3. **Replay-in-disguise:** account for all computation needed to infer or evaluate `Phi`; kill if its asymptotic and measured work is not materially below the strongest exact recomputation baseline.
4. **Passive gate reduction:** if `Phi` only accepts/rejects unchanged state, E-000093 applies and the arm is killed.
5. **Metadata dependency reduction:** source/dependency metadata supplied externally receives zero novelty credit.

## Phase B — frozen-backbone state screen

If Phase A survives, run on at least DistilGPT-2 and Pythia-70M before real-reader promotion.

For each backbone use multiple prompts and >=3 independent intervention seeds. Materialize old and fresh-current residual/Hidden/KV state and measure:

- exact transport error at every persisted layer/tensor used by inference;
- current-only downstream logits and top-1;
- state dependence of the learned/constructed transform;
- transform composition for UPDATE->UPDATE, UPDATE->REVOKE, DELETE->RESTORE and ABA;
- unrelated-Pod locality;
- stale-state forced-materiality control;
- transport cost, bytes touched and memory versus baselines A-H.

Kill if exactness fails in any interpreted arm, if a Pod-local/late-bound equivalent exists, or if measured mutation-to-ready advantage is not material.

## Phase C — real LINK->Pod reader

Only after Phase A/B survival. Every interpreted job must remeasure >=0.95 on **every** held-out Symlink template in that exact run.

Across >=3 genuine seeds and >=2 backbone families attack:

- UPDATE;
- RELINK;
- REVOKE;
- SHRED / RESIGN;
- DELETE;
- RESTORE;
- ABA / rollback;
- stale Bank replay;
- stale router replay;
- stale resolved-payload replay;
- stale post-read Hidden replay;
- stale KV replay;
- in-forward TOCTOU;
- key/reconstruction attacks.

Required programme gates remain:

- <=2% old/deleted-generation leakage;
- >=90% UNKNOWN in declared missing-key scope;
- exact generic bypass or <=0.05 nats generic divergence;
- independent J-space/J-lens content audit only;
- <=5% normal inference overhead;
- matched total memory;
- material mutation-to-ready advantage over the strongest guarantee-matched baseline.

## Current 2025-2026 boundary

The following receive no standalone novelty credit and must be treated as strong nearby art/baselines:

- KVEraser (2026): learned localized KV steering; explicitly approximate versus exact suffix recomputation.
- Models Take Notes at Prefill / programmable KV (2026): editable/composable downstream notes, errata and cache surgery with strong latency results; not an exact state-transport guarantee in the general lifecycle sense.
- AgentKVShift (2026): probe-guided per-memory residual correction for KV reuse; approximate task-quality objective.
- KV-Direct / residual-stream exact reconstruction (2026): exact KV reconstruction from smaller residual checkpoints, strengthening the exact recomputation baseline.
- Leyline (2026): agent-directed cache edit/splice primitives and exact positional repair where applicable.
- IBM WO2026087278A1 / US family: direct knowledge insertion/modification/deletion through a KV-cache network layer.
- Intel US20260080217A1: gauge-transform KV cache compression; transformation language alone is not novel.
- classical self-adjusting / incremental computation and dependency change propagation.

This is not an exhaustive patentability or freedom-to-operate conclusion.

## Survival meaning

E-000095 survival would **not** yet be a major invention. It would only establish a mechanism worth full lifecycle qualification: a state-dependent exact transform of genuinely mixed neural state that cannot be algebraically factored into late binding or a Pod-local action and that materially beats the strongest exact recompute/dependency baseline.

## Kill rule

Kill E-000095 if any of the following holds:

- exact fresh-state equivalence fails;
- correctness depends on passive freshness checks, masking, deletion or invalidation;
- the transform collapses to a Pod-local action, ordinary late binding, or generic dependency propagation;
- exact baseline recomputation from residual/checkpoint state is equal or cheaper under matched memory;
- the transport must inspect/recompute essentially the same affected work it claims to avoid;
- real-reader or lifecycle gates fail.
