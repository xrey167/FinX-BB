# E-000094 — Noninterference / Late-Binding Reduction — RESULT

Date: 2026-09-05
Status: **DECISIVE SCOPED FALSIFICATION — exact noninterference / causal-firewall separation alone is rejected as a major-invention seam**

This result is an architecture-level reduction and executable finite witness. It is **not** a latency lower bound, a universal impossibility theorem, or a claim against active state transformation, exact affected-work compression, or causal-lineage discovery.

## Registered question

Can a two-stream / causal-firewall architecture gain lifecycle novelty merely by enforcing that a large persistent neural state is exactly invariant to mutable Pod state while a small mutable stream remains available to the output computation?

Let static context be `x`, current Pod state be `p`, current query be `q`, persistent reusable state be `R(x,p)`, and output be `Y(x,p,q)`.

The candidate premise is exact Pod noninterference:

`R(x,p1) == R(x,p2)` for every allowed lifecycle state `p1,p2`.

Choose any `p0` and define `F(x) = R(x,p0)`. Then exact noninterference immediately gives

`R(x,p) = F(x)` for every `p`,

so the complete computation factors as

`Y(x,p,q) = G(F(x), p, q)`.

A correctly implemented co-located/external mutable-memory branch can cache the same `F(x)` and late-bind the same current `p` to the same `G`. Therefore the freshness/reuse guarantee obtained **solely from noninterference** is not stronger because the mutable branch happens to live inside transformer blocks.

If a cached substate `M(x,p)` actually varies with `p`, it is outside the exact noninterfering reusable domain and must receive lifecycle work: recomputation, transformation, invalidation, repair, or some other update. That is a different candidate class.

## CI result

GitHub Actions run `33989552119`, job `101369211008`, completed successfully from commit `163f83dce92ffef05960247ad5bc51d0196318fd`.

Focused regressions: **4/4 passed**.

Registered exhaustive assay:

| Check | Cases | Failures |
|---|---:|---:|
| Candidate vs late-bound sidecar direct equality | 23,040 | **0** |
| Registered lifecycle-trace equality | 92,160 | **0** |
| Pod-contamination negative control | 5,760 | **0 missed detections** |
| Pod-dependent mutable substate transitions | 11,520 | **0 unchanged cases** |

The result record emitted:

- `equality_mismatches = 0`
- `lifecycle_trace_mismatches = 0`
- `negative_control_detected = 5760 / 5760`
- `mutable_substate_changed = 11520 / 11520`
- `kill_screen_pass = true`
- `decision = KILL_NONINTERFERENCE_ALONE_AS_NOVELTY_SEAM`

The uploaded CI artifact digest was `63a81dce5cfa7f168568ad9d210c47ac26f99659645108e9a6be5bfb0ccf22b9`.

## Interpretation

This kills a broad family of tempting follow-ups **only when their claimed advantage is exact lifecycle isolation of a reusable stream**:

- causal-firewall state segregation;
- static/mutable two-stream transformer layouts;
- deep static state plus late mutable cross-attention;
- architectures claiming revocation locality solely because mutable Pod content never enters the cached static stream.

Those may still be useful engineering designs. They receive no major-invention credit from the isolation property itself.

The reduction is consistent with the already-tight prior-art boundary around external/layer-local memory, cross-attention, memory layers, prompt/prefix caching, and static/dynamic prompt segregation. Those systems are not asserted to be identical to every possible Symlink–Pod architecture; they establish that late-bound mutable memory is already a strong ordinary baseline.

## Interaction with earlier kills

E-000094 closes a different escape route from E-000093:

- **E-000093**: if freshness is only decoded from an otherwise unchanged cached neural artifact, a co-located sidecar can store the same sufficient statistic and make the same decision.
- **E-000094**: if mutable Pod state is instead prevented from entering the persistent reusable state altogether, the computation factors into static reusable state plus a late-bound mutable argument.

The older revocation-equivariance screen also does not escape this boundary in its pinned form. Its exact phase repair was independently shown to have an exact ordinary late-bind equivalent with the same one-action update complexity. That historical result remains a baseline warning rather than new current evidence.

## What remains open

A successor must change the computational frontier rather than the location of freshness information. The most defensible remaining classes are:

1. **State-dependent exact lifecycle transport:** transform already-materialized mixed neural state to the current lifecycle state using information from the actual joint state, while matching fresh recomputation and materially beating the strongest recompute / KV-edit / sidecar baselines.
2. **Exact affected-work compression beyond generic dependency/change propagation:** an architecture whose minimal repair work is demonstrably smaller than what a guarantee-matched incremental-computation baseline can achieve at matched memory.
3. **Neural causal-lineage discovery/certification:** derive a sound and sufficiently complete causal source set from computation rather than receiving ordinary source/dependency metadata.

The first class is already heavily crowded by approximate KV-editing work, so **exactness plus a material systems advantage** is mandatory; approximate steering or near-full-recompute task accuracy is not enough.

## Major-break gates remain unchanged

Any surviving successor still requires:

- every interpreted real-reader job >=0.95 on every held-out real-symlink template;
- >=3 genuine seeds;
- >=2 backbone families;
- <=2% old/deleted-generation leakage;
- >=90% UNKNOWN in declared missing-key scope;
- exact bypass or <=0.05 nats generic divergence;
- stale Bank/router/resolved-payload/Hidden/KV attacks;
- UPDATE, RELINK, REVOKE, SHRED, DELETE, RESTORE, ABA, rollback and in-forward TOCTOU;
- key/reconstruction attacks;
- independent J-space/J-lens content audit only;
- <=5% normal inference overhead;
- matched memory;
- and a material mutation-to-ready advantage over the strongest guarantee-matched baseline.

**Decision:** reject exact noninterference / causal-firewall segregation **by itself** as the next major-invention seam. Continue only with an active computational mechanism or a genuinely new exact-work / causal-lineage frontier.
