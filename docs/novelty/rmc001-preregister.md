# RMC-001 — native rounding-margin repair certificates

Registered 2026-09-05 BEFORE numerical execution. Parent FIR001: `b3cfa14069787893e0e934c490ca795f84978498`. Status at registration: UNRUN. Candidate-mechanism screen, not a major-invention claim.

## Motivation and candidate

E90 showed that global coupling can make exact mathematical dependency cones dense. E89 showed generic exact delta propagation can perform the same affected work as replay. QCR001 showed that exact lifecycle semantics must be evaluated in the prescribed native arithmetic and that rounding cells—not a generic error tolerance—determine whether an approximate represented-real value stores identically.

RMC001 tests a different possibility: a mathematically affected persistent coordinate need not be recomputed if a SOUND bound proves its entire revised pre-quantized value remains inside the same native storage cell. Rather than computing every coordinate's delta, maintain hierarchical block certificates containing a conservative sensitivity bound, a certified rounding margin, and accumulated uncertainty. If a source edit fits inside a block's remaining margin, all quantized writes in the block are reused exactly without evaluating their individual deltas. If a certificate fails, descend or perform exact sparse/full refresh. This is proof-directed reuse INSIDE a dense dependency cone, not graph independence.

The candidate is deliberately evaluated first in an exact integer/ties-to-even quantized persistence operator so the scientific question is separated from GPU/library nondeterminism. This is not yet BF16/FP32 pretrained evidence.

## Fixed exact operator screen

Five parameter seeds 0..4. Width d=1024, depth L=24, quantization step Q=4096, eight edit-active coordinates and dense nonzero mathematical couplings from those coordinates into every protected coordinate. Old preactivations are constructed strictly inside their quantization cells. Active rows carry strong source transport; protected rows carry dense small couplings so every protected preactivation changes as a real/integer quantity under a source edit while the stored quantized value can remain identical.

Use exact signed integer arithmetic and a hand-implemented round-to-nearest-ties-to-even quantizer. No floating tolerance is permitted. For each seed execute 64 sequential source edits including small updates, sign reversals, a source-to-zero/NEVER transition, and restoration. At every edit compare EVERY stored persistent layer write against a clean full rebuild from the revised source. Preserve accumulated certificate uncertainty; do not reset certificates between edits unless the algorithm actually pays a refresh.

## Strong baselines and cost accounting fixed in advance

A. Dense exact rebuild: recompute every dense matrix row at every layer.
B. Dense dependency-cone replay: same here because active input columns have nonzero weights into every row.
C. Strong exact sparse-delta baseline: cache each row's exact preactivation and, for every changed input coordinate, compute its exact contribution to EVERY downstream row before requantizing. This baseline gets the same old persistent states, weights, edit deltas, exact preactivation cache and total-array memory budget. It is stronger than graph replay and is the primary computational baseline.
D. Candidate block-margin certificate: for each protected block, update only a conservative uncertainty envelope using a stored block sensitivity maximum and the L1 size of the changed quantized input. A block may be reused only if the complete envelope remains strictly inside every represented rounding cell. Failed blocks pay exact refresh. Active rows use exact sparse delta. All comparison counts include certificate checks, exact active-row arithmetic and refresh work.

Report integer multiply/add-equivalent counts separately from Python wall time. No wall-time >=10x application claim may be made from Python. Candidate must beat sparse-delta operation count by >=10x across every seed to remain interesting. Full-rebuild ratios are secondary.

Matched auxiliary memory is reported explicitly. Giving the sparse-delta baseline the candidate's extra unused bytes is permitted; no candidate win is assigned merely to storing more state.

## Adversarial/correctness controls

- A certificate with a non-strict boundary test must fail a midpoint/tie control.
- Tampered sensitivity bound or tampered margin must be detected by an offline exact verifier; silent false-negative dependence invalidates the arm.
- An edit large enough to cross a protected block margin must force refresh; stale reuse is prohibited.
- Accumulated uncertainty across many individually safe edits must eventually trigger refresh when its certified cell enclosure is exhausted.
- An intentionally leaky/high-gain control should cause broad certificate failure and fall back toward sparse-delta/full work without sacrificing equality.
- Compare UPDATE, ZERO/NEVER and restoration; no-op edits must preserve exact state.

## Trained representation extension, only if operator screen survives

If the exact screen passes its >=10x sparse-delta operation gate, a separate post-registered extension may train a small quantized persistent network with an edit-margin regularizer that maximizes `rounding_margin - certified_source_influence` for non-source-active blocks while preserving task loss. Any such extension must be explicitly registered AFTER the operator result and BEFORE training. No trained result is part of this preregistration.

## Prior-art boundary before execution

No novelty credit is assigned to: interval/abstract-interpretation neural verification, quantization robustness certification, RLibm-style rounding intervals, Neural Caching activation-region reuse, generic neural incremental computation, sparse delta/event-driven inference, cache invalidation, ordinary dependency repair, or quantization itself. The novelty question—if the mechanism survives—is narrower: whether persistent neural lifecycle repair can use write-time native rounding-margin certificates to soundly prune mathematically dependent state without computing each dependent delta, and whether training can create such certified margins with useful semantics and low inference overhead.

No claim that this concept is novel is made before a direct 2025–2026 paper/patent search and the strongest-baseline experiments.

## Unchanged system gates

RMC001 alone cannot qualify >=10x end-to-end mutation-to-ready, <=5% inference-throughput loss, matched complete system memory, >=95% fresh/paraphrase/lifecycle reading, >=90% scoped UNKNOWN, <=0.05 nats generic divergence/exact bypass, independent J-space, generation/publication safety, three trained seeds, or a second backbone. Five operator seeds are not trained-reader seeds.
