# BHC-001 — bounded-horizon coalescence

Registered BEFORE numerical execution on 2026-09-05. Parent QCR001 f046318350206899ab1fd68d8b09be4f5cfc4fd4. Status: UNRUN. This is a candidate/baseline and correctness-boundary screen, not an invention claim.

## Candidate

Avoid fixed response charts by exactly replaying affected recurrent writes until the revised COMPLETE future-relevant state equals the corresponding cached old state, then reuse the identical-forcing suffix. This is conventional change propagation; give the strongest baseline the same stopping rule. Test whether real-valued contraction/fading memory certifies a finite exact replay horizon in native arithmetic. A finite-state map need not be synchronizing.

## Fixed tests

1. Four native arithmetic formats: BF16, FP16, FP32, FP64. Let u be the spacing above 1; choose b=1+u (odd significand) and a=b+u (even). Check the real contraction F(h)=b+(h-b)/2 and its explicitly staged native update. Exact rational round-to-nearest-even in this binade independently checks the proposed two fixed points. Include b=1 (even) as a coalescing positive control. A verified pair of distinct fixed points under constant forcing establishes non-coalescence for arbitrarily many repeats; no finite numerical loop is misreported as an infinity proof.
2. Five parameter seeds per format; four persistent layers of 16 coordinates. Nonnegative dyadic row-stochastic matrices (each row sums exactly to1). Synchronous nonlinear transition: first layer uses itself; later layers use the previous-time lower layer. Each output is b+0.5*tanh(W*(parent-b)), with prescribed native primitive ordering. In real arithmetic the complete transition is globally <=0.5-Lipschitz in max norm. A scalar source adds u to the first layer at the source boundary; no later source input. Measure every complete persistent write for256 steps, source-present versus NEVER source, first fixed-pair time, first complete-state equality and all layer differences. Check fixed pairs with one extra transition. No native pretrained system or trained seed is implied.
3. Keep the even-bias nonlinear controls whether or not they coalesce. A distinct true-contractivity and rounding argument must not be conflated with testing an entire real-valued nonlinear dynamical system using float64.
4. Positive hybrid reference: common EXOGENOUS event resets only the first layer at step32 in both worlds. Execute512 steps. Compare replay-until-COMPLETE-state-equality and a deliberately unsafe first-layer-only stop against every fresh write. Keep initial boundary in the trajectory. Conventional exact change propagation receives identical states, inputs, comparison and stopping rule. Count actual transition calls; no system-speed or novelty credit from a planted reset. Do not introduce a reset only in the repaired world.
5. Encode the exact scalar fixed-point witnesses as dyadic rationals and independently check with standard-library arithmetic. A subsequent diagnostic rescaling can expose a surviving ULP difference; this is not semantic factual recall or a claim of material normal-inference quality loss.

## Contract limits

Replay reuse needs equality of ALL state that can affect the future plus identical future inputs/randomness/source reads/execution configuration. This constructed recurrence has no additional KV cache, receipts, RNG state, token feedback or source reads. It cannot authorize reuse in a richer system whose hidden state differs. Earlier materialized writes must still be repaired/revoked; late equality alone does not clean earlier snapshots.

All repair equality gates use shape/dtype/bytes, not a small-error threshold. A proof for a constant suffix is not a certificate for arbitrary future source/context revisions. Contraction, quantized dynamical effects, exact coalescence and dependency propagation receive no novelty credit. Non-coalescence of one family does not rule out engineered synchronizing representations or efficient exact alternatives.

No >=10x complete mutation-to-ready, <=5% throughput loss, matched full memory, >=95% fresh/paraphrase/lifecycle reading, >=90% scoped UNKNOWN, <=0.05 nats generic divergence, independent J-space, generation-publication safety or trained-second-backbone gate is measured here.

Primary prior-art leads found before execution include Gated KalmaNet v3 (2511.21016,2026), Expansion Span (2025), quantized contraction algorithms (1006.3919), numerical coalescence (1912.04241), and Subtract, Transport, or Replay v2 (2607.27539). Their fading-memory/approximate-performance claims are not interpreted as universal exact-deletion claims.
