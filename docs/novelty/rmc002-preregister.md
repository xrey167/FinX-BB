# RMC-002 — source-anchored native-cell certificates

Registered 2026-09-05 BEFORE numerical execution. Parent RMC001 branch includes the preserved failed preregistered gate. Status: UNRUN. This is a mechanism-discrimination experiment, not a novelty claim.

## Architecture change from the failed RMC001

RMC001 accumulated absolute edit-path uncertainty. Edit/revert cycles therefore consumed certified slack even when the current source returned close to the exact anchor. Its fixed >=10x operation gate failed; this experiment does not retune that threshold or remove the adversarial edit sequence.

RMC002 replaces path-length uncertainty with a source-anchored invariant. For each layer, protected persistent blocks are certified relative to a shared exact source-active anchor vector. At each mutation the candidate computes the current source displacement from that anchor ONCE, forms a conservative block response bound, and reuses a block only when the entire represented-real response enclosure remains in its current native rounding cells. Edit/revert cancellation is therefore represented exactly at the certificate input. If any protected block in a layer cannot be certified, the entire protected layer is exactly refreshed and its anchor is reset to the current source-active vector; no mixed-anchor shortcut is permitted.

The scientific question is whether this proof-directed reuse beats the strongest exact sparse-delta baseline on the SAME mathematically dense operator and SAME edit path, not whether it beats full replay.

## Fixed exact operator screen

Reuse RMC001's five seeds 0..4, width 1024, depth 24, eight source-active coordinates, Q=4096 round-to-nearest-ties-to-even persistence, block width32, and the exact same 64-edit path including UPDATE, ZERO/NEVER, restoration, sign change, and edit/revert behavior. Protected rows have nonzero mathematical source coupling. Every stored persistent write must match a fresh full rebuild exactly.

Use exact integer arithmetic. Certificate cell geometry uses the hand-implemented ties-to-even quantizer and exact safe radii. No floating tolerance can authorize reuse.

## Candidate cost and strongest baselines

A. Dense full rebuild.
B. Dense dependency replay; the mathematical cone is dense.
C. Strong exact sparse-delta baseline from RMC001: exact preactivation cache for every row; computes every changed-input contribution for every downstream row then requantizes. This remains the primary baseline.
D. Source-anchored certificate candidate:
   - compute `||source_current - layer_anchor||_1` once per layer per mutation;
   - one proof check per protected block;
   - exact sparse update of the eight source-active rows;
   - if any protected block fails, refresh ALL protected rows in that layer exactly and reset the layer anchor.

Count integer multiply/add-equivalent work and certificate checks. Count computing an 8-coordinate anchor displacement explicitly. Report but do not use Python wall time as the application gate. The candidate must achieve >=10x operation reduction versus baseline C on EVERY seed on the unchanged main path. If it does not, this mechanism is rejected.

Candidate auxiliary state: one 8-coordinate anchor per layer, per-block certified safe radii/max-sensitivities, active-row exact preactivations, and persistent states. The sparse baseline gets any unused byte budget; no win from memory asymmetry.

## Correctness/adversarial controls

1. Main 64-edit sequence must be every-write exact without changing the RMC001 sequence.
2. Edit/revert to an anchor must restore zero displacement and must not consume margin.
3. A separately declared large source move must fail at least one first-layer block, force a full protected-layer refresh, reset the anchor, and remain every-write exact on subsequent edits.
4. A high-gain/leaky control must force broad refresh and converge toward conventional exact work rather than silently reuse stale blocks.
5. Tampering with a block sensitivity, safe radius, or layer anchor must be rejected by an independent exact verifier.
6. Ties-to-even midpoint parity controls remain mandatory.
7. No-op, UPDATE, ZERO/NEVER and restoration are checked.

## Critical baseline/novelty boundary

Even a >=10x operation result is NOT a breakthrough. The mechanism resembles certified perturbation reuse / robustness verification plus quantized-cell memoization. Before any novelty promotion, search and differentiate 2024-2026 work on interval/abstract neural verification, certified quantization robustness, input-region caching/memoization, incremental neural inference, activation reuse, and correctly-rounded interval methods. RLibm-style rounding intervals, Neural Caching, ordinary robustness certificates, and sparse-delta inference receive zero novelty credit.

A surviving RMC002 result would only justify a POST-REGISTERED trained extension asking whether a neural persistent representation can be trained to create large lifecycle-specific certified native margins at <=5% normal-inference overhead while retaining semantic capability. Training is not part of RMC002.

## Unchanged full-system gates

No operator result qualifies end-to-end >=10x mutation-to-ready, <=5% inference-throughput loss, matched total system memory, >=95% fresh/unseen-paraphrase/lifecycle reading, >=90% scoped UNKNOWN, <=0.05 nats generic divergence/exact bypass, independent J-space audit, generation/publication safety, three trained seeds, or a second trained backbone.
