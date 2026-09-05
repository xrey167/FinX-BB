# E-000097B — frozen semantic address + exact route + value-side calibration

Date: 2026-09-05
Status: preregistered before execution. **Do not execute as positive evidence unless E-000097A qualifies all three seeds.**
Breakthrough: false by construction. No novelty credit for curriculum, frozen teachers, argmax routing, distillation, exact retrieval, or value calibration.

## Question

E-000092 showed that post-hoc exact resolve+deref can make unrelated mutable-state changes byte-identically invisible, but can lose >0.02 held-out capability. E-000094/E-000095B showed that learning an exact boundary from scratch/jointly can collapse capability.

E-000097B tests one narrower explanation: the dense teacher may have already learned the correct immutable semantic identity, while its value/injection side has adapted to receiving a soft mixture. If so, freeze the learned semantic address function, switch executed routes to exact argmax, and retrain only the value/injection calibration under those exact routes.

## Preconditions

E-000097A must have completed on genuine training seeds 0,1,2 and every seed must satisfy:

- candidate >=0.95 on every held-out template 8..11;
- full-vocabulary top-1 >=0.95 on every held-out template;
- exact no-memory bypass;
- immutable alias-row resolve argmax >=0.95 on every held-out template.

If any seed fails, E97B remains blocked and its result cannot rescue the route.

## Student initialization and frozen boundary

For each qualified seed:

1. train the dense E81-style teacher using the same fixed regime;
2. retain all teacher weights;
3. replace resolve+deref forward distributions with E92's exact one-hot argmax hooks;
4. freeze the pretrained LM and every semantic-address / pointer-resolution parameter, including at minimum relation/key/query layers, null keys, q_deref, deref layer norms/scales/biases, v_link, marker validity machinery, and addressing scale;
5. train only target-payload/output calibration parameters under exact routes: `v_proj`, `o_proj`, `inject_gain`, and (if needed by existing UNKNOWN semantics) `null_value`;
6. no dense mutable attention is allowed during student inference.

The exact route itself has zero gradient and zero novelty credit. This is not straight-through routing.

## Training

- same public frozen backbone;
- same synthetic real-symlink world family;
- seeds 0,1,2;
- initial calibration budget: 1200 steps after the qualified 3000-step teacher;
- answer supervision on live/dangling/revoked/shred examples from the existing regime;
- no mutable payload is allowed into the semantic address scorer;
- teacher compute is reported separately from inference/mutation-time cost.

## Gates

Every seed must retain:

- candidate >=0.95 on each held-out template;
- full-vocab top-1 >=0.95 on each held-out template;
- <=0.02 per-template candidate degradation from its own qualified dense teacher;
- exact no-memory bypass;
- exact one-row mutable payload support per read.

Lifecycle cells on fresh independent worlds:

- unrelated Pod UPDATE: selected immutable row unchanged; routing/hidden/KV/full logits/stale continuation byte-identical;
- unrelated alias RELINK: same invariance;
- relevant target UPDATE: current answer changes to the new current payload;
- queried alias RELINK: answer follows new target;
- relevant REVOKE and SHRED: UNKNOWN/closed; old-object leakage <=0.02;
- rollback restores current answer;
- missing/dangling target -> UNKNOWN >=0.90.

No attack result is promoted unless all three reader gates pass.

## Failure decision

E97B is falsified if the frozen teacher address plus exact route cannot recover >=0.95 capability across all three seeds within the fixed calibration budget, or if recovery requires unfreezing the mutable/semantic address boundary.

If falsified, stop tuning exact one-row routing in this adapter family. The next architecture must use a different semantic identity representation or move the exact mutable boundary outside this neural adapter.
