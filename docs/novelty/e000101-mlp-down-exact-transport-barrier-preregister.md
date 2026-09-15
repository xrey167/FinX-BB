# E-000101 — First-MLP Exact-Transport Barrier

Date: 2026-09-06
Status: **PREREGISTERED STRUCTURAL KILL SCREEN / not a novelty claim**

## Trigger

E-000095/E-000096 killed calibration-fitted cross-context revision receipts, E-000097 killed generic associative recomposition, E-000098 killed a shared exact correction span even with oracle held-out coefficients, and E-000100 killed ordinary output-row/channel sparse exact repair as a material systems route in the registered two-block standard-transformer suffix.

The remaining standard-model idea is to exploit exact operator-local update algebra: propagate a Pod edit through LayerNorm/attention/residual operations using cached intermediates, hoping the correction remains compact enough to avoid ordinary dense suffix work.

E-000101 asks where that compactness first fails in the actual frozen backbones. It focuses on the first downstream MLP-down projection after a real nonlinear activation. This is the first point where an exact session-specific activation correction is multiplied by a large fixed dense matrix.

## Candidate class under test

The class is **ordinary-coordinate operator-local exact delta propagation** in an unchanged pretrained transformer:

1. keep the original DistilGPT-2 / Pythia-70M dense weights and nonlinearities;
2. inject one canonical old->new Pod payload at the same last-read-style residual site used by the current structural programme;
3. permit arbitrary exact cached intermediates and exact closed-form update algebra before the first downstream MLP-down;
4. at the MLP-down boundary, represent the changed activation in ordinary MLP coordinates and use the original dense projection;
5. permit a free oracle to identify unchanged MLP input coordinates and unchanged MLP output rows;
6. permit a fleet batch to factor the matrix of activation corrections across sessions if it has a compact exact linear rank.

The class explicitly excludes a genuinely new exact matrix representation/algorithm for the frozen MLP-down weight, a retrained lifecycle-native architecture, or a nonlinear symbolic quotient that does not materialize the ordinary MLP activation vector. Those remain possible escape routes.

## Fresh protocol

Backbones:

- `distilgpt2`;
- `EleutherAI/pythia-70m`.

Fresh intervention seeds: `20,21,22`.

Per seed:

- 128 independently generated contexts;
- sequence length 16;
- payload RMS 2.0;
- the read site is `max(0, n_blocks-3)`, leaving two complete nonlinear transformer blocks;
- one fixed old payload and one fixed new payload are used across all 128 contexts for that seed;
- capture the final-token **input** and **output** of the first downstream MLP-down projection, plus final logits;
- repeat the old forward exactly before interpreting any delta.

No threshold or seed may be changed after numerical results are observed.

## Measurements

For every context:

1. exact fraction of changed MLP-down input coordinates (`delta != 0`);
2. material fraction at `abs(delta) > 1e-6`;
3. exact fraction of changed MLP-down output coordinates;
4. material output fraction at `abs(delta) > 1e-6`;
5. **oracle Cartesian coefficient skip ceiling**

   `1 - changed_input_fraction * changed_output_fraction`,

   which grants the candidate free knowledge of both unchanged input columns and unchanged output rows and ignores all detection/indexing/memory costs;
6. final-logit materiality.

Across the 128 sessions, stack the MLP activation correction rows into matrix `D`. Compute singular values in float64 and the best rank-r Frobenius residual. Define the necessary Frobenius bound implied by per-entry state tolerance `1e-6` as

`1e-6 * sqrt(numel(D))`.

Record the smallest rank whose optimal Frobenius residual is no larger than that necessary bound. If no rank below the number of contexts qualifies, report full row-rank requirement. This is a necessary-condition screen for an exact fleet-shared linear factorization, not a proof about nonlinear representations.

## Validity controls

Every interpreted backbone×seed cell must satisfy:

- V1 final-logit material edit rate >= 0.95 at maxabs > 1e-4;
- V2 at least two full transformer blocks remain after the injection site;
- V3 repeated old forward is byte-identical for captured MLP input/output and logits;
- V4 first downstream block prefix positions before the edited final token remain byte-identical between old and new runs;
- V5 MLP-down input width is greater than model hidden width, confirming capture is the expanded nonlinear activation rather than a residual tensor.

If any control fails, that cell is VOID.

## Registered family kill

Kill **ordinary-coordinate operator-local exact delta propagation as the source of the major mutation-to-ready advantage** if all six valid backbone×seed cells satisfy all of:

1. median exact changed fraction of first MLP-down inputs >= 0.99;
2. median exact changed fraction of first MLP-down outputs >= 0.99;
3. maximum oracle Cartesian coefficient skip ceiling across the 128 contexts <= 0.05;
4. the smallest rank not excluded by the `1e-6` necessary Frobenius bound is >= 0.90 * 128.

Interpretation if the kill passes: exact LayerNorm/attention algebra may still save work before this point, but by the first downstream nonlinear MLP projection the canonical edit has become a dense, high-session-rank ordinary activation correction. A successor must therefore change the exact representation/algebra of that dense matrix-vector step, or change the architecture; merely carrying exact deltas through existing operators is not the breakthrough.

## Survival

Do **not** kill the class if any valid cell exposes >5% oracle coefficient headroom or a <=90%-of-context exact linear rank at the first MLP-down. Such a result would justify a targeted exact transport implementation exploiting that structure before broader claims.

## Strongest baseline and non-claims

The baseline remains exact read-site patch + minimal suffix recomputation / residual-stream-to-KV reconstruction, with ordinary dependency propagation and the strongest algebraic simplification allowed.

E-000101 does not claim an information-theoretic lower bound for matrix-vector multiplication. It does not exclude fast exact multiplication exploiting special structure in the frozen weight matrix, nonlinear/symbolic representations, polynomial/rational circuits, or a retrained lifecycle-native model. It also does not establish any real LINK->Pod lifecycle guarantee.

Fresh 2025-2026 boundary includes KV-Direct exact residual reconstruction, KVEraser approximate localized erasure, programmable/editable KV work, and the NeurIPS 2025 structural matrix-vector multiplication literature. Any later fast dense transform must compare against those and identify the specific matrix/state structure it exploits.

All major programme gates remain unchanged: real reader >=0.95 on every held-out template, >=3 genuine seeds, >=2 backbone families, <=2% old/deleted leakage, >=90% UNKNOWN in declared missing-key scope, exact bypass or <=0.05 nats generic divergence, stale Bank/router/resolved-payload/Hidden/KV attacks, UPDATE/RELINK/REVOKE/SHRED/DELETE/RESTORE/ABA/rollback/TOCTOU, key/reconstruction attacks, independent J-space/J-lens audit only, <=5% steady-state inference overhead, matched memory, and a material fleet-level mutation-to-ready advantage over the strongest guarantee-matched baseline.
