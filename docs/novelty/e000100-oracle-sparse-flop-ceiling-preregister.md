# E-000100 — Oracle Sparse-Coordinate FLOP Ceiling

Date: 2026-09-06
Status: **PREREGISTERED FALSIFICATION SCREEN — not a novelty claim**

## Trigger and independence

E-000099 showed extremely dense downstream influence on DistilGPT-2, but its deliberately severe `>=0.99` exact-QKV p05 rule did not fire on Pythia-70M because roughly 2–4% of registered QKV coordinates remained numerically unchanged in some contexts. Therefore E-000099 does **not** kill sparse ordinary-coordinate repair under its preregistered rule.

E-000100 asks the systems question that E-000099 did not preregister: **even granting an oracle that tells a repair engine exactly which output rows changed, is the maximum arithmetic that ordinary output-coordinate selective recomputation could skip material?**

To avoid post-hoc reuse of the E-000099 intervention sample, this run uses fresh intervention seeds `10,11,12`, new deterministic context streams, and different old/new payload token directions. No E-000099 numerical case is reused for the decision.

## Candidate class under test

The candidate class is exact repair that keeps the existing dense pretrained transformer operators and saves work by recomputing only output coordinates/rows that changed after the Pod edit.

The candidate receives an **oracle changed-row set** after the mutation. This is deliberately stronger than any deployable sparse detector: discovery cost is zero. If even this oracle cannot skip materially many dense-projection FLOPs, learned routing or dependency metadata cannot rescue the ordinary-coordinate sparsity route.

Excluded from the kill:

- a new compressed nonlinear coordinate system;
- low-complexity algebraic evaluation of dense changed outputs without ordinary row dot-products;
- architecture retraining that changes the operators themselves;
- exact state transforms whose cost advantage is not output-row skipping.

## Registered models and fresh interventions

Frozen `distilgpt2` and `EleutherAI/pythia-70m`.

Fresh seeds: `10,11,12`.

64 contexts per seed, sequence length 16, controlled payload RMS 2.0, read site `len(blocks)-3` so two nonlinear blocks remain.

Payload directions use fresh token formulas distinct from E-000095–E-000099. Context RNG is also distinct.

## Dense projections measured

For every suffix block capture target-token outputs from all major dense projections:

GPT-2 family:

- attention `c_attn` (QKV);
- attention `c_proj`;
- MLP `c_fc`;
- MLP `c_proj`.

GPT-NeoX/Pythia family:

- attention `query_key_value`;
- attention `dense` output projection;
- MLP `dense_h_to_4h`;
- MLP `dense_4h_to_h`.

For a projection with `N` stored weights and output width `m`, an ordinary output row costs `N/m` multiply-accumulate coefficients. For each context and projection, the oracle marks every output coordinate whose old and fresh-new values differ exactly. The most favorable possible ordinary-row selective repair cost is the sum of row costs for only those changed outputs.

Define:

`oracle_skip_fraction = 1 - changed_row_weight / total_projection_weight`.

This ignores indexing, detection, memory traffic, nonlinear functions, layer norms, attention softmax, residual adds, and all other repair work. It is therefore an **upper bound** on the FLOP fraction that this class could save relative to evaluating all registered dense projections.

Also report the same bound using `abs(delta)>1e-6`, but the exact-inequality bound governs lifecycle-grade equality.

## Controls

V1 final-logit material-edit rate >=0.95.

V2 at least two suffix blocks.

V3 repeated old forward is exactly deterministic for all captured projection outputs and final logits.

V4 first downstream block residual prefix positions before the edited final token remain exactly unchanged.

V5 both families and all three fresh seeds pass V1–V4.

## Kill rule

For each model/seed compute the oracle exact skip fraction per context across all registered dense projection weights and report mean, maximum, and p95.

Call a cell `NO_MATERIAL_SPARSE_FLOP_HEADROOM` if:

- the **maximum** oracle exact skip fraction over all 64 contexts is <=0.05; and
- the maximum oracle `>1e-6` skip fraction is <=0.05.

Using the maximum, rather than mean, is intentionally favorable to the sparse candidate: no tested session may offer >5% dense-projection arithmetic headroom.

Kill ordinary output-coordinate sparse exact repair as a major systems-advantage seam only if all six fresh backbone×seed cells satisfy this rule and V1–V5.

This does not say suffix recomputation is information-theoretically necessary. It says ordinary dense row skipping cannot provide a material fleet-level advantage in this registered standard-backbone regime, even with perfect free knowledge of affected rows.

## Baseline and novelty boundary

The strongest baseline remains exact read-site patch plus minimal suffix recomputation / KV-Direct-style exact reconstruction. Selective recomputation, pruning, sparse activation routing, Jacobian/JVP linearization, learned cache repair, KVEraser and CacheBlend are baseline/prior-art territory.

All major-break lifecycle, real-reader, audit, memory and overhead gates remain unchanged.
