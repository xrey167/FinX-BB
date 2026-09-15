# E-000099 — Dense Nonlinear Influence / Sparse Exact-Repair Kill Screen

Date: 2026-09-06
Status: **PREREGISTERED FALSIFICATION SCREEN — not a novelty claim**

## Trigger

E-000095/E-000096 falsified reusable cross-context correction receipts, E-000097 reduced associative exact recomposition to generic dynamic composition, and E-000098 falsified a shared exact linear correction span even when held-out coefficients were supplied by an oracle. The remaining standard-transformer transport idea is therefore session-specific exact repair from actual cached computation.

Before searching for a sophisticated exact transport operator, this screen asks a narrower systems question: **after one material Pod edit enters the last real memory-read-equivalent site, is the downstream nonlinear suffix sparse enough in ordinary transformer coordinates that an exact selective channel/neuron repair could skip most of the suffix work?**

This does not test semantic dependency elision, passive freshness, cache invalidation, or approximate steering.

## Candidate class under test

The killed class, if the screen passes, is limited to exact repair schemes that obtain their systems advantage by leaving most ordinary downstream scalar coordinates untouched and selectively recomputing only a sparse subset of residual, attention-QKV, or MLP-expansion coordinates in the existing pretrained backbone.

It does **not** kill:

- a new compact nonlinear coordinate system that exactly represents the affected computation;
- an algebraic transform that updates dense state with materially less work than evaluating the ordinary suffix;
- architectural retraining that enforces a new exact sparse structure while preserving capability;
- causal-lineage discovery that produces a smaller exact computational quotient not expressible as ordinary coordinate sparsity.

## Registered models and intervention

Frozen pretrained families:

- `distilgpt2`;
- `EleutherAI/pythia-70m`.

Seeds: `0,1,2`.

For each model/seed, generate 64 deterministic random token contexts of length 16. Inject one old or new controlled Pod payload at block `len(blocks)-3`, leaving two full nonlinear transformer blocks after the intervention. The old/new payloads use distinct output-embedding directions normalized to RMS 2.0, matching the controlled transport family used by E-000095 through E-000098.

No training or threshold tuning occurs after results are seen.

## Recorded downstream coordinates

For every suffix block, capture the target-token values of:

1. block residual output;
2. attention QKV projection output (`c_attn` on GPT-2-family blocks, `query_key_value` on GPT-NeoX-family blocks);
3. MLP expansion preactivation (`c_fc` on GPT-2-family blocks, `dense_h_to_4h` on GPT-NeoX-family blocks).

Also capture final logits.

For each coordinate family and each context, compare old versus fresh-new executions and report changed-entry fractions for absolute-delta thresholds:

- exact inequality (`delta != 0`);
- `1e-8`;
- `1e-6`;
- `1e-4`.

Report mean, minimum and fifth-percentile changed fraction across contexts. Report final-logit material-edit rate using max-absolute logit delta `>1e-4`.

## Controls

V1. **Material edit:** final-logit material-edit rate must be >=0.95.

V2. **Actual nonlinear suffix:** at least two full blocks remain after the read site.

V3. **Determinism:** repeating the old-payload forward pass must reproduce every registered captured tensor and final logits byte-for-byte / exact tensor equality.

V4. **Causal locality:** at the first downstream residual output, all positions before the final intervention position must be exactly unchanged between old and new runs. This rejects accidental whole-sequence corruption by the hook.

V5. Both backbone families and all three seeds must satisfy V1–V4 before a family-level kill is interpreted.

## Registered kill rule

Call a model/seed cell `DENSE_MATERIAL_INFLUENCE` only if, for **every** registered downstream residual/QKV/MLP coordinate family:

- the fifth-percentile exact-inequality fraction across contexts is >=0.99; and
- the fifth-percentile fraction with absolute delta >1e-6 is >=0.95.

Kill **ordinary sparse-coordinate exact repair as the major systems-advantage seam** only if all six backbone×seed cells are `DENSE_MATERIAL_INFLUENCE` and V1–V5 pass.

The consequence is scoped: a local Pod edit materially changes nearly all ordinary scalar coordinates of the nonlinear suffix, so an exact method that merely chooses a sparse set of ordinary channels/neurons to recompute cannot skip most suffix arithmetic on these backbones. The strongest baseline remains patching the exact read-site delta and performing the minimal suffix recomputation / exact residual-to-KV reconstruction.

## Non-kill outcome

If any model/seed exposes a reproducibly sparse ordinary-coordinate affected set under this registered intervention, do **not** kill the seam. Promote a follow-up that measures whether the sparse set can be discovered cheaply, remains stable across real LINK->Pod reads and lifecycle edits, and beats generic dependency/change propagation.

## Prior-art boundary

No novelty credit is assigned to selective recomputation, sparse attention, pruning, activation sparsity, Jacobian/JVP linearization, CacheBlend-style partial recomputation, KVEraser-style approximate steering, KV-Direct residual reconstruction, or ordinary dependency tracking.

Recent evidence already makes exact suffix recomputation the correct baseline: KVEraser states that exact local KV erasure requires recomputing the suffix and substitutes learned approximate steering; KV-Direct shows K/V are deterministic projections of the residual stream and supports bit-identical reconstruction in its evaluated architectures. E-000099 asks only whether ordinary-coordinate sparsity can materially shrink that exact affected work.

## Major-break gates remain unchanged

Even a failure of this kill screen is not a breakthrough. Any promoted mechanism still requires real LINK->Pod reader >=0.95 on every held-out template, >=3 seeds, >=2 backbone families, <=2% old/deleted leakage, >=90% UNKNOWN in declared missing-key scope, exact bypass or <=0.05 nats generic divergence, stale Bank/router/resolved-payload/Hidden/KV attacks, UPDATE/RELINK/REVOKE/SHRED/DELETE/RESTORE/ABA/rollback/TOCTOU, key/reconstruction attacks, independent J-space/J-lens audit only, <=5% normal inference overhead, matched memory, and a material fleet-level mutation-to-ready advantage over the strongest guarantee-matched baseline.
