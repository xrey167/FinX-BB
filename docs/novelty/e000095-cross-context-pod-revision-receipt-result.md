# E-000095 result — compact cross-context Pod revision receipts fail exact transport

Date: 2026-09-05
Status: **DECISIVE FALSIFICATION / REGISTERED FAMILY KILL**

## Decision

Kill the two preregistered compact E-000095 receipt families as exact active lifecycle-transport candidates:

1. one edit-level translation vector;
2. one edit-level per-dimension affine state-dependent receipt.

The same canonical old->new controlled Pod edit was evaluated across disjoint calibration and held-out contexts. The edit was materially behavioral on 100% of held-out contexts in every valid seed. Nevertheless neither receipt reproduced fresh-current final neural state within the preregistered exact thresholds on any seed of either backbone family.

Approximate KL/top-1 preservation is explicitly not lifecycle/deletion correctness and receives zero guarantee credit.

## Execution provenance

Initial workflow run `33989961557`:

- DistilGPT-2 completed scientifically and killed both registered receipt families on seeds 0,1,2.
- Pythia-70M was **VOID as scientific evidence** because reconstructed fp32 hidden states were passed directly to an fp16 LM head, producing a dtype runtime error before metrics were emitted.

Validity repair commit: `788b893ec6acccbf16e09c2988c97cbf13fc2638`.

The only repair was to cast reconstructed final hidden states to the output-head weight dtype for the LM-head application, then convert emitted logits back to fp32 for measurement. The candidate transforms, calibration/test split, seeds, payload, thresholds and decision rules were unchanged.

Corrected workflow run: `33991507053`. Both matrix jobs completed successfully and emitted artifacts.

## DistilGPT-2 — three registered seeds

All seeds:

- `material_edit_rate = 1.0`;
- 64 calibration and 64 disjoint held-out contexts;
- two nonlinear suffix blocks after the intervention layer;
- exact hidden fraction at `1e-6` = 0 for both receipt families;
- exact reconstructed-logit fraction at `1e-5` = 0 for both receipt families;
- decision = `KILL_REGISTERED_COMPACT_RECEIPTS`.

### Seed 0

Exact fresh correction varies materially by context: variation RMS `0.6218187809` versus total correction RMS `0.8605136275`.

- translation: hidden maxabs `33.0386810`, logit maxabs `25.2188454`, KL `0.00022012`, top1 `1.0`;
- diagonal affine: hidden maxabs `25.3917084`, logit maxabs `19.6509876`, KL `0.00024243`, top1 `1.0`.

### Seed 1

Context-variation RMS `0.5510361195`; correction RMS `0.6991489530`.

- translation: hidden maxabs `27.9596405`, logit maxabs `18.5203018`, KL `0.00088880`, top1 `1.0`;
- diagonal affine: hidden maxabs `28.2534332`, logit maxabs `18.0553207`, KL `0.00106982`, top1 `1.0`.

### Seed 2

Context-variation RMS `0.4286148548`; correction RMS `0.7780805826`.

- translation: hidden maxabs `18.7323685`, logit maxabs `14.6875648`, KL `0.0335070`, top1 `1.0`;
- diagonal affine: hidden maxabs `18.6863174`, logit maxabs `13.3321190`, KL `0.0370331`, top1 `1.0`.

The extremely good top-1/KL behavior on some seeds is a useful warning: task-level similarity can coexist with a neural-state error many orders of magnitude above the exact lifecycle contract.

## Pythia-70M — corrected run, three registered seeds

All seeds:

- `material_edit_rate = 1.0`;
- 64 calibration and 64 disjoint held-out contexts;
- two nonlinear suffix blocks;
- exact hidden fraction at `1e-6` = 0 for both receipt families;
- exact reconstructed-logit fraction at `1e-5` = 0 for both receipt families;
- decision = `KILL_REGISTERED_COMPACT_RECEIPTS`.

### Seed 0

Context-variation RMS `0.2175867856`; correction RMS `1.2883870602`.

- translation: hidden maxabs `1.435546875`, logit maxabs `3.0`, KL `0.01608731`, top1 `1.0`;
- diagonal affine: hidden maxabs `1.5010509491`, logit maxabs `3.0`, KL `0.01567239`, top1 `1.0`.

### Seed 1

Context-variation RMS `0.2441389114`; correction RMS `1.5132197142`.

- translation: hidden maxabs `1.8454589844`, logit maxabs `3.0`, KL `0.05247493`, top1 `1.0`;
- diagonal affine: hidden maxabs `1.8066921234`, logit maxabs `3.0`, KL `0.04977756`, top1 `1.0`.

### Seed 2

Context-variation RMS `0.2838684916`; correction RMS `1.6013216972`.

- translation: hidden maxabs `1.86328125`, logit maxabs `5.0`, KL `0.07867634`, top1 `0.96875`;
- diagonal affine: hidden maxabs `2.2574768066`, logit maxabs `5.0`, KL `0.06968119`, top1 `0.96875`.

The final-hidden failure alone is already decisive and is independent of the fp16 output-head quantization details.

## What this kills

Do not promote a context-independent edit translation or a per-dimension affine receipt as an exact reusable old->new Pod transport across already-materialized sessions. The exact correction is materially context dependent after even a two-block nonlinear suffix.

Do not reinterpret the low KL or preserved top-1 cases as lifecycle success. They are approximate behavioral steering only.

This result does **not** prove that every compact nonlinear state-dependent transport is impossible. It specifically closes the two registered E-000095 families.

## Successor boundary

The next valid lane must be genuinely state dependent and nonlinear. However, a generic learned cache-repair MLP, Jacobian/JVP approximation, selective recomputation, cache blending, KVEraser-style steering, AgentKVShift-style residual correction, or KV-Direct-style exact reconstruction receives no novelty credit by itself.

A successor must be evaluated against:

1. fresh full recomputation;
2. exact delta patched at the last real memory-read site followed by minimal suffix recomputation;
3. exact residual/KV reconstruction where applicable;
4. strongest generic learned nonlinear repair at the same memory budget.

The candidate must match fresh neural state, not merely answers or logits approximately, and must ultimately provide a material fleet-level mutation-to-ready advantage under the unchanged major-break lifecycle gates.

## Current prior-art boundary

Current 2026 work already occupies broad learned-cache-repair territory:

- AgentKVShift: shared memory-level residual correction plus token-wise fluctuation for approximate K/V repair;
- KVEraser: learned query-agnostic steering of contaminated KV while reusing the suffix cache;
- Models Take Notes at Prefill: editable/composable downstream KV conclusions;
- KV-Direct / The Residual Stream Is All You Need: exact reconstruction of K/V from residual-stream checkpoints.

These references do not establish the exact cross-session lifecycle transport sought here, but they prevent broad novelty claims for "learn one correction and apply it to cached neural state".

## Programme consequence

**E-000095 is closed.**

Proceed only to a preregistered genuinely state-dependent nonlinear active-transport screen. No major invention is promoted by E-000095.
