# R350b — Prospective Rebinding Loss (PRL)

Executed in GitHub Actions on 2026-09-14, run `34899352790`, commit `918c78a5065f3bc8488ac3f4550c2c3954519914`.

## Passing result

- Contract: **PASS**
- Five independent seeds: 17, 29, 43, 71, 101
- Ordinary current-snapshot training, mean future rebind accuracy: **0.3896218754**
- PRL mean future rebind accuracy: **1.0**
- Future rebinding gain: **+0.6103781246**
- Baseline descriptor cosine drift after world rewrite: **0.5969720244**
- PRL descriptor cosine drift: **0.0000668802**
- Baseline low-two-bit value leakage probe accuracy: **0.58755**
- PRL value leakage probe accuracy: **0.256375** (approximately four-class chance)
- Minimum PRL future rebind accuracy across seeds and future-world cycles: **1.0**
- Inference-time world-rebinding gradient steps: **0**
- Report SHA256: `c34ef7511acb2f20d7bab193b2c2d5bf9a91670583d422484093581325af18be`
- Evidence ZIP SHA256: `697925ac4bca3e4b87a083f0eec3e55fda4e7fd5b4789aa4d02cc3a9c61a2ef7`

## Mechanism under test

For one semantic query, training samples two independently redrawn world snapshots `W1` and `W2`. The compiler is deliberately allowed to see both the semantic representation and current payload features, so ordinary training can leak current-world values into the retained descriptor. PRL adds two constraints:

1. **cross-world descriptor/value swap:** the descriptor compiled under `W1` must execute correctly using late-bound values from `W2`, and vice versa;
2. **descriptor invariance:** the two descriptors for the same query semantics are penalized for drifting with the world payload.

The passing gate therefore shows that this objective can force a compact learned state to discard mutable payload information while preserving the semantic operation required to act on future values.

## Claim boundary

Counterfactual training, invariant representation learning, domain randomization, consistency regularization and swap-based disentanglement all have substantial prior art. This result does **not** establish novelty for those ingredients. The narrower research hypothesis is the use of cross-world swap consistency specifically to train a retained neural activation to be **future-bindable** to authoritative mutable knowledge. A direct literature/patent audit and a pretrained-Transformer version remain mandatory before any novelty claim.
