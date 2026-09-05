# E-000052 — BOS plus two-channel null: joint reading/deletion/locality test

This composes E-000050's BOS training fix with E-000022's match gate + two-channel null, with no new mechanism.
Training and evaluation generic templates are disjoint; overlap = 0 by runtime assertion.

**Joint breakthrough criterion: FAIL**

| criterion | required | worst seed | result |
|---|---|---|---|
| on/train/active_correct | >= 0.95 | 0.9969 | PASS |
| on/heldout/active_correct | >= 0.95 | 0.9712 | PASS |
| on/revoke_train_min | >= 0.95 | 1.0000 | PASS |
| on/revoke_heldout_min | >= 0.95 | 0.9900 | PASS |
| on/shred_train_min | >= 0.95 | 1.0000 | PASS |
| on/shred_heldout_min | >= 0.95 | 0.9900 | PASS |
| on/broken1_unknown | >= 0.9 | 0.8750 | FAIL |
| on/heldout/revoked_deleted_object | <= 0.02 | 0.0000 | PASS |
| on/generic/kl_to_base | <= 0.05 | 1.1901 | FAIL |
| off/heldout/active_correct | <= 0.85 | 0.4950 | PASS |

Decision rule:

JOINT PASS only if every on/* criterion passes on the worst seed and the reverse-control row also passes. That licenses only: 'this frozen-GPT-2 external memory simultaneously met the programme's reading, deletion/refusal and unrelated-text locality bars under the registered synthetic fact protocol.' It does not license a novelty claim, LLM-scale claim, or pretrained-weight unlearning claim. If generic KL remains > 0.05 while reading/deletion pass, locality is the remaining blocker. If reading/deletion regress while KL improves, the two fixes do not compose. If the reverse control fails, the E-000050 position-0 explanation did not transport cleanly and the joint result is not interpreted as a BOS composition.

Scope: frozen GPT-2 small; synthetic, single-token facts; external mutable memory. No pretrained-weight unlearning claim.
