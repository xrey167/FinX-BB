# R295b Result — Twin-Plane Separation

Status: **real-backbone separation gate passed; not DoD.**

Backbone: frozen `Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.

The earlier integrity check was invalid because it compared a probe extracted inside a left-padded batch against a standalone probe, measuring a different numerical execution shape. R295b corrected this by repeating the exact same standalone `batch=1` B-plane input before and after all Port/bridge/world-state work.

## Corrected result

- held-out entity/value-pair accuracy: **100%**;
- late-bound alias accuracy: **100%**;
- online world-update accuracy: **100%** over 10,000 post-training world writes;
- optimizer steps after world updates: **0**;
- optimizer owns any backbone parameter: **false**;
- sampled backbone parameter witness unchanged: **true**;
- B-plane hidden-state max delta after Port/world changes: **0.0**;
- B-plane logits max delta after Port/world changes: **0.0**.

Artifact ZIP SHA256: `c03d3f6da6d5e9d9323e6e3477989f1030206e3b676b98c9a686323725f423cb`.

## Architectural decision

Keep the **Twin-Plane invariant**:

- B-plane: immutable language/skill computation;
- P/J-plane: mutable world-state payload and derived causal workspace.

A future entity/value binding is runtime data, not something the frozen backbone must relearn. A world edit can therefore change the Port plane while the exact tested B-plane query state remains unchanged.

R295b still uses a trained sidecar classifier and synthetic operations. It establishes separation, not free-form language generation. R300 moves the Port-conditioned J-space residual in front of Qwen's frozen LM head so token selection is model-native rather than delegated to an external classifier.
