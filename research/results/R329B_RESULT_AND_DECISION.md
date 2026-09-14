# R329b Result — Strong Parametric-World Control

Status: **strong controlled future-rebinding gate passed; not DoD and not a general model-editing/RAG comparison.**

## Why R329b exists

R329's query-only monolithic Transformer underfit the home world, so its future collapse was not a strong enough counterfactual.

R329b holds the neural language compiler and identity-blind J Transformer common between both systems. The only architectural difference is where mutable world state lives:

- **CKCA/WIT:** values come from the current external generation-scoped world;
- **parametric control:** all 64 entity->value bindings are memorized in a trainable parameter table and receive no update after the world is rebound.

## Result

Three seeds.

Strong parametric world table:

- home binding fit: **100%**.

Shared compiler/executor quality:

- held-out language-template plan accuracy: **97.21%**;
- J Transformer accuracy with held-out numeric values: **99.73%**.

### CKCA/WIT external-world path

- home accuracy: **97.70%**;
- fully deranged future-world accuracy: **97.15%**;
- future world + held-out language template: **94.62%**;
- gradient steps for world rebinding: **0**.

### Strong parametric-world control

- home accuracy: **97.70%**;
- fully deranged future-world accuracy: **31.42%**;
- future world + held-out language template: **31.23%**.

Both paths therefore have the same home task quality and the same compiler/J machinery, while the future-world gap is **+65.73 percentage points** for the external current-world path.

The value-only parametric table also cannot distinguish a same-value generation rewrite without adding external temporal metadata; identical payload parameters do not encode the fact that a lifecycle transition occurred.

Report SHA256: `f53777716013584133b72be8f31b16abcb03bd60700590d2800b48d1ca962026`.

## Architectural decision

The R329 neural shape survives the stronger control:

```text
language Transformer
  -> verified source pointers
  -> trusted linker
  -> current World ABI values
  -> identity-blind J Transformer
```

The result isolates the core benefit more cleanly than R329: **future world rebinding remains a data update rather than a parameter-generalization problem when the mutable world is outside reusable neural skill state.**

This is still a controlled synthetic architecture gate. It does not establish superiority over model editing, RAG, editable memory, or continual learning on realistic language tasks. Those remain explicit DoD baselines.
