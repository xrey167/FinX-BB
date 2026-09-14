# R329 Result — World-Interfaced Dual-Rail Transformer (WIDRT)

Status: **neural dual-rail mechanism gate passed; parametric control is not strong enough yet, so R329b is required before using the baseline gap as evidence. Not DoD.**

## Construction

This is no longer an MLP-only separation gate.

A small B-rail Transformer reads language-like query sequences and is permitted to emit only:

- an operation opcode;
- a pointer to the first lexical entity source token;
- a pointer to the second lexical entity source token.

A trusted linker maps those source positions to canonical entities. Current values are then read from the external world and passed to a separate J-rail Transformer as bit-tokenized typed values.

The J Transformer has **no entity-identity channel**. The B Transformer has **no current-world-value channel**. There is no J->B edge.

## Result

Three seeds, 64 entities, 32 values, four operations.

B compiler:

- train-template full-plan accuracy: **99.57%**
- held-out word-order/template full-plan accuracy: **97.21%**

J executor:

- accuracy when at least one numeric value is from the held-out set `{28,29,30,31}`: **99.73%**

Full CKCA neural pipeline:

- home-world accuracy: **97.70%**
- fully deranged future-world accuracy: **97.15%**
- future world + held-out language template accuracy: **94.62%**
- future plan/link accuracy: **97.41%**
- max B hidden/logit state delta across a world change: **0.0**
- mean cached-plan execution speedup vs compile+link+execute: **3.28x**
- gradient steps required for world update: **0**

Query-only monolithic Transformer control:

- home-world accuracy: **72.80%**
- fully deranged future-world accuracy: **28.40%**

The external-world pipeline therefore remained stable under complete world rebinding, but the monolithic control did not fit the home task strongly enough to serve as the final strong parametric baseline.

Report SHA256: `be212f099f16dca428e9c527cfc6cda3206897095bd86c3a2008ab64bbfd6c5c`.
Artifact ZIP SHA256: `85dc04d7545b7434758721442638031977226a314912708b73871031056fbc6b`.

## Decision

Promote the **neural shape**, not the baseline claim:

```text
language Transformer (B)
   -> opcode + source pointers
   -> trusted linker
   -> current World ABI values
   -> identity-blind J Transformer
```

This directly instantiates the CKCA boundary with trainable Transformers while preserving exact B-world invariance and no-gradient rebinding.

However, R329b must replace the weak query-only monolith with a deliberately strong **parametric-world control** that memorizes the home entity->value table nearly perfectly and then uses the same compiler/executor. That isolates whether the advantage comes from where mutable world state lives rather than from control underfitting.
