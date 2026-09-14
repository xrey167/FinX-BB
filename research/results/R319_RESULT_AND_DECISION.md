# R319 Result — Generational Causal Rewind Decoding (GCRD)

Status: **causal-rewind mechanism gate passed; not DoD and not a standalone novelty claim.**

## Idea

A mutable generation records the earliest decode position at which it becomes causally relevant. If that generation changes, CKCA need not throw away the whole answer or conversation state. It rewinds to the earliest affected neural checkpoint, reuses the safe prefix, and regenerates only the dependent suffix against current capabilities.

The safety oracle is full current-world recomputation, including the **generation read-set**, not just output bytes.

## Result

Two policies, **60,000 update trials each** over 96-token autoregressive traces:

### Early binding — mutable state can enter at token 4

- affected-update trials: **15,769**
- partial-rewind output exactness vs full recompute: **100%**
- partial-rewind lifetime/read-set exactness: **100%**
- mean recomputed fraction for affected updates: **50.97%**
- mean recomputed fraction over all updates: **13.40%**
- same-value read witnesses: **11,365**
- same output bytes but changed lifetime witnesses: **6,785**

### Late binding — mutable state can enter at token 67

- affected-update trials: **5,570**
- partial-rewind output exactness vs full recompute: **100%**
- partial-rewind lifetime/read-set exactness: **100%**
- mean recomputed fraction for affected updates: **15.80%**
- mean recomputed fraction over all updates: **1.47%**
- same-value read witnesses: **4,009**
- same output bytes but changed lifetime witnesses: **2,474**

Moving the mutable boundary later reduced the mean recomputed fraction on affected updates by **35.17 percentage points** compared with early binding.

Report SHA256: `b11c98f3bcc6ab0806140cf0e467614660907cb6d2dfa216589f1e54c1dd98e4`.
Artifact ZIP SHA256: `a1c1f5918d29fe8f4d4f700973e8c64b2c5cd37ba7c362ade87754c2f4dbde8f`.

## Architectural decision

Promote **Generational Causal Rewind** as the repair primitive for long-running decode/reasoning state:

```text
changed generation
    -> reverse lifetime index
    -> earliest causal neural checkpoint
    -> preserve earlier prefix
    -> refresh current capabilities
    -> regenerate dependent suffix only
    -> attach fresh lifetime
```

This is the performance counterpart to strict lifecycle safety. Instead of weakening invalidation to save latency, CKCA preserves exact generation semantics and reduces the amount of neural state that must be repaired.

The result also strengthens the late-binding principle: late materialization is useful not only because it reduces the amount of mutable neural state, but because it moves the earliest legal rewind point forward.

Checkpointing, incremental recomputation, MVCC and dependency invalidation are established. R319 tests their generation-lifetime composition at an autoregressive boundary; standalone novelty is not claimed.
