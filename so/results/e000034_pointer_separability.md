# E-000034 — a pointer is separable from an object by its norm alone

Seeds [0, 1, 2]. The recorded arm trains nothing: it encodes the E-000015 banks with the
recorded one-slot checkpoints and asks how well a single threshold on the value vector's L2
norm tells alias rows from fact rows.

## What the store gives away for free

| value projection | pointer norm | object norm | gap (pooled sd) | best single threshold | linear probe | direct read (worst seed) | alias read (worst seed) |
|---|---|---|---|---|---|---|---|
| recorded (separate v_link) | 18.557 | 11.917 | 7.6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

E-000015's record says *the model is never told that a value it has read is a pointer -- that
it must learn*. `encode_bank` builds a fact row's value as `v_fwd(ent_emb(obj))` and an alias
row's as `v_link(ln_key(...))`: two projections, one input layer-normalised and one not. The
outputs sit at different scales, and one number separates them.

The claim that survives is narrower and still real: **recognising a pointer is free; only
following one is learned.** E-000016's one-slot arm refuses a two-link chain rather than
inventing an answer, which a branch on a flag would not do.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| recorded/norm_threshold_accuracy | >= 0.99 | 1.0000 | PASS |
