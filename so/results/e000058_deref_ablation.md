# E-000058 — ablating the dereference slot: is it what resolves the pointer?

The control §31.53 found missing from the pointer claim: the dereference slot is directly
supervised and had never been ablated. Same checkpoints, same world, nothing trained; the only
change is `cfg.n_deref = 0` at inference. Worst seed.

| template | alias, slot on (worst seed) | alias, slot off (worst seed) | direct, slot on | direct, slot off |
|---|---|---|---|---|
| t0 | 0.9300 | 0.0000 | 0.9967 | 0.9967 |
| t1 | 0.8400 | 0.0000 | 0.9700 | 0.9833 |
| t2 | 0.9350 | 0.0000 | 1.0000 | 1.0000 |
| t3 | 0.9100 | 0.0000 | 0.9967 | 0.9967 |
| t4 | 0.9500 | 0.0000 | 1.0000 | 1.0000 |
| t5 | 0.9300 | 0.0000 | 1.0000 | 1.0000 |
| t6 | 0.9450 | 0.0000 | 1.0000 | 1.0000 |
| t7 | 0.9050 | 0.0000 | 0.9967 | 0.9967 |
| t8 | 0.9300 | 0.0000 | 0.9933 | 0.9933 |
| t9 | 0.8500 | 0.0000 | 0.8200 | 0.8233 |
| t10 | 0.9300 | 0.0000 | 0.9933 | 0.9933 |
| t11 | 0.9250 | 0.0000 | 0.9933 | 0.9933 |

## The rule, fixed before the run

Worst seed over three, every template. UNREADABLE if the trained arm does not reproduce the battery (alias below 0.80 or direct below 0.90) or if the ablation costs direct reading more than 0.05 at any template -- then the arm is destructive rather than surgical and nothing about pointers can be read from it. With both holding: MEASURED if alias reading falls by at least 0.50 at every template -- the dereference slot is what resolves the pointer, and the claim's attribution stops being architectural. REFUTED if alias reading survives the ablation (drop below 0.50 at any template) -- something other than the dereference hop is resolving the alias, and the claim credits the wrong component and must say so. The magnitudes are recorded whatever the reading. Fixed before the run.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| DEREF/alias_min | >= 0.8 | 0.8400 | PASS |
| DEREF/direct_min | >= 0.9 | 0.8200 | FAIL |
| direct_drop_max | <= 0.05 | 0.0000 | PASS |
| alias_drop_min | >= 0.5 | 0.8400 | PASS |
