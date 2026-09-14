# R321 Result — Generation-Gated KV Overlay

Status: **strict numerical gate failed; masked append-only overlay is rejected as the exact CKCA cache primitive.**

## Attempt

The experiment kept a superseded mutable suffix physically resident inside Qwen2.5-0.5B's actual legacy KV tensor, masked those stale slots out of attention, appended the current generation, and reset the current suffix to the superseded suffix's logical/RoPE positions.

This tested whether CKCA could separate:

`physical cache slot != logical token position != authority`

without compacting/gathering the live KV working set.

## Result

Seven real-model generation transitions:

- top-token match across full recompute, direct current splice, masked overlay and corrupted-stale overlay: **100%**;
- max full-vocabulary logit delta, direct splice vs masked overlay: **0.125**;
- max delta after aggressively corrupting the masked stale physical K/V slots: **0.125**;
- max full-recompute vs masked overlay delta: **0.125**;
- median direct current splice: **1.6010 s**;
- median append-only masked overlay: **1.6022 s**;
- median stale KV bytes deliberately retained: **61,440 bytes**.

Report SHA256: `46062b055c1514513f467de0551913329372b30eb444463128fe75ab90ef1943`.
Artifact ZIP SHA256: `d71a17159e640e663542246af224697605dc1e55ce04a96d4e247364a9c36748`.

## Interpretation

The fact that corrupting the masked stale K/V values did **not** increase the 0.125 delta is useful: stale slot contents were not the source of the difference. The discrepancy comes from changing the physical attention/cache geometry rather than from stale-value influence.

But CKCA's optimized path is supposed to match the full-current oracle at a much tighter boundary. **100% top-1 agreement is not enough.** Therefore the masked in-tensor overlay is not promoted.

## Decision

Replace authority-by-mask with **authority-by-KV page-table admission**:

- physical pages may remain append-only and stale;
- a generation-aware page table chooses which physical pages are admitted into the logical attention working set;
- stale pages are excluded before attention rather than represented as masked positions inside the sequence;
- current repaired suffix then runs over the exact same logical cache geometry as a clean execution.

This is R321b. The goal is `0.0` full-vocabulary delta while still proving that physically resident stale pages can be arbitrarily corrupted without affecting the current path.
