# E-000046 — the currency of tracelessness: deletions, repairs, or an opaque interface

E-000041 measured T = k over 105 of 105 cells and carried one caveat: that it held for a
store which exports a link's target key and goes on exporting it after the target is gone.
This is that caveat, tested. Mechanical, no model.

| semantics | T | T = k | T = U | exported view clean | **raw store still discloses** |
|---|---|---|---|---|---|
| exporting | 6.00 | 1.0000 | 0.2000 | 1.0000 | 0.0000 |
| compacting | 6.00 | 1.0000 | 0.2000 | 1.0000 | 0.0000 |
| opaque | 3.50 | 0.2000 | 1.0000 | 1.0000 | 0.8000 |

The last column is the experiment. Under OPAQUE the exported view is clean by construction,
so an experiment that stopped at the fourth column would have measured its own definition.
What decides whether opacity is erasure or access control is whether the removed key is
still recoverable from the store itself.

## The rule, fixed before the run

T = k under EXPORTING and COMPACTING with T = U under OPAQUE, and the raw store still disclosing -> k is not a cost canonicalisation imposes but one that can be paid in three currencies, deletions, repairs, or an interface that declines to show the reference, and the third is not payment. raw_discloses at 0 under OPAQUE -> opacity is erasure and the law must be restated. T != k under COMPACTING -> repairs are cheaper than deletions and the law is about deletions specifically. Fixed before the run.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| exporting/T_equals_k | >= 1.0 | 1.0000 | PASS |
| exporting/unreachable | >= 1.0 | 1.0000 | PASS |
| compacting/T_equals_k | >= 1.0 | 1.0000 | PASS |
| opaque/exported_clean | >= 1.0 | 1.0000 | PASS |
| opaque/raw_discloses | >= 0.5 | 0.8000 | PASS |
