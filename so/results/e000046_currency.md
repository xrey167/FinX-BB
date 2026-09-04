# E-000046 — the currency of tracelessness: deletions, repairs, or an opaque interface

E-000041 measured T = k over 105 of 105 cells and carried one caveat: that it held for a
store which exports a link's target key and goes on exporting it after the target is gone.
This is that caveat, tested. Mechanical, no model.

| semantics | T | T = k | T = U | exported view clean | raw store still discloses | rows left live | **history independent (exported)** | residue rows | history independent (raw) |
|---|---|---|---|---|---|---|---|---|---|
| exporting | 6.00 | 1.0000 | 0.2000 | 1.0000 | 0.8000 | 40.0 | 1.0000 | 0.0 | 0.0000 |
| compacting | 6.00 | 1.0000 | 0.2000 | 1.0000 | 0.0000 | 42.5 | 0.2000 | 2.5 | 0.0000 |
| opaque | 3.50 | 0.2000 | 1.0000 | 1.0000 | 0.8000 | 42.5 | 0.2000 | 2.5 | 0.0000 |

`raw store still discloses` is referential: does a surviving version still hold the removed
key. Under OPAQUE the exported view is clean by construction, so an experiment that stopped
at the fourth column would have measured its own definition. `history independent (exported)`
is the property §31.31 adopted as the meaning of traceless (Naor and Teague 2001, Def. 2.1):
is `bank()` identical to that of a store that never held the fact. `residue rows` counts the
exported rows that exist only because it did. `history independent (raw)` compares
`store.cells`, the operation log and the next id as well, and an MVCC store fails it by
design. The first two versions of this report had only the referential column and read it
as the history-independence one (ledger §31.35).

## Post hoc, labelled as such: the same column over cells that have an alias

The registered `exported_hi` rows for COMPACTING and OPAQUE came back at 0.2000, not 0.0, and
FAIL as registered. The 0.2 is exactly the cells with `n_links = 0` -- a pod made of copies
has no alias to blank or to leave dangling, so evicting its closure leaves nothing behind
under every semantics. The criterion should have conditioned on `n_links >= 1`; it did not,
and it is not rewritten. The same quantity over the cells the prediction was about:

| semantics | cells with an alias | history independent (exported) | residue rows |
|---|---|---|---|
| exporting | 48 | 1.0000 | 0.0 |
| compacting | 48 | 0.0000 | 3.1 |
| opaque | 48 | 0.0000 | 3.1 |

## The rule, fixed before the run

T = k under EXPORTING and COMPACTING with T = U under OPAQUE, and the raw store still disclosing -> k is not a cost canonicalisation imposes but one that can be paid in three currencies, deletions, repairs, or an interface that declines to show the reference, and the third is not payment. raw_discloses at 0 under OPAQUE -> opacity is erasure and the law must be restated. T != k under COMPACTING -> repairs are cheaper than deletions and the law is about deletions specifically. Fixed before the run. THIRD RUN: exported_hi at 0 under COMPACTING and at 1 under EXPORTING -> 'referentially clean' and 'history independent' are different properties, repair buys the first and deletion the second, and §31.30's 'strictly stronger AND less destructive' is withdrawn. exported_hi at 1 under COMPACTING -> blanked rows are not a residue and §31.30 stands. raw_hi at 1 anywhere -> the fresh-store comparison is broken, since the log alone distinguishes the two stores. Fixed before the third run.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| exporting/T_equals_k | >= 1.0 | 1.0000 | PASS |
| exporting/unreachable | >= 1.0 | 1.0000 | PASS |
| compacting/T_equals_k | >= 1.0 | 1.0000 | PASS |
| compacting/raw_discloses | <= 0.0 | 0.0000 | PASS |
| exporting/raw_discloses | >= 0.5 | 0.8000 | PASS |
| opaque/exported_clean | >= 1.0 | 1.0000 | PASS |
| opaque/raw_discloses | >= 0.5 | 0.8000 | PASS |
| compacting/exported_hi | <= 0.0 | 0.2000 | FAIL |
| opaque/exported_hi | <= 0.0 | 0.2000 | FAIL |
| exporting/exported_hi | >= 1.0 | 1.0000 | PASS |
| exporting/raw_hi | <= 0.0 | 0.0000 | PASS |
| compacting/raw_hi | <= 0.0 | 0.0000 | PASS |
