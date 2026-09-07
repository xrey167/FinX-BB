# NOV-004 result — the vacuity pattern is in a third experiment, and the sweep found it

Date: 2026-09-06
Decision: **SCANNER_VALID** — 3 confirmed Class A sites across 100 experiments
Major-invention claim: **NO**

## Why a sweep

Three experiments were read by hand — E-000103, E-000104, E-000105 — and two distinct defect classes
came out of three readings. That rate makes hand-reading the remaining twenty both slow and a poor
instrument: whoever reads them will find what they are looking for and stop. So this scans all 100
recorded experiments mechanically, for exactly the two patterns already demonstrated to occur.

## Validity floor, and what it caught

The scanner must rediscover the two Class A sites established by hand at `e000104:224-225` and
`e000105:176-177`. Its first run reported `SCANNER_INVALID_KNOWN_SITES_MISSED` — because it globbed
`e0000*.py`, a pattern that silently excludes **every experiment from E-000100 onward**, which is
where both known sites live. The floor did exactly the job E-000019 built it for, on the file that
was auditing everyone else.

Fixed to a regex over the real naming pattern: 100 files scanned, both known sites found,
`validity_floor_met` true.

## Class A — a comparison whose two sides share a source

**3 confirmed sites.**

| file | lines | call | names | in kill predicate |
|---|---|---|---|---|
| `e000104_tensor_train_lifecycle_quotient_reduction.py` | 224–225 | `local_update_multiplies(rank)` | `c_work`, `g_work` | yes |
| `e000105_stable_gate_cohort_transport_reduction.py` | 176–177 | `mutation_multiplies(net)` | `candidate_work`, `generic_work` | yes |
| **`e000107_state_only_causal_lineage_identifiability.py`** | **93–94** | **`fn(values, delete_target(h, target))`** | **`oracle`, `direct`** | **yes** |

The third is new. E-000107 lines 88–96:

```python
# Oracle history is sufficient by construction: exact recompute from h.
oracle_failures = 0
for target in range(len(values)):
    for h in histories:
        oracle = fn(values, delete_target(h, target))
        direct = fn(values, delete_target(h, target))
        if oracle != direct:
            oracle_failures += 1
```

`delete_target` copies a tuple and sets one index; `fn` is one of the four pure state functions. Both
calls are identical and pure, so `oracle_failures` is **identically zero for every input**. It is
nonetheless:

1. reported as a measured field, `oracle_history_failures`;
2. a conjunct of the kill predicate — line 187, `kill = all_have and distinct_family_targets >= 4 and
   oracle_failures == 0`;
3. asserted in `test_oracle_history_resolves_registered_ambiguity` as though it were evidence.

The comment says "by construction", so the tautology was understood when written. What was not
carried through is that a quantity true by construction must not then be reported, screened and
tested as a measurement. **§31.15 for the fifth time, and the third distinct experiment.** E-000107's
substantive result — the indistinguishable-history witnesses — is untouched by this; what falls is
one of the three conjuncts guarding it, which never guarded anything.

### What the discriminator had to add

The first version flagged **87** sites on syntactic identity alone, and most were legitimate:
`bank_from_store(store)` before and after a deletion is the same expression and a different value,
because `store` moved in between — that is the comparison working, not failing.

Two refinements brought it to 3:

- **A window that closes on mutation.** The pair must sit in one block with nothing between them that
  rebinds any of the call's free names, and nothing that is not a plain assignment.
- **The results must actually be compared.** A duplicated call is only a defect when something reads
  the difference. Allocating two zero vectors, or drawing twice from a generator, duplicates the
  expression without duplicating the value. 18 such incidental duplicates are recorded separately and
  are not findings.

## Class B — loop-invariant work inside a loop

**61 candidates after two scanner fixes, and they are candidates, not findings.** The first run
reported 94; the triage section below explains what came out and why.

This is the pattern NOV-003 found in E-000103, where `woodbury_rankk_solution` rebuilds `inv_u` and
`middle_inv` on every one of 24 sessions. It is a defect *only when it sits on the baseline arm*,
where it inflates the candidate's margin for free. On the candidate arm it is merely slow, and in
plenty of cases the call is doing necessary work the heuristic cannot see.

Reading this list as 94 defects would be §31.46's error repeated — scoring something the instrument
cannot resolve. It is a work queue for review, ordered by nothing, and nothing in this document
claims any entry beyond E-000103 is real.

## Class B triaged, 94 → 61 → 18 → 2

The Class B list was worked the same way Class A was, and it cost the scanner two more bugs.

**Bug 1: mutation that rebinds nothing.** `_bound_names` counted only `ast.Name` stores, so
`aff[idx] = new` — a write through a subscript — did not count as moving `aff`. The scanner therefore
reported `compose_all(aff, ...)` in E-000097 as loop-invariant when `aff` is written on the line
directly above it. Method calls have the same problem: `tree.update(idx, new)` mutates its receiver
and rebinds nothing. Counting in-place mutation removed **every** E-000097 site and took the total
from 94 to 70. This is the same error the Class A window would make if it ignored intervening writes,
and it produced false positives for the same reason.

**Bug 2: nondeterministic calls.** `idx = rng.randrange(length)` is loop-invariant by the syntax and
draws a different value every iteration. Filtering those took 70 to 61.

Then the triage. Of the 61, **18** are in the reduction family (E-000086 … E-000108), and of those
exactly **2** sit on a baseline arm — the only place a loop-invariant call inflates a candidate's
margin. Both are in E-000102, and only one of them is cost-relevant:

- **`e000102:250`** — `old_fresh = fresh_numeric(family, coeffs, old_values)` inside `for edited in
  edited_indices`, where `old_values` does not move. Genuinely invariant, and **not a defect**:
  `old_fresh` feeds a correctness assertion, not the work comparison.
- **`e000102:296`** — `generic = GenericDependencyProduct(factors)` inside the same loop shape, where
  `factors` does not move. This one *is* on the cost-compared arm. Its `__init__` builds the whole
  product DAG at `size − 1` multiplications, and `generic_ops = generic.update(...)` counts **only
  the update walk**, so the generic's per-event cost is understated by the construction. The
  candidate, `specialized_lifecycle_product_update`, is a pure function needing no such reset.

**And the honest reading of that second one, which cuts against the obvious conclusion.** Charging
the construction would move E-000102 *toward* the candidate — the opposite direction from E-000105.
But it would be charging the baseline for the harness's choice rather than the algorithm's: a real
generic keeps one DAG and updates it incrementally, and rebuilds only because this loop needs to
reset mutable state per edit. Charging it would be NOV-003's lesson run backwards — denying the
baseline an optimisation, then billing it for the denial.

So the finding is narrow and it is about measurement, not about the verdict: **E-000102's work
comparison counts update cost on both arms while one arm additionally pays an uncounted O(n) reset.**
The fix is to the harness — construct the generic once outside the loop and reset it cheaply — after
which the comparison measures what it claims. Until then the number is not wrong so much as not the
number being described. No verdict is revisited on this evidence.

## What this settles

The vacuity pattern is not a one-off in the transport family: three experiments, found in two
different ways (twice by reading, once by sweep), and the sweep is now cheap to re-run. `so/screen.py`
already makes it unrepresentable in new work; NOV-004 is how the existing corpus gets checked.

**No verdict is revisited.** E-000107's kill rests on its indistinguishable-history witnesses, which
this does not touch — and the same was true of E-000104 and E-000105, whose kills NOV-002 showed
survive a proper accounting. The pattern is a reporting and screening defect, not, so far, a wrong
answer.

## Limits

Two patterns, both chosen because each was demonstrated by hand first. A file absent from both lists
has not been cleared — it has been checked for two things. Nothing here detects mismatched arms,
domains that hide a crossover (§31.46), or cost coordinates nobody counted, which is the class
NOV-003 had to find by reading. Class B is heuristic throughout, and Class A's window rule is
conservative: a vacuous pair separated by a function call would be missed.
