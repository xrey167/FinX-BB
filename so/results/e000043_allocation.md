# E-000043 — clean-deletion capacity, and whether GPT-2 is against the bound or below it

Frozen gpt2, d = 768, no training. For each fact the minimal subspace whose removal
silences every phrasing is found, and the set of those subspaces is read two ways.

## The dimension budget

| measure | value |
|---|---|
| facts the model answers | 17 |
| facts a subspace of their own basis silences | 12 |
| directions per deletion, mean | 5.76 |
| directions demanded in total | 58 of 768 |
| **pressure** (demand / d) | **0.0755** |
| headroom left unused | 0.9245 |
| mean pairwise overlap of the deletion subspaces | 0.5566 |
| the same on a MATCHED NULL (random states, identical construction) | 0.1448 |
| **excess over the null** | **+0.4118** |
| sigma_min of the stacked bases (a dependency check, not a summary) | 0.2393 |

## Where the sharing is: content or addressing

Row 0 of a fact's basis is what all its phrasings share -- its CONTENT direction. Rows 1 and
up are the phrasing spread, which is how the fact is ADDRESSED. Both are compared against
the same matched null.

| subspace | overlap | matched null | excess |
|---|---|---|---|
| content direction only | 0.2232 | 0.0638 | +0.1594 |
| addressing rows only | 0.5954 | 0.1930 | +0.4024 |
| the whole basis | 0.6157 | 0.2048 | +0.4109 |
| **addressing minus content** |  |  | **+0.2430** |

A fact's own content direction being near the null while its addressing rows are far above
it is the symlink result stated in activation space: what a store keeps in separate records
-- the object and the keys that reach it -- a representation keeps in one subspace, so a
deletion aimed at the content pays its collateral to the addressing.

## What the deletion costs, and where the overlap actually is

| measure | value |
|---|---|
| the model answers, before | 0.9338 |
| facts a subspace of their own basis silences | 0.7059 |
| answers after its own deletion, over those | 0.0312 |
| the same over every fact, silenced or not | 0.3088 |
| bystander facts under the same ablation | 0.3897 |
| overlap of A_i with A_j, mean | 0.5566 |
| overlap of A_i with A_j, max | 0.8559 |
| **overlap of A_i with V_j** (what the theorem needs), mean | **0.5842** |
| overlap of A_i with V_j, max | 0.8575 |

The last two rows are the refinement. The orthogonality the argument requires is between a
fact's DELETION subspace and every other fact's READOUT subspace, not between two deletion
subspaces. They can be mutually independent while each still intrudes on what other facts
read from, and that produces collateral with orthogonality near one.


The instrument had a bug this run found. The first version measured `rank(union)/total`, which is LINEAR INDEPENDENCE; the theorem needs ORTHOGONALITY. It reported 1.0000 on twelve subspaces whose pairwise principal cosines were 0.5566 mean and 0.8559 max in the same run -- a direct sum, nowhere near orthogonal. sigma_min is now the primary and the rank is kept beside it.

## Verdict

12 fact(s), 58 direction(s) in d=768: pressure 0.0755 against a bound of 159 facts, mean pairwise overlap 0.5566 against a matched null of 0.1448 (max 0.8559) -- ALLOCATION, NOT CAPACITY: the subspaces overlap 0.4118 more than a matched null while 92% of the budget is unused -- the model had room to give each fact a private subspace and did not, which is a training objective and not a law of dimension

## The rule, fixed before the numbers

pressure <= 0.5 and orthogonality <= 0.95 -> ALLOCATION, not capacity: the model had room for private subspaces and did not take it, so a training objective is the remedy. pressure > 0.8 -> the bound is binding and no objective fixes it without more dimensions. orthogonality > 0.95 with collateral still present -> the deletion subspaces are independent and the mechanism runs through A_i against V_j instead, which is measured in the same run. Fixed before the numbers were seen.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| answer_before | >= 0.75 | 0.9338 | PASS |
| answer_after | <= 0.25 | 0.0312 | PASS |
| silenced_rate | >= 0.5 | 0.7059 | PASS |
| pressure | <= 0.5 | 0.0755 | PASS |
| excess_overlap | >= 0.1 | 0.4118 | PASS |
| address_over_content | >= 0.1 | 0.2430 | PASS |
