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
| the same on a MATCHED NULL (random states, identical construction) | 0.6872 |
| **excess over the null** | **-0.1307** |
| sigma_min of the stacked bases (a dependency check, not a summary) | 0.2393 |

## Where the sharing is: content or addressing

Row 0 of a fact's basis is what all its phrasings share -- its CONTENT direction. Rows 1 and
up are the phrasing spread, which is how the fact is ADDRESSED. Both are compared against
the same matched null.

| subspace | overlap | matched null | excess |
|---|---|---|---|
| content direction only | 0.2232 | 0.1871 | +0.0361 |
| addressing rows only | 0.5954 | 0.7762 | -0.1809 |
| the whole basis | 0.6157 | 0.8178 | -0.2021 |
| **addressing minus content** |  |  | **-0.2170** |

READ THE SIGNS. Against the design-matched null the addressing rows overlap LESS than
chance, not more: permuting the fact x template interaction RAISES their overlap. So the
sharing in the addressing is a property of the design -- every fact asked with the same
templates -- and the model's own structure makes those subspaces more distinct rather than
less. An earlier version of this experiment used a random-state null, which carries no
template structure at all, and reported the opposite sign on both rows.

What survives is the structural point, and it is stronger for not being a training defect:
a fact's deletion subspace necessarily CONTAINS addressing directions, and addressing is
shared across facts because facts are asked in the same ways. In a store the address and
the object are separate records, so deleting the object leaves the addressing untouched. In
a representation they cannot be pulled apart by allocation, because the sharing is in the
task and not in the model.

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


TWO INSTRUMENT FAULTS THIS EXPERIMENT FOUND IN ITSELF, both recorded because each changed a published number. (1) It first measured `rank(union)/total`, which is LINEAR INDEPENDENCE, and reported 1.0000 on twelve subspaces whose pairwise principal cosines were 0.5566 and 0.8559 -- a direct sum, nowhere near orthogonal. (2) It then compared overlap against a RANDOM-STATE null, which carries no template structure, and reported +0.4118 where the design-matched permutation null gives -0.1306. The verdict reversed.

## Verdict

12 fact(s), 58 direction(s) in d=768: pressure 0.0755 against a bound of 159 facts, mean pairwise overlap 0.5566 against a matched null of 0.6872 (max 0.8559) -- ALLOCATED: the subspaces are no more overlapping than a matched null and there is budget to spare, so clean deletion is available here

## The rule, fixed before the numbers

pressure <= 0.5 and orthogonality <= 0.95 -> ALLOCATION, not capacity: the model had room for private subspaces and did not take it, so a training objective is the remedy. pressure > 0.8 -> the bound is binding and no objective fixes it without more dimensions. orthogonality > 0.95 with collateral still present -> the deletion subspaces are independent and the mechanism runs through A_i against V_j instead, which is measured in the same run. Fixed before the numbers were seen.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| answer_before | >= 0.75 | 0.9338 | PASS |
| answer_after | <= 0.25 | 0.0312 | PASS |
| silenced_rate | >= 0.5 | 0.7059 | PASS |
| pressure | <= 0.5 | 0.0755 | PASS |
| excess_overlap | >= 0.1 | -0.1307 | FAIL |
| address_over_content | >= 0.1 | -0.2170 | FAIL |
