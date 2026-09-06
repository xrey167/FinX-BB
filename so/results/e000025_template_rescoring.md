# E-000025 — re-scoring the symlink checkpoints across all twelve templates

No training. The checkpoints of E-000020 (link adapter) and E-000017-B (link-free adapter, eight
trained templates) are loaded from disk and scored at every one of the twelve templates on one
alias world, in both stores.

E-000020's headline numbers — direct 0.5667, alias 0.5067 — are template 0 only, because
`E20._answers` defaults to it. The table below is what the same checkpoints do everywhere else.

## Reading, per template (mean over seeds)

| template | direct (link adapter) | alias, shared object | alias, duplicated | direct (link-free) | alias, duplicated (link-free) |
|---|---|---|---|---|---|
| t0 (trained) | 0.6122 | 0.6167 | 0.6233 | 0.7889 | 0.7900 |
| t1 (trained) | 0.9633 | 0.8367 | 0.9817 | 0.9967 | 0.9933 |
| t2 (trained) | 0.6322 | 0.6017 | 0.6217 | 0.7789 | 0.7650 |
| t3 (trained) | 0.9956 | 0.8933 | 0.9983 | 1.0000 | 1.0000 |
| t4 (trained) | 0.9967 | 0.9350 | 0.9967 | 0.9989 | 1.0000 |
| t5 (trained) | 0.9989 | 0.9250 | 0.9967 | 1.0000 | 1.0000 |
| t6 (trained) | 0.6422 | 0.6317 | 0.6317 | 0.7778 | 0.7633 |
| t7 (trained) | 0.9989 | 0.8917 | 0.9967 | 0.9989 | 0.9950 |
| t8 (held out) | 0.4500 | 0.4283 | 0.4133 | 0.5622 | 0.5200 |
| t9 (held out) | 0.9567 | 0.9033 | 0.9817 | 0.9622 | 0.9650 |
| t10 (held out) | 0.9989 | 0.9250 | 1.0000 | 1.0000 | 1.0000 |
| t11 (held out) | 0.3078 | 0.3433 | 0.2883 | 0.4056 | 0.4200 |

`alias, shared object` and `alias, duplicated` are the *same adapter* answering the *same questions*
against a store that shares one object and a store that holds independent copies: their difference is
the cost of sharing. The difference between `alias, duplicated (link-free)` and `alias, duplicated` is
the cost of having trained on links at all.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| train/alias_max | >= 0.75 | 0.8950 | PASS |
| heldout/alias_mean | >= 0.55 | 0.6013 | PASS |
| all/cost_of_sharing | <= 0.1 | 0.0954 | PASS |
| all/cost_of_link_training | <= 0.25 | 0.0688 | PASS |

Disclosure: alias reading at templates 1, 8 and 9 was already recorded in E-000020, so these
thresholds were set knowing three of the sixty cells above. This record confirms a reading of
existing numbers; it is not an independent prediction.

## Aggregates

| measure | mean over seeds | worst seed | pooled n | 95% CI lower | 95% CI upper |
|---|---|---|---|---|---|
| train/direct_mean | 0.8550 | 0.8313 | - | - | - |
| train/alias_mean | 0.7915 | 0.7325 | - | - | - |
| train/dup_mean | 0.8558 | 0.8387 | - | - | - |
| train/linkfree_direct_mean | 0.9175 | 0.9104 | - | - | - |
| train/linkfree_dup_mean | 0.9133 | 0.8894 | - | - | - |
| heldout/direct_mean | 0.6783 | 0.6617 | - | - | - |
| heldout/alias_mean | 0.6500 | 0.6013 | - | - | - |
| heldout/dup_mean | 0.6708 | 0.6500 | - | - | - |
| heldout/linkfree_direct_mean | 0.7325 | 0.7175 | - | - | - |
| heldout/linkfree_dup_mean | 0.7262 | 0.6987 | - | - | - |
| all/direct_mean | 0.7961 | 0.7747 | - | - | - |
| all/alias_mean | 0.7443 | 0.6888 | - | - | - |
| all/dup_mean | 0.7942 | 0.7812 | - | - | - |
| all/linkfree_direct_mean | 0.8558 | 0.8461 | - | - | - |
| all/linkfree_dup_mean | 0.8510 | 0.8258 | - | - | - |
| train/alias_max | 0.9367 | 0.8950 | - | - | - |
| train/alias_min | 0.6017 | 0.4850 | - | - | - |
| train/cost_of_sharing | 0.0644 | 0.1063 | - | - | - |
| train/cost_of_link_training | 0.0575 | 0.0725 | - | - | - |
| heldout/alias_max | 0.9250 | 0.8700 | - | - | - |
| heldout/alias_min | 0.3433 | 0.3000 | - | - | - |
| heldout/cost_of_sharing | 0.0208 | 0.0737 | - | - | - |
| heldout/cost_of_link_training | 0.0554 | 0.0613 | - | - | - |
| all/alias_max | 0.9383 | 0.8950 | - | - | - |
| all/alias_min | 0.3433 | 0.3000 | - | - | - |
| all/cost_of_sharing | 0.0499 | 0.0954 | - | - | - |
| all/cost_of_link_training | 0.0568 | 0.0688 | - | - | - |

## Provenance

a forced re-run of E-000020 overwrote its seed-0 and seed-1 checkpoints after that record was written; only seed 2 still matches the SHA-256 recorded there. The SHA of every checkpoint scored here is in per_seed.
