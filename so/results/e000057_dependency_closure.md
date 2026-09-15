# E-000057 — what a deleted row still contributes to a bystander's answer

Synthetic E-000015 reader, seeds [0, 1, 2], 100 pods, trains nothing. One row is
silenced through the reader's own cell mask; both compared arms silence exactly one row, so the
row-count channel of §31.41 cannot act. Worst seed.

| arm (one row silenced) | mean routing coefficient | max-abs logit change | mean KL to the unablated forward | top-1 flip rate |
|---|---|---|---|---|
| DEL | 0.00020 | 0.1073 | 0.0000 | 0.0000 |
| LIVE | 0.00020 | 0.5596 | 0.0192 | 0.0013 |
| LIVE2 | 0.00020 | 0.5923 | 0.0209 | 0.0013 |
| REV | 0.00000 | 0.0000 | 0.0000 | 0.0000 |
| TOP | 0.01376 | 17.8188 | 1.0683 | 0.0842 |

| paired comparison | dominance (worst seed) | median ratio | sign-test z |
|---|---|---|---|
| del_vs_live | 0.000 | 0.313 | -6.94 |
| floor | 0.468 | 0.997 | -0.14 |

## The rule, fixed before the run

Worst seed. Every row is a PAIRED comparison over pods: the effect of silencing one row against the effect of silencing another of the same routing mass, on the same queries. VOID if the REV zero control moves the forward at all (the mask or the routability flag is wrong and nothing else is readable), if silencing the top-coefficient row does not move the forward (a one-row ablation is invisible, so no row below it can be read), or if the FLOOR dominance exceeds 0.60 (two coefficient-matched live rows already dominate one another and the pairing is not matched). With all three holding: NO-EFFECT if DEL vs LIVE dominance is at or below 0.60 -- a deleted row that is still routable shapes a bystander's answer exactly as a live row of the same routing mass does, the other branch's dependency-closure requirement has no measurable basis on this substrate, and the entry is a refutation of it. SUB-BEHAVIOURAL if DEL vs LIVE is at least 0.75 while the flip rate stays under 0.02 and the KL under 0.05 nats: the deleted row's contribution is distinguishable in the logits and does not reach answers, which is the same shape as §31.45's residue and is reported with the magnitudes beside it. BEHAVIOURAL if both fire: a deleted row changes what the model answers about other pods, at a measured rate, and the revocation unit for this reader is the routable set and not the queried pod. Anything in (0.60, 0.75) is inconclusive at this n. The magnitudes (max-abs, KL, flip rate, mean coefficient) are recorded for every arm whatever the reading, because the number the other branch published has no null beside it. Fixed before the run.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| REV/maxabs_max | <= 1e-06 | 0.0000 | PASS |
| TOP/maxabs | >= 1.0 | 15.9716 | PASS |
| floor/dominance | <= 0.6 | 0.5213 | PASS |
| del_vs_live/dominance | >= 0.75 | 0.0000 | FAIL |
| DEL/flip | >= 0.02 | 0.0000 | FAIL |
