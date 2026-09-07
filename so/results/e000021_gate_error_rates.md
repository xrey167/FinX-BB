# E-000021 — The verification gate as a classifier

The standing audit's objection that the deletion certificate is a learned classifier whose false-accept rate is reported only as a worst-seed maximum.

Nothing was trained. Every recorded checkpoint of the verified-gate family is loaded and its gate scored on freshly drawn markers; the rest of the model is not involved, because the gate is a function of the marker alone.

**Pooled over 11 checkpoints and 2,200,000 markers per class: 1867 false accepts (rate 8.49e-04, 95% interval [8.11e-04, 8.88e-04]) and 0 false rejects (rate 0.00e+00).**

| family | false accepts | rate | 95% interval | false rejects | max score on an unsigned marker | min score on a signed marker |
|---|---|---|---|---|---|---|
| e000010 | 811/1000000 | 8.11e-04 | [7.56e-04, 8.69e-04] | 0/1000000 | 0.8610 | 0.9926 |
| e000014 | 576/600000 | 9.60e-04 | [8.83e-04, 1.04e-03] | 0/600000 | 0.8774 | 0.9902 |
| e000019 | 480/600000 | 8.00e-04 | [7.30e-04, 8.75e-04] | 0/600000 | 0.8929 | 0.9878 |

Pre-registered criteria:

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| false_accept_rate | <= 0.001 | 0.0008 | PASS |
| false_reject_rate | <= 0.001 | 0.0000 | PASS |
| false_accept_ci_upper | <= 0.01 | 0.0009 | PASS |

This is the gate's error rate on markers drawn from the same two distributions the programme uses. It is not a security claim: an adversary who can choose the marker is not modelled here, and a gate that separates two fixed distributions says nothing about one that must resist a search for a passing vector.
