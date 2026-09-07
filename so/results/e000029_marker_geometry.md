# E-000029 — what the marker gate actually certifies

11 recorded checkpoints, no training.

E-000021 reported the gate's false-accept rate as 8.49e-04 and called it the bound on the deletion
guarantee. Its unsigned class comes from `invalid_markers`, which rejects every draw within 0.7 of
the centre, while the store calls everything beyond 0.35 deleted. The band in between was
measured by nothing. These are the three distributions side by side.

## The gate's accept rate, by where the marker is

| marker distribution | accepted | of | rate | 95% CI lower | 95% CI upper |
|---|---|---|---|---|---|
| uniform | 5843 | 5500000 | 1.062e-03 | 1.035e-03 | 1.090e-03 |
| annulus | 2199996 | 2200000 | 1.000e+00 | 1.000e+00 | 1.000e+00 |
| rejection_sampled | 1881 | 2200000 | 8.550e-04 | 8.168e-04 | 8.945e-04 |

## Accept rate by distance from the centre

| distance from the centre | accept rate (mean over checkpoints) | mean gate score |
|---|---|---|
| 0.10 | 1.0000 | 0.9985 |
| 0.20 | 1.0000 | 0.9978 |
| 0.30 | 1.0000 | 0.9962 |
| 0.40 | 1.0000 | 0.9917 |
| 0.50 | 1.0000 | 0.9777 |
| 0.60 | 1.0000 | 0.9300 |
| 0.70 | 0.9999 | 0.7725 |
| 0.80 | 0.2191 | 0.4320 |
| 0.90 | 0.0000 | 0.1299 |
| 1.00 | 0.0000 | 0.0254 |
| 1.10 | 0.0000 | 0.0044 |
| 1.20 | 0.0000 | 0.0008 |
| 1.30 | 0.0000 | 0.0002 |
| 1.40 | 0.0000 | 0.0000 |
| 1.50 | 0.0000 | 0.0000 |
| 1.60 | 0.0000 | 0.0000 |
| 1.70 | 0.0000 | 0.0000 |
| 1.80 | 0.0000 | 0.0000 |
| 1.90 | 0.0000 | 0.0000 |
| 2.00 | 0.0000 | 0.0000 |

Operational radius (smallest shell with an accept rate of exactly zero), worst checkpoint: 0.90 against a declared 0.35.

## The centre is not a secret

A signed marker is `normalise(centre + N(0, 0.05^2))`, so the mean of 950 of them
estimates the centre to 0.0076 on the worst checkpoint, well inside
the 0.35 radius. Markers minted from that estimate pass `marker_valid` at 0.9999 and the learned gate at 1.0000. `make_centre` derives the centre from `10_000 + seed`,
and every checkpoint stores it verbatim, so shipping a model ships the ability to mint signatures.

## Pre-registered criteria

| criterion (worst checkpoint) | required | observed | result |
|---|---|---|---|
| annulus_accept_rate | <= 0.001 | 1.0000 | FAIL |
| uniform_accept_rate | <= 0.001 | 0.0013 | FAIL |
