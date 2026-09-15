# E-000056 — the epoch capacity of a frozen learned acceptance gate

Recorded checkpoints, nothing trained, nothing written to a store. The question is where a
version check can live in a memory-augmented model of this shape: a frozen gate can carry as
many epochs as its acceptance function has disjoint accepting regions. Worst checkpoint.

| checkpoint | radial accept bands | operational radius | monotone | tangential bands (max) | min accept fraction on a circle | epoch capacity | equidistant gap | accept, key 0 / key 1 |
|---|---|---|---|---|---|---|---|---|
| e000010_seed0 | 1 | 0.850 | yes | 1 | 1.000 | 0 | 0.268 | 1.000 / 1.000 |
| e000010_seed1 | 1 | 0.860 | yes | 1 | 1.000 | 0 | 0.074 | 1.000 / 1.000 |
| e000010_seed2 | 1 | 0.850 | yes | 1 | 1.000 | 0 | 0.189 | 1.000 / 1.000 |
| e000010_seed3 | 1 | 0.840 | yes | 1 | 1.000 | 0 | 0.253 | 1.000 / 1.000 |
| e000010_seed4 | 1 | 0.850 | yes | 1 | 1.000 | 0 | 0.490 | 1.000 / 1.000 |
| e000014_seed0 | 1 | 0.860 | yes | 1 | 1.000 | 0 | 0.368 | 1.000 / 1.000 |
| e000014_seed1 | 1 | 0.870 | yes | 1 | 1.000 | 0 | 0.108 | 1.000 / 1.000 |
| e000014_seed2 | 1 | 0.880 | yes | 1 | 1.000 | 0 | 0.285 | 1.000 / 1.000 |
| e000019_seed5 | 1 | 0.840 | yes | 1 | 1.000 | 0 | 0.189 | 1.000 / 1.000 |
| e000019_seed6 | 1 | 0.840 | yes | 1 | 1.000 | 0 | 0.397 | 1.000 / 1.000 |
| e000019_seed7 | 1 | 0.890 | yes | 1 | 1.000 | 0 | 0.144 | 1.000 / 1.000 |

## The rule, fixed before the run

Worst checkpoint over every family and seed. The epoch rows cannot come out otherwise -- the gate is a pure function of the row's own marker, so a retained row's verdict is constant in time -- and are recorded, not scored: `epoch/capacity` at most 1 and `epoch/equidistant_max_gap` are reported for the record. What is read: VOID if the radial profile has no finite operational radius (the gate accepts everywhere, and E-000029's instrument reading is wrong). RE-OPENS E-000053 if arm K fails -- if the gate does not accept two HMAC keys of the same content alike (`key/accept_key*` below 0.99 or a gap over 0.01), then E-000053's 1.000 acceptance was a property of one key and the content-marker option is re-measured before anything else is read. DISCONNECTED if the radial or tangential band count exceeds 1: the accepting set of a trained gate is not a cap, which is a fact about the instrument worth recording (it still does not buy a freshness predicate, per the proposition). CAP otherwise: one accepting region, of the recorded radius, indifferent to the signing key -- the instrument behind E-000029, E-000053 and the proposition, in one table. Fixed before the run.

## Pre-registered criteria

| criterion (worst checkpoint) | required | observed | result |
|---|---|---|---|
| radial/operational_radius | <= 2.0 | 0.8900 | PASS |
| radial/n_bands | <= 1.0 | 1.0000 | PASS |
| tangential/n_bands_max | <= 1.0 | 1.0000 | PASS |
| key/accept_key0 | >= 0.99 | 1.0000 | PASS |
| key/accept_key1 | >= 0.99 | 1.0000 | PASS |
| key/accept_gap | <= 0.01 | 0.0000 | PASS |
| epoch/capacity | <= 1.0 | 0.0000 | PASS |
