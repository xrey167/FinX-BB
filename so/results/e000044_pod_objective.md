# E-000044 — training the pod objective: is the allocation failure fixable, and at what price

Two arms on the same worlds and seeds, 700 steps, 3 seed(s). Arm B
adds the pod objective (`so/pod.py`): every access path of one fact pulled onto one carrier,
carriers of different facts pushed apart, hinged at the larger of the Welch bound and the
centring floor. Everything else is identical.

## Arm A (baseline) against arm B (pod objective)

| measure | A | B |
|---|---|---|
| accuracy | 1.0000 | 1.0000 |
| closure | 1.0000 | 1.0000 |
| silenced_rate | 1.0000 | 1.0000 |
| collateral | 0.9983 | 1.0000 |
| excess_full | 0.2399 | 0.1998 |
| excess_content | 0.0510 | 0.0325 |
| excess_address | 0.2484 | 0.2172 |
| pressure | 0.1250 | 0.1250 |

## The differences the criteria are written on

| measure | mean over seeds | worst seed |
|---|---|---|
| accuracy ratio B/A (the price) | 1.0000 | 1.0000 |
| drop in excess overlap | +0.0401 | +0.0361 |
| drop in excess ADDRESSING overlap | +0.0312 | +0.0243 |
| gain in bystander accuracy under deletion | +0.0017 | +0.0000 |
| drop in closure size | +0.0000 | +0.0000 |

## The rule, fixed before the run

excess_full_drop >= 0.10 with accuracy_ratio >= 0.95 -> allocation is trainable and E-000043's verdict has its constructive half. A drop that only arrives with accuracy_ratio below 0.95 -> the dimensions were not free and the capacity reading was closer to right. No drop -> allocation is not trainable by this objective, and the diagnosis stands without a remedy. Fixed before the run.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| A/accuracy | >= 0.6 | 1.0000 | PASS |
| accuracy_ratio | >= 0.95 | 1.0000 | PASS |
| excess_full_drop | >= 0.1 | 0.0361 | FAIL |
| collateral_gain | >= 0.05 | 0.0000 | FAIL |
