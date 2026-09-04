# E-000030 — a deletion certificate for the recorded checkpoints

Seeds [0, 1, 2], 3 targets, the recorded E-000010 checkpoints, no training.
For each lifecycle operation, every value the deleted payload could hold is swept and what the
model computes is compared. The payload domain has 256 values, so the sweep
is every case rather than a sample.

## What survives the deletion

| operation | certified for every query (interface) | certified on the swept queries | mediation premise | first quantity that moves | encodings swept |
|---|---|---|---|---|---|
| revoke | no | yes | consistent | encode_bank[v_f] | 2 |
| shred | no | no | consistent | encode_bank[v_f] | 2 |
| delete | yes (structural) | yes (structural) | n/a | the row is not in the bank | 0 |
| revoke (GPT-2, soft gate) | yes | - | - | - | 782 |
| shred (GPT-2, soft gate) | no | - | - | encode_bank[values] | 2  (residual 1.39e-02) |
| revoke (GPT-2, hard gate) | yes | - | - | - | 782 |
| shred (GPT-2, hard gate) | yes | - | - | - | 782 |

`interface` compares `encode_bank`'s output. The forward reads the bank only there, so an
invariant encoding means an invariant computation FOR EVERY POSSIBLE QUERY, not just the swept
ones. `outputs` compares the returned logits over an exhaustive single-hop query domain
`mediation` is the falsification check on the premise the interface column rests on: it looks
for an output that moves while the encoding does not, and voids the certificate if it finds one.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| shred/outputs_certified | >= 1.0 | 0.0000 | FAIL |
| revoke/outputs_certified | >= 1.0 | 1.0000 | PASS |
| revoke/mediation_consistent | >= 1.0 | 1.0000 | PASS |
| shred/mediation_consistent | >= 1.0 | 1.0000 | PASS |
