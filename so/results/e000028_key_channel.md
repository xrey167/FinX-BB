# E-000028 — the channel SHRED does not close

Seeds [0, 1, 2, 3, 4], 100 targets, the recorded E-000010 checkpoints, no training.
The attacker knows a cell's subject and relation, finds its column from the routing of the
ordinary forward question, then sweeps every candidate object through a REVERSE query and
takes the candidate that steers the read onto that column.

## Object recovery through the key channel (worst seed)

| condition | object recovered top-1 | top-5 | mean rank | chance top-1 | chance rank |
|---|---|---|---|---|---|
| active | 1.0000 | 1.0000 | 0.0 | 0.0039 | 127.5 |
| shred | 1.0000 | 1.0000 | 0.0 | 0.0039 | 127.5 |
| revoke | 0.0200 | 0.0400 | 120.6 | 0.0039 | 127.5 |
| delete | 0.0200 | 0.0400 | 120.6 | 0.0039 | 127.5 |

`active` is the validity control: if the sweep cannot recover a live object, the shredded
number means nothing. `revoke` and `delete` remove the row from routing altogether.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| column_located | >= 0.9 | 1.0000 | PASS |
| active/object_top1 | >= 0.2 | 1.0000 | PASS |

## What F4 would require of this channel

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| shred/object_top1 | <= 0.00390625 | 1.0000 | FAIL |

## All measures

| measure | mean over seeds | worst seed | pooled n | 95% CI lower | 95% CI upper |
|---|---|---|---|---|---|
| column_located | 1.0000 | 1.0000 | 500 | 0.9926 | 1.0000 |
| shred/column_located | 1.0000 | 1.0000 | 500 | 0.9926 | 1.0000 |
| active/column_mass | 0.9980 | 0.9977 | - | - | - |
| active/object_top1 | 1.0000 | 1.0000 | 500 | 0.9926 | 1.0000 |
| active/object_top5 | 1.0000 | 1.0000 | 500 | 0.9926 | 1.0000 |
| active/object_mean_rank | 0.0000 | 0.0000 | - | - | - |
| active/margin_mean | 0.6195 | 0.5968 | - | - | - |
| shred/object_top1 | 1.0000 | 1.0000 | 500 | 0.9926 | 1.0000 |
| shred/object_top5 | 1.0000 | 1.0000 | 500 | 0.9926 | 1.0000 |
| shred/object_mean_rank | 0.0000 | 0.0000 | - | - | - |
| shred/margin_mean | 0.6195 | 0.5968 | - | - | - |
| revoke/object_top1 | 0.0040 | 0.0200 | 500 | 0.0005 | 0.0144 |
| revoke/object_top5 | 0.0220 | 0.0400 | 500 | 0.0110 | 0.0390 |
| revoke/object_mean_rank | 128.0200 | 120.6300 | - | - | - |
| revoke/margin_mean | 0.0022 | 0.0016 | - | - | - |
| delete/object_top1 | 0.0040 | 0.0200 | 500 | 0.0005 | 0.0144 |
| delete/object_top5 | 0.0220 | 0.0400 | 500 | 0.0110 | 0.0390 |
| delete/object_mean_rank | 128.0200 | 120.6300 | - | - | - |
| delete/margin_mean | 0.0022 | 0.0016 | - | - | - |
