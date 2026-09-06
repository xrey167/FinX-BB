# SIT-001 — the audit window

Pre-registered at `docs/novelty/sit001-preregister.md`. Bars: arrival ratio ≥ 0.50, lens |cos| to the unembedding ≤ 0.90.

## arm `recorded` — read_layers [8, 10]

| site | moves(shred) | arrival | lens cos | in window | jprobe(A−N) | jprobe(S−N) |
|---|---|---|---|---|---|---|
| `site8` | 2.80 | 0.034 | 0.784 | no | -0.0134 | -0.0089 |
| `site9` | 4.18 | 0.052 | 0.822 | no | -0.0223 | -0.0089 |
| `site10` | 83.53 | 1.000 | 1.000 | no | +0.9152 | +0.0759 |
| `final` | 27.05 | 0.325 | 1.000 | no | +0.9152 | +0.0804 |

Worst seed: `active_alias_correct` 0.8795, `shred_alias_unknown` 0.9955, `shred_alias_true_object` 0.0000, `never_alias_true_object` 0.0000, `bystander_top1_agree` 1.0000

| check | result |
|---|---|
| `P0_first_read_site_fails_arrival` | True |
| `P1_window_non_empty` | False |
| `P2_jprobe_active_minus_never_ge_030` | None |
| `P3_jprobe_shred_minus_never_le_005` | None |
| `VE1_active_alias_correct_ge_080` | True |
| `VE2_shred_alias_true_object_le_005` | True |
| `VE2_shred_alias_unknown_ge_090` | True |
| `VE3_bystander_top1_agree_ge_098` | True |
| `kill1_never_floor_le_005` | True |

Window on all seeds: `EMPTY`; registered site: `None`.
