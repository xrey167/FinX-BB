# SIT-001 — the audit window

Pre-registered at `docs/novelty/sit001-preregister.md`. Bars: arrival ratio ≥ 0.50, lens |cos| to the unembedding ≤ 0.90.

## arm `early` — read_layers [4, 6]

| site | moves(shred) | arrival | lens cos | in window | jprobe(A−N) | jprobe(S−N) |
|---|---|---|---|---|---|---|
| `site4` | 4.09 | 0.080 | 0.592 | no | -0.0491 | +0.0045 |
| `site5` | 17.67 | 0.343 | 0.643 | no | -0.0312 | -0.0045 |
| `site6` | 32.56 | 0.644 | 0.694 | **yes** | +0.7723 | +0.0179 |
| `site7` | 31.78 | 0.628 | 0.748 | **yes** | +0.8080 | +0.0491 |
| `site8` | 32.73 | 0.648 | 0.784 | **yes** | +0.8125 | +0.0312 |
| `site9` | 34.69 | 0.686 | 0.822 | **yes** | +0.7946 | +0.0491 |
| `site10` | 44.86 | 0.883 | 1.000 | no | +0.7902 | +0.0268 |
| `final` | 49.18 | 0.952 | 1.000 | no | +0.7188 | +0.0045 |

Worst seed: `active_alias_correct` 0.8482, `shred_alias_unknown` 0.9732, `shred_alias_true_object` 0.0000, `never_alias_true_object` 0.0000, `bystander_top1_agree` 1.0000

| check | result |
|---|---|
| `P0_first_read_site_fails_arrival` | True |
| `P1_window_non_empty` | True |
| `P2_jprobe_active_minus_never_ge_030` | True |
| `P3_jprobe_shred_minus_never_le_005` | True |
| `VE1_active_alias_correct_ge_080` | True |
| `VE2_shred_alias_true_object_le_005` | True |
| `VE2_shred_alias_unknown_ge_090` | True |
| `VE3_bystander_top1_agree_ge_098` | True |
| `kill1_never_floor_le_005` | True |

Window on all seeds: `['site6', 'site7', 'site8', 'site9']`; registered site: `site6`.
