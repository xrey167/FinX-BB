# E-000002 — Weight-memorisation control (copy problem)

Evidence level: **E4** (Controlled neural-network evidence). Seeds: [0, 1, 2]. Fixed-world regimes trained for 2000 steps; resampled regime = E-000001-B models.

| training regime | direct (layer intact) | layer fully masked | target leak after REVOKE | target UNKNOWN after REVOKE | control after REVOKE |
|---|---|---|---|---|---|
| resampled | 100.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| fixed_routing | 100.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| fixed_no_routing | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% |

Reading: 'layer fully masked' is what the weights answer on their own. A leak after REVOKE is knowledge that survived in the weights — the copy problem the ledger warns about (sections 9, 28). The mechanism's deletion guarantee therefore depends on the training regime keeping facts out of the weights. n = 100 targets per seed (leak of 0 in 300 pooled trials -> failure rate below 1.3% at 95%).

Pre-registered criteria (worst seed; leak-type metrics use the max):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| resampled/bank_removed_acc | <= 0.02 | 0.0000 | PASS |
| resampled/target_after_revoke_leak | <= 0.02 | 0.0000 | PASS |
| resampled/control_after_revoke | >= 0.99 | 1.0000 | PASS |
| fixed_no_routing/target_after_revoke_leak | >= 0.5 | 1.0000 | PASS |
| fixed_no_routing/direct | >= 0.5 | 1.0000 | PASS |

Caveats: Only 'fixed_routing' is an empirical control: 'resampled' cannot memorise by construction and 'fixed_no_routing' cannot read the layer by construction. The fixed-world regimes see the same random lifecycle states per step as the re-sampled regime (only the world is held fixed), so the no-routing model receives inconsistent labels for revoked/shredded cells and settles on the majority label. Fixed regimes are trained for fewer steps than the re-sampled E-000001-B models.

Per seed:

**resampled**

| seed | direct | bank_removed_acc | target_before | target_after_revoke_leak | target_after_revoke_unknown | control_after_revoke |
|---|---|---|---|---|---|---|
| 0 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 1 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 2 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

**fixed_routing**

| seed | direct | bank_removed_acc | target_before | target_after_revoke_leak | target_after_revoke_unknown | control_after_revoke |
|---|---|---|---|---|---|---|
| 0 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 1 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 2 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

**fixed_no_routing**

| seed | direct | bank_removed_acc | target_before | target_after_revoke_leak | target_after_revoke_unknown | control_after_revoke |
|---|---|---|---|---|---|---|
| 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
