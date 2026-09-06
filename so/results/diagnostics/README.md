# Diagnostics — the raw numbers §31.48 cites

These are NOT records. A record in `so/results/` carries pre-registered criteria, an evidence level and
a checkpoint SHA, and is never edited; these files are the output of `so/diagnostics/`, which reads a
checkpoint and prints a table. They are kept because §31.48 cites their numbers.

Every file here was produced on 2026-09-05 from FRESH trainings of the named recipes (no checkpoint in
this repository survives: `so/results/checkpoints/` is git-ignored). `e20_anchor_seed0.json` is the
reproduction anchor for that retrain against the recorded E-000020: same shape, different level
(direct 0.6867 against the record's 0.5667). A checkpoint SHA in these files ties a number to a file on
one machine, not to a record.

| file | what it is | substrate |
| --- | --- | --- |
| `e20_anchor_seed0.json` | E-000020's own battery on the retrained seed-0 adapter | `e000020_gpt2_seed0.pt` |
| `q1_crosslayer_seed0.json`, `q1b_derefmass_seed0.json` | routing mass per read-layer slot; layer-8 and layer-10 passthrough arms (§31.48 R2) | same |
| `q1c_layer8_seed0.json` | direct reads with the layer-8 injection zeroed (the zero-ablation calibration) | same |
| `pilot_entity_seed0.json` | entity-table swap: held-out subjects and objects (§31.48 R4) | same |
| `blank_attrib_seed0.json` | BLANK export arms with the bare-LM and alias-masked floors, McNemar (§31.48 R5) | same |
| `blank_option_check_seed0.json` | `MVCCStore(blank_export=...)` end to end through the store path | same |
| `e56_clockgate_v2.json` | recorded-gate acceptance by shell; the clock-gate benchmark (§31.48 R3) | `e000015_deref1_seed{0,1,2}.pt` |
