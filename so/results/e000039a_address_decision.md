# E-000039-A — Addressing versus transport in the paraphrase gap

E-000017-B's three checkpoints are evaluated as recorded; nothing is trained.

routing_share >= 0.7 -> train the address arm alone; <= 0.3 -> train the read arm alone; in between -> train both. Fixed before either arm was trained.

| measure | mean over seeds | worst seed |
|---|---|---|
| heldout/gap | 1.1100 | 1.0400 |
| heldout/residual_gap | nan | nan |
| heldout/routing_share | nan | nan |
