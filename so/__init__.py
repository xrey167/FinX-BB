"""SO — Modular Neural Operating System: experimental research package.

Sub-modules:

- ``so.world``         synthetic knowledge worlds with exact ground truth
- ``so.mvcc``          the mutable knowledge layer (versioned cells, Neural-MVCC lifecycle)
- ``so.reference``     the mechanical reference resolver (experiment E-000001-A)
- ``so.model``         the Mini-Transformer neural core with a mutable knowledge interface
- ``so.train``         resampled-world training so that facts cannot be copied into weights
- ``so.evaluation``    the test families (direct, multi-hop, provenance, lifecycle, locality, noise)
- ``so.attacks``       reconstruction attacks (paraphrase, reverse, forced-choice, probes, dependencies)
- ``so.interventions`` causal interventions on cells (disable / swap / restore / replace)
- ``so.ledger``        result recording (JSON + Markdown) with evidence levels
"""

__version__ = "0.1.0"
