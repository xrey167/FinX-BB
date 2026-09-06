"""Diagnostics: measurements that are NOT records (ledger §31.48).

Everything in ``so/experiments/`` writes a pre-registered record into ``so/results/`` and is never
edited afterwards. The scripts here do not: they read a checkpoint, print a table, and write their raw
numbers to a path the caller names (default: the current directory). They exist because §31.48 cites
their numbers and a cited number that cannot be re-run is a number on trust.

Each one names the substrate it needs. None of them trains anything; ``so/results/checkpoints/`` is
ignored by git, so retrain the named recipe first (``e000015_symlink_cells``, ``e000020_symlink_gpt2``)
and expect a fresh adapter, not the recorded one: this session measured direct 0.6867 against the
record's 0.5667 on the same recipe, seed and budget.
"""
