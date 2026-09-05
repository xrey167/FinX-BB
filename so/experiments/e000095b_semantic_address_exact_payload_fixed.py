"""E-000095B — execution-only repair for E-000095.

The first E95 Actions attempt terminated before training because the temporary
route-target monkeypatch called E20.route_targets_slots after replacing that
symbol, recursively calling itself. This wrapper captures the original function
before E95.run applies its monkeypatch and replaces only the broken helper.
No architecture, loss, data, threshold, seed, step budget or evaluation gate is
changed. The failed first attempt remains preserved as a harness negative.
"""
from __future__ import annotations

import torch

from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments import e000095_semantic_address_exact_payload as E95

_BASE_ROUTE_TARGETS = E20.route_targets_slots


def _fixed_identity_route_targets(queries, bank, world, n_reads: int, n_deref: int) -> torch.Tensor:
    base = _BASE_ROUTE_TARGETS(queries, bank, world, n_reads, n_deref).clone()
    if n_deref != 1:
        raise ValueError("E95 is pinned to one identity-selection slot per read")
    for rr in range(n_reads):
        resolve = base[:, 2 * rr]
        base[:, 2 * rr + 1] = resolve
    return base


E95._identity_route_targets = _fixed_identity_route_targets

if __name__ == "__main__":
    E95.main()
