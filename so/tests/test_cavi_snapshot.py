import threading
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from so.cavi_snapshot import ForwardSnapshotConsumptionGuard


class FakeLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = SimpleNamespace(h=nn.ModuleList([nn.Identity(), nn.Identity()]))

    def forward(self, x):
        for block in self.transformer.h:
            x = block(x)
        return x


class FakeAdapter:
    def __init__(self):
        self.lm = FakeLM()
        self.cfg = SimpleNamespace(read_layers=(0, 1))
        self._ctx = {"allowed": torch.tensor([True, True])}


def test_forward_snapshot_blocks_inter_read_mutation_and_samples_once():
    adapter = FakeAdapter()
    lock = threading.RLock()
    calls = []
    mutation_started = threading.Event()
    mutation_finished = threading.Event()
    state = {"finished_between": None}

    def mask_fn():
        calls.append(1)
        return np.asarray([True, False], dtype=bool)

    def mutate():
        mutation_started.set()
        with lock:
            mutation_finished.set()

    thread_box = {}

    def between(module, inputs):
        th = threading.Thread(target=mutate, daemon=True)
        thread_box["thread"] = th
        th.start()
        assert mutation_started.wait(1.0)
        # The forward-wide guard must still own the lock here.
        state["finished_between"] = mutation_finished.wait(0.05)

    with ForwardSnapshotConsumptionGuard(adapter, mask_fn, lock):
        h = adapter.lm.transformer.h[1].register_forward_pre_hook(between)
        try:
            adapter.lm(torch.zeros(1, 1))
        finally:
            h.remove()

    th = thread_box["thread"]
    th.join(timeout=1.0)
    assert not th.is_alive()
    assert calls == [1]
    assert state["finished_between"] is False
    assert mutation_finished.is_set()
    assert adapter._ctx["allowed"].tolist() == [True, False]
    assert adapter._ctx["_cavi_forward_snapshot"] is True


def test_forward_snapshot_is_noop_without_memory_context():
    adapter = FakeAdapter()
    adapter._ctx = None
    lock = threading.RLock()
    calls = []
    with ForwardSnapshotConsumptionGuard(adapter, lambda: calls.append(1) or np.asarray([True]), lock):
        adapter.lm(torch.zeros(1, 1))
    assert calls == []
