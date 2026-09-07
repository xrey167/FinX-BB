"""Serving lifetime regressions; random tiny GPT-2 is NOT capability evidence."""
import threading
import numpy as np
import pytest
import torch
from so.cavi_snapshot import ForwardSnapshotConsumptionGuard
from so.tests.test_cavi_snapshot import FakeAdapter
from so.tests.test_llm_adapter import _adapter, _bank, _prompt


def lock_available_from_another_thread(lock):
    box = []
    def worker():
        ok = lock.acquire(timeout=.1)
        box.append(ok)
        if ok: lock.release()
    th = threading.Thread(target=worker, daemon=True); th.start(); th.join(timeout=1.)
    assert not th.is_alive()
    return box == [True]


def test_snapshot_releases_when_early_block_raises_before_terminal_hook():
    a = FakeAdapter(); lock = threading.RLock()
    g = ForwardSnapshotConsumptionGuard(a, lambda: np.ones(2, dtype=bool), lock)
    def fail(*args): raise RuntimeError('controlled early block failure')
    h = a.lm.transformer.h[0].register_forward_hook(fail)
    try:
        with pytest.raises(RuntimeError, match='controlled early'):
            a.lm(torch.zeros(1, 1))
        assert lock_available_from_another_thread(lock)
    finally:
        h.remove(); g.__exit__(None, None, None)


def test_adapter_discards_memory_context_after_exception():
    a = _adapter(); bank = _bank(); ids, am, last = _prompt()
    def fail(*args): raise RuntimeError('controlled core failure')
    h = a.lm.transformer.h[0].register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match='controlled core'):
            a(bank, ids, am, last)
        assert a._ctx is None
    finally:
        h.remove(); a._ctx = None


def test_snapshot_does_not_make_shared_adapter_request_state_isolated():
    torch.set_num_threads(1)
    a = _adapter(); bank = _bank(); ids, am, last = _prompt(B=1)
    with torch.no_grad():
        expected_memory = a(bank, ids, am, last)[1].clone()
        expected_bypass = a(None, ids, am, last)[1].clone()
    assert float((expected_memory-expected_bypass).abs().max()) > 1e-4
    paused, release = threading.Event(), threading.Event()
    result, errors = {}, []
    def pause_memory(module, args):
        if threading.current_thread().name == 'memory-request':
            paused.set()
            if not release.wait(3.): raise RuntimeError('test coordination timed out')
    def worker(name, b):
        try:
            with torch.no_grad(): result[name] = a(b, ids, am, last)[1].clone()
        except Exception as e: errors.append(repr(e))
    h = a.lm.register_forward_pre_hook(pause_memory)
    g = ForwardSnapshotConsumptionGuard(a, lambda: np.ones(len(bank['subject']), dtype=bool), threading.RLock())
    th = threading.Thread(target=worker, name='memory-request', args=('memory', bank), daemon=True)
    try:
        th.start(); assert paused.wait(2.)
        worker('bypass', None)
        release.set(); th.join(timeout=3.)
        assert not th.is_alive() and not errors
        torch.testing.assert_close(result['bypass'], expected_bypass, rtol=0., atol=0.)
        torch.testing.assert_close(result['memory'], expected_memory, rtol=0., atol=0.)
    finally:
        release.set(); th.join(timeout=3.); h.remove(); g.close()
