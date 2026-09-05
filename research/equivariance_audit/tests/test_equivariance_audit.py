"""Numerical regression tests. None are real-reader/security/J-lens tests."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit", ROOT / "src/equivariance_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
SOURCE = ROOT / "upstream/e000084_revocation_equivariance_screen.py"
if not SOURCE.exists():
    SOURCE = ROOT.parents[1] / "so/experiments/e000084_revocation_equivariance_screen.py"

@pytest.fixture(scope="module")
def original():
    return audit.load_original(SOURCE)

@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("depth", [1, 8, 24, 96])
def test_exact_matched_ordinary_alternative(original, seed, depth):
    row = audit.numerical_cell(original, seed, depth, channels=16, pods=8,
                               subset_trials=16, edits=32)
    assert row["pass"], row

@pytest.mark.parametrize("seed", range(5))
def test_original_positive_and_nonlinear_negative_preserved(original, seed):
    row = original.run_seed(seed, channels=64, depth=24, pods=32, delete_trials=16)
    assert row["pass"]
    assert row["min_generic_nonlinear_inverse_error"] >= 1e-4

def test_source_edit_cannot_silently_change_assay(tmp_path):
    changed = tmp_path / "source.py"
    changed.write_bytes(SOURCE.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Unqualified"):
        audit.load_original(changed)

@pytest.mark.parametrize("dims", [(0, 24, 32), (64, 0, 32), (64, 24, 0)])
def test_invalid_dimensions_rejected(dims):
    with pytest.raises(ValueError):
        audit.world(0, *dims)

@pytest.mark.parametrize("seed", range(3))
def test_nonlinear_readout_does_not_distinguish_equal_hidden_states(original, seed):
    z, theta, alpha, bias = audit.world(seed, 16, 24, 8)
    rng = np.random.default_rng(50 + seed)
    w = rng.normal(size=(4, 32))
    def readout(h):
        return np.tanh(w @ np.concatenate([h.real, h.imag]))
    for mask in range(256):
        active = [(mask >> i) & 1 for i in range(8)]
        total = theta[np.array(active, dtype=bool)].sum(0)
        early = original._equivariant_deep(original._rotate(z, total), alpha, bias)
        late = original._rotate(original._equivariant_deep(z, alpha, bias), total)
        np.testing.assert_allclose(readout(early), readout(late), rtol=0, atol=1e-10)

@pytest.mark.parametrize("seed", range(3))
def test_content_sensitive_gate_breaks_simple_inverse(original, seed):
    z, theta, alpha, bias = audit.world(seed, 32, 24, 8)
    def sensitive(x):
        out = x.copy()
        for a, b in zip(alpha, bias):
            out *= np.exp(.025 * np.tanh(a * np.abs(out)**2 + b + .3 * out.real))
        return out
    total = theta.sum(0)
    old = sensitive(original._rotate(z, total))
    repaired = original._rotate(old, -theta[0])
    reference = sensitive(original._rotate(z, total-theta[0]))
    assert audit.maxabs(repaired-reference) > 1e-4

@pytest.mark.parametrize("seed", range(3))
def test_orthogonal_reparameterization_cannot_hide_collapse(original, seed):
    z, theta, alpha, bias = audit.world(seed, 8, 8, 4)
    rng = np.random.default_rng(seed+90)
    q, _ = np.linalg.qr(rng.normal(size=(16,16)))
    def enc(x): return q @ np.concatenate([x.real,x.imag])
    def dec(x):
        y=q.T@x
        return y[:8]+1j*y[8:]
    def f(x): return enc(original._equivariant_deep(dec(x),alpha,bias))
    def t(x, angle): return enc(original._rotate(dec(x),angle))
    total=theta.sum(0)
    np.testing.assert_allclose(f(t(enc(z),total)),t(f(enc(z)),total),rtol=0,atol=1e-10)

def test_benchmark_compares_equal_outputs_and_charges_setup(original):
    row=audit.benchmark_cell(original,0,8,channels=16,pods=8,events=8,rounds=3)
    assert row["output_equivalence_max_error"] < 1e-10
    assert row["deep_layers_per_update_ordinary_late"] == 0
    assert row["deep_layers_per_update_inverse"] == 0
    assert all(len(t)==3 for t in row["total_sequence_including_setup_ns_raw"].values())
