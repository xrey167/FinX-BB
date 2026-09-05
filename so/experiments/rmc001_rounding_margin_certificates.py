"""RMC001: proof-directed native-cell reuse inside a dense dependency cone.

Exact integer/ties-to-even operator screen. This is a candidate mechanism assay,
not BF16/FP32 or trained-reader evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

D = 1024
L = 24
A = 8
Q = 4096
BLOCK = 32
EDITS = 64


def qround_scalar(n: int, q: int = Q) -> int:
    sign = -1 if n < 0 else 1
    x = abs(int(n))
    a, r = divmod(x, q)
    twice = 2 * r
    if twice > q or (twice == q and (a & 1)):
        a += 1
    return sign * a


def qround(z: np.ndarray, q: int = Q) -> np.ndarray:
    z = np.asarray(z, dtype=np.int64)
    sign = np.where(z < 0, -1, 1).astype(np.int64)
    x = np.abs(z)
    a = x // q
    r = x % q
    inc = (2 * r > q) | ((2 * r == q) & ((a & 1) == 1))
    return sign * (a + inc.astype(np.int64))


def safe_radius(z: int, y: int, q: int = Q) -> int:
    """Largest symmetric integer radius around exact z staying in y's RN-even cell."""
    assert qround_scalar(z, q) == y
    lo, hi = 0, q + 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if qround_scalar(z - mid, q) == y and qround_scalar(z + mid, q) == y:
            lo = mid
        else:
            hi = mid
    return lo


def exact_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


@dataclass
class Model:
    weights: list[np.ndarray]
    bias: list[np.ndarray]
    input_background: np.ndarray
    old_source: np.ndarray
    old_h: list[np.ndarray]
    old_z: list[np.ndarray]


@dataclass
class BlockCert:
    layer: int
    start: int
    end: int
    max_abs_weight: int
    safe_radius: int
    uncertainty: int = 0
    refreshes: int = 0


@dataclass
class CandidateState:
    source: np.ndarray
    h: list[np.ndarray]
    active_z: list[np.ndarray]
    certs: list[list[BlockCert]]


@dataclass
class SparseState:
    source: np.ndarray
    h: list[np.ndarray]
    z: list[np.ndarray]


def _even_random(rng, n):
    return (2 * rng.integers(-3, 4, size=n)).astype(np.int64)


def build_model(seed: int, leaky: bool = False) -> Model:
    rng = np.random.default_rng(900000 + seed + (1000 if leaky else 0))
    weights, bias, old_h, old_z = [], [], [], []
    source = np.zeros(A, dtype=np.int64)
    background = _even_random(rng, D)
    background[:A] = source
    prev = background.copy()
    for layer in range(L):
        # Protected rows have dense mathematical dependency. Active rows carry
        # the source exactly on a small quantized channel.
        w = rng.choice(np.array([-1, 1], dtype=np.int64), size=(D, D))
        w[:A] = 0
        for i in range(A):
            w[i, i] = Q
        w[A:, :A] = (Q // 2 if leaky else 1)
        desired = _even_random(rng, D)
        desired[:A] = 0
        offset = np.zeros(D, dtype=np.int64)
        # One first-layer guard block has small certified slack. Alternating
        # edits exhaust its envelope and force honest refresh without making
        # all layers expensive.
        if layer == 0 and not leaky:
            offset[A:A + BLOCK] = Q // 2 - 24
        target_z = Q * desired + offset
        b = target_z - w @ prev
        z = w @ prev + b
        h = qround(z)
        assert exact_equal(h, desired)
        weights.append(w)
        bias.append(b)
        old_z.append(z.copy())
        old_h.append(h.copy())
        prev = h
    return Model(weights, bias, background, source, old_h, old_z)


def full_rebuild(model: Model, source: np.ndarray):
    h = model.input_background.copy()
    h[:A] = source
    hs, zs = [], []
    for w, b in zip(model.weights, model.bias):
        z = w @ h + b
        h = qround(z)
        zs.append(z.copy()); hs.append(h.copy())
    return hs, zs


def make_certs(model: Model) -> list[list[BlockCert]]:
    out = []
    for layer in range(L):
        certs = []
        z, h, w = model.old_z[layer], model.old_h[layer], model.weights[layer]
        for start in range(A, D, BLOCK):
            end = min(D, start + BLOCK)
            radius = min(safe_radius(int(z[j]), int(h[j])) for j in range(start, end))
            maxw = int(np.max(np.abs(w[start:end])))
            certs.append(BlockCert(layer, start, end, maxw, radius))
        out.append(certs)
    return out


def verify_certs(model: Model, certs: list[list[BlockCert]]) -> bool:
    for layer, group in enumerate(certs):
        for c in group:
            w = model.weights[layer][c.start:c.end]
            z = model.old_z[layer][c.start:c.end]
            h = model.old_h[layer][c.start:c.end]
            if c.max_abs_weight != int(np.max(np.abs(w))):
                return False
            actual_radius = min(safe_radius(int(a), int(b)) for a, b in zip(z, h))
            if c.safe_radius != actual_radius:
                return False
    return True


def init_candidate(model: Model) -> CandidateState:
    certs = make_certs(model)
    assert verify_certs(model, certs)
    return CandidateState(model.old_source.copy(), [x.copy() for x in model.old_h],
                          [z[:A].copy() for z in model.old_z], certs)


def init_sparse(model: Model) -> SparseState:
    return SparseState(model.old_source.copy(), [x.copy() for x in model.old_h],
                       [x.copy() for x in model.old_z])


def edit_path(seed: int):
    source = np.zeros(A, dtype=np.int64)
    path = []
    # Alternating unit edits conservatively exhaust the first guard envelope.
    for t in range(48):
        i = t % A
        source = source.copy()
        source[i] = 1 - source[i] if t % 2 == 0 else -source[i]
        path.append(source.copy())
    # Explicit UPDATE, ZERO/NEVER, restoration, sign changes, then more unit edits.
    source = source.copy(); source[3] += 2; path.append(source.copy())
    source = np.zeros(A, dtype=np.int64); path.append(source.copy())
    restore = np.zeros(A, dtype=np.int64); restore[1] = -2; restore[5] = 1
    path.append(restore.copy()); source = restore
    source = -source; path.append(source.copy())
    while len(path) < EDITS:
        i = len(path) % A
        source = source.copy(); source[i] += 1 if len(path) % 2 == 0 else -1
        path.append(source.copy())
    return path[:EDITS]


def sparse_edit(model: Model, state: SparseState, new_source: np.ndarray):
    prev_old = model.input_background.copy(); prev_old[:A] = state.source
    prev_new = model.input_background.copy(); prev_new[:A] = new_source
    new_hs, new_zs = [], []
    ops = 0
    changed_counts = []
    for layer in range(L):
        changed = np.flatnonzero(prev_new != prev_old)
        delta = prev_new[changed] - prev_old[changed]
        z = state.z[layer].copy()
        if len(changed):
            z += model.weights[layer][:, changed] @ delta
            ops += D * len(changed)
        h = qround(z)
        new_hs.append(h.copy()); new_zs.append(z.copy())
        changed_counts.append(int(np.count_nonzero(h != state.h[layer])))
        prev_old, prev_new = state.h[layer], h
    state.source = new_source.copy(); state.h = new_hs; state.z = new_zs
    return ops, changed_counts


def candidate_edit(model: Model, state: CandidateState, new_source: np.ndarray):
    prev_old = model.input_background.copy(); prev_old[:A] = state.source
    prev_new = model.input_background.copy(); prev_new[:A] = new_source
    new_hs, new_active_z = [], []
    ops = 0; checks = 0; refresh_ops = 0; refreshed = 0; changed_counts = []
    for layer in range(L):
        old_layer = state.h[layer]
        h = old_layer.copy()
        changed = np.flatnonzero(prev_new != prev_old)
        delta = prev_new[changed] - prev_old[changed]
        # Exact sparse delta for source-active rows only.
        az = state.active_z[layer].copy()
        if len(changed):
            az += model.weights[layer][:A, changed] @ delta
            ops += A * len(changed)
        h[:A] = qround(az)
        l1 = int(np.sum(np.abs(delta), dtype=np.int64))
        for cert in state.certs[layer]:
            checks += 1
            if l1 == 0:
                continue
            bound = cert.max_abs_weight * l1
            candidate_uncertainty = cert.uncertainty + bound
            if candidate_uncertainty <= cert.safe_radius:
                cert.uncertainty = candidate_uncertainty
                continue
            # Certificate exhausted: pay exact dense refresh for this block.
            z = model.weights[layer][cert.start:cert.end] @ prev_new + model.bias[layer][cert.start:cert.end]
            fresh = qround(z)
            h[cert.start:cert.end] = fresh
            cert.safe_radius = min(safe_radius(int(a), int(b)) for a, b in zip(z, fresh))
            cert.uncertainty = 0
            cert.refreshes += 1
            refreshed += 1
            block_rows = cert.end - cert.start
            refresh_ops += block_rows * D
        ops += checks - (0 if layer else 0)  # each block proof check counts one unit
        new_hs.append(h.copy()); new_active_z.append(az.copy())
        changed_counts.append(int(np.count_nonzero(h != old_layer)))
        prev_old, prev_new = old_layer, h
        checks = 0
    state.source = new_source.copy(); state.h = new_hs; state.active_z = new_active_z
    return ops + refresh_ops, refresh_ops, refreshed, changed_counts


def exact_costs(model: Model, path):
    sparse = init_sparse(model); candidate = init_candidate(model)
    sparse_total = candidate_total = dense_total = 0
    rows = []
    for edit_index, source in enumerate(path):
        sh, _ = full_rebuild(model, source)
        sop, schanged = sparse_edit(model, sparse, source)
        cop, refresh_ops, refreshed, cchanged = candidate_edit(model, candidate, source)
        assert all(exact_equal(a, b) for a, b in zip(sparse.h, sh))
        assert all(exact_equal(a, b) for a, b in zip(candidate.h, sh))
        sparse_total += sop; candidate_total += cop; dense_total += L * D * D
        rows.append(dict(edit=edit_index, source=source.tolist(), sparse_ops=sop,
                         candidate_ops=cop, refresh_ops=refresh_ops, refreshed_blocks=refreshed,
                         sparse_changed=schanged, candidate_changed=cchanged,
                         every_write_exact=True))
    return dict(sparse_ops=sparse_total, candidate_ops=candidate_total, dense_ops=dense_total,
                sparse_over_candidate=sparse_total / candidate_total,
                dense_over_candidate=dense_total / candidate_total,
                edits=rows,
                total_refreshes=sum(c.refreshes for g in candidate.certs for c in g),
                candidate_aux_scalars=sum(len(g) * 3 for g in candidate.certs) + L * A,
                sparse_exact_preactivation_scalars=L * D,
                candidate_persistent_state_scalars=L * D,
                sparse_persistent_state_scalars=L * D)


def leaky_control(seed: int):
    # Smaller depth of reporting but same declared architecture dimensions.
    model = build_model(seed, leaky=True)
    path = edit_path(seed)[:4]
    result = exact_costs(model, path)
    return dict(edits=len(path), sparse_over_candidate=result['sparse_over_candidate'],
                total_refreshes=result['total_refreshes'],
                all_exact=all(r['every_write_exact'] for r in result['edits']))


def boundary_controls():
    # Naively accepting the geometric midpoint for an odd output is wrong under ties-to-even.
    odd_y = 1
    z = odd_y * Q
    naive = Q // 2
    assert qround_scalar(z + naive) != odd_y
    exact_r = safe_radius(z, odd_y)
    assert exact_r == naive - 1
    even_y = 2
    assert safe_radius(even_y * Q, even_y) == naive
    return dict(odd_midpoint_naive_radius=naive, odd_exact_safe_radius=exact_r,
                even_exact_safe_radius=safe_radius(even_y * Q, even_y),
                naive_non_strict_midpoint_is_unsound=True)


def benchmark(seed: int, rounds=5):
    model = build_model(seed)
    path = edit_path(seed)[:8]
    def run_sparse():
        state = init_sparse(model)
        for p in path: sparse_edit(model, state, p)
    def run_candidate():
        state = init_candidate(model)
        for p in path: candidate_edit(model, state, p)
    vals = {}
    for name, fn in [('sparse', run_sparse), ('candidate', run_candidate)]:
        fn(); samples=[]
        for _ in range(rounds):
            t=time.perf_counter_ns(); fn(); samples.append(time.perf_counter_ns()-t)
        vals[name+'_ns']=float(np.median(samples))
    vals['sparse_over_candidate_wall']=vals['sparse_ns']/vals['candidate_ns']
    vals['wall_not_application_gate']=True
    return vals


def run_seed(seed: int):
    model = build_model(seed)
    result = exact_costs(model, edit_path(seed))
    assert result['sparse_over_candidate'] >= 10.0, result['sparse_over_candidate']
    assert all(r['every_write_exact'] for r in result['edits'])
    return dict(seed=seed, main=result, leaky=leaky_control(seed), benchmark=benchmark(seed))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    ap.add_argument('--output',type=Path,default=Path('rmc001-results.json'))
    a=ap.parse_args()
    result=dict(experiment='RMC-001',status='candidate_operator_screen_not_invention',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=platform.python_version(),numpy=np.__version__,dimensions=dict(width=D,depth=L,active=A,q=Q,block=BLOCK,edits=EDITS),
        boundary=boundary_controls(),seeds=[run_seed(s) for s in a.seeds],
        trained_backbones=0,full_system_gates='NOT_EVALUATED')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([dict(seed=r['seed'],ratio=r['main']['sparse_over_candidate'],
        dense=r['main']['dense_over_candidate'],refreshes=r['main']['total_refreshes'],
        leaky_ratio=r['leaky']['sparse_over_candidate'],wall=r['benchmark']['sparse_over_candidate_wall']) for r in result['seeds']],indent=2))

if __name__=='__main__': main()
