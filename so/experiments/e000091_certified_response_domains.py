"""E91: exact finite-response regions; a baseline screen, not an invention.

All source revisions use the same immutable context and weights. Integer
arithmetic separates exact finite response from floating-point reassociation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

LIMIT = int(np.iinfo(np.int64).max)


def maxabs(a: np.ndarray) -> int:
    """Safe even for int64 minimum (convert to Python before abs)."""
    return max(abs(int(a.min(initial=0))), abs(int(a.max(initial=0))))


def safe_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    bound = maxabs(a) * maxabs(b) * a.shape[-1]
    if bound > LIMIT:
        raise OverflowError("Conservative integer matmul bound exceeded")
    return a @ b


def safe_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if maxabs(a) + maxabs(b) > LIMIT:
        raise OverflowError("Conservative integer addition bound exceeded")
    return a + b


def identical(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


@dataclass
class Network:
    weights: np.ndarray
    bias: np.ndarray
    context: np.ndarray
    source_basis: np.ndarray
    source: np.ndarray

    def input_at(self, p: np.ndarray) -> np.ndarray:
        return safe_add(self.context, safe_matmul(self.source_basis, p))


@dataclass
class Response:
    old_z: np.ndarray
    old_h: np.ndarray
    maps: np.ndarray
    active: np.ndarray
    intercept: np.ndarray
    source: np.ndarray
    max_delta: int
    max_source: int


def case(seed: int, d: int = 256, layers: int = 8, rank: int = 2) -> Network:
    rng = np.random.default_rng(seed)
    w = rng.integers(-1, 2, size=(layers, d, d), dtype=np.int64)
    bias = rng.integers(-16, 17, size=(layers, d), dtype=np.int64)
    context = rng.integers(-10000, 10001, size=d, dtype=np.int64)
    basis = np.zeros((d, rank), dtype=np.int64)
    basis[:rank] = np.eye(rank, dtype=np.int64)
    source = rng.integers(-20000, 20001, size=rank, dtype=np.int64)
    return Network(w, bias, context, basis, source)


def full(n: Network, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = n.input_at(p)
    zs, hs = [], []
    for w, b in zip(n.weights, n.bias):
        z = safe_add(safe_matmul(w, h), b)
        h = np.maximum(z, 0)
        zs.append(z)
        hs.append(h)
    return np.stack(zs), np.stack(hs)


def full_kernel_prevalidated(n: Network, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tight conventional replay for the tested, independently checked inputs.

    Arithmetic safety is not established by this kernel. The experiment first
    verifies its entire trajectory against arbitrary precision for each input.
    This permissive timing control removes repeated weight-array bound scans;
    it is NOT an arbitrary-input certifier or a deployed lifecycle runtime.
    """
    h = n.context + n.source_basis @ p
    zs, hs = [], []
    for w, b in zip(n.weights, n.bias):
        z = w @ h + b
        h = np.maximum(z, 0)
        zs.append(z)
        hs.append(h)
    return np.stack(zs), np.stack(hs)


def full_bigint(n: Network, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Independent arbitrary-precision reference, not timed as a baseline."""
    h = n.context.astype(object) + n.source_basis.astype(object) @ p.astype(object)
    zs, hs = [], []
    for w, b in zip(n.weights, n.bias):
        z = w.astype(object) @ h + b.astype(object)
        # Explicit Python scalar activation, independent of int64 vector kernel.
        h = np.array([max(0, int(v)) for v in z], dtype=object)
        zs.append(z.copy())
        hs.append(h.copy())
    return np.stack(zs), np.stack(hs)


def compile_response(n: Network) -> Response:
    old_z, old_h = full(n, n.source)
    active = old_z > 0
    response = n.source_basis.copy()
    maps = []
    for w, gate in zip(n.weights, active):
        pre = safe_matmul(w, response)
        maps.append(pre)
        response = pre * gate[:, None]
    maps = np.stack(maps)
    offset = safe_matmul(maps, n.source)
    intercept = safe_add(old_z, -offset)
    # Conservative declared arithmetic domains, checked cheaply at evaluation.
    per_unit = maxabs(maps) * maps.shape[-1]
    max_delta = (LIMIT - maxabs(old_z)) // max(1, per_unit)
    max_source = (LIMIT - maxabs(intercept)) // max(1, per_unit)
    return Response(old_z, old_h, maps, active, intercept, n.source.copy(), max_delta, max_source)


def check_source(r: Response, p: np.ndarray) -> None:
    if p.shape != r.source.shape or p.dtype != np.int64:
        raise ValueError("Require an int64 source vector with the compiled shape")


def proposed(r: Response, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    check_source(r, p)
    # Ensure subtraction cannot wrap; Python scalar arithmetic is intentional.
    values = [int(x)-int(y) for x, y in zip(p, r.source)]
    if max(map(abs, values), default=0) > r.max_delta:
        raise OverflowError("Revision outside certified arithmetic domain")
    delta = np.array(values, dtype=np.int64)
    z = r.old_z + r.maps @ delta
    return z, z * r.active


def ordinary_affine(r: Response, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Conventional cached affine-region baseline. Same maps/guard budget."""
    check_source(r, p)
    if maxabs(p) > r.max_source:
        raise OverflowError("Source outside certified arithmetic domain")
    z = r.intercept + r.maps @ p
    valid = bool(np.array_equal(z > 0, r.active))
    return z, z * r.active, valid


def certified(r: Response, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    z, h = proposed(r, p)
    valid = bool(np.array_equal(z > 0, r.active))
    return z, h, valid


def hybrid(n: Network, r: Response, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    z, h = proposed(r, p)
    mismatch = np.flatnonzero(np.any((z > 0) != r.active, axis=1))
    if not len(mismatch):
        return z, h, len(h)
    first = int(mismatch[0])
    # Discard predictions from first unverified layer onward. Earlier layers
    # are valid by induction; the continuation is ordinary dense cone replay.
    prev = n.input_at(p) if first == 0 else h[first-1].copy()
    for layer in range(first, len(h)):
        z[layer] = safe_add(safe_matmul(n.weights[layer], prev), n.bias[layer])
        prev = np.maximum(z[layer], 0)
        h[layer] = prev
    return z, h, first


def timed(function: Callable[[], object], rounds: int = 41) -> float:
    for _ in range(5):
        function()
    out = []
    for _ in range(rounds):
        t = time.perf_counter_ns()
        function()
        out.append(time.perf_counter_ns()-t)
    return float(np.median(out))


def smooth_control(seed: int) -> dict:
    rng = np.random.default_rng(9000+seed)
    d, layers, rank = 64, 4, 2
    weights = rng.normal(size=(layers,d,d))*(.7/np.sqrt(d))
    bias = rng.normal(scale=.2, size=(layers,d))
    context = rng.normal(scale=.2, size=d)
    basis = rng.normal(scale=.1, size=(d,rank))
    source = rng.normal(scale=.5, size=rank)
    delta = rng.normal(scale=.05, size=rank)
    def run(p):
        h = context + basis @ p
        states = []
        for w,b in zip(weights,bias):
            h = np.tanh(w @ h + b)
            states.append(h.copy())
        return np.stack(states)
    old = run(source)
    u = basis.copy()
    maps = []
    for w,h in zip(weights,old):
        u = (1-h*h)[:,None] * (w @ u)
        maps.append(u.copy())
    patch = old + np.stack(maps) @ delta
    fresh = run(source+delta)
    return dict(frozen_response_maxabs=float(np.max(np.abs(patch-fresh))),
                frozen_response_exact=identical(patch,fresh),
                no_op_exact=identical(old,run(source)),
                second_language_backbone=False)


def run_seed(seed: int, d: int = 256, layers: int = 8, rounds: int = 41) -> dict:
    n = case(seed,d,layers)
    r = compile_response(n)
    # Independently verify compiled response matrices, including intermediates.
    response = n.source_basis.astype(object)
    for i,w in enumerate(n.weights):
        pre = w.astype(object) @ response
        assert np.array_equal(pre, r.maps[i].astype(object))
        response = pre * r.active[i,:,None]
    rng = np.random.default_rng(10000+seed)
    tasks = [(str(scale), n.source+rng.integers(-scale,scale+1,size=2,dtype=np.int64))
             for scale in (1,16,256,4096) for _ in range(16)]
    tasks.append(("delete_to_never",np.zeros(2,dtype=np.int64)))
    rows = []
    accepted_sample = rejected_sample = None
    for index,(magnitude,p) in enumerate(tasks):
        zref,href = full(n,p)
        zraw,hraw = full_kernel_prevalidated(n,p)
        assert identical(zraw,zref) and identical(hraw,href)
        zbig,hbig = full_bigint(n,p)
        assert np.array_equal(zref.astype(object),zbig)
        assert np.array_equal(href.astype(object),hbig)
        z,h,valid = certified(r,p)
        za,ha,va = ordinary_affine(r,p)
        assert identical(z,za) and identical(h,ha) and valid == va
        zr,hr,prefix = hybrid(n,r,p)
        assert identical(zr,zref) and identical(hr,href)
        exact = identical(h,href) and identical(z,zref)
        if valid:
            assert exact
            if accepted_sample is None and np.any(p != n.source):
                accepted_sample = p.copy()
        elif rejected_sample is None:
            rejected_sample = p.copy()
        rows.append(dict(index=index,magnitude=magnitude,source=p.tolist(),
                         source_delta=(p-n.source).tolist(),certified=valid,
                         proposed_every_write_exact=exact,
                         proposed_maxabs=maxabs(h-href),
                         hybrid_every_write_exact=True,
                         verified_prefix_layers=prefix,
                         hybrid_dense_suffix_layers=layers-prefix,
                         changed_persistent_coordinates=int(np.count_nonzero(href != r.old_h)),
                         changed_preactivation_coordinates=int(np.count_nonzero(zref != r.old_z)),
                         structural_preactivation_coordinates=layers*d,
                         reference_bigint_exact=True))
    if accepted_sample is None:
        accepted_sample = n.source.copy()  # Label timings as no-op if needed.
    values = {
        "full_replay_ns":timed(lambda: full(n,accepted_sample),rounds),
        "prevalidated_full_kernel_ns":timed(lambda: full_kernel_prevalidated(n,accepted_sample),rounds),
        "certified_response_ns":timed(lambda: certified(r,accepted_sample),rounds),
        "ordinary_affine_ns":timed(lambda: ordinary_affine(r,accepted_sample),rounds),
        "compile_with_full_forward_ns":timed(lambda: compile_response(n),max(7,rounds//3)),
        "accepted_timing_is_nontrivial_edit":bool(np.any(accepted_sample != n.source)),
    }
    if rejected_sample is not None:
        values["rejected_hybrid_ns"] = timed(lambda: hybrid(n,r,rejected_sample),rounds)
        values["rejected_full_ns"] = timed(lambda: full(n,rejected_sample),rounds)
    values["dense_over_certified"] = values["full_replay_ns"]/values["certified_response_ns"]
    values["prevalidated_dense_over_certified"] = values["prevalidated_full_kernel_ns"]/values["certified_response_ns"]
    values["ordinary_over_certified"] = values["ordinary_affine_ns"]/values["certified_response_ns"]
    values["compile_over_prevalidated_full"] = values["compile_with_full_forward_ns"]/values["prevalidated_full_kernel_ns"]
    values["compile_over_full"] = values["compile_with_full_forward_ns"]/values["full_replay_ns"]
    # Each alternative gets old z/h and maps; one anchor vector is enough.
    common_bytes = r.old_z.nbytes+r.old_h.nbytes+r.maps.nbytes+r.active.nbytes+r.source.nbytes
    return dict(seed=seed,d=d,layers=layers,source_rank=2,source_old=n.source.tolist(),
                events=rows,timings=values,
                memory=dict(model_array_bytes=sum(x.nbytes for x in (n.weights,n.bias,n.context,n.source_basis)),
                            response_matrices_bytes=r.maps.nbytes,
                            candidate_cache_bytes=common_bytes,
                            ordinary_matched_budget_bytes=common_bytes,
                            baseline_optional_intercept_replaces_old_z=True,
                            harness_holds_both_anchors_not_deployment_allocation=True),
                smooth_control=smooth_control(seed))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds",nargs="+",type=int,default=[0,1,2,3,4])
    p.add_argument("--d",type=int,default=256)
    p.add_argument("--layers",type=int,default=8)
    p.add_argument("--rounds",type=int,default=41)
    p.add_argument("--output",type=Path,default=Path("e000091-results.json"))
    args = p.parse_args()
    results = dict(experiment="E-000091",status="candidate_baseline_screen_not_invention",
                   python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   seeds=[run_seed(s,args.d,args.layers,args.rounds) for s in args.seeds],
                   trained_backbones=0,full_system_utility="NOT_EVALUATED")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2)+"\n")
    for row in results["seeds"]:
        print(json.dumps(dict(seed=row["seed"],accepted=sum(e["certified"] for e in row["events"]),
                              total=len(row["events"]),timings=row["timings"])))


if __name__ == "__main__":
    main()
