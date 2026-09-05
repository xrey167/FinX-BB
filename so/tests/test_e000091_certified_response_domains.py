"""E91 regression and adversarial controls, not trained-language-model tests."""
from dataclasses import replace
import numpy as np
import pytest
from so.experiments.e000091_certified_response_domains import (
    LIMIT, Network, case, certified, compile_response, full, full_bigint,
    full_kernel_prevalidated, hybrid, identical, ordinary_affine,
    safe_matmul, safe_add, maxabs, smooth_control,
)


@pytest.mark.parametrize("seed",range(5))
@pytest.mark.parametrize("scale",[1,16,256,4096])
def test_finite_updates_against_bigint_at_every_write(seed,scale):
    n = case(seed,d=32,layers=4)
    r = compile_response(n)
    p = n.source+np.array([scale,-scale],dtype=np.int64)
    zf,hf = full(n,p)
    zb,hb = full_bigint(n,p)
    zraw,hraw = full_kernel_prevalidated(n,p)
    assert np.array_equal(zf.astype(object),zb)
    assert np.array_equal(hf.astype(object),hb)
    assert identical(zraw,zf) and identical(hraw,hf)
    z,h,ok = certified(r,p)
    za,ha,oka = ordinary_affine(r,p)
    assert identical(z,za) and identical(h,ha) and ok == oka
    if ok:
        assert identical(z,zf) and identical(h,hf)
    zh,hh,prefix = hybrid(n,r,p)
    assert identical(zh,zf) and identical(hh,hf)
    assert 0 <= prefix <= 4


@pytest.mark.parametrize("seed",range(5))
def test_delete_to_never_and_no_op(seed):
    n = case(seed,d=32,layers=4)
    r = compile_response(n)
    z,h,ok = certified(r,n.source)
    assert ok and identical(z,r.old_z) and identical(h,r.old_h)
    p = np.zeros(2,dtype=np.int64)
    z,h,_ = hybrid(n,r,p)
    zf,hf = full(n,p)
    assert identical(z,zf) and identical(h,hf)


@pytest.mark.parametrize("seed",range(5))
def test_recompiled_sequential_revision_chain(seed):
    n = case(seed,d=32,layers=4)
    original = n.source.copy()
    for p in [original+1,np.zeros(2,dtype=np.int64),-original,original]:
        r = compile_response(n)
        z,h,_ = hybrid(n,r,p)
        zf,hf = full(n,p)
        assert identical(z,zf) and identical(h,hf)
        n = replace(n,source=p.copy())


def scalar_case(source):
    return Network(np.array([[[1]],[[1]]],dtype=np.int64),
                   np.zeros((2,1),dtype=np.int64),np.zeros(1,dtype=np.int64),
                   np.ones((1,1),dtype=np.int64),np.array([source],dtype=np.int64))


def test_crossing_requires_fallback_not_stale_linear_response():
    n = scalar_case(1)
    r = compile_response(n)
    p = np.array([-1],dtype=np.int64)
    _,unsafe,ok = certified(r,p)
    assert not ok and int(unsafe[0,0]) == -1
    _,fixed,prefix = hybrid(n,r,p)
    assert prefix == 0
    assert np.array_equal(fixed,np.zeros((2,1),dtype=np.int64))


def test_zero_boundary_is_conservatively_replayed():
    n = scalar_case(1)
    r = compile_response(n)
    _,_,ok = certified(r,np.array([0],dtype=np.int64))
    assert not ok  # Reuse is sufficient, not exactly minimal.
    n = scalar_case(0)
    r = compile_response(n)
    _,_,ok = certified(r,np.array([-1],dtype=np.int64))
    assert ok


def test_bad_arithmetic_is_rejected_before_wrap():
    with pytest.raises(OverflowError):
        safe_matmul(np.array([[LIMIT]],dtype=np.int64),np.array([2],dtype=np.int64))
    with pytest.raises(OverflowError):
        safe_add(np.array([LIMIT],dtype=np.int64),np.array([1],dtype=np.int64))
    assert maxabs(np.array([-LIMIT-1],dtype=np.int64)) == LIMIT+1
    r = compile_response(scalar_case(1))
    with pytest.raises(OverflowError):
        certified(r,np.array([-LIMIT-1],dtype=np.int64))


@pytest.mark.parametrize("bad",[np.array([1.0]),np.array([1,2],dtype=np.int64)])
def test_source_shape_and_dtype_are_part_of_contract(bad):
    r = compile_response(scalar_case(1))
    with pytest.raises(ValueError):
        certified(r,bad)
    with pytest.raises(ValueError):
        ordinary_affine(r,bad)


def test_gate_checks_do_not_authenticate_learned_or_tampered_maps():
    n = scalar_case(10)
    r = compile_response(n)
    forged = replace(r,maps=r.maps+1)
    p = np.array([11],dtype=np.int64)
    z,h,ok = certified(forged,p)
    zf,hf = full(n,p)
    assert ok  # Establishes the trusted-compilation premise, not a safe input.
    assert not identical(h,hf)


def test_context_change_requires_recompilation_even_if_gates_stay_same():
    n = scalar_case(10)
    r = compile_response(n)
    changed = replace(n,context=np.array([2],dtype=np.int64))
    p = np.array([11],dtype=np.int64)
    _,h,ok = certified(r,p)
    _,hf = full(changed,p)
    assert ok and not identical(h,hf)


@pytest.mark.parametrize("seed",range(5))
def test_smooth_response_requires_more_than_a_frozen_jacobian(seed):
    row = smooth_control(seed)
    assert row["no_op_exact"]
    assert not row["frozen_response_exact"]
    assert row["frozen_response_maxabs"] > 0


@pytest.mark.parametrize("seed",range(5))
def test_independent_torch_integer_stack(seed):
    torch = pytest.importorskip("torch")
    n = case(seed,d=32,layers=4)
    p = n.source+np.array([3,-4],dtype=np.int64)
    h = torch.from_numpy(n.context.copy())+torch.from_numpy(n.source_basis.copy()) @ torch.from_numpy(p)
    zs,hs = [],[]
    for w,b in zip(n.weights,n.bias):
        z = torch.from_numpy(w.copy()) @ h+torch.from_numpy(b.copy())
        h = torch.relu(z)
        zs.append(z.numpy().copy()); hs.append(h.numpy().copy())
    zf,hf = full(n,p)
    assert identical(np.stack(zs),zf) and identical(np.stack(hs),hf)
