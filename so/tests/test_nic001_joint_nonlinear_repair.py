from fractions import Fraction
from types import SimpleNamespace
import itertools
import numpy as np
import pytest
from so.experiments.nic001_joint_nonlinear_repair import (
    exact_screen,mobius,reconstruct,tensor_screen,identical,cache_arrays,snapshots_identical,
)

@pytest.mark.parametrize('n',[1,2,3,4,5])
@pytest.mark.parametrize('seed',range(3))
def test_exact_subset_transform_roundtrip(n,seed):
    rng=np.random.default_rng(seed)
    states=np.array([[Fraction(int(x),7) for x in row] for row in rng.integers(-20,21,size=(2**n,4))],dtype=object)
    co=mobius(states)
    for mask in range(2**n):
        assert np.array_equal(reconstruct(co,mask,n),states[mask])

@pytest.mark.parametrize('size',[0,3,5,7])
def test_bad_world_count_rejected(size):
    with pytest.raises(ValueError):
        mobius(np.zeros((size,2)))


def test_triple_invisible_to_proper_subset_tests_and_commutes():
    rows={x['family']:x for x in exact_screen()}
    triple=rows['pure_triple']
    assert all(triple['exact_by_order']['2'][:7])
    assert not triple['exact_by_order']['2'][7]
    assert triple['all_increment_orders_agree'] and not triple['increment_orders_correct']
    assert triple['final_error_by_order']['2']=='-11'


def test_pairwise_and_separable_positive_controls():
    rows={x['family']:x for x in exact_screen()}
    assert all(rows['separable']['exact_by_order']['1'])
    assert all(rows['pairwise']['exact_by_order']['2'])
    assert not all(rows['pairwise']['exact_by_order']['1'])
    assert rows['compact_joint']['three_way_interaction']!='0'


@pytest.mark.parametrize('seed',range(5))
def test_exact_pair_charts_can_miss_material_triple(seed):
    rng=np.random.default_rng(seed)
    base=rng.normal(size=(2,3,4))
    directions=rng.normal(size=(3,2,3,4))
    triple=rng.normal(size=base.shape)
    states=[]
    for mask in range(8):
        state=base.copy()
        for i in range(3):
            if mask&(1<<i): state+=directions[i]
        if mask==7: state+=triple
        states.append(state)
    stats=tensor_screen(states)
    assert stats['orders']['2']['material']
    assert stats['full_order_roundoff']['maxabs']<1e-12
    assert all(x['maxabs']<1e-12 for x in stats['pair_removal_independent_errors'].values())


def test_byte_comparison_strict():
    assert not identical(np.array([0.]),np.array([-0.]))


def test_cache_snapshot_copies_and_stage_compare():
    import torch
    cache=((torch.zeros((1,2,3,4),dtype=torch.float64),torch.ones((1,2,3,4),dtype=torch.float64)),)
    out=SimpleNamespace(past_key_values=cache)
    a=cache_arrays(out)
    b=cache_arrays(out)
    assert snapshots_identical([a],[b])
    cache[0][0].fill_(8)
    assert snapshots_identical([a],[b])
    c=cache_arrays(out)
    assert not snapshots_identical([a],[c])


def test_append_statistics_isolate_new_not_retained_changes():
    states=[]
    for mask in range(8):
        x=np.zeros((1,1,4,2))
        if mask==7: x[...,:2,:]=1.
        states.append(x)
    assert tensor_screen(states)['orders']['2']['material']
    assert not tensor_screen([x[...,2:,:] for x in states])['orders']['2']['material']


def test_missing_snapshot_components_do_not_pass_vacuously():
    t=np.zeros((1,1,2,2))
    full=[[[t,t]]]
    assert not snapshots_identical(full,[])
    assert not snapshots_identical(full,[[]])
    assert not snapshots_identical(full,[[[t]]])
