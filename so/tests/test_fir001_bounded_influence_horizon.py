import numpy as np
import pytest
from so.experiments.fir001_bounded_influence_horizon import (
    K, STEPS, equal_state, fraction_control, payload, run_world, seed_case, source_step,
)

@pytest.mark.parametrize('seed',range(5))
def test_complete_state_coalesces_exactly_at_structural_horizon(seed):
    a,_=run_world(seed,payload(seed,0)); b,_=run_world(seed,payload(seed,1))
    assert not equal_state(a[K-1],b[K-1])
    assert equal_state(a[K],b[K])
    assert all(equal_state(a[t],b[t]) for t in range(K,STEPS))

@pytest.mark.parametrize('seed',range(5))
def test_late_recall_requires_reinjection_or_retained_source(seed):
    pa,pb=payload(seed,0),payload(seed,1)
    _,ra=run_world(seed,pa);_,rb=run_world(seed,pb)
    assert ra[64]==rb[64]
    _,ra=run_world(seed,pa,query_reread_at=64);_,rb=run_world(seed,pb,query_reread_at=64)
    assert ra[64]!=rb[64]

@pytest.mark.parametrize('seed',range(5))
def test_leaky_long_term_recall_breaks_exact_horizon(seed):
    a,_=run_world(seed,payload(seed,0),leaky=True);b,_=run_world(seed,payload(seed,1),leaky=True)
    assert not any(equal_state(x,y) for x,y in zip(a,b))

@pytest.mark.parametrize('seed',range(5))
def test_fraction_support_extinguishes_exactly(seed):
    h=fraction_control(seed,payload(seed,0))
    assert all(x=='0' for x in h[K])


def test_source_shift_is_structurally_nilpotent_without_injection():
    z=[np.ones(32,dtype=object)*(i+1) for i in range(K)]
    for _ in range(K): z=source_step(z,None)
    assert all(np.count_nonzero(s)==0 for s in z)

@pytest.mark.parametrize('seed',range(5))
def test_full_seed_contract(seed):
    r=seed_case(seed)
    assert r['first_complete_coalescence_write']==K
    assert not r['late_no_reread_source_distinguishable']
    assert r['late_reread_source_distinguishable']
    assert not r['leaky_control_ever_coalesces']
