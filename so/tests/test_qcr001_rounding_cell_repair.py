from copy import deepcopy
from fractions import Fraction as F
import numpy as np
import pytest
import torch
from so.experiments.qcr001_rounding_cell_repair import (
    rn_grid,certify_grid,scalar_screen,round_native,compare,same,
    rounding_box,fraction_solve,make_witness,verify_witness,separation_search,oracle,
)

@pytest.mark.parametrize('x,expected',[(F(1,2),0),(F(3,2),2),(F(5,2),2),(-F(1,2),0),(-F(3,2),-2),(-F(5,2),-2)])
def test_exact_even_ties(x,expected):
    assert rn_grid(x,F(1))==expected

@pytest.mark.parametrize('j',range(8))
def test_interval_containment_with_nonzero_error(j):
    q=F(1,128);c=F(5,4)+j*q
    assert certify_grid(c+q/16,q/8,q)==c
    for i in range(-8,9):
        assert rn_grid(c+q/16+i*q/64,q)==c

@pytest.mark.parametrize('power',[10,20,40,100])
def test_tiny_errors_can_cross_cells(power):
    q=F(1,128);m=F(5,4)+q/2;e=q/2**power
    assert rn_grid(m-e,q)!=rn_grid(m+e,q)
    assert certify_grid(m,e,q) is None


def test_invalid_interval_rejected():
    with pytest.raises(ValueError):certify_grid(1,-1)
    with pytest.raises(ValueError):rn_grid(1,0)


def test_scalar_controls():
    r=scalar_screen()
    assert r['nonzero_error_exact_rounding_controls']==64
    assert r['staged_result']!=r['final_only_rounding_result']

@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float32])
def test_true_native_rounding_box(dtype):
    values=round_native(np.array([1.25,-1.25,16.0,-16.]),dtype)
    low,high=rounding_box(values,dtype)
    assert np.all(low<values) and np.all(values<high)
    assert same(round_native((values+low)/2,dtype),values)
    assert same(round_native((values+high)/2,dtype),values)


def test_rational_linear_solve():
    a=[[F(2),F(1)],[F(1),F(3)]]
    x=fraction_solve(a,[1,2])
    assert x==[F(1,5),F(3,5)]
    with pytest.raises(ValueError):fraction_solve([[1,1],[1,1]],[1,2])


def test_exact_box_separation_and_tamper():
    u=np.array([[1.,0.],[0.,1.],[1.,1.]])
    lo=np.array([0.,0.,1.]);hi=np.array([.1,.1,1.1])
    w=make_witness(u,lo,hi,[0,1,2])
    assert w and verify_witness(w)
    bad=deepcopy(w);bad['extra_row'][0]=str(F(bad['extra_row'][0])+1)
    assert not verify_witness(bad)
    bad=deepcopy(w);bad['anchor_low'][0]='-100'
    assert not verify_witness(bad)


def test_feasible_box_does_not_certify_separation():
    u=np.array([[1.,0.],[0.,1.],[1.,1.]])
    assert make_witness(u,np.array([0.,0.,0.]),np.array([1.,1.,2.]),[0,1,2]) is None


def test_exact_shift_is_not_float_subtraction_assumption():
    u=np.array([[1.],[1.]])
    lo=np.array([1.,2.]);hi=lo.copy();old=np.array([2.**100,2.**100])
    # float subtraction loses the difference; Fraction box offsets do not.
    assert np.array_equal(lo-old,hi-old)
    w=make_witness(u,lo,hi,[0,1],old)
    assert w and verify_witness(w)


def test_lp_proposal_requires_exact_witness():
    # Four scalar outputs cannot lie on this represented-real line and all
    # round to mutually incompatible bf16 values.
    u=np.ones((4,1))/2
    old=np.zeros(4);target=np.array([1.,1.,2.,2.])
    r=separation_search(u,old,target,torch.bfloat16)
    assert r['status']=='EXACT_SEPARATION'
    assert verify_witness(r['witness'])


def test_oracle_rounding_positive_and_native_noop():
    x=(np.arange(33)-16)/8
    states=1+x[:,None]*np.array([[.125,.25]])
    states=round_native(states,torch.bfloat16)
    row=oracle(states,torch.bfloat16,1)
    assert row['ranks'][0]['all_nontrivial_exact']==32
    assert row['ranks'][0]['new_nontrivial_exact']==32


def test_comparison_is_not_tolerance_or_shape_broadcast():
    assert not same(np.array([0.]),np.array([-0.]))
    with pytest.raises(ValueError):compare(np.ones(2),np.ones(3))
    r=compare(np.ones(1000),np.concatenate((np.ones(999),[1.0001])))
    assert r['match_fraction']==.999 and not r['byte_identical']
