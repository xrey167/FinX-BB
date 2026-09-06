"""BHC001: scope, native countermodels and exact reference controls."""
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from so.experiments.bhc001_exact_coalescence import (
    FORMATS,machine,rollout,replay_to_join,rn_binade,same,scalar_screen,neural_screen,
)
from tools.bhc001_check_fixed_points import check


@pytest.mark.parametrize('name',FORMATS)
def test_scalar_pair_is_fixed_and_even_bias_coalesces(name):
    r=scalar_screen(name)
    assert r['native_low_fixed'] and r['native_high_fixed']
    assert r['even_bias_one_step_coalesces']


@pytest.mark.parametrize('name',FORMATS)
@pytest.mark.parametrize('seed',range(5))
def test_nonlinear_native_states_and_positive_hybrid(seed,name):
    r=neural_screen(seed,name)
    assert r['different_final_coordinates']==64
    assert r['first_adjacent_fixed_pair_time']==4
    assert r['even_bias_first_complete_join']==1
    assert r['unit_rescaled_old_read_values']==[1.]
    assert r['unit_rescaled_never_read_values']==[0.]
    c=r['reset_control']
    assert c['actual_transition_calls']==c['ordinary_transition_calls']==35
    assert c['complete_join_at']==35 and c['unsafe_first_layer_join_at']==32
    assert c['unsafe_mismatched_coordinates']==48


@pytest.mark.parametrize('name',FORMATS)
def test_source_wave_matches_independent_closed_form(name):
    m=machine(3,name)
    history=rollout(m,True,8)
    for t,state in enumerate(history):
        expected=m.initial(False)
        expected[:min(t+1,4)]+=m.unit
        assert same(state,expected)


@pytest.mark.parametrize('seed',range(5))
def test_real_operator_contraction_has_stochastic_row_premise(seed):
    m=machine(seed,'float64')
    assert torch.all(m.weights>=0)
    assert torch.equal(m.weights.sum(-1),torch.ones((4,16),dtype=torch.float64))
    # A numerical autograd diagnostic in addition to the analytic max-row proof.
    g=torch.Generator().manual_seed(seed)
    x=torch.randn((4,16),generator=g,dtype=torch.float64,requires_grad=True)
    jac=torch.autograd.functional.jacobian(m.step,x).reshape(64,64)
    assert float(jac.abs().sum(1).max())<=.5+1e-14


def test_earlier_retained_writes_are_not_cleaned_by_late_join():
    m=machine(0,'float32')
    old,fresh=rollout(m,True,64,32),rollout(m,False,64,32)
    assert same(old[35:],fresh[35:])
    assert not same(old[:35],fresh[:35])
    fixed,_,_=replay_to_join(m,old,32)
    assert same(fixed,fresh)


def test_no_join_does_not_terminate_or_claim_success_early():
    m=machine(0,'bfloat16')
    old,fresh=rollout(m,True,64),rollout(m,False,64)
    fixed,calls,join=replay_to_join(m,old,None)
    assert calls==64 and join is None and same(fixed,fresh)


@pytest.mark.parametrize('mid,index',[(F(257,256),0),(F(259,256),2)])
def test_independent_ties_to_even(mid,index):
    assert rn_binade(mid,7)==1+F(index,128)


@pytest.mark.parametrize('value',[F(0),F(3)])
def test_binade_checker_rejects_out_of_scope(value):
    with pytest.raises(ValueError):rn_binade(value,7)


def witnesses():return [scalar_screen(n) for n in FORMATS]


def test_independent_all_format_verifier():assert check(witnesses())['exact_scalar_witnesses']==4


@pytest.mark.parametrize('field,value',[
    ('low','1'),('high','1'),('half_step','1'),('mantissa_bits',99),
    ('format','not-a-dtype'),('real_gap_bound_after_64','1')])
def test_tampered_witness_rejected(field,value):
    rows=witnesses();rows[0][field]=value
    with pytest.raises(ValueError):check(rows)


@pytest.mark.parametrize('rows',[[],[{}],[{}, {}, {}, {}]])
def test_missing_witnesses_do_not_pass(rows):
    with pytest.raises((ValueError,KeyError)):check(rows)


def test_duplicate_format_rejected():
    rows=witnesses();rows[1]=deepcopy(rows[0])
    with pytest.raises(ValueError):check(rows)


def test_checker_remains_active_under_optimized_python(tmp_path):
    p=tmp_path/'bad.json';p.write_text(json.dumps({'scalar':[]}))
    script=Path(__file__).resolve().parents[2]/'tools/bhc001_check_fixed_points.py'
    proc=subprocess.run([sys.executable,'-O',str(script),str(p)],capture_output=True)
    assert proc.returncode!=0


def test_dtype_and_signed_zero_are_not_discarded():
    assert not same(torch.tensor([0.]),torch.tensor([-0.]))
    assert not same(torch.tensor([1.]),torch.tensor([1.],dtype=torch.float64))
