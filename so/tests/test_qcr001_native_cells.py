from fractions import Fraction as F
from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest
import torch
from tools.qcr001_check_native_cells import decode, encode_rn, cell, MANTISSA


@pytest.mark.parametrize('m', [7,23])
@pytest.mark.parametrize('v', [F(0),F(1),F(-1),F(3,2),F(-7,8),F(2)**-126,F(2)**-133,F(2)**-149])
def test_exact_encoder_matches_native_cast(m,v):
    dtype=torch.bfloat16 if m==7 else torch.float32
    got=decode(encode_rn(v,m),m)
    expected=F(float(torch.tensor(float(v),dtype=dtype)))
    assert got==expected


@pytest.mark.parametrize('m', [7,23])
@pytest.mark.parametrize('v', [F(1),F(-1),F(2),F(-2),F(129,128),F(3,4)])
def test_complete_cell_matches_independent_torch_neighbors(m,v):
    dtype=torch.bfloat16 if m==7 else torch.float32
    t=torch.tensor(float(v),dtype=dtype)
    prev=torch.nextafter(t,torch.tensor(-float('inf'),dtype=dtype))
    nxt=torch.nextafter(t,torch.tensor(float('inf'),dtype=dtype))
    expected=((F(float(prev))+F(float(t)))/2,(F(float(nxt))+F(float(t)))/2)
    assert cell(encode_rn(v,m),m)==expected


def test_all_finite_bfloat16_values_roundtrip():
    for bits in range(65536):
        if ((bits>>7)&255)==255 or bits==32768:
            continue  # nonfinite and negative-zero rational sign excluded
        assert encode_rn(decode(bits,7),7)==bits


@pytest.mark.parametrize('m', [7,23])
def test_ties_to_even(m):
    one=127 << m
    low,high=cell(one,m)
    assert encode_rn(low,m)==one
    assert encode_rn(high,m)==one
    odd=one+1
    lo,hi=cell(odd,m)
    assert encode_rn(lo,m)==one
    assert encode_rn(hi,m)==one+2


def test_zero_nonfinite_and_subnormal_cells_fail_closed():
    with pytest.raises(ValueError): cell(0,7)
    with pytest.raises(ValueError): cell(255<<7,7)
    with pytest.raises(ValueError): cell(1,7)
