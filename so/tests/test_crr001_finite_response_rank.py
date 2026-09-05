from fractions import Fraction
import numpy as np
import pytest
from so.experiments.crr001_finite_response_rank import (
    response, modular_rank_det, fraction_rank, oracle_screen, identical,
)

@pytest.mark.parametrize('d',[2,4,8,16,32,64])
def test_exact_rank_exceeds_scalar_source_dimension(d):
    a=[[response(i+1,j+2) for j in range(d)] for i in range(d)]
    r,det=modular_rank_det(a)
    assert r==d and det!=0
    if d<=8:
        assert fraction_rank(a)==d

@pytest.mark.parametrize('u',[1,2,7,31])
@pytest.mark.parametrize('v',[Fraction(1,3),Fraction(1),Fraction(2),Fraction(9)])
def test_nonlinear_decoder_is_exact(u,v):
    actual=Fraction(u*v-1,u*v+1)-Fraction(u-1,u+1)
    assert response(u,v)==actual


def test_zero_revision_and_repeated_source_values_are_not_full_rank():
    a=[[response(i+1,1) for j in range(4)] for i in range(4)]
    assert modular_rank_det(a)[0]==0
    a=[[response(i+1,2) for j in range(4)] for i in range(4)]
    assert modular_rank_det(a)[0]==1


def test_invalid_modulus_denominator_rejected():
    with pytest.raises(ValueError):
        modular_rank_det([[Fraction(1,7)]],prime=7)


def test_modular_zero_determinant_does_not_claim_rational_singularity():
    a=[[Fraction(7),Fraction(0)],[Fraction(0),Fraction(1)]]
    assert modular_rank_det(a,prime=7)[0]==1
    assert fraction_rank(a)==2


def test_linear_oracle_control_and_no_op():
    x=np.arange(33,dtype=np.float64)-16
    states=x[:,None]*np.arange(1,17)[None,:]+2
    row=oracle_screen(states,16,8,ranks=[1])
    assert row['numerical_ranks']['1e-10']==1
    assert row['oracle'][0]['maxabs']<1e-10


def test_oracle_rejects_material_nonlinear_residual():
    states=np.array([[float(response(i+1,j+2)) for i in range(16)] for j in range(16)])
    row=oracle_screen(states,0,1,ranks=[1,2])
    assert row['oracle'][0]['maxabs']>1e-5
    assert row['oracle'][1]['maxabs']>1e-7
    assert row['oracle'][0]['relative_frobenius_error']>=row['oracle'][1]['relative_frobenius_error']


def test_strict_byte_equality():
    assert not identical(np.array([0.]),np.array([-0.]))
    assert not identical(np.array([1.]),np.array([1.],dtype=np.float32))
