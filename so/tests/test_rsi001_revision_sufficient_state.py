"""RSI001 regressions: interface semantics, not trained reader qualification."""
from fractions import Fraction
import itertools
import numpy as np
import pytest
import torch
from so.experiments.rsi001_revision_sufficient_state import (
    ALPHABET, DTYPES, decode_receipt, delta_write, encode_receipt, exact_screen,
    finite_screen, fraction_suffix, identical, interface_bytes, neural_build,
    neural_screen, setup, tensor_bytes,
)


def test_exact_fiber_has_256_distinct_targets():
    r=exact_screen()
    assert r['distinct_current_interfaces']==1
    assert r['distinct_final_rebuild_states']==256
    assert r['minimum_auxiliary_bits']==8


@pytest.mark.parametrize('code', [0,1,17,63,127,128,255])
def test_packed_receipt_is_lossless(code):
    assert encode_receipt(decode_receipt(code))==code


@pytest.mark.parametrize('code',[-1,256,3.5,True])
def test_invalid_receipts_rejected(code):
    with pytest.raises(ValueError):
        decode_receipt(code)


def test_lossless_receipt_roundtrip_exhaustive():
    for c in itertools.product(ALPHABET,repeat=4):
        assert decode_receipt(encode_receipt(c))==c


def test_rational_half_gate_is_injective():
    xs=list(range(-128,128))
    ys=[(Fraction(x)+1)/2 for x in xs]
    assert len(set(ys))==256
    assert [2*y-1 for y in ys]==xs


@pytest.mark.parametrize('seed',range(5))
def test_complete_current_interface_does_not_determine_rebuild(seed):
    r=neural_screen(seed)
    assert r['full_interface_byte_identical']
    assert r['old_all_write_trajectories_identical']
    assert r['distinct_never_final_states']
    assert all(e>1e-8 for e in r['final_layer_maxabs'])
    assert all(r['exact_receipt_plus_replay'])


@pytest.mark.parametrize('seed',range(5))
def test_extra_value_information_changes_interface(seed):
    w,f,x,a,b=setup(seed)
    sa=neural_build(a,w,f,x,True)[-1]
    sb=neural_build(b,w,f,x,True)[-1]
    i1,i2=interface_bytes(sa,seed),interface_bytes(sb,seed)
    assert i1==i2
    assert i1+tensor_bytes(a)!=i2+tensor_bytes(b)
    # Merely invalidating a generation cannot reconstruct either old context.
    assert i1+b'revoked:8'==i2+b'revoked:8'


@pytest.mark.parametrize('seed',range(5))
def test_wrong_receipt_and_first_layer_only_are_not_repairs(seed):
    w,f,x,a,b=setup(seed)
    fresh=neural_build(a,w,f,x,False)[-1]
    wrong=neural_build(torch.zeros(8,dtype=torch.float64),w,f,x,False)[-1]
    assert not identical(wrong,fresh)
    wrong[0]=fresh[0]
    assert not identical(wrong,fresh)


@pytest.mark.parametrize('dtype',list(DTYPES))
def test_normal_finite_numbers_collide_at_an_interior_gate(dtype):
    r=finite_screen(dtype)
    assert r['distinct_float_outputs']<256
    assert r['rational_outputs_distinct']==256
    assert r['gate']==.5
    assert r['first_collision_target_gap']>0


def test_delta_projector_control_matches_independent_formula():
    s=torch.arange(32,dtype=torch.float64).reshape(4,8)/4
    k=torch.eye(4,dtype=torch.float64)[0]
    v=torch.zeros(8,dtype=torch.float64)
    got=delta_write(s,k,v,1.)
    ref=(torch.eye(4,dtype=torch.float64)-torch.outer(k,k)) @ s + torch.outer(k,v)
    assert identical(got,ref)
    assert identical(got[1:],s[1:])


def test_byte_equality_checks_signed_zero_and_dtype():
    assert not identical(torch.tensor([0.]),torch.tensor([-0.]))
    assert not identical(torch.tensor([1.]),torch.tensor([1.],dtype=torch.float64))


def test_a_decoder_cannot_solve_both_colliding_histories():
    # Exhaust every deterministic output for this two-element target alphabet.
    target=[fraction_suffix((-3,1,1,1),False),fraction_suffix((3,1,1,1),False)]
    assert target[0]!=target[1]
    for output in target:
        assert sum(output==y for y in target)==1
