"""RMC002 exact operator regressions; not trained-reader tests."""
from copy import deepcopy
import numpy as np
import pytest

from so.experiments import rmc001_rounding_margin_certificates as R
from so.experiments import rmc002_anchor_cell_certificates as M


def test_anchor_state_initial_certificates_verify():
    model=R.build_model(0)
    state=M.init_anchor_state(model)
    assert M.verify_all_layers(model,state)


@pytest.mark.parametrize('seed',range(3))
def test_unit_edit_and_revert_are_exact_without_margin_consumption(seed):
    row=M.edit_revert_control(seed)
    assert row['return_displacement_zero'] and row['refreshes']==0


def test_tampering_anchor_radius_and_sensitivity_is_detected():
    row=M.tamper_control(0)
    assert all(row.values())


@pytest.mark.parametrize('seed',range(2))
def test_large_move_forces_refresh_then_followup_remains_exact(seed):
    row=M.large_refresh_control(seed)
    assert row['first_refreshed_layers']>=1
    assert row['first_refresh_ops']>0 and row['exact']


def test_high_gain_control_falls_back_exactly():
    row=M.leaky_control(0)
    assert row['refreshed_layers']>0 and row['refresh_ops']>0 and row['exact']


def test_one_main_edit_matches_full_rebuild():
    model=R.build_model(1)
    state=M.init_anchor_state(model)
    target=np.zeros(R.A,dtype=np.int64); target[0]=1
    fresh,_=R.full_rebuild(model,target)
    cell=M.anchor_edit(model,state,target)
    assert cell['refreshed_layers']==0
    assert all(R.exact_equal(a,b) for a,b in zip(state.h,fresh))


def test_shared_anchor_rejects_integrity_tamper_before_reuse():
    model=R.build_model(2); state=M.init_anchor_state(model)
    state.layers[3].source_anchor[1]=7
    target=np.zeros(R.A,dtype=np.int64); target[0]=1
    with pytest.raises(ValueError):
        M.anchor_edit(model,state,target)


def test_preregistered_main_gate_survives_seed_zero():
    row=M.main_path(0)
    assert row['sparse_over_candidate']>=10.0
    assert row['total_refreshed_layers']==0
    assert all(x['every_write_exact'] for x in row['edits'])


def test_cost_model_counts_anchor_coordinates_conservatively():
    model=R.build_model(0); state=M.init_anchor_state(model)
    target=np.zeros(R.A,dtype=np.int64)
    cell=M.anchor_edit(model,state,target)
    # Every layer pays 3 operations per source coordinate plus one per block,
    # even on a no-op source edit.
    blocks=((R.D-R.A+R.BLOCK-1)//R.BLOCK)
    floor=R.L*(M.ANCHOR_COORD_COST*R.A+blocks)
    assert cell['ops']>=floor
