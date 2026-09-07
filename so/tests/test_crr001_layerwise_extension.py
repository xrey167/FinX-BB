from types import SimpleNamespace
import numpy as np
import torch
from so.experiments.crr001_layerwise_extension import cache_arrays, flatten, REVISIONS
from so.experiments.crr001_finite_response_rank import oracle_screen


def test_separate_nonlinear_coefficients_are_stronger_than_shared_basis():
    s=np.arange(9,dtype=np.float64)
    a=s[:,None]*np.array([[1.,2.]])
    b=(s*s)[:,None]*np.array([[3.,4.]])
    aggregate=oracle_screen(np.concatenate((a,b),axis=1),0,1,ranks=[1])
    assert aggregate['oracle'][0]['maxabs']>1e-3
    for tensor in (a,b):
        assert oracle_screen(tensor,0,1,ranks=[1])['oracle'][0]['maxabs']<1e-10


def test_snapshots_copy_values_and_flatten_in_layer_key_value_order():
    pairs=tuple((torch.full((1,2),float(2*i)),torch.full((1,2),float(2*i+1))) for i in range(3))
    arrays=cache_arrays(SimpleNamespace(past_key_values=pairs))
    pairs[0][0].fill_(99.)
    assert np.array_equal(flatten(arrays),np.repeat(np.arange(6,dtype=np.float32),2))


def test_initial_checkpoint_revisions_are_pinned():
    assert len(REVISIONS)==2
    assert all(len(v)==40 and all(c in '0123456789abcdef' for c in v) for v in REVISIONS.values())
