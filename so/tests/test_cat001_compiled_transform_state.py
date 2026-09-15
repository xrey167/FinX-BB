import random
import pytest
from so.experiments.cat001_compiled_transform_state import (
    P,I,CompiledMemory,ProductTree,combine,det,encode,fold,interaction_control,inv,mm,
    naive_patch_control,project,query_outputs,safe_queries,seed_size,
)


def test_matrix_inverse_and_identity():
    for i in range(20):
        m=encode(i,12345+i)
        assert det(m)!=0
        assert mm(m,inv(m))==I and mm(inv(m),m)==I


def test_tree_root_matches_ordered_fold():
    leaves=[encode(i,100+i) for i in range(64)]
    tree=ProductTree(leaves)
    root,_=fold(leaves)
    assert tree.root==root


@pytest.mark.parametrize('n',[8,16,64,256])
def test_every_point_update_matches_full_rebuild(n):
    leaves=[encode(i,1000+i) for i in range(n)]
    t=ProductTree(leaves)
    for pos in [1,n//3,n//2,n-2]:
        value=encode(pos,900000+pos)
        calls=t.update(pos,value); leaves[pos]=value
        root,full=fold(leaves)
        assert t.root==root
        assert calls==(n.bit_length()-1)
        assert full==n-1


def test_noncommutative_interaction_controls():
    leaves=[encode(i,5000+i) for i in range(32)]
    r=interaction_control(leaves,10)
    assert r['nonzero_projective_interactions']>0
    assert r['adjacent_swap_changes_root']
    assert r['adjacent_swap_changes_queries']


def test_naive_context_free_inverse_patch_fails_interior_edit():
    leaves=[encode(i,6000+i) for i in range(32)]
    r=naive_patch_control(leaves,15,encode(15,999999))
    assert not r['naive_equals_fresh']


def test_generation_epoch_rejects_aba_even_if_numerical_root_returns():
    payloads=[100+i for i in range(16)]
    m=CompiledMemory(payloads); snap=m.snapshot(); p=7; old=payloads[p]
    m.edit(p,payload=old+500); m.edit(p,payload=old)
    assert m.tree.root==snap[0]
    with pytest.raises(RuntimeError):
        m.consume(snap,safe_queries([snap[0]],8))


def test_revoke_identity_matches_clean_rebuild():
    payloads=[200+i for i in range(16)]
    m=CompiledMemory(payloads); p=6
    m.edit(p,revoke=True)
    leaves=[encode(i,x) for i,x in enumerate(payloads)]; leaves[p]=I
    root,_=fold(leaves)
    assert m.tree.root==root


def test_query_consumes_root_only():
    leaves=[encode(i,7000+i) for i in range(64)]
    root,_=fold(leaves); qs=safe_queries([root],32)
    out=query_outputs(root,qs)
    assert len(out)==32 and all(v is not None for v in out)


@pytest.mark.parametrize('seed',range(5))
def test_strongest_baseline_has_identical_operation_count(seed):
    r=seed_size(seed,64,updates=4)
    assert r['operation_conventional_over_candidate']==1.0
    assert r['operation_full_over_candidate']>10
    assert r['candidate_and_conventional_same_algorithm']
    assert r['stale_epoch_rejected']


def test_projective_readout_is_not_additive_in_matrix_entries():
    m=encode(1,123); x=17
    y=project(m,x)
    shifted=((m[0]+1)%P,m[1],m[2],m[3])
    assert project(shifted,x)!=y
