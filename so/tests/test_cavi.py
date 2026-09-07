import numpy as np

from so.cavi import CAVIAuthority, ResolveWitness, RowManifest, Scope


def manifest(auth, pod_ids, alias_ids):
    return RowManifest(
        pod_id=np.asarray(pod_ids, dtype=np.int64),
        pod_incarnation=np.asarray([auth.pod_incarnation(p) for p in pod_ids], dtype=np.int64),
        alias_id=np.asarray(alias_ids, dtype=np.int64),
        alias_incarnation=np.asarray([auth.alias_incarnation(a) if a >= 0 else 0 for a in alias_ids], dtype=np.int64),
    )


def test_alias_relink_falsifies_pod_only_baseline():
    a=CAVIAuthority(); a.create_pod(1); a.create_pod(2); a.create_alias(10,1)
    w=a.witness(10)
    a.relink_alias(10,2)
    assert a.validate_pod_only(w) is True      # old referent is still live/current
    assert a.validate_witness(w) is False      # but the reference itself changed


def test_row_mask_preserves_bystander_and_rejects_stale_alias():
    a=CAVIAuthority(); a.create_pod(1); a.create_pod(2); a.create_alias(10,1); a.create_alias(20,2)
    m=manifest(a,[1,1,2,2],[-1,10,-1,20])
    a.relink_alias(10,2)
    got=a.row_mask(m,full=True)
    assert got.tolist()==[True,False,True,True]


def test_pod_update_invalidates_canonical_and_alias_rows():
    a=CAVIAuthority(); a.create_pod(1); a.create_alias(10,1)
    m=manifest(a,[1,1],[-1,10])
    a.update_pod(1)
    assert a.row_mask(m,full=True).tolist()==[False,False]


def test_aba_old_witness_stays_dead():
    a=CAVIAuthority(); a.create_pod(1); a.create_alias(10,1)
    w=a.witness(10)
    a.delete_pod(1); a.recreate_pod_same_id(1)
    assert not a.validate_witness(w)
    assert a.pod_incarnation(1)>w.pod_incarnation


def test_scope_is_explicit():
    a=CAVIAuthority(); a.create_pod(1); a.create_alias(10,1)
    w=a.witness(10)
    assert a.scope(in_scope=False,witness=None) is Scope.BYPASS
    assert a.scope(in_scope=True,witness=w) is Scope.RESOLVE
    a.revoke_alias(10)
    assert a.scope(in_scope=True,witness=w) is Scope.UNKNOWN
