"""Symlink semantics: sharing versus duplication, at store and bank level (E-000015)."""
import numpy as np
import pytest
import torch

from so.data import bank_from_store
from so.experiments.e000015_symlink_cells import (AliasSpec, bank_with_links, encode_slots, load_arm,
                                                  model_config, sample_alias_world)
from so.mvcc import MVCCStore, Status
from so.model import MutableKnowledgeTransformer
from so.reference import ReferenceResolver
from so.train import make_centre
from so.world import Query, UNKNOWN


def _arms(seed=0, n_base=60, n_groups=5, n_alias=2):
    rng = np.random.default_rng(seed)
    centre = make_centre(seed, 16)
    world, spec = sample_alias_world(rng, n_base, n_groups, n_alias, n_entities=64, n_relations=4, n_synonyms=2)
    sym, sym_kids = load_arm(world, spec, centre, seed, symlink=True)
    dup, dup_kids = load_arm(world, spec, centre, seed, symlink=False)
    return world, spec, (sym, sym_kids), (dup, dup_kids)


def test_both_arms_start_from_identical_ground_truth():
    world, spec, (sym, _), (dup, _) = _arms()
    assert sym.index_view() == dup.index_view() == {f.key: f.obj for f in world.facts}


def test_one_update_reaches_every_alias_only_in_the_symlink_arm():
    world, spec, (sym, sk), (dup, dk) = _arms()
    target, aliases = spec.groups[0]
    new = (world.index[target] + 7) % world.n_entities
    sym.update(sk[target], new); dup.update(dk[target], new)
    assert all(sym.index_view()[a] == new for a in aliases)
    assert all(dup.index_view()[a] == world.index[target] for a in aliases)   # copies are untouched
    assert dup.index_view()[target] == new


def test_one_shred_deletes_every_alias_only_in_the_symlink_arm():
    world, spec, (sym, sk), (dup, dk) = _arms()
    target, aliases = spec.groups[0]
    sym.shred(sk[target]); dup.shred(dk[target])
    sv, dv = sym.index_view(), dup.index_view()
    assert all(a not in sv for a in aliases) and target not in sv
    assert all(dv[a] == world.index[target] for a in aliases)                # the copies still answer
    assert target not in dv


def test_revoking_one_alias_leaves_target_and_sibling_intact():
    world, spec, (sym, sk), _ = _arms()
    target, aliases = spec.groups[0]
    sym.revoke(sk[aliases[0]])
    v = sym.index_view()
    assert aliases[0] not in v
    assert v[aliases[1]] == world.index[target] and v[target] == world.index[target]


def test_deleting_the_target_leaves_a_dangling_pointer_in_the_bank():
    world, spec, (sym, sk), _ = _arms()
    target, aliases = spec.groups[0]
    sym.delete(sk[target])
    v = sym.index_view()
    assert all(a not in v for a in aliases)
    bank = bank_from_store(sym)
    row = int(np.where(bank.kid == sk[aliases[0]])[0][0])
    assert bool(bank.is_link[row])
    assert (int(bank.link_subject[row]), int(bank.link_relation[row])) == target   # the pointer is NOT erased


def test_relink_and_rollback_move_one_access_path_only():
    world, spec, (sym, sk), _ = _arms()
    (t0, a0), (t1, _) = spec.groups[0], spec.groups[1]
    sym.relink(sk[a0[0]], sk[t1])
    v = sym.index_view()
    assert v[a0[0]] == world.index[t1] and v[a0[1]] == world.index[t0]
    sym.rollback(sk[a0[0]], 1)
    assert sym.index_view()[a0[0]] == world.index[t0]


def test_chain_and_cycle_are_bounded():
    centre = make_centre(0, 16)
    st = MVCCStore(marker_dim=16, seed=0, marker_centre=centre)
    f = st.write(1, 0, 5)
    a = st.link(2, 0, f)
    b = st.link(3, 0, a)
    assert st.resolve_key((3, 0))[0] == 5 and len(st.resolve_key((3, 0))[1]) == 3
    c1 = st.write(10, 0, 0); c2 = st.write(11, 0, 0)
    st.relink(a, c1) if False else None
    x = st.link(20, 1, c1); y = st.link(21, 1, x)
    st.relink(x, y)                                   # x -> y -> x
    assert st.resolve_key((20, 1))[0] is None         # cycle detected, no crash


def test_reference_resolver_follows_aliases_and_traces_both_cells():
    world, spec, (sym, sk), _ = _arms()
    ref = ReferenceResolver(sym)
    target, aliases = spec.groups[0]
    q = Query("fwd", aliases[0][0], (aliases[0][1],), (world.surface_of(aliases[0][1], 0),))
    r = ref.resolve(q)
    assert r.answer == world.index[target]
    assert r.trace == (sk[aliases[0]], sk[target])


def test_slot_route_targets_name_alias_then_target():
    rng = np.random.default_rng(0)
    centre = make_centre(0, 16)
    world, spec = sample_alias_world(rng, 60, 5, 2, n_entities=64, n_relations=4, n_synonyms=2)
    bank = bank_with_links(rng, world, spec, centre, p_revoked=0.0, p_shred=0.0, p_stale=0.0, p_dangling=0.0)
    alias = spec.alias_keys[0]
    q = Query("fwd", alias[0], (alias[1],), (world.surface_of(alias[1], 0),))
    batch = encode_slots([q], bank, world, max_hops=3, n_deref=1)
    route = batch.route[0].tolist()
    assert route[0] == bank.trace_of_key[alias][0]          # resolve slot -> the alias row
    assert route[1] == bank.trace_of_key[alias][1]          # dereference slot -> the target row
    fact = [f.key for f in world.facts if f.key not in spec.alias_of][0]
    qf = Query("fwd", fact[0], (fact[1],), (world.surface_of(fact[1], 0),))
    rf = encode_slots([qf], bank, world, max_hops=3, n_deref=1).route[0].tolist()
    assert rf[1] == -1                                       # a fact hop dereferences nothing: passthrough


def test_default_model_is_unchanged_by_the_symlink_extension():
    from so.model import ModelConfig
    a = MutableKnowledgeTransformer(ModelConfig(n_entities=32))
    b = MutableKnowledgeTransformer(model_config(1))
    assert not any(n.startswith(("v_link", "link_rev_key", "deref")) for n, _ in a.named_parameters())
    assert any(n.startswith("deref") for n, _ in b.named_parameters())
