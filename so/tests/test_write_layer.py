"""E-000084: address deep, write late — the contract, on a tiny random GPT-2 (no download).

The option under test separates WHERE the memory is addressed (the read layers, unchanged) from WHERE
it is written (one late block). Three things are by construction and are pinned here so the experiment
can declare them as pipeline checks rather than claim rows:

  1. with ``write_layer=None`` nothing changes: every recorded configuration is bit-identical;
  2. with ``write_layer`` = the final block, every hidden state and every persisted K/V tensor up to and
     including that block is bit-identical to the no-memory forward (exposure exactly 0.0), while the
     logits still move (the memory is material) and still depend on the bank's payload (UPDATE reaches
     the answer without touching the cache);
  3. the routing tensors — the addressing decision — are bit-identical between the in-place and the
     deferred configuration for the same weights, because the read layers compute exactly the same thing
     up to the point where the in-place variant adds its read to the residual and the deferred one does not.
"""
import numpy as np
import pytest
import torch

from so.data import bank_from_world
from so.llm_adapter import AdapterConfig, KnowledgeAdapterLM
from so.train import make_centre
from so.world import World

transformers = pytest.importorskip("transformers")

N_ENT, UNK = 20, 5
N_LAYER = 4


@pytest.fixture(autouse=True)
def _single_threaded():
    """Every assertion here is bit-exact equality between two SEPARATELY allocated models.

    Repeated forwards of one module are bit-identical at any thread count (checked), but two models
    allocated independently can take different vectorised kernel paths under intra-op parallelism, so
    the same arithmetic reduces in a different order and the last bits differ. That is a property of
    the CPU kernels, not of the code path under test, and it made this file fail under load. The
    tolerance is not relaxed: the comparisons stay exact and the thread count is pinned instead.
    """
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(n)


def _lm():
    cfg = transformers.GPT2Config(vocab_size=64, n_positions=16, n_embd=32, n_layer=N_LAYER, n_head=2)
    torch.manual_seed(0)
    return transformers.GPT2LMHeadModel(cfg).eval()


def _adapter(**kw):
    cfg = AdapterConfig(read_layers=(1, 2), d_key=16, marker_dim=16, **kw)
    torch.manual_seed(1)
    return KnowledgeAdapterLM(_lm(), cfg, list(range(10, 10 + N_ENT)), UNK).eval()


def _bank(seed=0):
    rng = np.random.default_rng(seed)
    world = World.sample(rng, N_ENT, 4, 30, 2)
    return bank_from_world(rng, world, make_centre(seed, 16), 0.0, 0.0, 0.1).tensors()


def _prompt(B=3, T=7):
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(0, 64, (B, T), generator=g)
    am = torch.ones(B, T, dtype=torch.long)
    return ids, am, torch.full((B,), T - 1, dtype=torch.long)


def _cache_and_hidden(m, bank, ids, am, last):
    """Run the frozen core through the adapter and collect what it persists: hidden states and K/V."""
    out = {}
    m._ctx = m.make_ctx(bank, last) if bank is not None else None
    with torch.no_grad():
        o = m.lm(input_ids=ids, attention_mask=am, output_hidden_states=True, use_cache=True)
    m._ctx = None
    out["hidden"] = [h.clone() for h in o.hidden_states]
    pkv = o.past_key_values
    kv = []
    for i in range(N_LAYER):
        if hasattr(pkv, "layers"):
            kv.append((pkv.layers[i].keys.clone(), pkv.layers[i].values.clone()))
        elif hasattr(pkv, "key_cache"):
            kv.append((pkv.key_cache[i].clone(), pkv.value_cache[i].clone()))
        else:
            kv.append((pkv[i][0].clone(), pkv[i][1].clone()))
    out["kv"] = kv
    out["logits"] = o.logits.clone()
    return out


def test_write_layer_none_is_the_recorded_configuration():
    a = _adapter()
    b = _adapter(write_layer=None)
    b.load_state_dict(a.state_dict())      # identical weights by construction, not by seeding
    assert a.cfg.write_layer is None and b.cfg.write_layer is None
    bank = _bank(); ids, am, last = _prompt()
    with torch.no_grad():
        ca, fa, ra, ha = a(bank, ids, am, last)
        cb, fb, rb, hb = b(bank, ids, am, last)
    assert torch.equal(fa, fb) and torch.equal(ra, rb) and torch.equal(ha, hb)


def test_write_layer_must_not_precede_the_last_read_layer():
    with pytest.raises(ValueError):
        _adapter(write_layer=1)
    with pytest.raises(ValueError):
        _adapter(write_layer=N_LAYER)


@pytest.mark.parametrize("n_deref", [0, 1])
@pytest.mark.parametrize("fallback", ["unknown", "prior"])
def test_final_block_write_leaves_every_persisted_tensor_bit_identical_to_no_memory(n_deref, fallback):
    m = _adapter(write_layer=N_LAYER - 1, n_deref=n_deref, fallback=fallback, use_links=(n_deref > 0))
    bank = _bank(); ids, am, last = _prompt()
    with torch.no_grad():
        # make the read material: a learned adapter would do this; here we just make the gain large
        m.inject_gain.fill_(3.0)
    mem = _cache_and_hidden(m, bank, ids, am, last)
    base = _cache_and_hidden(m, None, ids, am, last)
    # hidden_states[i] is the input of block i (0 = embeddings); hidden_states[N_LAYER] is the output of the
    # last block AFTER hooks and after ln_f in HF's GPT-2, so it is the only one allowed to differ.
    for i in range(N_LAYER):
        assert torch.equal(mem["hidden"][i], base["hidden"][i]), f"block input {i} moved"
    for i in range(N_LAYER):
        assert torch.equal(mem["kv"][i][0], base["kv"][i][0]), f"K of block {i} moved"
        assert torch.equal(mem["kv"][i][1], base["kv"][i][1]), f"V of block {i} moved"
    # the memory is material: the logits at the last position moved
    B = ids.shape[0]
    ar = torch.arange(B)
    assert not torch.equal(mem["logits"][ar, last], base["logits"][ar, last])
    # and only at the last position: every other position's logits are the frozen model's
    for t in range(ids.shape[1] - 1):
        assert torch.equal(mem["logits"][:, t], base["logits"][:, t])


def test_update_reaches_the_answer_without_touching_the_cache():
    m = _adapter(write_layer=N_LAYER - 1)
    with torch.no_grad():
        m.inject_gain.fill_(3.0)
    bank = _bank(); ids, am, last = _prompt()
    before = _cache_and_hidden(m, bank, ids, am, last)
    updated = {k: v.clone() for k, v in bank.items()}
    updated["obj"] = (updated["obj"] + 1) % N_ENT          # UPDATE every pod's payload
    after = _cache_and_hidden(m, updated, ids, am, last)
    for i in range(N_LAYER):
        assert torch.equal(before["kv"][i][0], after["kv"][i][0])
        assert torch.equal(before["kv"][i][1], after["kv"][i][1])
    B = ids.shape[0]
    ar = torch.arange(B)
    assert not torch.equal(before["logits"][ar, last], after["logits"][ar, last])


def test_deferred_and_in_place_variants_address_identically():
    """Same weights, two write placements: the routing (the addressing decision) is bit-identical."""
    a = _adapter()
    d = _adapter(write_layer=N_LAYER - 1)
    d.load_state_dict(a.state_dict())
    bank = _bank(); ids, am, last = _prompt()
    with torch.no_grad():
        _, fa, ra, _ = a(bank, ids, am, last)
        _, fd, rd, _ = d(bank, ids, am, last)
    # the FIRST read layer sees the same residual in both variants, so its routing is bit-identical;
    # the second read layer sees a residual the in-place variant has already written into, so it may differ
    assert torch.equal(ra[:, 0], rd[:, 0])
    # and the two placements are different computations, not the same one relabelled
    assert not torch.equal(fa, fd)


def test_two_deferred_placements_see_identical_residuals_at_every_read():
    """The confound control for E-000084 arm D.

    Arm C writes after the last block, arm D after the second read's own block. Neither writes in
    place at the first read layer, so both read layers see exactly the residual the frozen model
    produces: the routing — the addressing decision at both slots — must be bit-identical between the
    two placements. That is what lets arm D vary only the depth of processing after the write, while
    arm A (in-place) differs at the second slot because its first write has already moved the residual.
    """
    c = _adapter(write_layer=N_LAYER - 1)
    d = _adapter(write_layer=N_LAYER - 2)
    a = _adapter()
    d.load_state_dict(c.state_dict())
    a.load_state_dict(c.state_dict())
    bank = _bank(); ids, am, last = _prompt()
    with torch.no_grad():
        _, fc, rc, _ = c(bank, ids, am, last)
        _, fd, rd, _ = d(bank, ids, am, last)
        _, _, ra, _ = a(bank, ids, am, last)
    assert torch.equal(rc, rd), "the two deferred placements must address identically at every slot"
    assert torch.equal(rc[:, 0], ra[:, 0]), "the first read is upstream of every write"
    assert not torch.equal(rc[:, 1], ra[:, 1]), "the in-place arm's second read sees its own first write"
    assert not torch.equal(fc, fd), "different write depths are different computations"


def _ref_adapter():
    cfg = AdapterConfig(read_layers=(1, 2), d_key=16, marker_dim=16, status_gated=True,
                        reference_carrier=True, write_layer=N_LAYER - 1)
    torch.manual_seed(1)
    return KnowledgeAdapterLM(_lm(), cfg, list(range(10, 10 + N_ENT)), UNK).eval()


def _ref_bank(seed=0):
    b = _bank(seed)
    b["resolved_idx"] = torch.arange(b["obj"].shape[0])
    return b


def test_reference_carrier_needs_a_write_layer():
    with pytest.raises(ValueError):
        _adapter(reference_carrier=True)


def test_handles_are_knowledge_free_and_untrained():
    m = _ref_adapter()
    assert "handle_basis" in dict(m.named_buffers()), "the basis must be a buffer, not a trained parameter"
    assert not any(n.endswith("handle_basis") for n, _ in m.named_parameters())
    b1, b2 = _ref_bank(0), _ref_bank(1)
    assert not torch.equal(b1["obj"], b2["obj"])
    before = m.handle_basis.clone()
    with torch.no_grad():
        ids, am, last = _prompt()
        m(b1, ids, am, last)
        m(b2, ids, am, last)
    assert torch.equal(before, m.handle_basis), "reading banks must not move the carrier basis"


def test_a_handle_is_a_function_of_the_identity_alone():
    """Distinct identities give distinct handles; the same identity gives the same handle everywhere."""
    m = _ref_adapter()
    ids = torch.tensor([0, 1, 2, 7, 31, 1_000_003, -1])
    h = m.handles_for(ids)
    assert h.shape == (ids.shape[0], m.d)
    # well separated, not merely unequal: the closest pair is a large fraction of a handle's own norm
    norm = float(h.norm(dim=-1).mean())
    closest = min(float((h[i] - h[j]).norm()) for i in range(ids.shape[0]) for j in range(i + 1, ids.shape[0]))
    assert closest > 0.1 * norm, f"identities crowd together: closest {closest:.4f} against norm {norm:.4f}"
    # order- and position-independent. Equality here is numerical, not bit-exact: the handle is a matmul,
    # and a different batch shape reduces the same sum in a different order, which is float32 rounding
    # (measured at 7e-09) and not a dependence on position.
    assert torch.allclose(m.handles_for(ids.flip(0)).flip(0), h, atol=1e-7, rtol=0)
    assert torch.allclose(m.handles_for(ids[2:3]), h[2:3], atol=1e-7, rtol=0)


def test_reordering_the_store_leaves_every_persisted_tensor_bit_identical():
    """Keying by identity, not by position, is what makes this true.

    A store that reorders, compacts or grows its rows changes every row position. If a handle were the
    row's position, a reference already written into a cache would silently come to name a different
    pod — a different pod's value, not a rounding difference. Keyed by identity, the same identities
    addressed in a different order produce the same persisted state.

    This one is NUMERICAL, not bit-exact, and the difference from the lifecycle rows is the point.
    Permuting rows reorders the softmax denominator and the handle mixture, so float32 reduces the
    same sums in a different order: measured at 7.5e-08 on the cache here. The lifecycle invariance
    above is exactly 0.0 because there the tensors are literally unchanged.
    """
    m = _ref_adapter()
    with torch.no_grad():
        m.inject_gain.fill_(3.0)
    bank = _ref_bank(); ids, am, last = _prompt()
    before = _cache_and_hidden(m, bank, ids, am, last)

    n = bank["obj"].shape[0]
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(n, generator=g)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(n)
    permuted = {}
    for k, v in bank.items():
        permuted[k] = v[perm].clone() if torch.is_tensor(v) and v.shape[:1] == (n,) else v
    # resolved_idx holds POSITIONS, so it must be renumbered into the new layout, not just reordered
    permuted["resolved_idx"] = inverse[bank["resolved_idx"][perm]]
    after = _cache_and_hidden(m, permuted, ids, am, last)

    TOL = 1e-6      # two orders of magnitude above the measured 7.5e-08, still far below any semantic change
    kv_moved = max(float((before["kv"][i][j] - after["kv"][i][j]).abs().max())
                   for i in range(N_LAYER) for j in (0, 1))
    assert kv_moved < TOL, f"reordering moved the cache by {kv_moved:.3e}, which is a rebinding, not rounding"
    ar = torch.arange(ids.shape[0])
    answer_moved = float((before["logits"][ar, last] - after["logits"][ar, last]).abs().max())
    assert answer_moved < TOL, f"reordering changed the answer by {answer_moved:.3e}"
    # and the contrast that makes this meaningful: a payload UPDATE moves the cache by EXACTLY zero
    # while moving the answer, so the two invariances are different in kind and are reported as such.
    updated = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in bank.items()}
    updated["obj"] = (updated["obj"] + 1) % N_ENT
    upd = _cache_and_hidden(m, updated, ids, am, last)
    for i in range(N_LAYER):
        assert torch.equal(before["kv"][i][0], upd["kv"][i][0])
        assert torch.equal(before["kv"][i][1], upd["kv"][i][1])
    assert float((before["logits"][ar, last] - upd["logits"][ar, last]).abs().max()) > TOL


def test_a_handle_for_a_removed_identity_does_not_name_a_surviving_row():
    """The ABA case that position keying would have got wrong.

    Remove the identity a handle names. Under position keying the vector would still index a row —
    whichever slid into that slot — so a cached reference would bind to an unrelated pod's value.
    Under identity keying the handle is simply not among the identities present, which is what lets a
    stale reference fail rather than silently resolve to something else.
    """
    m = _ref_adapter()
    bank = _ref_bank()
    ids_present = bank["handle_id"]
    gone = int(ids_present[0])
    survivors = ids_present[1:]
    stale = m.handles_for(torch.tensor([gone]))
    table = m.handles_for(survivors)
    # the stale handle is not reproduced by any surviving identity
    assert not any(torch.allclose(stale[0], table[i]) for i in range(table.shape[0]))
    # and the identities that survive keep exactly the handles they had
    kept = m.handles_for(ids_present)[1:]
    assert torch.equal(kept, table)


@pytest.mark.parametrize("op,mutate", [
    ("update_payload", lambda b: b.__setitem__("obj", (b["obj"] + 1) % N_ENT)),
    ("relink", lambda b: b.__setitem__("resolved_idx", torch.roll(b["resolved_idx"], 3))),
    ("shred_markers", lambda b: b.__setitem__("marker", torch.zeros_like(b["marker"]))),
])
def test_lifecycle_operations_leave_every_persisted_tensor_bit_identical(op, mutate):
    """The property the reference carrier exists for.

    A handle names a row; it never carries the row's payload, the row it resolves to, or its marker.
    So a payload UPDATE, an alias RELINK and a SHRED all leave the persisted K/V byte-for-byte
    unchanged — no cache entry has to be invalidated, recomputed or lineage-tracked — while the answer
    the model gives still changes, because the value is bound after the last cache-writing block.
    """
    m = _ref_adapter()
    with torch.no_grad():
        m.inject_gain.fill_(3.0)
        m.bind_gain.fill_(3.0)
    bank = _ref_bank(); ids, am, last = _prompt()
    before = _cache_and_hidden(m, bank, ids, am, last)
    after_bank = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in bank.items()}
    mutate(after_bank)
    after = _cache_and_hidden(m, after_bank, ids, am, last)
    for i in range(N_LAYER):
        assert torch.equal(before["kv"][i][0], after["kv"][i][0]), f"{op} moved K of block {i}"
        assert torch.equal(before["kv"][i][1], after["kv"][i][1]), f"{op} moved V of block {i}"
    ar = torch.arange(ids.shape[0])
    assert not torch.equal(before["logits"][ar, last], after["logits"][ar, last]), f"{op} changed nothing"


def test_the_reference_carrier_actually_participates():
    """What separates arm E from arm C: the handle rides through the frozen blocks.

    Arm C's persisted state is identical to the no-memory forward — nothing of the memory takes part
    in the frozen computation. Here the handle is injected in place at the read layers, so the K/V the
    model persists genuinely differs from the no-memory forward, while still being independent of every
    mutable field (the test above).
    """
    m = _ref_adapter()
    with torch.no_grad():
        m.inject_gain.fill_(3.0)
    bank = _ref_bank(); ids, am, last = _prompt()
    mem = _cache_and_hidden(m, bank, ids, am, last)
    base = _cache_and_hidden(m, None, ids, am, last)
    moved = max(float((a[0] - b[0]).abs().max()) for a, b in zip(mem["kv"], base["kv"]))
    assert moved > 0.0, "the handle must participate, otherwise this is arm C with extra steps"

    c = _adapter(write_layer=N_LAYER - 1)
    with torch.no_grad():
        c.inject_gain.fill_(3.0)
    c_mem = _cache_and_hidden(c, _bank(), ids, am, last)
    c_base = _cache_and_hidden(c, None, ids, am, last)
    assert max(float((a[0] - b[0]).abs().max()) for a, b in zip(c_mem["kv"], c_base["kv"])) == 0.0


def test_no_memory_forward_is_untouched_by_the_write_hook():
    m = _adapter(write_layer=N_LAYER - 1)
    lm = m.lm
    ids, am, last = _prompt()
    with torch.no_grad():
        _, full, routing, _ = m(None, ids, am, last)
        ref = lm(input_ids=ids, attention_mask=am).logits[torch.arange(ids.shape[0]), last]
    assert routing is None
    assert torch.equal(full, ref)
