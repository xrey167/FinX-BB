"""The premise every certificate in ``so/audit.py`` rests on, made explicit and pinned.

WHERE THIS COMES FROM. A parallel research branch (``research/cavi-continuation-audit``) audited this
architecture and recorded two things about it: (1) a bank exported before a SHRED can be replayed
afterwards and still answers with the deleted object -- a *stale snapshot* -- because the exported
tensors carry no live generation the reader could check; (2) an adapter that revalidates authority
independently at each read layer can consume two different generations inside one forward pass -- a
*torn read*.

Both are statements about a premise this repository states in prose (``so/audit.py``
``certify_encoding``: "the forward is therefore a deterministic function of (encoding, query)") and
has never tested. These tests state it as behaviour:

  A. THE FORWARD IS A PURE FUNCTION OF THE MATERIALISED BANK. Mutating the store after materialisation
     -- SHRED, REVOKE, BLANK, EVICT -- does not move a single bit of the forward that consumes the
     already-materialised tensors. So every certificate in ``so/audit.py`` is a statement about ONE
     export and never about the store's later state, and a retained export keeps answering. That is
     what a snapshot is; it is recorded here so no certificate can be read as more than it is.

  B. THE DELETION IS VISIBLE ONLY THROUGH RE-MATERIALISATION. The same store, re-exported after the
     mutation, moves the forward. So the certificate's window is exactly one ``bank_from_store``.

  C. THIS SUBSTRATE IS FORWARD-ATOMIC BY CONSTRUCTION, so the torn read cannot arise here:
     ``KnowledgeAdapterLM.forward`` calls ``encode_bank`` once (``so/llm_adapter.py:333``) and stashes
     the result in ``self._ctx``; both read-layer hooks consume only that stash, never the bank. A
     mutation cannot linearise between the read sites of one forward because there is only one read of
     the store per forward. The hazard the parallel branch measured belongs to a design that consults
     live authority per read site -- which is what its own remedy then had to undo.

None of this is a claim. It is the premise of the certificate ladder, pinned so that it fails loudly
if a future change makes the reader consult the store more than once per forward.
"""

import numpy as np
import pytest
import torch

from so.data import bank_from_store, bank_from_world
from so.model import ModelConfig, MutableKnowledgeTransformer
from so.mvcc import MVCCStore
from so.train import make_centre
from so.world import World

N_ENT = 32


def _model():
    cfg = ModelConfig(n_entities=N_ENT, n_relations=4, d_model=32, marker_dim=16, n_core_layers=1, n_heads=2)
    torch.manual_seed(0)
    return MutableKnowledgeTransformer(cfg).eval()


def _store_with_a_pod(seed: int = 0):
    """A fact plus two aliases pointing at it -- a pod, in this repository's sense.

    Returns the store, the fact's kid and the first alias's kid.
    """
    st = MVCCStore(marker_dim=16, seed=seed, marker_centre=make_centre(seed, 16))
    kid = st.write(3, 0, 7, provenance="pod")
    alias = st.link(11, 0, kid, provenance="alias")
    st.link(12, 0, kid, provenance="alias")
    for s in range(20, 28):                       # bystanders, so the bank is not one pod wide
        st.write(s, 1, (s * 5) % N_ENT, provenance="filler")
    return st, kid, alias


def _queries(bank_tensors, n: int = 6):
    """The same questions in every run: E-000030's rule, so the audit never perturbs its own input.

    Built exactly as ``so/tests/test_audit.py`` builds them, from the bank's own subjects and objects.
    """
    mode = torch.tensor([0, 0, 0, 1, 1, 1][:n])
    start = torch.stack([bank_tensors["subject"][:3], bank_tensors["obj"][:3]]).reshape(-1)[:n]
    rels = bank_tensors["relation"][:n].reshape(n, 1).repeat(1, 3)
    hop_valid = torch.tensor([[True, False, False]] * n)
    return mode, start, rels, hop_valid


def _cand(model, tensors, q):
    """One forward through the synthetic reader on the given (already materialised) tensors."""
    mode, start, rels, hop_valid = q
    with torch.no_grad():
        out = model(tensors, mode, start, rels, hop_valid)
    return out[0] if isinstance(out, tuple) else out


def test_a_mutation_after_materialisation_does_not_move_the_forward():
    """A. The certificate's subject is one export, and a retained export keeps answering."""
    model = _model()
    for mutate in ("shred", "revoke", "blank", "evict"):
        st, kid, alias = _store_with_a_pod()
        stale = bank_from_store(st)
        qs = _queries(stale.tensors())
        before = _cand(model, stale.tensors(), qs).clone()
        if mutate == "shred":
            st.shred(kid)
        elif mutate == "revoke":
            st.revoke(kid)
        elif mutate == "blank":
            st.blank(alias)
        else:
            st.evict(kid)
        after = _cand(model, stale.tensors(), qs)
        assert torch.equal(before, after), f"{mutate} moved a forward over the already-materialised bank"


def test_the_deletion_is_visible_only_through_re_materialisation():
    """B. The window is exactly one ``bank_from_store``."""
    model = _model()
    st, kid, _ = _store_with_a_pod()
    stale = bank_from_store(st)
    qs = _queries(stale.tensors())
    before = _cand(model, stale.tensors(), qs).clone()
    st.shred(kid)
    replayed = _cand(model, stale.tensors(), qs)
    fresh = _cand(model, bank_from_store(st).tensors(), qs)
    assert torch.equal(before, replayed)              # the retained export is unchanged, bit for bit
    assert not torch.equal(before, fresh)             # the store's own export moved


def test_the_adapter_reads_the_store_once_per_forward():
    """C. Forward-atomicity by construction: one ``encode_bank`` per forward, whatever the read layers.

    Counted on the synthetic reader, which has the same shape as ``KnowledgeAdapterLM`` (a single
    ``encode_bank`` whose outputs every read site consumes). A design that re-encoded per read site
    would count more than one and could observe two generations inside one forward.
    """
    model = _model()
    st, _, _ = _store_with_a_pod()
    tensors = bank_from_store(st).tensors()
    qs = _queries(tensors)
    calls = {"n": 0}
    original = model.encode_bank

    def counted(bank, *a, **kw):
        calls["n"] += 1
        return original(bank, *a, **kw)

    model.encode_bank = counted
    try:
        _cand(model, tensors, qs)
    finally:
        model.encode_bank = original
    assert calls["n"] == 1, f"the reader consulted the bank {calls['n']} times in one forward"


def test_the_two_read_site_adapter_also_reads_the_store_once_per_forward():
    """C, on the shape that matters: two read layers, one materialisation.

    ``KnowledgeAdapterLM`` injects at every layer in ``read_layers``. The torn read a per-read-site
    revalidation design admits -- two authority generations inside one inference -- is impossible here
    because both hooks consume the single ``self._ctx`` stashed by ``forward`` (so/llm_adapter.py:333),
    and the bank never re-enters. Counted on a tiny randomly initialised GPT-2, no download.
    """
    transformers = pytest.importorskip("transformers")
    from so.llm_adapter import AdapterConfig, KnowledgeAdapterLM

    lm_cfg = transformers.GPT2Config(vocab_size=64, n_positions=16, n_embd=32, n_layer=2, n_head=2)
    torch.manual_seed(0)
    lm = transformers.GPT2LMHeadModel(lm_cfg).eval()
    torch.manual_seed(1)
    model = KnowledgeAdapterLM(lm, AdapterConfig(read_layers=(0, 1), d_key=16, marker_dim=16),
                               list(range(10, 10 + N_ENT)), 5).eval()
    rng = np.random.default_rng(0)
    world = World.sample(rng, N_ENT, 4, 30, 2)
    tensors = bank_from_world(rng, world, make_centre(0, 16), 0.3, 0.2, 0.1).tensors()
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(0, 64, (4, 6), generator=g)
    am = torch.ones(4, 6, dtype=torch.long)
    last = torch.full((4,), 5, dtype=torch.long)

    calls = {"n": 0}
    original = model.encode_bank

    def counted(bank, *a, **kw):
        calls["n"] += 1
        return original(bank, *a, **kw)

    model.encode_bank = counted
    try:
        with torch.no_grad():
            model(tensors, ids, am, last)
    finally:
        model.encode_bank = original
    assert len(model.cfg.read_layers) == 2
    assert calls["n"] == 1, f"two read sites materialised the bank {calls['n']} times in one forward"
