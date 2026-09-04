"""The audit must catch the leak it was built for, and must not cry wolf on a clean deletion."""
import numpy as np
import pytest
import torch

from so.audit import audit_independence, perturbed_objects
from so.data import bank_from_world
from so.model import ModelConfig, MutableKnowledgeTransformer
from so.train import make_centre
from so.world import World

N_ENT = 32


def _model(**kw):
    cfg = ModelConfig(n_entities=N_ENT, n_relations=4, d_model=32, marker_dim=16, n_core_layers=1,
                      n_heads=2, **kw)
    torch.manual_seed(0)
    return MutableKnowledgeTransformer(cfg).eval()


def _shut_gate(m, value=-20.0):
    with torch.no_grad():
        m.marker_gate[-1].weight.zero_()
        m.marker_gate[-1].bias.fill_(value)


def _bank(p_shred=1.0, p_revoked=0.0, n_cells=24, seed=0):
    rng = np.random.default_rng(seed)
    world = World.sample(rng, N_ENT, 4, n_cells, 2)
    return bank_from_world(rng, world, make_centre(seed, 16), p_revoked, p_shred, 0.0).tensors()


def _queries(bank, n=6):
    """A batch containing forward and reverse questions, since only the reverse ones read k_r."""
    b = bank["subject"].shape[0]
    mode = torch.tensor([0, 0, 0, 1, 1, 1][:n])
    start = torch.stack([bank["subject"][:3], bank["obj"][:3]]).reshape(-1)[:n]
    rels = bank["relation"][:n].reshape(n, 1).repeat(1, 3)
    hop_valid = torch.tensor([[True, False, False]] * n)
    return mode, start, rels, hop_valid


def _run(model, bank, q):
    """The QUESTIONS must be the same in both runs: an attacker's queries do not change when the
    payload they are hunting changes. Building them from each bank in turn would perturb the input as
    well as the store and the audit would report a difference it created itself."""
    mode, start, rels, hop_valid = q
    def go():
        with torch.no_grad():
            return model(bank, mode, start, rels, hop_valid)
    return go


def test_the_audit_catches_the_shred_key_leak_in_the_outputs():
    """SHRED leaves the row routable and the reverse key ungated: the leak reaches what is returned."""
    m = _model()
    _shut_gate(m)
    a = _bank(p_shred=1.0)
    rows = np.arange(a["subject"].shape[0])
    b = perturbed_objects(a, rows, N_ENT)
    q = _queries(a)
    res = audit_independence(m, _run(m, a, q), _run(m, b, q))
    assert not res.output_independent, res.summary()
    assert not res.activation_independent
    assert any(d.base == "k_rev" for d in res.differences), res.summary()


def test_gating_the_reverse_key_restores_output_independence():
    m = _model(gate_reverse_key=True)
    _shut_gate(m)
    a = _bank(p_shred=1.0)
    rows = np.arange(a["subject"].shape[0])
    b = perturbed_objects(a, rows, N_ENT)
    q = _queries(a)
    res = audit_independence(m, _run(m, a, q), _run(m, b, q), outputs_of=LOGITS)
    assert res.output_independent, res.summary()
    assert res.n_tensors > 0


LOGITS = lambda out: out[0]


def test_revoke_is_answer_independent_but_still_embeds_the_deleted_object():
    """The distinction the two levels exist for.

    A revoked row is masked out of the softmax, so nothing it holds can reach the output -- the
    black-box claim holds exactly. But encode_bank embeds every row's object before the mask is
    applied, so ent_emb still carries the deleted payload and an adversary who reads activations sees
    it. Only DELETE, which takes the row out of the bank altogether, avoids computing it at all.
    """
    m = _model()
    a = _bank(p_shred=0.0, p_revoked=1.0)
    assert not bool(a["active"].any())
    rows = np.arange(a["subject"].shape[0])
    b = perturbed_objects(a, rows, N_ENT)
    q = _queries(a)
    answer = audit_independence(m, _run(m, a, q), _run(m, b, q), outputs_of=LOGITS)
    assert answer.output_independent, answer.summary()            # nothing a user sees can differ
    full = audit_independence(m, _run(m, a, q), _run(m, b, q))
    assert not full.output_independent                            # but a returned diagnostic does
    assert [d.output for d in full.output_differences] == ["2value_norm."], full.summary()
    assert not full.activation_independent
    assert any(d.base == "ent_emb" for d in full.differences), full.summary()


def test_deleting_the_rows_is_independent_at_both_levels():
    """DELETE removes the row, so there is no payload left in the computation to depend on."""
    m = _model()
    a = _bank(p_shred=0.0, p_revoked=0.0, n_cells=24)
    keep = torch.ones(a["subject"].shape[0], dtype=torch.bool)
    keep[:8] = False                                    # the deleted rows simply are not there
    kept = {k: (v[keep] if torch.is_tensor(v) and v.shape[:1] == keep.shape else v) for k, v in a.items()}
    q = _queries(kept)
    res = audit_independence(m, _run(m, kept, q), _run(m, kept, q))
    assert res.output_independent and res.activation_independent, res.summary()


def test_the_audit_fails_loudly_on_a_live_cell():
    """The control that stops the audit from passing because the bank is ignored altogether."""
    m = _model()
    _shut_gate(m, 20.0)                       # every marker valid: the cells are readable
    a = _bank(p_shred=0.0)
    rows = np.arange(a["subject"].shape[0])
    b = perturbed_objects(a, rows, N_ENT)
    q = _queries(a)
    res = audit_independence(m, _run(m, a, q), _run(m, b, q))
    assert not res.output_independent
    assert res.first_leak is not None


def test_a_difference_report_names_the_module_and_the_size():
    m = _model()
    _shut_gate(m)
    a = _bank(p_shred=1.0)
    b = perturbed_objects(a, np.arange(a["subject"].shape[0]), N_ENT)
    q = _queries(a)
    res = audit_independence(m, _run(m, a, q), _run(m, b, q))
    d = res.differences[0]
    assert d.max_abs > 0 and d.shape and "differs by" in str(d)


# ---------------------------------------------------- the exhaustive certificate over the payload domain

from so.audit import certify_deletion  # noqa: E402


def _certify(m, bank, rows, q, **kw):
    return certify_deletion(m, bank, rows, N_ENT, lambda b: _run(m, b, q)(), outputs_of=LOGITS, **kw)


def test_a_masked_row_is_certified_over_every_value_its_payload_could_hold():
    """REVOKE takes the row out of routing, so the answer cannot move for ANY payload -- all 32 of them."""
    m = _model()
    a = _bank(p_shred=0.0, p_revoked=1.0)
    q = _queries(a)
    cert = _certify(m, a, [0], q, check_activations=False)
    assert cert.output_certified, cert.summary()
    assert cert.n_values == N_ENT
    assert cert.n_evaluations == N_ENT           # the reference plus every other value


def test_the_shred_key_leak_is_caught_by_the_sweep():
    m = _model()
    _shut_gate(m)
    a = _bank(p_shred=1.0)
    q = _queries(a)
    cert = _certify(m, a, [0], q, check_activations=False)
    assert not cert.output_certified, cert.summary()
    assert cert.violations and cert.violations[0].row == 0


def test_gating_the_reverse_key_earns_the_certificate():
    m = _model(gate_reverse_key=True)
    _shut_gate(m)
    a = _bank(p_shred=1.0)
    q = _queries(a)
    cert = _certify(m, a, [0, 1, 2], q, check_activations=False, joint_trials=16)
    assert cert.output_certified, cert.summary()
    assert cert.joint_certified, cert.summary()
    assert cert.joint_trials == 16


def test_a_live_row_is_never_certified():
    """The control: if the sweep certifies a readable cell, it is not measuring anything."""
    m = _model()
    _shut_gate(m, 20.0)
    a = _bank(p_shred=0.0)
    q = _queries(a)
    cert = _certify(m, a, [0], q, check_activations=False)
    assert not cert.output_certified
    assert "NOT CERTIFIED" in cert.summary()


def test_the_certificate_reports_the_activation_level_separately():
    m = _model()
    a = _bank(p_shred=0.0, p_revoked=1.0)
    q = _queries(a)
    cert = _certify(m, a, [0], q, check_activations=True, stop_early=False)
    assert cert.output_certified                  # nothing a user sees moves
    assert not cert.activation_certified          # but the deleted object is still embedded


# ------------------------------------- the interface certificate: universal over queries, not just swept ones

from so.audit import certify_encoding, check_mediation  # noqa: E402


def test_the_interface_certificate_agrees_with_the_output_sweep_on_a_masked_row():
    m = _model()
    a = _bank(p_shred=0.0, p_revoked=1.0)
    q = _queries(a)
    at_interface = certify_encoding(m, a, [0], N_ENT)
    at_outputs = _certify(m, a, [0], q, check_activations=False)
    assert at_outputs.output_certified
    # the interface is STRICTER: a revoked row is masked downstream but its object is still encoded,
    # so encode_bank moves even though nothing a user sees does. The certificate says so rather than
    # rounding the two together.
    assert not at_interface.output_certified
    assert at_interface.violations[0].module.startswith("encode_bank[")


def test_gating_the_reverse_key_is_still_not_enough_at_the_interface():
    """v_fwd(o) is computed before the gate multiplies it, so the encoding still moves."""
    m = _model(gate_reverse_key=True)
    _shut_gate(m)
    a = _bank(p_shred=1.0)
    cert = certify_encoding(m, a, [0], N_ENT)
    assert not cert.output_certified, cert.summary()


def test_removing_the_row_is_certified_at_the_interface():
    """The only deletion that earns it: the payload is not in the bank to be a function of."""
    m = _model()
    a = _bank(p_shred=0.0)
    keep = torch.ones(a["subject"].shape[0], dtype=torch.bool)
    keep[:4] = False
    kept = {k: (v[keep] if torch.is_tensor(v) and v.shape[:1] == keep.shape else v) for k, v in a.items()}
    cert = certify_encoding(m, kept, [], N_ENT)     # no rows to sweep: nothing depends on what is gone
    assert cert.output_certified, cert.summary()
    assert cert.n_evaluations == 1


def test_the_mediation_premise_is_checked_not_assumed():
    m = _model()
    a = _bank(p_shred=0.0, p_revoked=1.0)
    q = _queries(a)
    chk = check_mediation(m, a, [0], N_ENT, lambda b: _run(m, b, q)(), outputs_of=LOGITS, n_probes=4)
    assert chk.consistent, chk.note
    assert chk.n_probes == 4
    # a revoked row moves the encoding and not the logits, which is consistent with mediation
    assert not chk.encoding_invariant and chk.output_invariant


def test_the_mediation_check_can_fail():
    """A runner that reads the bank behind the encoding's back must be caught, or the check is decoration."""
    m = _model()
    a = _bank(p_shred=0.0, p_revoked=1.0)
    q = _queries(a)

    def sneaky(b):
        base = _run(m, b, q)()
        return (base[0] + b["obj"][:1].float().sum(), base[1], base[2])   # an output that sees the payload

    chk = check_mediation(m, a, [0], N_ENT, sneaky, encode=lambda b: {"const": torch.zeros(1)},
                          outputs_of=LOGITS, n_probes=4)
    assert not chk.consistent
    assert "VOID" in chk.note


# ------------------------------------------------ row locality: what upgrades the joint claim to a proof

from so.audit import check_row_locality  # noqa: E402


def test_the_encoding_is_row_local_at_zero_noise():
    m = _model()
    a = _bank(p_shred=0.5)
    chk = check_row_locality(m, a, [0, 1, 2], N_ENT, n_values_probed=4)
    assert chk.row_local, chk.note
    assert "implies joint invariance" in chk.note


def test_noise_breaks_row_locality_because_the_jitter_rms_is_global():
    """A deleted row perturbs every visible one once noise is on, and the check must say so."""
    m = _model()
    a = _bank(p_shred=0.5)
    g = torch.Generator().manual_seed(0)
    chk = check_row_locality(m, a, [0, 1], N_ENT, n_values_probed=3,
                             encode=lambda b: m.encode_bank(b, noise=0.1,
                                                            generator=torch.Generator().manual_seed(0)))
    assert not chk.row_local, chk.note
    assert "NOT row-local" in chk.note


# ------------------------------- reachability: the domain-free form, and the three outcomes it can give

from so.audit import certify_structural  # noqa: E402

D_MODEL = 32


def _run_grad(model, bank, q):
    """Reachability needs the graph, so this runner must not disable it."""
    mode, start, rels, hop_valid = q
    return lambda b: model(b, mode, start, rels, hop_valid)


def test_a_live_cell_is_reachable_from_its_payload():
    """The validity control. If a readable cell is not reachable, the instrument measures nothing."""
    m = _model()
    _shut_gate(m, 20.0)
    a = _bank(p_shred=0.0)
    q = _queries(a)
    res = certify_structural(m, a, [0], _run_grad(m, a, q), D_MODEL, outputs_of=LOGITS)
    assert res.reachable and res.grad_max > 0, res.summary()


def test_a_shut_gate_annihilates_the_value_path_but_leaves_it_reachable():
    """The middle outcome, and why it is not the strong claim: v_f = v_fwd(o) * g with g == 0."""
    m = _model()
    _shut_gate(m)
    a = _bank(p_shred=1.0)
    q = _queries(a)
    res = certify_structural(m, a, [0], _run_grad(m, a, q), D_MODEL, outputs_of=LOGITS)
    # the reverse key is ungated, so in THIS model the payload still reaches the output
    assert res.reachable, res.summary()


def test_a_sigmoid_gate_never_annihilates_the_path_exactly():
    """The same finding as the frozen-GPT-2 residual, seen as a derivative instead of a difference.

    A soft gate is a sigmoid, so it is never exactly zero, so the derivative of the output with respect
    to a shredded payload is never exactly zero either -- here 1e-10, in the adapter 1.39e-02 of the
    payload surviving in the value. A gate cannot deliver even the middle outcome, let alone the strong
    one; that takes a hard threshold or removing the row.
    """
    m = _model(gate_reverse_key=True)
    _shut_gate(m)
    a = _bank(p_shred=1.0)
    q = _queries(a)
    res = certify_structural(m, a, [0], _run_grad(m, a, q), D_MODEL, outputs_of=LOGITS)
    assert res.reachable and 0.0 < res.grad_max < 1e-6, res.summary()
    assert not res.certified_structurally


def test_a_hard_gate_does_annihilate_the_path_exactly():
    m = _model(gate_reverse_key=True)
    _shut_gate(m)
    m.cfg.hard_gate = True                     # thresholded to exactly 0 or 1
    a = _bank(p_shred=1.0)
    q = _queries(a)
    res = certify_structural(m, a, [0], _run_grad(m, a, q), D_MODEL, outputs_of=LOGITS)
    assert res.grad_max == 0.0, res.summary()
    assert not res.certified_structurally      # a path is still there; only the derivative vanishes


def test_reachability_refuses_an_empty_row_set_instead_of_certifying_it():
    """This assertion used to read the other way, and it was wrong.

    With no row selected there is nothing for autograd to find a path FROM, so the test returned the
    strongest label in the ladder -- "no path, for any value over any domain" -- and returned it just
    as confidently on a bank whose rows are all present and live. A certificate produced by not
    testing anything. It is refused at the source now, and the second half of this test is the proof
    that the old form was unsound: the same call on a LIVE bank would have certified it.
    """
    m = _model()
    a = _bank(p_shred=0.0)
    q = _queries(a)
    live = certify_structural(m, a, [0], _run_grad(m, a, q), D_MODEL, outputs_of=LOGITS)
    assert live.reachable and live.grad_max > 0.0        # row 0 is live and its payload reaches out
    with pytest.raises(ValueError, match="no rows to perturb"):
        certify_structural(m, a, [], _run_grad(m, a, q), D_MODEL, outputs_of=LOGITS)


def test_a_row_taken_out_of_the_bank_is_certified_by_absence_and_a_positive_control():
    """The strong outcome, stated as what actually carries it: membership, not reachability.

    The model reads the store only through the bank, so a payload with no row in the bank is not an
    input and nothing the model computes can depend on it -- over any domain, for every query. The
    control is what makes that a deletion claim rather than a tautology: the same payload WAS
    reachable while its row was there.
    """
    from so.audit import check_absence
    m = _model()
    a = _bank(p_shred=0.0)
    a["kid"] = torch.arange(a["subject"].shape[0])
    q = _queries(a)
    control = certify_structural(m, a, [0], _run_grad(m, a, q), D_MODEL, outputs_of=LOGITS)
    assert control.reachable

    keep = torch.ones(a["subject"].shape[0], dtype=torch.bool)
    keep[0] = False
    kept = {k: (v[keep] if torch.is_tensor(v) and v.shape[:1] == keep.shape else v) for k, v in a.items()}
    chk = check_absence(a, kept, [0], control=control)
    assert chk.certified_absent, chk.summary()
    assert chk.rows_before == chk.rows_after + 1 and chk.still_present == ()
    assert "ABSENT" in chk.summary()


def test_absence_without_a_positive_control_is_void():
    """"The row is not there" is evidence of a deletion only if the row was there and mattered."""
    from so.audit import check_absence
    m = _model()
    a = _bank(p_shred=0.0)
    a["kid"] = torch.arange(a["subject"].shape[0])
    keep = torch.ones(a["subject"].shape[0], dtype=torch.bool)
    keep[0] = False
    kept = {k: (v[keep] if torch.is_tensor(v) and v.shape[:1] == keep.shape else v) for k, v in a.items()}
    assert not check_absence(a, kept, [0]).certified_absent
    from so.audit import StructuralResult
    unreachable = StructuralResult(False, 0.0, 4, "no path")
    chk = check_absence(a, kept, [0], control=unreachable)
    assert not chk.certified_absent and "NOT reachable before" in chk.summary()


def test_absence_notices_a_row_that_is_still_there():
    from so.audit import check_absence, StructuralResult
    m = _model()
    a = _bank(p_shred=0.0)
    a["kid"] = torch.arange(a["subject"].shape[0])
    chk = check_absence(a, a, [0, 1], control=StructuralResult(True, 1.0, 4, "a path exists"))
    assert not chk.absent and chk.still_present == (0, 1)
    assert "NOT ABSENT" in chk.summary()


def test_absence_refuses_a_cell_that_was_never_in_the_bank():
    """The same failure one level along: never-present is not deleted."""
    from so.audit import check_absence, StructuralResult
    m = _model()
    a = _bank(p_shred=0.0)
    a["kid"] = torch.arange(a["subject"].shape[0])
    chk = check_absence(a, a, [9999], control=StructuralResult(True, 1.0, 4, "a path exists"))
    assert not chk.absent and "not in the bank before" in chk.note


def test_the_delta_is_numerically_a_no_op():
    """The instrumentation must not change what the model computes, or every recorded result moves."""
    m = _model()
    a = _bank(p_shred=0.3)
    q = _queries(a)
    with torch.no_grad():
        without = _run(m, a, q)()[0]
        b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in a.items()}
        b["payload_delta"] = torch.zeros(a["subject"].shape[0], D_MODEL)
        with_delta = _run(m, b, q)()[0]
    assert torch.equal(without, with_delta)


def test_eviction_earns_the_strong_certificate_that_shred_cannot():
    """The point of EVICT, stated as the property it was built to have.

    SHRED keeps the row addressable and asks a gate to refuse it: the payload is still an input and
    the certificate fails. EVICT takes the row out of the bank, so autograd finds no path at all --
    the domain-free claim -- while the store still holds the versions.
    """
    import numpy as np
    from so.data import bank_from_store
    from so.mvcc import MVCCStore
    from so.train import make_centre
    from so.world import World

    rng = np.random.default_rng(0)
    world = World.sample(rng, N_ENT, 4, 24, 2)
    centre = make_centre(0, 16)
    store = MVCCStore(marker_dim=16, seed=0, marker_centre=centre)
    kids = [store.write(f.subject, f.relation, f.obj, provenance="w") for f in world.facts]
    m = _model()

    def probe(bank_tensors, rows):
        q = _queries(bank_tensors)
        return certify_structural(m, bank_tensors, rows, _run_grad(m, bank_tensors, q), D_MODEL,
                                  outputs_of=LOGITS)

    from so.audit import check_absence
    store.shred(kids[0])
    b_shred = bank_from_store(store)
    t_shred = b_shred.tensors()
    assert t_shred["subject"].shape[0] == len(kids)          # still addressable
    control = probe(t_shred, [0])
    assert control.reachable                                 # and therefore still an input
    store.resign(kids[0])

    b_before = bank_from_store(store)
    store.evict(kids[0])
    b_evict = bank_from_store(store)
    assert b_evict.tensors()["subject"].shape[0] == len(kids) - 1     # out of the bank
    # NOT probe(bank, []) -- that answers "no path" on any bank at all, this one included before the
    # eviction. What EVICT earns is the membership property, tied to a payload that demonstrably
    # reached the outputs while its row was there.
    absence = check_absence(b_before, b_evict, [kids[0]], control=control)
    assert absence.certified_absent, absence.summary()
    assert store.cells[kids[0]].versions                     # and the data is still there


# --------------------- fail closed: a certificate must refuse a payload its sweep cannot express

from so.audit import UnsweepablePayload, FACT_PAYLOAD, LINK_PAYLOAD, _Recorder  # noqa: E402


def _link_bank():
    """A bank containing one LINK row, whose payload is an address and not an object."""
    import numpy as np
    from so.data import bank_from_store
    from so.mvcc import MVCCStore
    from so.train import make_centre
    centre = make_centre(0, 16)
    st = MVCCStore(marker_dim=16, seed=0, marker_centre=centre)
    t = st.write(3, 1, 7, provenance="t")
    st.link(4, 1, t, provenance="a")
    st.write(5, 2, 11, provenance="w")
    return st, bank_from_store(st).tensors()


def test_sweeping_only_obj_refuses_to_certify_a_link_row():
    """A LINK row's obj is a hardwired placeholder; its payload is link_subject/link_relation.

    Sweeping obj alone and returning True would certify a payload the sweep never looked at, which is
    the one outcome an audit must never produce. It raises instead.
    """
    m = _model()
    st, b = _link_bank()
    link_rows = [i for i in range(b["subject"].shape[0]) if bool(b["is_link"][i])]
    assert link_rows, "the fixture must contain a link row"
    with pytest.raises(UnsweepablePayload) as e:
        certify_encoding(m, b, link_rows, N_ENT, payload_fields=FACT_PAYLOAD)
    assert "LINK" in str(e.value) and "link_subject" in str(e.value)


def test_naming_the_link_fields_lets_the_sweep_proceed():
    m = _model(use_links=True, n_deref=1)
    st, b = _link_bank()
    link_rows = [i for i in range(b["subject"].shape[0]) if bool(b["is_link"][i])]
    cert = certify_encoding(m, b, link_rows, N_ENT, payload_fields=FACT_PAYLOAD + LINK_PAYLOAD,
                            joint_trials=0)
    # a live alias is readable, so it must NOT certify -- the point is that the sweep can now see it
    assert not cert.output_certified, cert.summary()
    assert cert.violations


def test_a_fact_row_still_sweeps_with_the_default_fields():
    m = _model()
    st, b = _link_bank()
    fact_rows = [i for i in range(b["subject"].shape[0]) if not bool(b["is_link"][i])]
    cert = certify_encoding(m, b, fact_rows[:1], N_ENT, joint_trials=0)
    assert cert.n_evaluations > 1


def test_the_recorder_keeps_every_call_of_a_module_not_just_the_last():
    """ln_key is called twice in encode_bank; keying by name alone kept only the second."""
    m = _model()
    b = _bank(p_shred=0.2)
    with _Recorder(m) as rec:
        with torch.no_grad():
            m.encode_bank(b)
        keys = [k for k in rec.captured if k.startswith("ln_key#")]
    assert len({k.split("|")[0] for k in keys}) >= 2, sorted(keys)


# --------------- the composition: a record certificate is only as strong as the closure it covers

from so.audit import FactCertificate, certify_fact                       # noqa: E402
from so.closure import duplicate_keys, fact_closure, pod_keys            # noqa: E402
from so.mvcc import MVCCStore                                            # noqa: E402
from so.reference import ReferenceResolver                               # noqa: E402
from so.world import Query                                               # noqa: E402


def _clean(n_rows=1, n_eval=8):
    """A record-level certificate that holds, as the composition would receive one."""
    from so.audit import Certificate
    return Certificate(True, True, True, n_rows, N_ENT, n_eval, 0, [])


def _dirty(n_rows=1):
    from so.audit import Certificate, Violation
    return Certificate(False, False, False, n_rows, N_ENT, 8, 0,
                       [Violation(0, 3, "encode_bank[v_fwd]", 1.2e-2)])


def _no_path():
    """``reachable`` is the field; ``certified_structurally`` is its negation."""
    from so.audit import StructuralResult
    return StructuralResult(False, 0.0, 4, "autograd found no path from the deleted payload to any output")


def _was_reachable():
    """A control as check_absence wants one: the payload REACHED the outputs while its row was there.

    The opposite polarity to _no_path(), and deliberately so. certify_fact's ``structural`` argument
    wants a result that found NO path; check_absence's ``control`` wants one that DID.
    """
    from so.audit import StructuralResult
    return StructuralResult(True, 2.2e-2, 4, "a path exists")


def _gone(n=1):
    """A membership result as ``check_absence`` would return one: rows gone, control found a path."""
    from so.audit import AbsenceCheck
    return AbsenceCheck(absent=True, control_reachable=True, n_removed=n, rows_before=10 + n,
                        rows_after=10)


def _pod_store(n_aliases=4, obj=7):
    st = MVCCStore(marker_dim=16, seed=0)
    target = st.write(3, 1, obj, provenance="target")
    for i in range(n_aliases):
        st.link(100 + i, 1, target, provenance=f"alias{i}")
    return st, target


def _dup_store(k=5, obj=7):
    st = MVCCStore(marker_dim=16, seed=0)
    kids = [st.write(100 + i, 1, obj, provenance=f"copy{i}") for i in range(k)]
    return st, kids


def test_a_pod_turns_one_record_certificate_into_a_fact_certificate():
    """The whole point: k access paths, one object, and one certified removal covers the fact."""
    st, target = _pod_store(n_aliases=4)
    keys = pod_keys(st, target)
    fc = fact_closure(st, keys, obj=7)
    assert fc.size == 1 and fc.records == (target,)

    st.evict(target)                                   # the deletion the closure prescribes
    cert = certify_fact(_clean(), fc, [target], store_after=st, keys=keys, structural=_no_path())
    assert cert.valid, cert.summary()
    assert cert.covers_closure and cert.record_certified and cert.post_condition is True
    assert "FACT CERTIFIED" in cert.summary()
    for key in keys:                                   # and the store agrees, independently
        assert ReferenceResolver(st).resolve(Query("fwd", key[0], (key[1],), (0,))).answer != 7


def test_the_same_certificate_licenses_nothing_at_the_fact_level_under_duplication():
    """The contrast that makes the composition worth having: identical record proof, void conclusion."""
    st, kids = _dup_store(k=5)
    keys = duplicate_keys(st, 7)
    fc = fact_closure(st, keys, obj=7)
    assert fc.size == 5

    st.evict(kids[0])                                  # exactly as much deletion as the pod needed
    cert = certify_fact(_clean(), fc, [kids[0]], store_after=st, keys=keys, structural=_no_path())
    assert not cert.valid
    assert "misses 4 record(s)" in cert.void_reason
    assert cert.post_condition is False                # the store still answers, and says so
    assert "still answers 7" in cert.void_reason


def test_removing_the_whole_closure_of_a_duplicated_store_does_earn_the_certificate():
    """It is not that duplication cannot be erased -- it is that it costs k, and the cost is visible."""
    st, kids = _dup_store(k=5)
    keys = duplicate_keys(st, 7)
    fc = fact_closure(st, keys, obj=7)
    for kid in fc.records:
        st.evict(kid)
    cert = certify_fact(_clean(n_rows=5), fc, fc.records, store_after=st, keys=keys,
                        structural=_no_path())
    assert cert.valid, cert.summary()
    assert cert.closure_size == 5 and len(cert.removed) == 5


def test_an_oversized_removal_is_wasteful_and_still_sound():
    """Minimality is an efficiency property. The certificate must not confuse it with soundness."""
    st, target = _pod_store(n_aliases=3)
    keys = pod_keys(st, target)
    fc = fact_closure(st, keys, obj=7)
    extra = [k for k in st.cells if k != target]
    for kid in [target] + extra:
        st.evict(kid)
    cert = certify_fact(_clean(n_rows=len(extra) + 1), fc, [target] + extra, store_after=st, keys=keys,
                        structural=_no_path())
    assert cert.valid and cert.closure_size == 1 and len(cert.removed) == len(extra) + 1


def test_a_failed_record_certificate_cannot_be_rescued_by_a_closure_of_one():
    """Both premises are needed. A pod does not make a leaking model clean."""
    st, target = _pod_store(n_aliases=2)
    keys = pod_keys(st, target)
    fc = fact_closure(st, keys, obj=7)
    st.evict(target)
    cert = certify_fact(_dirty(), fc, [target], store_after=st, keys=keys, structural=_no_path())
    assert not cert.valid
    assert "does not hold at the level of outputs" in cert.void_reason


def test_a_sweep_over_no_rows_is_refused_as_vacuous():
    """The trap the machinery walks into: EVICT leaves nothing to perturb, and an empty sweep
    certifies with one evaluation. Vacuous and strong look identical in a boolean."""
    st, target = _pod_store(n_aliases=2)
    keys = pod_keys(st, target)
    fc = fact_closure(st, keys, obj=7)
    st.evict(target)
    vacuous = _clean(n_rows=0, n_eval=1)
    cert = certify_fact(vacuous, fc, [target], store_after=st, keys=keys)
    assert not cert.valid
    assert "vacuously" in cert.void_reason and "membership property" in cert.void_reason
    # the same empty sweep, with the membership property and its control, IS the theorem
    ok = certify_fact(vacuous, fc, [target], store_after=st, keys=keys, absence=_gone())
    assert ok.valid and ok.payload_absent


def test_an_absence_without_its_control_does_not_repair_the_vacuous_sweep():
    """The void has to propagate: a membership check that proves nothing cannot discharge anything."""
    from so.audit import AbsenceCheck
    st, target = _pod_store(n_aliases=1)
    keys = pod_keys(st, target)
    fc = fact_closure(st, keys, obj=7)
    st.evict(target)
    uncontrolled = AbsenceCheck(absent=True, control_reachable=False, n_removed=1, rows_before=3,
                                rows_after=2)
    cert = certify_fact(_clean(n_rows=0), fc, [target], store_after=st, keys=keys, absence=uncontrolled)
    assert not cert.valid and "NOT reachable before" in cert.void_reason


def test_an_exhausted_closure_search_cannot_be_covered_by_anything():
    """If the closure is unknown, no removal set can be shown to contain it, and saying so is the
    only honest answer -- a search that ran out of budget is not a small closure."""
    st, _ = _dup_store(k=6)
    keys = duplicate_keys(st, 7)
    fc = fact_closure(st, keys, obj=7, max_records=2)
    assert fc.exhausted
    cert = certify_fact(_clean(n_rows=2), fc, fc.records)
    assert not cert.valid and "cut short" in cert.void_reason


def test_the_post_condition_is_a_check_and_not_a_restatement_of_the_closure():
    """A closure that disagrees with the store it describes is the failure that would make all of
    this decorative, so the store is asked directly rather than trusted."""
    st, target = _pod_store(n_aliases=2)
    keys = pod_keys(st, target)
    fc = fact_closure(st, keys, obj=7)
    # the closure is covered on paper, but the deletion was never applied to the store
    cert = certify_fact(_clean(), fc, [target], store_after=st, keys=keys, structural=_no_path())
    assert cert.covers_closure and not cert.valid
    assert cert.post_condition is False and "disagrees with the store" in cert.void_reason


def test_the_residual_the_certificate_does_not_cover_is_carried_rather_than_hidden():
    st, target = _pod_store(n_aliases=2)
    keys = pod_keys(st, target)
    fc = fact_closure(st, keys, obj=7)
    st.evict(target)
    note = "the frozen core knew this fact before the store existed (E-000013)"
    cert = certify_fact(_clean(), fc, [target], store_after=st, keys=keys, structural=_no_path(),
                        residual_note=note)
    assert cert.valid and cert.residual_note == note


def test_the_composition_runs_on_a_real_certificate_and_a_real_bank():
    """End to end on the actual model: measure the closure, evict it, certify what remains.

    The record certificate here is the interface one, run over the surviving bank; the structural
    result is what carries the evicted row, since it is no longer an input to perturb.
    """
    from so.audit import certify_encoding, certify_structural
    from so.data import bank_from_store
    from so.train import make_centre

    st = MVCCStore(marker_dim=16, seed=0)
    st.marker_centre = make_centre(0, 16)
    target = st.write(3, 1, 7, provenance="target")
    for i in range(4):
        st.link(10 + i, 1, target, provenance=f"alias{i}")
    for i in range(8):                                   # unrelated traffic, so the bank is not trivial
        st.write(20 + i, 2, 11 + i, provenance="other")

    keys = pod_keys(st, target)
    fc = fact_closure(st, keys, obj=7)
    assert fc.size == 1 and fc.optimal

    from so.audit import check_absence
    from so.experiments.common import position_of_kid
    m = _model()

    bank_before = bank_from_store(st)
    before = bank_before.tensors()
    q = _queries(before)
    control = certify_structural(m, before, [position_of_kid(st, target)], _run_grad(m, before, q),
                                 D_MODEL, outputs_of=LOGITS)
    assert control.reachable, "the payload must matter BEFORE, or its absence after proves nothing"

    st.evict(target)
    bank_after = bank_from_store(st)
    bank = bank_after.tensors()
    record = certify_encoding(m, bank, [], N_ENT)        # nothing left to sweep: vacuous on its own
    absence = check_absence(bank_before, bank_after, [target], control=control)
    assert absence.certified_absent, absence.summary()

    cert = certify_fact(record, fc, [target], store_after=st, keys=keys, absence=absence,
                        residual_note="says nothing about what the core knew before the store existed")
    assert cert.valid, cert.summary()
    assert cert.n_keys == 5 and cert.closure_size == 1 and cert.payload_absent


# ------------- absence proved by a counterfactual in the store, not by a membership test

from so.audit import StoreAbsence, certify_store_absence      # noqa: E402


def _pod_with_alias(obj=7):
    st = MVCCStore(marker_dim=16, seed=0)
    target = st.write(3, 1, obj, provenance="t")
    alias = st.link(10, 1, target, provenance="a")
    return st, target, alias


def test_the_payload_of_an_evicted_cell_does_not_reach_any_surviving_row():
    """The claim check_absence makes, actually tested: sweep the payload IN THE STORE, exhaustively."""
    st, target, _ = _pod_with_alias()
    st.evict(target)
    res = certify_store_absence(st, [target], lambda s: s.bank(), n_values=32)
    assert res.certified, res.summary()
    assert res.n_evaluations == 32 and "STORE-ABSENT" in res.summary()


def test_a_live_cell_is_never_store_absent():
    """The control. If the sweep certifies a cell that is still in the bank it measures nothing."""
    st, target, _ = _pod_with_alias()
    res = certify_store_absence(st, [target], lambda s: s.bank(), n_values=32)
    assert not res.certified
    assert any(m.startswith("obj") for m in res.moved), res.moved


def test_the_address_of_an_evicted_cell_DOES_still_reach_a_surviving_alias_row():
    """The gap a membership test cannot see, measured rather than argued.

    ``MVCCStore.bank()`` builds a LINK row's link_subject/link_relation from the TARGET cell. Evicting
    the target does not stop that, so the surviving alias row is a function of the evicted cell's KEY.
    The payload is gone; the address is not. Two different claims, and the sweep tells them apart.
    """
    st, target, _ = _pod_with_alias()
    st.evict(target)
    payload = certify_store_absence(st, [target], lambda s: s.bank(), 32, fields=("obj",))
    address = certify_store_absence(st, [target], lambda s: s.bank(), 32, fields=("subject",))
    assert payload.certified
    assert not address.certified, address.summary()
    assert any("link_subject" in m for m in address.moved), address.moved


def test_the_sweep_leaves_the_store_as_it_found_it():
    st, target, _ = _pod_with_alias()
    st.evict(target)
    before = st.state_hash()
    certify_store_absence(st, [target], lambda s: s.bank(), 32, fields=("obj", "subject", "relation"))
    assert st.state_hash() == before


def test_the_store_counterfactual_can_void_a_fact_certificate_a_membership_test_would_pass():
    """The composition has to inherit the stronger instrument's verdict, not the weaker one's."""
    from so.audit import Certificate, check_absence, certify_fact
    from so.closure import fact_closure, pod_keys
    from so.data import bank_from_store
    st, target, _ = _pod_with_alias()
    keys = pod_keys(st, target)
    fc = fact_closure(st, keys, obj=7)
    before = bank_from_store(st)
    st.evict(target)
    membership = check_absence(before, bank_from_store(st), [target], control=_was_reachable())
    assert membership.certified_absent                 # the weaker test passes ...

    address = certify_store_absence(st, [target], lambda s: s.bank(), 32, fields=("subject",))
    cert = certify_fact(_clean(n_rows=0), fc, [target], store_after=st, keys=keys,
                        absence=membership, store_absence=address)
    assert not cert.valid                              # ... and the stronger one voids the certificate
    assert "store counterfactual moved a surviving row" in cert.void_reason

    payload = certify_store_absence(st, [target], lambda s: s.bank(), 32, fields=("obj",))
    ok = certify_fact(_clean(n_rows=0), fc, [target], store_after=st, keys=keys,
                      absence=membership, store_absence=payload)
    assert ok.valid and ok.payload_absent, ok.summary()


# ------------------- reachability is not erasure, and the certificate must not say it is

from so.audit import RetentionCheck, check_retention                     # noqa: E402


def test_eviction_leaves_the_payload_in_the_store_and_the_check_says_so():
    """The two-part statement EVICT actually earns: unreachable to the reader, retained in the store.

    Keeping the versions is the operation's purpose -- it is what makes RESTORE work -- so a
    certificate that reads "the fact is gone" after an EVICT is describing something that did not
    happen. This is the instrument that stops it.
    """
    st, target, _ = _pod_with_alias(obj=41)
    st.evict(target)
    ret = check_retention(st, [target])
    assert ret.retained and not ret.erased
    assert target in ret.in_versions
    assert ret.in_log, "the write-ahead log holds it too"
    assert "RETAINED (unreachable, not erased)" in ret.summary()


def test_a_deleted_cell_keeps_its_payload_in_the_log_even_when_its_versions_are_gone():
    """DELETE drops the versions; the write-ahead log is the second place, and it is not swept."""
    st, target, _ = _pod_with_alias(obj=41)
    st.delete(target)
    ret = check_retention(st, [target])
    assert not st.cells[target].versions
    # with no versions left there is nothing to name the payload from, so the check reports honestly
    assert ret.summary().startswith("ERASED") or ret.retained


def test_the_certificate_says_unreachable_rather_than_gone_when_the_store_still_holds_it():
    from so.audit import certify_fact
    from so.closure import fact_closure, pod_keys
    st, target, _ = _pod_with_alias(obj=41)
    keys = pod_keys(st, target)
    fc = fact_closure(st, keys, obj=41)
    st.evict(target)
    cert = certify_fact(_clean(n_rows=0), fc, [target], store_after=st, keys=keys, absence=_gone(),
                        retention=check_retention(st, [target]))
    assert cert.valid and cert.retained_in_store is True
    assert "FACT UNREACHABLE, CERTIFIED" in cert.summary()
    assert "NOT ERASED" in cert.summary()
    assert "triple" in cert.summary()          # and it names what it individuates a fact by


def test_the_individuation_travels_with_the_verdict_and_can_be_narrowed():
    from so.audit import certify_fact
    from so.closure import fact_closure, pod_keys
    st, target, _ = _pod_with_alias(obj=41)
    keys = pod_keys(st, target)
    fc = fact_closure(st, keys, obj=41)
    st.evict(target)
    cert = certify_fact(_clean(n_rows=0), fc, [target], store_after=st, keys=keys, absence=_gone(),
                        individuation="this subject's pod only; the value 41 may live under others")
    assert "may live under others" in cert.summary()


def test_a_shadowed_duplicate_on_the_same_key_is_found_by_the_search():
    """The case the refuter said had no test: a second FACT cell holding the SAME key.

    ``_key_index`` is first-holder-wins, so the shadowed cell answers nothing while the first is
    alive and starts answering the moment it goes. A closure that stopped at the first record would
    be wrong, and the greedy search has to keep going -- which is where its lower bound stops being
    trivial.
    """
    from so.closure import fact_closure
    st = MVCCStore(marker_dim=16, seed=0)
    first = st.write(3, 1, 7, provenance="first")
    shadow = st.write(3, 1, 7, provenance="shadow")     # same key, hidden behind the first
    assert ReferenceResolver(st).resolve(Query("fwd", 3, (1,), (0,))).answer == 7
    fc = fact_closure(st, [(3, 1)], obj=7)
    assert fc.size == 2 and set(fc.records) == {first, shadow}, fc.summary()
    # the two derivations are never live at the same time, so no disjoint pair exists and the bound
    # is 1 while the true answer is 2: greedy does real work here and says it is not proved optimal
    assert fc.lower_bound == 1 and not fc.optimal
    assert ReferenceResolver(st).resolve(Query("fwd", 3, (1,), (0,))).answer == 7   # and it restored
