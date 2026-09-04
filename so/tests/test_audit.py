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
    assert any("k_rev" in d.module for d in res.differences), res.summary()


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
    assert any(d.module.endswith("ent_emb") for d in full.differences), full.summary()


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
