"""A deletion audit: does anything the model computes still depend on the deleted payload?

Every deletion result in this repository is an attack that failed. That is the weakest kind of
evidence: it says a particular adversary did not recover the fact, and it says nothing about the
adversary nobody thought of. E-000028 is what that costs -- SHRED passed a probe, forced choice, rank
and top-1, and an attack invented afterwards recovered the object at 100% through the routing keys,
which no attack had read.

This module asks the question the other way round, and mechanically. Take the store after the
deletion, build a second store identical except that the deleted cells hold DIFFERENT objects, and run
the model on both. Register a forward hook on every submodule and compare every tensor either produces.

If no tensor differs, the deletion is complete with respect to this model: the two computations are
the same computation, so no function of the model -- no probe, no sweep, no attack that exists or will
be invented -- can tell the two payloads apart. That is an independence result, not a failed attack.

If something does differ, the audit names the module, which is the part an engineer can act on.

The check is exact rather than statistical. A difference of 1e-9 is still a difference, and floating
point is deterministic here: the two runs execute the same kernels on the same shapes, so an unchanged
quantity is bit-identical rather than nearly so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class Difference:
    module: str
    output: str
    max_abs: float
    shape: Tuple[int, ...]

    @property
    def base(self) -> str:
        """The module name without the call index. ``ent_emb#2`` -> ``ent_emb``.

        Captures are keyed by call index because a module invoked more than once used to keep only its
        last output. Callers that care which module leaked, not which invocation, want this."""
        return self.module.split("#", 1)[0]

    def __str__(self) -> str:
        return f"{self.module}[{self.output}] differs by {self.max_abs:.3e} (shape {tuple(self.shape)})"


@dataclass
class AuditResult:
    """Two levels, because two adversaries.

    ``output_independent`` is the black-box statement: what the model returns is identical under both
    payloads, so nothing an adversary can compute FROM ITS OUTPUTS can tell them apart. That is the
    threat model of every attack in this repository's battery.

    ``activation_independent`` is the white-box statement: no tensor anywhere in the forward pass
    differs. It is strictly stronger, and it is the one that matters against an adversary who can read
    activations -- which is exactly the linear-probe adversary the battery already assumes elsewhere.
    A masked cell still has its object embedded before it is masked, so a deletion can be
    output-independent and activation-dependent at the same time; the audit says which.
    """

    output_independent: bool
    activation_independent: bool
    n_modules: int
    n_tensors: int
    differences: List[Difference] = field(default_factory=list)
    output_differences: List[Difference] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """The black-box claim, which is what the recorded deletion criteria are about."""
        return self.output_independent

    @property
    def first_leak(self) -> Optional[str]:
        return self.differences[0].module if self.differences else None

    def summary(self) -> str:
        out = ("outputs identical" if self.output_independent
               else f"OUTPUTS DIFFER ({len(self.output_differences)} tensors)")
        if self.activation_independent:
            act = f"all {self.n_tensors} activations from {self.n_modules} modules bit-identical"
        else:
            worst = max(self.differences, key=lambda d: d.max_abs)
            act = (f"{len(self.differences)} of {self.n_tensors} activations move with the payload; "
                   f"first at {self.differences[0].module}, largest {worst}")
        return f"{out}; {act}"


def _flatten(x: Any, prefix: str = "") -> List[Tuple[str, torch.Tensor]]:
    """Every tensor inside a module's output, whatever container it came in."""
    out: List[Tuple[str, torch.Tensor]] = []
    if torch.is_tensor(x):
        out.append((prefix or "0", x))
    elif isinstance(x, (tuple, list)):
        for i, v in enumerate(x):
            out += _flatten(v, f"{prefix}{i}." if prefix else f"{i}")
    elif isinstance(x, dict):
        for k, v in x.items():
            out += _flatten(v, f"{prefix}{k}." if prefix else str(k))
    return out


class _Recorder:
    """Captures every tensor every submodule emits during one forward pass."""

    def __init__(self, model: nn.Module, skip: Sequence[str] = ()):
        self.model = model
        self.skip = tuple(skip)
        self.captured: Dict[str, torch.Tensor] = {}
        self._calls: Dict[str, int] = {}
        self._handles: List[Any] = []
        self.n_modules = 0

    def __enter__(self) -> "_Recorder":
        for name, mod in self.model.named_modules():
            if name == "" or any(name.startswith(s) for s in self.skip):
                continue
            self.n_modules += 1
            self._handles.append(mod.register_forward_hook(self._make(name)))
        return self

    def _make(self, name: str) -> Callable:
        def hook(module, inputs, output):
            # Keyed by CALL INDEX as well as name. Keying by name alone kept only a module's last
            # invocation, so "every tensor every submodule emits" was false wherever a module is
            # called more than once -- ln_key twice in encode_bank, the hop block once per hop -- and
            # the comparison silently skipped the earlier calls.
            i = self._calls.get(name, 0)
            self._calls[name] = i + 1
            for suffix, t in _flatten(output):
                if t.is_floating_point():
                    self.captured[f"{name}#{i}|{suffix}"] = t.detach().clone()
            return None
        return hook

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def audit_independence(model: nn.Module, run_a: Callable[[], Any], run_b: Callable[[], Any],
                       skip: Sequence[str] = (), atol: float = 0.0,
                       outputs_of: Optional[Callable[[Any], Any]] = None) -> AuditResult:
    """Run the model twice and compare every tensor every submodule produced.

    ``run_a`` and ``run_b`` differ only in the payload of the deleted cells -- and in nothing else, the
    QUESTIONS included: an attacker's queries do not change when the payload they are hunting changes,
    so building them from each store in turn would report a difference the audit created itself.

    ``outputs_of`` selects what counts as "the output" from what the run returns, which decides which
    adversary the black-box level is about. The default is everything returned, which for this
    repository's models includes diagnostics such as per-cell value norms -- observables the biomarker
    experiments use, and rightly counted, but not what a user of the model sees. Pass a selector to
    ask the narrower question about the answer alone. ``skip`` drops module name
    prefixes from the comparison -- used for the frozen core of a hooked LM, whose blocks legitimately
    differ once an injection has happened, when only the knowledge layer is under audit. ``atol`` above
    zero is a deliberate tolerance and should be justified wherever it is used; the default demands
    bit-identity.
    """
    with _Recorder(model, skip) as rec_a:
        out_a = run_a()
        a = dict(rec_a.captured)
        n_modules = rec_a.n_modules
    with _Recorder(model, skip) as rec_b:
        out_b = run_b()
        b = dict(rec_b.captured)

    diffs: List[Difference] = []
    shared = [k for k in a if k in b]
    for key in shared:
        ta, tb = a[key], b[key]
        if ta.shape != tb.shape:
            diffs.append(Difference(key.split("|")[0], key.split("|")[1], float("inf"), tuple(ta.shape)))
            continue
        d = float((ta - tb).abs().max().item()) if ta.numel() else 0.0
        if d > atol:
            diffs.append(Difference(key.split("|")[0], key.split("|")[1], d, tuple(ta.shape)))
    for key in a:
        if key not in b:
            diffs.append(Difference(key.split("|")[0], key.split("|")[1], float("inf"), tuple(a[key].shape)))

    out_diffs: List[Difference] = []
    if outputs_of is not None:
        out_a, out_b = outputs_of(out_a), outputs_of(out_b)
    fa, fb = dict(_flatten(out_a)), dict(_flatten(out_b))
    for name, ta in fa.items():
        tb = fb.get(name)
        if tb is None or ta.shape != tb.shape:
            out_diffs.append(Difference("<returned>", name, float("inf"), tuple(ta.shape)))
            continue
        if not ta.is_floating_point():
            if not torch.equal(ta, tb):
                out_diffs.append(Difference("<returned>", name, float("inf"), tuple(ta.shape)))
            continue
        d = float((ta - tb).abs().max().item()) if ta.numel() else 0.0
        if d > atol:
            out_diffs.append(Difference("<returned>", name, d, tuple(ta.shape)))

    return AuditResult(output_independent=not out_diffs, activation_independent=not diffs,
                       n_modules=n_modules, n_tensors=len(shared),
                       differences=diffs, output_differences=out_diffs)


def perturbed_objects(bank: Dict[str, torch.Tensor], rows: np.ndarray, n_entities: int,
                      shift: int = 1) -> Dict[str, torch.Tensor]:
    """A copy of ``bank`` in which the named rows hold different objects and nothing else changes."""
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in bank.items()}
    idx = torch.as_tensor(rows, dtype=torch.long)
    out["obj"] = out["obj"].clone()
    out["obj"][idx] = (out["obj"][idx] + shift) % n_entities
    return out


# --------------------------------------------------------------------------- an exhaustive certificate

@dataclass
class Violation:
    row: int
    value: int
    module: str
    max_abs: float

    def __str__(self) -> str:
        return f"row {self.row} at payload {self.value}: {self.module} differs by {self.max_abs:.3e}"


@dataclass
class Certificate:
    """What a completed sweep licenses, and what it does not.

    ``output_certified`` states: for every deleted row, and for EVERY value that row's payload field
    could hold, the model returns bit-identical tensors on the query set swept. Over a finite payload
    domain that is not a sample -- it is every case -- so within the swept query set no function of the
    model's outputs can depend on the deleted payload. That is the strongest form the black-box claim
    can take, and it is stronger than any number of failed attacks.

    ``activation_certified`` says the same of every tensor every submodule emitted, which is the claim
    against an adversary who reads activations.

    Three limits, stated because a certificate that hides them is worse than none:
      * It is exhaustive over the payload domain and over whatever query set the caller sweeps. Make
        the query set exhaustive too and the statement becomes exhaustive in both; a sampled query set
        gives an exhaustive statement about a sampled set of questions.
      * Rows are swept ONE AT A TIME with the others held at their stored values, so it certifies each
        deletion marginally. Joint dependence across several deleted rows is covered only by the
        ``joint_trials`` random assignments, which ARE a sample. ``joint_certified`` says so honestly.
      * It certifies the model, not the system. The store still holds the payload after SHRED or
        REVOKE, and anyone who can read the store does not need the model.
    """

    output_certified: bool
    activation_certified: bool
    joint_certified: bool
    n_rows: int
    n_values: int
    n_evaluations: int
    joint_trials: int
    violations: List[Violation] = field(default_factory=list)

    def summary(self) -> str:
        head = (f"{self.n_evaluations} evaluations: {self.n_rows} rows x {self.n_values} payload values"
                f" + {self.joint_trials} joint trials")
        if self.output_certified and self.activation_certified:
            return f"CERTIFIED at both levels ({head})"
        if self.output_certified:
            return (f"outputs certified, activations NOT ({len(self.violations)} violations, "
                    f"first {self.violations[0]}) ({head})")
        return f"NOT CERTIFIED ({len(self.violations)} violations, first {self.violations[0]}) ({head})"


def _fingerprint(x: Any) -> List[Tuple[str, torch.Tensor]]:
    return [(k, v.detach().clone()) for k, v in _flatten(x) if torch.is_tensor(v)]


def _compare(ref: List[Tuple[str, torch.Tensor]], got: List[Tuple[str, torch.Tensor]],
             atol: float) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    gd = dict(got)
    for name, a in ref:
        b = gd.get(name)
        if b is None or a.shape != b.shape:
            out.append((name, float("inf")))
            continue
        if not a.is_floating_point():
            if not torch.equal(a, b):
                out.append((name, float("inf")))
            continue
        d = float((a - b).abs().max().item()) if a.numel() else 0.0
        if d > atol:
            out.append((name, d))
    return out


# A row's payload is not always one field. A FACT row carries its object in ``obj``; a LINK row's obj
# is a hardwired placeholder and its payload -- the address it points at -- lives in ``link_subject``
# and ``link_relation`` (so/mvcc.py bank()). A sweep over ``obj`` alone therefore says nothing about a
# link row, and a certificate that returned True for such a row would be certifying a payload it never
# looked at. Both domains are finite, so exhaustiveness survives; what does not survive is guessing.
FACT_PAYLOAD: Tuple[str, ...] = ("obj",)
LINK_PAYLOAD: Tuple[str, ...] = ("link_subject", "link_relation")


class UnsweepablePayload(RuntimeError):
    """Raised when a row's payload is not covered by the fields being swept.

    Failing closed is the whole point. A certificate that cannot express what a row holds must refuse
    to certify it, because the alternative -- returning True -- is the one outcome an audit must never
    produce for a case it did not examine.
    """


def payload_fields_for(bank: Dict[str, torch.Tensor], rows: Sequence[int],
                       fields: Sequence[str]) -> Tuple[str, ...]:
    """Check the named fields cover every deleted row's payload, and refuse if they do not."""
    fields = tuple(fields)
    if "is_link" in bank and len(rows):
        idx = torch.as_tensor([int(r) for r in rows], dtype=torch.long)
        if bool(bank["is_link"][idx].any()):
            missing = [f for f in LINK_PAYLOAD if f not in fields]
            if missing:
                raise UnsweepablePayload(
                    f"rows {[int(r) for r in rows]} include LINK cells, whose payload is "
                    f"{LINK_PAYLOAD} and not {fields}; sweeping {fields} would certify a payload the "
                    f"sweep never looked at. Pass payload_fields={FACT_PAYLOAD + LINK_PAYLOAD}.")
    for f in fields:
        if f not in bank:
            raise UnsweepablePayload(f"the bank has no field {f!r} to sweep")
    return fields


def _domain(bank: Dict[str, torch.Tensor], field: str, n_values: int) -> int:
    """How many values a field can take. Relations are a smaller domain than entities."""
    return 4 if field.endswith("relation") else n_values


def _assignments(bank: Dict[str, torch.Tensor], fields: Sequence[str], row: int,
                 n_values: int) -> List[Dict[str, int]]:
    """Every assignment of the payload fields for one row, excluding the one it already holds.

    The product stays small because the domains are: 256 for an entity, 4 for a relation. Exhaustive
    over the product is what makes the result a statement about every case rather than a sample.
    """
    import itertools as _it
    ranges = [range(_domain(bank, f, n_values)) for f in fields]
    current = tuple(int(bank[f][row].item()) for f in fields)
    out: List[Dict[str, int]] = []
    for combo in _it.product(*ranges):
        if combo == current:
            continue
        out.append({f: int(v) for f, v in zip(fields, combo)})
    return out


def certify_deletion(model: nn.Module, bank: Dict[str, torch.Tensor], deleted_rows: Sequence[int],
                     n_values: int, run: Callable[[Dict[str, torch.Tensor]], Any],
                     payload_fields: Sequence[str] = FACT_PAYLOAD,
                     outputs_of: Optional[Callable[[Any], Any]] = None,
                     joint_trials: int = 64, seed: int = 0, atol: float = 0.0,
                     skip: Sequence[str] = (), check_activations: bool = True,
                     stop_early: bool = True) -> Certificate:
    """Sweep every value the deleted payload could hold and check the computation does not move.

    ``run(bank) -> outputs`` must hold the QUESTIONS fixed and vary only the bank, or the sweep will
    report a difference it created itself. ``n_values`` is the size of the payload domain -- for this
    repository, the entity count -- and the sweep is over ``range(n_values)``, which is why the result
    is a statement about every case rather than about a sample.
    """
    rows = [int(r) for r in deleted_rows]
    fields = payload_fields_for(bank, rows, payload_fields)
    rng = np.random.default_rng(seed)
    base = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in bank.items()}

    def evaluate(b: Dict[str, torch.Tensor]) -> Tuple[List, List]:
        if check_activations:
            with _Recorder(model, skip) as rec:
                out = run(b)
                acts = sorted(rec.captured.items())
        else:
            out = run(b)
            acts = []
        picked = outputs_of(out) if outputs_of is not None else out
        return _fingerprint(picked), [(k, v) for k, v in acts]

    ref_out, ref_act = evaluate(base)
    violations: List[Violation] = []
    out_ok = act_ok = True
    n_eval = 1

    for row in rows:
        for assignment in _assignments(base, fields, row, n_values):
            probe = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}
            for f, value in assignment.items():
                probe[f][row] = value
            value = next(iter(assignment.values()))
            got_out, got_act = evaluate(probe)
            n_eval += 1
            for name, d in _compare(ref_out, got_out, atol):
                out_ok = False
                violations.append(Violation(row, value, f"<returned>{name}", d))
            if check_activations:
                for name, d in _compare(ref_act, got_act, atol):
                    act_ok = False
                    violations.append(Violation(row, value, name.split("|")[0], d))
            if violations and stop_early:
                # the sweep has already answered the question; finishing it would only enumerate
                return Certificate(False, False, False, len(rows), n_values, n_eval, 0, violations)

    joint_ok = True
    if len(rows) > 1 and joint_trials > 0:
        for _ in range(joint_trials):
            probe = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}
            for row in rows:
                for f in fields:
                    probe[f][row] = int(rng.integers(_domain(base, f, n_values)))
            got_out, got_act = evaluate(probe)
            n_eval += 1
            diffs = _compare(ref_out, got_out, atol)
            if check_activations:
                diffs += _compare(ref_act, got_act, atol)
            if diffs:
                joint_ok = False
                violations.append(Violation(-1, -1, diffs[0][0].split("|")[0], diffs[0][1]))
                if stop_early:
                    break
    elif len(rows) <= 1:
        joint_ok = out_ok and act_ok           # nothing to interact with

    return Certificate(out_ok, act_ok and check_activations, joint_ok, len(rows), n_values, n_eval,
                       joint_trials if len(rows) > 1 else 0, violations)


# ------------------------------------------------- the certificate at the interface the bank enters through

@dataclass
class MediationCheck:
    """Evidence for the premise the interface certificate rests on, or against it."""
    consistent: bool
    n_probes: int
    encoding_invariant: bool
    output_invariant: bool
    note: str = ""


def certify_encoding(model: nn.Module, bank: Dict[str, torch.Tensor], deleted_rows: Sequence[int],
                     n_values: int, encode: Optional[Callable[[Dict[str, torch.Tensor]], Any]] = None,
                     payload_fields: Sequence[str] = FACT_PAYLOAD, joint_trials: int = 64, seed: int = 0,
                     atol: float = 0.0, stop_early: bool = True,
                     interface_keys: Optional[Sequence[str]] = None) -> Certificate:
    """Certify at the interface the bank enters the computation through, not at the outputs.

    Both models here read the store in exactly one place. ``MutableKnowledgeTransformer.forward``
    computes ``enc = self.encode_bank(bank)`` and from then on touches only ``enc["k_f"]``,
    ``enc["v_f"]``, ``enc["k_r"]``, ``enc["v_r"]`` and ``enc["active"]``, the query tensors and its own
    parameters; ``KnowledgeAdapterLM.forward`` does the same with ``enc["keys"]``, ``enc["values"]``
    and ``enc["active"]``. The forward is therefore a deterministic function of (encoding, query).

    That buys something the output sweep cannot. If the encoding is bit-identical for every value the
    deleted payload could take, then every downstream quantity is identical **for every possible
    query** -- multi-hop, reverse, unseen phrasings, questions nobody has written yet -- and not merely
    for the queries somebody thought to sweep. The cost is one cheap encoding per payload value
    instead of a full forward over a query set.

    ``interface_keys`` names WHICH of the encoding's outputs the forward actually consumes, and getting
    it wrong breaks the certificate in one direction or the other. Too narrow and the certificate is
    unsound -- it would ignore a quantity the model reads. Too wide and it is over-strict: the adapter's
    ``encode_bank`` also returns ``values_payload``, the UNGATED payload, which ``forward`` never reads
    (`so/llm_adapter.py:323` takes only ``keys``, ``values`` and the allowed set), and comparing it
    reports a leak through a diagnostic. Name the consumed set explicitly, cite the line, and let
    ``check_mediation`` guard the choice: if the named subset is invariant while an output moves, the
    subset was too narrow and the certificate is void.

    The premise is a claim about the model, so it is checked rather than assumed: pass the result to
    ``check_mediation`` with a runner, which falsifies it if an output ever moves while the encoding
    does not.
    """
    def select(enc: Any) -> Any:
        if interface_keys is None or not isinstance(enc, dict):
            return enc
        missing = [k for k in interface_keys if k not in enc]
        if missing:
            raise KeyError(f"interface_keys names {missing}, which the encoding does not return")
        return {k: enc[k] for k in interface_keys}
    _enc = encode if encode is not None else (lambda b: model.encode_bank(b))
    enc_fn = lambda b: select(_enc(b))
    rows = [int(r) for r in deleted_rows]
    fields = payload_fields_for(bank, rows, payload_fields)
    rng = np.random.default_rng(seed)
    base = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in bank.items()}

    with torch.no_grad():
        ref = _fingerprint(enc_fn(base))
    violations: List[Violation] = []
    ok = True
    n_eval = 1
    for row in rows:
        for assignment in _assignments(base, fields, row, n_values):
            probe = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}
            for f, value in assignment.items():
                probe[f][row] = value
            value = next(iter(assignment.values()))
            with torch.no_grad():
                got = _fingerprint(enc_fn(probe))
            n_eval += 1
            for name, d in _compare(ref, got, atol):
                ok = False
                violations.append(Violation(row, value, f"encode_bank[{name}]", d))
            if violations and stop_early:
                return Certificate(False, False, False, len(rows), n_values, n_eval, 0, violations)

    joint_ok = ok
    if len(rows) > 1 and joint_trials > 0 and ok:
        for _ in range(joint_trials):
            probe = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}
            for row in rows:
                for f in fields:
                    probe[f][row] = int(rng.integers(_domain(base, f, n_values)))
            with torch.no_grad():
                got = _fingerprint(enc_fn(probe))
            n_eval += 1
            diffs = _compare(ref, got, atol)
            if diffs:
                joint_ok = False
                violations.append(Violation(-1, -1, f"encode_bank[{diffs[0][0]}]", diffs[0][1]))
                if stop_early:
                    break
    return Certificate(ok, ok, joint_ok, len(rows), n_values, n_eval,
                       joint_trials if len(rows) > 1 else 0, violations)


def check_mediation(model: nn.Module, bank: Dict[str, torch.Tensor], deleted_rows: Sequence[int],
                    n_values: int, run: Callable[[Dict[str, torch.Tensor]], Any],
                    encode: Optional[Callable[[Dict[str, torch.Tensor]], Any]] = None,
                    payload_fields: Sequence[str] = FACT_PAYLOAD, n_probes: int = 8, seed: int = 0,
                    atol: float = 0.0, outputs_of: Optional[Callable[[Any], Any]] = None) -> MediationCheck:
    """Try to falsify the premise: an output that moves while the encoding does not.

    ``certify_encoding`` is sound only if the bank reaches the computation through the encoding alone.
    That is read off the source, and reading source is how the two defects of E-000028 and E-000029 got
    into the record in the first place. So it is also measured: run the full forward at ``n_probes``
    sampled payload values and compare both levels. Encoding invariant and outputs invariant is
    consistent; encoding invariant and an output moving falsifies the premise and voids the
    certificate. The check samples -- it can refute the premise, not establish it.
    """
    enc_fn = encode if encode is not None else (lambda b: model.encode_bank(b))
    rows = [int(r) for r in deleted_rows]
    fields = payload_fields_for(bank, rows, payload_fields)
    rng = np.random.default_rng(seed + 1)
    base = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in bank.items()}
    with torch.no_grad():
        ref_enc = _fingerprint(enc_fn(base))
        ref_out = _fingerprint(outputs_of(run(base)) if outputs_of else run(base))

    enc_moved = out_moved = False
    for _ in range(n_probes):
        probe = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}
        for row in rows:
            for f in fields:
                probe[f][row] = int(rng.integers(_domain(base, f, n_values)))
        with torch.no_grad():
            enc_d = _compare(ref_enc, _fingerprint(enc_fn(probe)), atol)
            out_d = _compare(ref_out, _fingerprint(outputs_of(run(probe)) if outputs_of else run(probe)), atol)
        enc_moved = enc_moved or bool(enc_d)
        out_moved = out_moved or bool(out_d)

    consistent = not (out_moved and not enc_moved)
    note = ("an output moved while the encoding did not: the bank reaches the computation somewhere else "
            "and the interface certificate is VOID for this model"
            if not consistent else
            "no probe found an output moving while the encoding held still")
    return MediationCheck(consistent, n_probes, not enc_moved, not out_moved, note)


@dataclass
class LocalityCheck:
    """Whether a row's encoding depends only on that row.

    This is what turns the joint claim from a sample into a proof. ``certify_encoding`` sweeps rows one
    at a time, so on its own it certifies each deletion marginally and covers joint dependence with
    random assignments. But if the encoding is ROW-LOCAL -- perturbing row i moves row i's encoding and
    nothing else -- then per-row invariance over the whole payload domain implies invariance under
    every joint assignment, because the rows do not interact. No sampling required.

    Both models are row-local at noise = 0: ``encode_bank`` embeds, LayerNorms and projects per row,
    and the marker gate is a function of that row's marker. They are NOT row-local at noise > 0:
    ``jitter`` in ``so/model.py`` takes its rms over ALL rows, masked ones included, so a deleted row
    perturbs the jitter of every visible one. The check reports which regime it was run in rather than
    assuming the benign one.
    """

    row_local: bool
    n_rows_probed: int
    n_values_probed: int
    offending_rows: List[int] = field(default_factory=list)
    note: str = ""


def check_row_locality(model: nn.Module, bank: Dict[str, torch.Tensor], probe_rows: Sequence[int],
                       n_values: int, encode: Optional[Callable[[Dict[str, torch.Tensor]], Any]] = None,
                       payload_fields: Sequence[str] = FACT_PAYLOAD, n_values_probed: int = 8, seed: int = 0,
                       atol: float = 0.0) -> LocalityCheck:
    """Perturb one row and check that every OTHER row's encoding is untouched."""
    enc_fn = encode if encode is not None else (lambda b: model.encode_bank(b))
    rng = np.random.default_rng(seed + 7)
    base = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in bank.items()}
    fields = payload_fields_for(bank, probe_rows, payload_fields)
    n_rows = int(base[fields[0]].shape[0])
    with torch.no_grad():
        ref = _fingerprint(enc_fn(base))
    offending: List[int] = []

    for row in [int(r) for r in probe_rows]:
        others = torch.ones(n_rows, dtype=torch.bool)
        others[row] = False
        for _ in range(n_values_probed):
            probe = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}
            for f in fields:
                probe[f][row] = int(rng.integers(_domain(base, f, n_values)))
            with torch.no_grad():
                got = _fingerprint(enc_fn(probe))
            for (name, a), (_, b) in zip(ref, got):
                if a.shape != b.shape or a.shape[:1] != (n_rows,):
                    continue                       # not a per-row tensor; the sweep covers it
                if a.is_floating_point():
                    d = (a[others] - b[others]).abs()
                    moved = bool(d.numel()) and float(d.max().item()) > atol
                else:
                    moved = not torch.equal(a[others], b[others])   # bools and ints do not subtract
                if moved:
                    offending.append(row)
                    break
            if row in offending:
                break

    ok = not offending
    return LocalityCheck(ok, len(list(probe_rows)), n_values_probed, sorted(set(offending)),
                         "row-local: per-row invariance over the whole domain implies joint invariance"
                         if ok else
                         "NOT row-local: one row's payload moves another row's encoding, so the joint "
                         "claim stays a sample. so/model.py jitter() takes its rms over all rows, which "
                         "is the known cause at noise > 0.")


# ------------------------------------------------- independence without a finite domain to sweep

@dataclass
class StructuralResult:
    """Independence as a property of the computation graph rather than of a swept domain.

    ``certify_encoding`` and ``certify_deletion`` sweep every value the payload could hold. That is
    exact, but it needs the domain to be small and finite -- here an entity id with 256 values. Real
    stores hold text, embeddings, images: domains no sweep can exhaust. So the sharpest objection to
    the whole instrument is that it works only on a toy payload.

    Reachability answers the same question without the domain. Make the payload a differentiable input
    and ask autograd whether any output is reachable from it. Three outcomes, in decreasing strength:

      no path        ``grad`` comes back None: the payload is not in the output's computation graph at
                     all. This is a THEOREM about every value the payload could take, over any domain,
                     finite or not, and it costs one backward pass. It is what DELETE achieves and what
                     a gate never can.
      zero gradient  A path exists and the derivative is exactly zero here. That is what multiplying by
                     a gate of exactly zero gives, and it holds wherever the gate stays zero -- but the
                     gate is a function of the marker, so the claim is only as good as the argument that
                     the marker does not move. Weaker than no path, stronger than a sample.
      nonzero        The output moves with the payload. Dependent, and the size says how much.

    The three compose: reachability makes the domain-free claim, the sweep makes the exact claim on a
    finite domain, and where both are available they must agree -- a disagreement means one of them is
    wrong, which is worth knowing.
    """

    reachable: bool
    grad_max: float
    n_outputs: int
    note: str = ""

    @property
    def certified_structurally(self) -> bool:
        return not self.reachable

    def summary(self) -> str:
        if not self.reachable:
            return "NO PATH: the payload is not in the output's graph, for any value over any domain"
        if self.grad_max == 0.0:
            return ("path exists, gradient exactly zero: annihilated here, and only as global as the "
                    "argument that the annihilating factor does not move")
        return f"REACHABLE: the output moves with the payload, |grad| up to {self.grad_max:.3e}"


def certify_structural(model: nn.Module, bank: Dict[str, torch.Tensor], deleted_rows: Sequence[int],
                       run: Callable[[Dict[str, torch.Tensor]], Any], d_payload: int,
                       outputs_of: Optional[Callable[[Any], Any]] = None,
                       delta_key: str = "payload_delta") -> StructuralResult:
    """Ask autograd whether any output is reachable from the deleted rows' payload.

    Both ``encode_bank`` implementations add ``bank[delta_key]`` to the payload embedding when the key
    is present. It is zeros, so nothing changes numerically; it exists so that the payload -- otherwise
    an integer index, which autograd cannot differentiate -- becomes a leaf the graph can be questioned
    about. Rows outside ``deleted_rows`` get a delta too, and it is detached, so only the deleted rows'
    payloads are under test.
    """
    n_rows = int(next(v for v in bank.values() if torch.is_tensor(v)).shape[0])
    if not len(list(deleted_rows)):
        # This branch used to return the strongest outcome in the ladder -- "no path, for any value
        # over any domain" -- and it fires on a bank whose rows are all present and live, because
        # with no row selected there is nothing for autograd to find a path FROM. It certified the
        # deletion by not testing it. Refused, for the same reason a gradient-free runner is refused.
        # When the row is genuinely gone, the claim is a MEMBERSHIP property and ``check_absence`` is
        # the instrument for it; reachability is for a row that is still in the bank.
        raise ValueError(
            "certify_structural was given no rows to perturb, so autograd would report 'no path' "
            "whatever the bank contains -- including a bank where the row is still live. If the row "
            "was removed, its independence is a membership property: use check_absence, which takes "
            "the bank before and after and requires a positive control showing the payload WAS "
            "reachable while the row was there.")
    eps = torch.zeros(len(deleted_rows), d_payload, requires_grad=True)
    delta = torch.zeros(n_rows, d_payload)
    probe = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in bank.items()}
    rows = torch.as_tensor([int(r) for r in deleted_rows], dtype=torch.long)
    probe[delta_key] = delta.index_put((rows,), eps)

    with torch.enable_grad():
        out = run(probe)
    picked = outputs_of(out) if outputs_of is not None else out
    tensors = [t for _, t in _flatten(picked) if torch.is_tensor(t) and t.is_floating_point()]
    if not tensors:
        return StructuralResult(False, 0.0, 0, "no floating-point output to differentiate")
    if not any(t.requires_grad for t in tensors):
        # Distinguish "no path" from "no graph". A runner that wraps its forward in torch.no_grad()
        # produces outputs with no grad_fn at all, and reading that as independence would certify
        # every deletion trivially -- the most dangerous failure this instrument could have.
        raise RuntimeError(
            "the outputs carry no autograd graph, so reachability cannot be decided. The runner passed "
            "to certify_structural must NOT wrap its forward in torch.no_grad(); certifying on a "
            "gradient-free run would report 'no path' for everything.")
    total = sum(t.sum() for t in tensors)
    grad = torch.autograd.grad(total, eps, allow_unused=True, retain_graph=False)[0]
    if grad is None:
        return StructuralResult(False, 0.0, len(tensors),
                                "autograd found no path from the deleted payload to any output")
    g = float(grad.abs().max().item())
    return StructuralResult(True, g, len(tensors),
                            "a path exists" + (" but the derivative is exactly zero" if g == 0.0 else ""))


# ------------------------------------------------- a row that is gone: membership, not reachability

@dataclass
class AbsenceCheck:
    """Independence by ABSENCE: the payload is not in the bank the model reads.

    This exists because the reachability instrument cannot state it. Ask autograd for a path from a
    row that is no longer in the bank and there is no row to perturb, so it answers "no path" -- and
    it answers "no path" just as confidently on a bank where the row is present and live, because the
    absence of a perturbation is not the absence of a dependence. That is a certificate produced by
    not testing anything, and it is refused at the source.

    What actually carries the claim after EVICT or DELETE is a membership property, and it is worth
    stating exactly because it is not a technicality: the model reads the store only through the bank
    -- ``MutableKnowledgeTransformer.forward`` at ``so/model.py:246``, ``KnowledgeAdapterLM.forward``
    at ``so/llm_adapter.py:323``, and ``check_mediation`` is the standing falsification of that premise
    -- so a payload with no row in the bank is not an input, and no function of the model can depend
    on it. Over any payload domain, finite or not, for every query. That is stronger than the sweep
    and stronger than the gradient, and it costs a set difference.

    The positive control is mandatory rather than optional. "The row is not there" is only evidence of
    a deletion if the row was there before and MATTERED: ``control_reachable`` records that the same
    reachability test, run on the bank BEFORE the removal at the rows that were removed, found a path.
    Without it the check would pass on a row that was never in the bank to begin with, which is the
    same failure one level along.
    """

    absent: bool
    control_reachable: bool
    n_removed: int
    rows_before: int
    rows_after: int
    still_present: Tuple[int, ...] = ()
    note: str = ""

    @property
    def certified_absent(self) -> bool:
        return bool(self.absent and self.control_reachable)

    def summary(self) -> str:
        if not self.control_reachable:
            return ("VOID: the payload was NOT reachable before the removal either, so its absence "
                    "afterwards is not evidence that anything was deleted")
        if not self.absent:
            return f"NOT ABSENT: {len(self.still_present)} removed cell(s) still hold a bank row"
        return (f"ABSENT: {self.n_removed} cell(s) hold no row in the bank the model reads "
                f"({self.rows_before} rows before, {self.rows_after} after), and the same payload was "
                f"reachable while they were there")


def _kids_of(b: Any, kid_key: str) -> List[int]:
    """The cell ids a bank holds, from whichever of the three shapes the caller has.

    ``Bank.tensors()`` drops the ``kid`` column -- the model has no use for it -- so a caller working
    with tensors must pass the ``Bank`` itself or the id list. Guessing is worse than asking.
    """
    if hasattr(b, kid_key) and not isinstance(b, dict):
        return [int(x) for x in getattr(b, kid_key)]
    if isinstance(b, dict):
        if kid_key not in b:
            raise KeyError(
                f"the bank has no {kid_key!r} column, so membership cannot be decided. Bank.tensors() "
                f"drops it; pass the Bank object or the list of cell ids instead.")
        return [int(x) for x in b[kid_key]]
    return [int(x) for x in b]


def check_absence(bank_before: Any, bank_after: Any, removed_kids: Sequence[int],
                  control: Optional["StructuralResult"] = None,
                  kid_key: str = "kid") -> AbsenceCheck:
    """Did the removal actually take the rows out of the bank, and did they matter while they were in?

    ``bank_before`` and ``bank_after`` may each be a ``Bank``, a dict carrying a ``kid`` column, or a
    plain sequence of cell ids.

    ``control`` is a ``certify_structural`` result computed on ``bank_before`` at the rows about to be
    removed. It must find a path -- ``reachable`` true -- or the absence proves nothing.
    """
    before, after = _kids_of(bank_before, kid_key), _kids_of(bank_after, kid_key)
    want = [int(k) for k in removed_kids]
    missing_before = [k for k in want if k not in set(before)]
    still = tuple(k for k in want if k in set(after))
    control_ok = bool(control is not None and control.reachable)
    note = ""
    if missing_before:
        note = (f"{len(missing_before)} of the cells named were not in the bank before the removal "
                f"either, so their absence afterwards is not evidence of anything")
    return AbsenceCheck(absent=not still and not missing_before, control_reachable=control_ok,
                        n_removed=len(want), rows_before=len(before), rows_after=len(after),
                        still_present=still, note=note)


# --------------------------------------------------- from a record certificate to a fact certificate

@dataclass
class FactCertificate:
    """What a record-level certificate plus a store-level closure jointly license.

    Everything above certifies independence from a RECORD: the model computes the same thing for every
    value the deleted row's payload could hold. That is exactly what a data subject does NOT ask for.
    They ask whether the FACT is gone, and the two statements come apart the moment the fact is
    reachable another way -- through a duplicate row, through an alias, through a derivation. This
    repository has the extreme case on record: ``dependency/derivable_recovery_after_revoke_K3 = 1.0``
    in every seed of E-000019, a certified record deletion under which every derivable fact survives.

    The composition closes that gap, and it is the point of the pod:

        record-level certificate over R  +  R covers the fact closure of the fact
        ------------------------------------------------------------------------
                          fact-level certificate

    The second premise is a property of the STORE, computed by ``so.closure.fact_closure`` with the
    mechanical resolver and no model at all. So the guarantee factorises into one expensive model-side
    proof and one cheap store-side search -- and the search is what canonicalisation makes trivial. In
    a pod the closure is a single record for any number of access keys, so ONE record certificate is
    already a fact certificate. Under duplication the closure is k, so certifying one record licenses
    nothing at the fact level, however exhaustive that certificate was.

    Three things this deliberately does NOT require, each for a reason:
      * ``closure.optimal`` is not required. An oversized removal set is wasteful, never unsound: what
        matters is that R COVERS the closure, not that it is minimal.
      * The closure is a statement about the store's own semantics. A frozen core that knew the fact
        before the store existed is outside it -- E-000013 measures that separately -- and
        ``residual_note`` is where a caller records that limit rather than letting the word
        "certificate" swallow it.
      * The closure is the resilience of a QUERY SET, and the certificate is only as wide as that set.
        ``FactClosure.workload`` names it and ``summary`` prints it, because the default workload is
        one single-hop question per key: a fact a multi-hop derivation still reaches is outside the
        claim, and E-000019 records that case at ``derivable_recovery_after_revoke_K3 = 1.0``.
        Canonicalisation collapses the duplication term of the closure and does not touch the
        derivation term.
      * A verified post-condition (no key resolves to the object afterwards) is checked when the caller
        supplies the store, because a closure argument that disagrees with the store it describes is
        the one failure mode that would make all of this decorative.

    One trap is closed explicitly, because the machinery walks straight into it -- twice. When a
    deletion takes the row OUT of the bank -- which is what EVICT and DELETE do, and is the only
    deletion that earns the domain-free claim -- there is no row left to perturb. A payload sweep over
    an empty row set then certifies with a single evaluation and no violations, and a reachability
    test over an empty row set answers "no path", the strongest label in the ladder. BOTH are vacuous:
    each fires unchanged on a bank whose rows are all present and live, because the absence of a
    perturbation is not the absence of a dependence.

    So a record certificate covering no rows is refused unless an ``AbsenceCheck`` is supplied -- the
    membership property, with its mandatory positive control showing the payload WAS reachable while
    the row was in the bank. A ``StructuralResult`` discharges the same requirement only when it was
    computed over rows that exist; ``certify_structural`` now refuses an empty row set outright, so a
    result that reaches here was obtained by perturbing something.
    """

    obj: int
    n_keys: int
    closure_size: int
    removed: Tuple[int, ...]
    covers_closure: bool
    record_certified: bool
    post_condition: Optional[bool] = None
    valid: bool = False
    void_reason: str = ""
    payload_absent: bool = False
    workload: str = ""
    residual_note: str = ""

    def summary(self) -> str:
        # The workload travels with the verdict. A closure is the resilience of a query set, so a
        # fact-level claim that does not name the query set it was proved over is not a claim.
        scope = f" over [{self.workload}]" if self.workload else ""
        if self.valid:
            post = "" if self.post_condition is None else ", post-condition verified"
            return (f"FACT CERTIFIED{scope}: object {self.obj} unreachable through all {self.n_keys} "
                    f"key(s); {len(self.removed)} record(s) removed covering a closure of "
                    f"{self.closure_size}{post}")
        return f"NOT A FACT CERTIFICATE{scope}: {self.void_reason}"


def certify_fact(record: "Certificate", closure: "FactClosure", removed: Sequence[int],
                 store_after: Optional[Any] = None, keys: Optional[Sequence[Tuple[int, int]]] = None,
                 structural: Optional["StructuralResult"] = None,
                 absence: Optional["AbsenceCheck"] = None,
                 residual_note: str = "") -> FactCertificate:
    """Compose a record-level certificate with a store-level closure.

    ``record``  the certificate obtained over the rows that were removed (``certify_deletion`` or
                ``certify_encoding``; the interface certificate is the one that generalises over
                queries, so prefer it).
    ``closure`` the fact closure measured on the store BEFORE the deletion.
    ``removed`` the cells actually removed. Must cover ``closure.records``.
    ``store_after`` optional: the store after the deletion, used to VERIFY rather than infer that no
                key still yields the object.
    ``structural`` optional: a ``certify_structural`` result over rows that are still in the bank.
    ``absence`` optional: a ``check_absence`` result. This is the one that applies when the deletion
                REMOVED the rows, and one of the two is required then, since a payload sweep over no
                rows is vacuous.

    Every way this can fail is named in ``void_reason`` rather than folded into a boolean, because a
    certificate whose failure modes are invisible is worse than no certificate.
    """
    removed_set = {int(x) for x in removed}
    needed = {int(x) for x in closure.records}
    covers = needed.issubset(removed_set)
    swept = record.n_rows > 0
    structural_ok = bool(structural is not None and structural.certified_structurally)
    absent_ok = bool(absence is not None and absence.certified_absent)
    record_ok = (bool(record.output_certified and record.activation_certified)
                 and (swept or structural_ok or absent_ok))

    reasons: List[str] = []
    if not swept and not (structural_ok or absent_ok):
        why = ""
        if absence is not None:
            why = f" ({absence.summary()})"
        reasons.append("the record-level sweep covered no rows, which certifies vacuously; when the "
                       "deletion removes the row from the bank the claim is a membership property and "
                       "needs a check_absence result with its positive control" + why)
    if closure.exhausted:
        reasons.append("the closure search was cut short, so the closure is unknown and no set can be "
                       "shown to cover it")
    if not covers:
        missing = sorted(needed - removed_set)
        reasons.append(f"the removal misses {len(missing)} record(s) of the closure: {missing[:8]}")
    if not (record.output_certified and record.activation_certified):
        which = "outputs" if not record.output_certified else "activations"
        reasons.append(f"the record-level certificate does not hold at the level of {which} "
                       f"({len(record.violations)} violation(s))")

    post: Optional[bool] = None
    if store_after is not None and keys is not None:
        from .closure import _query          # local import: audit must not depend on the resolver
        from .reference import ReferenceResolver
        from .world import UNKNOWN
        resolver = ReferenceResolver(store_after)
        still = [k for k in keys
                 if resolver.resolve(_query(k)).answer == closure.obj and closure.obj != UNKNOWN]
        post = not still
        if still:
            reasons.append(f"the store still answers {closure.obj} for {len(still)} key(s) after the "
                           f"removal, so the closure disagrees with the store: {still[:8]}")

    return FactCertificate(
        obj=int(closure.obj), n_keys=len(closure.keys), closure_size=closure.size,
        removed=tuple(sorted(removed_set)), covers_closure=covers, record_certified=record_ok,
        post_condition=post, valid=not reasons, void_reason="; ".join(reasons),
        payload_absent=absent_ok, workload=getattr(closure, "workload", ""),
        residual_note=residual_note)
