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
            for suffix, t in _flatten(output):
                if t.is_floating_point():
                    self.captured[f"{name}|{suffix}"] = t.detach().clone()
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
