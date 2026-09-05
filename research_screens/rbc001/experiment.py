"""RBC-001: source-level countermodels; NOT trained-reader attack measurements.

Runs the pinned, unmodified CAVI authority and consumption guard with genuine
PyTorch hooks and an intentionally small numerical fixture. No transformer,
J-lens, language capability, leakage rate or novelty result is claimed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import hmac
from itertools import combinations
import json
import math
from pathlib import Path
import platform
import secrets
import sys
import threading
from types import ModuleType, SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from so import cavi
from so.cavi import CAVIAuthority, NeuralConsumptionGuard, ResolveWitness, RowManifest

EXPECTED_CAVI_SHA256 = "df9ca7ad21d7b783a56f2148009385a2ec42a6ac0f6ca9dc54102fcb728632a6"
MUTATIONS = (
    "alias_relink", "pod_update", "alias_revoke", "pod_shred",
    "alias_revoke_restore", "pod_delete_recreate", "unrelated_pod_update",
)
PHASES = ("before_guard", "after_guard", "block_body")
MODES = ("original", "late_check_comparator")


def verify_source() -> dict[str, Any]:
    path = Path(cavi.__file__)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_CAVI_SHA256:
        raise RuntimeError(f"CAVI source changed: {digest}; register a new source audit")
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    return {"sha256": digest, "git_blob": blob, "bytes": len(raw)}


def authority_fixture() -> tuple[CAVIAuthority, ResolveWitness, RowManifest]:
    auth = CAVIAuthority()
    for pid in (1, 2, 3):
        auth.create_pod(pid)
    auth.create_alias(10, 1)
    w = auth.witness(10)
    manifest = RowManifest(
        np.array([w.pod_id], dtype=np.int64),
        np.array([w.pod_incarnation], dtype=np.int64),
        np.array([w.alias_id], dtype=np.int64),
        np.array([w.alias_incarnation], dtype=np.int64),
    )
    return auth, w, manifest


def mutate(auth: CAVIAuthority, name: str) -> None:
    if name == "alias_relink":
        auth.relink_alias(10, 2)
    elif name == "pod_update":
        auth.update_pod(1)
    elif name == "alias_revoke":
        auth.revoke_alias(10)
    elif name == "pod_shred":
        auth.shred_pod(1)
    elif name == "alias_revoke_restore":
        auth.revoke_alias(10)
        auth.restore_alias(10)
    elif name == "pod_delete_recreate":
        auth.delete_pod(1)
        auth.recreate_pod_same_id(1)
    elif name == "unrelated_pod_update":
        auth.update_pod(3)
    else:
        raise ValueError(name)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.callback: Callable[[], None] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.callback is not None:
            self.callback()
        return x.clone()


class _NumericReadFixture:
    """One row and actual hook ordering, not a stand-in capability benchmark."""
    def __init__(self, auth, witness, manifest, seed: int, mode: str):
        self.auth, self.witness, self.manifest = auth, witness, manifest
        self.block = _Block()
        self.lm = SimpleNamespace(blocks=[self.block])
        self.cfg = SimpleNamespace(read_layers=[0])
        self._ctx = {"allowed": torch.ones(1, dtype=torch.bool)}
        self.mode = mode
        self.trace: list[str] = []
        self.observed: dict[str, Any] = {}
        self.payload = torch.randn(4, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
        self.read_handle = self.block.register_forward_hook(self._read)

    def _read(self, module, inputs, output):
        # This is the numerical fixture; the guard implementation is unmodified.
        # The comparator has no intervening callback between final check and use.
        with self.auth.lock:
            mask = bool(self._ctx["allowed"][0])
            valid = self.auth.validate_witness(self.witness)
            allowed = mask
            if self.mode == "late_check_comparator":
                allowed = mask and bool(self.auth.row_mask(self.manifest)[0])
            self.trace.append("consume")
            self.observed = {
                "precomputed_mask_at_consume": mask,
                "authoritative_valid_at_consume": valid,
                "consumed": allowed,
                "stale_consumed": bool(allowed and not valid),
            }
            return output + self.payload if allowed else output


def _new_guard(fixture: _NumericReadFixture) -> NeuralConsumptionGuard:
    # Only adapt transformer_blocks to the deliberately tiny module fixture.
    # No source or method of CAVIAuthority/NeuralConsumptionGuard is replaced.
    module = ModuleType("so.llm_adapter")
    module.transformer_blocks = lambda lm: lm.blocks
    with patch.dict(sys.modules, {"so.llm_adapter": module}):
        return NeuralConsumptionGuard(
            fixture, lambda: fixture.auth.row_mask(fixture.manifest), lock=fixture.auth.lock
        )


def schedule_case(seed: int, mutation: str, phase: str, mode: str) -> dict[str, Any]:
    if phase not in PHASES or mode not in MODES or mutation not in MUTATIONS:
        raise ValueError((mutation, phase, mode))
    auth, witness, manifest = authority_fixture()
    fixture = _NumericReadFixture(auth, witness, manifest, seed, mode)
    handles = []
    guard = None

    def callback(*_args):
        fixture.trace.append("mutation_requested")
        mutate(auth, mutation)
        fixture.trace.append("mutation_committed")

    try:
        if phase == "before_guard":
            handles.append(fixture.block.register_forward_pre_hook(callback))
        guard = _new_guard(fixture)
        if phase == "after_guard":
            handles.append(fixture.block.register_forward_pre_hook(callback))
        if phase == "block_body":
            fixture.block.callback = callback
        output = fixture.block(torch.zeros(4, dtype=torch.float64))
    finally:
        for handle in handles:
            handle.remove()
        if guard is not None:
            guard.close()
        fixture.read_handle.remove()
    relevant = mutation != "unrelated_pod_update"
    expected_stale = relevant and phase != "before_guard" and mode == "original"
    return {
        "seed": seed, "mutation": mutation, "phase": phase, "mode": mode,
        **fixture.observed, "trace": fixture.trace,
        "output_maxabs": float(output.abs().max()),
        "expected_stale_counterexample": expected_stale,
        "matches_registered_prediction": fixture.observed["stale_consumed"] == expected_stale,
    }


def other_thread_control(seed: int) -> dict[str, Any]:
    auth, witness, manifest = authority_fixture()
    fixture = _NumericReadFixture(auth, witness, manifest, seed, "original")
    attempted, done = threading.Event(), threading.Event()
    box: dict[str, Any] = {}

    def worker():
        # A nonblocking failed acquire proves this thread met the held lock.
        acquired = auth.lock.acquire(blocking=False)
        box["competing_acquire_blocked"] = not acquired
        if acquired:
            auth.lock.release()
        attempted.set()
        try:
            auth.update_pod(1)
            fixture.trace.append("other_thread_mutation_committed")
        except BaseException as exc:
            box["worker_error"] = repr(exc)
        finally:
            done.set()

    thread = threading.Thread(target=worker, daemon=True)
    guard = _new_guard(fixture)

    def launch(module, inputs):
        thread.start()
        if not attempted.wait(5):
            raise RuntimeError("worker did not attempt the lock")
        box["mutation_done_before_consume"] = done.is_set()

    handle = fixture.block.register_forward_pre_hook(launch)
    try:
        fixture.block(torch.zeros(4, dtype=torch.float64))
    finally:
        handle.remove()
        guard.close()
        fixture.read_handle.remove()
    thread.join(timeout=5)
    if thread.is_alive() or "worker_error" in box:
        raise RuntimeError(f"worker did not finish cleanly: {box}")
    return {
        "seed": seed, **box, **fixture.observed,
        "old_witness_invalid_after_join": not auth.validate_witness(witness),
        "trace": fixture.trace,
    }


def _envelope_bytes(payload: torch.Tensor, witness: ResolveWitness) -> bytes:
    payload = payload.detach().cpu().contiguous()
    header = json.dumps({
        "domain": "RBC-001-test-envelope-v1",
        "shape": list(payload.shape), "dtype": str(payload.dtype),
        "witness": asdict(witness),
    }, sort_keys=True, separators=(",", ":")).encode()
    return len(header).to_bytes(8, "big") + header + payload.numpy().tobytes()


def binding_case(seed: int) -> dict[str, Any]:
    auth, old_witness, _ = authority_fixture()
    old = torch.randn(4, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    fresh = old + 1.0
    key = secrets.token_bytes(32)  # private to the reference issuer, never serialized

    def seal(payload, witness):
        return hmac.digest(key, _envelope_bytes(payload, witness), "sha256")

    def authenticated_accept(payload, witness, signature):
        bound = hmac.compare_digest(seal(payload, witness), signature)
        return bound and auth.validate_witness(witness)

    old_signature = seal(old, old_witness)
    auth.relink_alias(10, 2)
    fresh_witness = auth.witness(10)
    fresh_signature = seal(fresh, fresh_witness)
    # The adversary here only substitutes externally supplied envelope fields;
    # it does not alter authority, verifier code or the private issuer key.
    substituted_signature_accepts = authenticated_accept(old, fresh_witness, fresh_signature)
    wrong_lineage_but_issuer_signed = seal(old, fresh_witness)
    return {
        "seed": seed,
        "old_witness_rejected": not auth.validate_witness(old_witness),
        "current_witness_with_old_payload_accepted_by_witness_predicate": auth.validate_witness(fresh_witness),
        "old_and_fresh_payload_differ": not torch.equal(old, fresh),
        "fresh_bound_envelope_accepted": authenticated_accept(fresh, fresh_witness, fresh_signature),
        "stale_authentic_envelope_rejected": not authenticated_accept(old, old_witness, old_signature),
        "payload_substitution_rejected_by_binding": not substituted_signature_accepts,
        "witness_substitution_rejected_by_binding": not authenticated_accept(old, fresh_witness, old_signature),
        "issuer_signed_wrong_lineage_still_accepted": authenticated_accept(old, fresh_witness, wrong_lineage_but_issuer_signed),
        "scope": "untrusted replaceable cache envelope; not a trusted closure or service exploit",
        "baseline_limit": "authentication binds a producer claim; it does not certify completeness/correctness of that claim",
    }


def observation_collision_case(seed: int) -> dict[str, Any]:
    auth, old_witness, _ = authority_fixture()
    gen = torch.Generator().manual_seed(1000 + seed)
    payload = torch.randn(4, generator=gen, dtype=torch.float64)
    w = torch.randn(6, 4, generator=gen, dtype=torch.float64)
    b = torch.randn(6, generator=gen, dtype=torch.float64)
    k = torch.randn(4, 6, generator=gen, dtype=torch.float64)
    v = torch.randn(4, 6, generator=gen, dtype=torch.float64)
    p = torch.randn(6, 6, generator=gen, dtype=torch.float64)
    u = torch.randn(8, 6, generator=gen, dtype=torch.float64)

    def observe():
        h = torch.tanh(w @ payload + b)
        def suffix(z):
            return u @ (torch.sigmoid(p @ z) * z)
        return {
            "hidden": h, "key": k @ h, "value": v @ h,
            "logits": suffix(h),
            "jacobian": torch.autograd.functional.jacobian(suffix, h),
        }

    old = observe()
    # A new canonical generation with exactly the same numerical content.
    auth.update_pod(1)
    fresh_witness = auth.witness(10)
    fresh = observe()
    equality = {name: old[name].contiguous().numpy().tobytes() == fresh[name].contiguous().numpy().tobytes() for name in old}
    digest = hashlib.sha256(b"".join(old[n].contiguous().numpy().tobytes() for n in sorted(old))).hexdigest()
    return {
        "seed": seed, "old_witness_valid": auth.validate_witness(old_witness),
        "fresh_witness_valid": auth.validate_witness(fresh_witness),
        "same_numerical_observations": equality,
        "all_observations_byte_equal": all(equality.values()),
        "observation_sha256": digest,
        "maxabs": {name: float((old[name] - fresh[name]).abs().max()) for name in old},
        "scope": "history-free numerical observer; not an actual trained J-lens run",
    }


def membership_bound(n: int, k: int) -> dict[str, Any]:
    if not 0 <= k <= n:
        raise ValueError("require 0 <= k <= n")
    histories = math.comb(n, k)
    return {"N": n, "k": k, "possible_histories": str(histories),
            "minimum_exact_auxiliary_bits": (histories - 1).bit_length()}


def small_membership_collision() -> dict[str, Any]:
    n, k = 8, 3
    histories = list(combinations(range(n), k))
    value = torch.tensor([0.25, -0.5, 1.0], dtype=torch.float64)
    outputs = [torch.stack([value for _ in history]).sum(0) for history in histories]
    signatures = {tuple(int(i in history) for i in range(n)) for history in histories}
    return {**membership_bound(n, k),
            "all_outputs_byte_equal": all(outputs[0].numpy().tobytes() == x.numpy().tobytes() for x in outputs),
            "distinct_required_membership_vectors": len(signatures)}


def run(seeds: list[int]) -> dict[str, Any]:
    rows = [schedule_case(s, m, p, mode) for s in seeds
            for m in MUTATIONS for p in PHASES for mode in MODES]
    threads = [other_thread_control(s) for s in seeds]
    binding = [binding_case(s) for s in seeds]
    collisions = [observation_collision_case(s) for s in seeds]
    return {
        "experiment": "RBC-001", "candidate_only": True,
        "evidence_type": "source-level synchronization and identifiability countermodels",
        "source": verify_source(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform()},
        "numeric_fixture_seeds": seeds,
        "schedule_rows": rows, "other_thread_controls": threads,
        "binding_rows": binding, "observation_collision_rows": collisions,
        "membership_collision": small_membership_collision(),
        "scale_counting_bound": membership_bound(1_000_000, 8),
        "summary": {
            "schedule_cases": len(rows),
            "original_guard_stale_consumptions": sum(r["stale_consumed"] for r in rows if r["mode"] == "original"),
            "late_check_comparator_stale_consumptions": sum(r["stale_consumed"] for r in rows if r["mode"] == "late_check_comparator"),
            "all_schedule_predictions_matched": all(r["matches_registered_prediction"] for r in rows),
            "all_other_thread_controls_passed": all(r["competing_acquire_blocked"] and not r["mutation_done_before_consume"] and not r["stale_consumed"] and r["old_witness_invalid_after_join"] for r in threads),
            "all_observation_collisions_exact": all(r["all_observations_byte_equal"] and not r["old_witness_valid"] and r["fresh_witness_valid"] for r in collisions),
        },
        "trained_reader_gate": "NOT_RUN: none of these fixture results count as a learned-reader attack result",
        "novelty": "NONE: RLock semantics, effect-boundary checks, authenticated envelopes and indistinguishability are standard boundaries",
        "utility_gates": "UNMEASURED: no language accuracy, leakage, UNKNOWN, speedup, overhead, memory parity or J-lens certification claim",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--results-dir", type=Path, default=Path("so/results/rbc001"))
    args = ap.parse_args()
    torch.set_num_threads(1)
    result = run(args.seeds)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "rbc001-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    if not all(result["summary"][k] for k in (
        "all_schedule_predictions_matched", "all_other_thread_controls_passed", "all_observation_collisions_exact"
    )):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
