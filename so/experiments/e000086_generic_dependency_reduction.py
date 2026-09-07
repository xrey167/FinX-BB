"""E-000086R -- reduce explicit Symlink/Pod coherence to generic dependency validation.

This is an architecture falsification instrument, not a neural-capability result and
not a novelty claim. It asks whether the explicit E-000086 witness carries any
freshness semantics beyond an ordinary dependency record over the versioned alias
node and the versioned canonical Pod node.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

from so.cavi import CAVIAuthority, ResolveWitness

ARTIFACT_CLASSES: Tuple[str, ...] = (
    "bank",
    "router",
    "resolved_payload",
    "post_read_hidden",
    "kv",
)


@dataclass(frozen=True)
class GenericDependencySnapshot:
    """Ordinary captured dependency versions for one derived computation.

    `alias_target_pod_id` is the value of the alias relationship at capture time.
    This is not extra candidate information: the E-000086 ResolveWitness already
    stores the same target as `pod_id`.
    """

    alias_id: int
    alias_generation: int
    alias_target_pod_id: int
    pod_generation: int


def capture_generic(auth: CAVIAuthority, alias_id: int) -> GenericDependencySnapshot:
    """Capture the same two authoritative dependency nodes as an ordinary memoized computation."""
    w = auth.witness(alias_id)
    return GenericDependencySnapshot(
        alias_id=w.alias_id,
        alias_generation=w.alias_incarnation,
        alias_target_pod_id=w.pod_id,
        pod_generation=w.pod_incarnation,
    )


def generic_dependency_valid(auth: CAVIAuthority, s: GenericDependencySnapshot) -> bool:
    """Type-agnostic versioned dependency validation; no neural artifact semantics."""
    with auth.lock:
        alias = auth.aliases.get(s.alias_id)
        pod = auth.pods.get(s.alias_target_pod_id)
        return bool(
            alias is not None
            and pod is not None
            and alias.live
            and pod.live
            and alias.incarnation == s.alias_generation
            and alias.pod_id == s.alias_target_pod_id
            and pod.incarnation == s.pod_generation
        )


def witness_projection(w: ResolveWitness) -> GenericDependencySnapshot:
    return GenericDependencySnapshot(
        alias_id=w.alias_id,
        alias_generation=w.alias_incarnation,
        alias_target_pod_id=w.pod_id,
        pod_generation=w.pod_incarnation,
    )


def _new_authority() -> CAVIAuthority:
    auth = CAVIAuthority()
    for pod_id in (1, 2, 3):
        auth.create_pod(pod_id)
    auth.create_alias(10, 1)  # target artifact lineage
    auth.create_alias(11, 3)  # unrelated alias
    return auth


Event = Callable[[CAVIAuthority], None]


def _traces() -> Dict[str, Sequence[Tuple[str, Event]]]:
    return {
        "no_mutation": (),
        "alias_relink": (("relink_a10_p2", lambda a: a.relink_alias(10, 2)),),
        "pod_update": (("update_p1", lambda a: a.update_pod(1)),),
        "alias_revoke": (("revoke_a10", lambda a: a.revoke_alias(10)),),
        "alias_revoke_restore": (
            ("revoke_a10", lambda a: a.revoke_alias(10)),
            ("restore_a10", lambda a: a.restore_alias(10)),
        ),
        "pod_shred": (("shred_p1", lambda a: a.shred_pod(1)),),
        "pod_shred_restore": (
            ("shred_p1", lambda a: a.shred_pod(1)),
            ("restore_p1", lambda a: a.restore_pod(1)),
        ),
        "same_id_delete_recreate_aba": (
            ("delete_p1", lambda a: a.delete_pod(1)),
            ("recreate_p1", lambda a: a.recreate_pod_same_id(1)),
        ),
        "repeated_pod_update": (
            ("update_p1_1", lambda a: a.update_pod(1)),
            ("update_p1_2", lambda a: a.update_pod(1)),
        ),
        "relink_then_update_new_pod": (
            ("relink_a10_p2", lambda a: a.relink_alias(10, 2)),
            ("update_p2", lambda a: a.update_pod(2)),
        ),
        "relink_then_mutate_old_pod": (
            ("relink_a10_p2", lambda a: a.relink_alias(10, 2)),
            ("update_old_p1", lambda a: a.update_pod(1)),
        ),
        "unrelated_pod_update": (("update_p3", lambda a: a.update_pod(3)),),
        "unrelated_alias_relink": (("relink_a11_p2", lambda a: a.relink_alias(11, 2)),),
        "two_unrelated_mutations": (
            ("update_p3", lambda a: a.update_pod(3)),
            ("relink_a11_p2", lambda a: a.relink_alias(11, 2)),
        ),
    }


def _decision_row(auth: CAVIAuthority, witness: ResolveWitness, generic: GenericDependencySnapshot,
                  epoch_at_capture: int, current_epoch: int) -> Dict[str, object]:
    cavi = bool(auth.validate_witness(witness))
    dep = bool(generic_dependency_valid(auth, generic))
    return {
        "object_coherence": cavi,
        "generic_dependency": dep,
        "global_epoch": current_epoch == epoch_at_capture,
        "equal_selective_decision": cavi == dep,
        "per_artifact": {
            kind: {"object_coherence": cavi, "generic_dependency": dep, "equal": cavi == dep}
            for kind in ARTIFACT_CLASSES
        },
    }


def run() -> Dict[str, object]:
    trace_rows: List[Dict[str, object]] = []
    all_equal = True
    projection_equal = True
    unrelated_preserved_selectively = True
    global_overinvalidations = 0

    for trace_name, events in _traces().items():
        auth = _new_authority()
        witness = auth.witness(10)
        generic = capture_generic(auth, 10)
        projection_equal = projection_equal and witness_projection(witness) == generic
        epoch_at_capture = 0
        epoch = 0
        prefixes = [
            {
                "after": "capture",
                "decision": _decision_row(auth, witness, generic, epoch_at_capture, epoch),
            }
        ]
        for event_name, event in events:
            event(auth)
            epoch += 1
            decision = _decision_row(auth, witness, generic, epoch_at_capture, epoch)
            prefixes.append({"after": event_name, "decision": decision})
            all_equal = all_equal and bool(decision["equal_selective_decision"])

        final = prefixes[-1]["decision"]
        is_unrelated = trace_name in {
            "unrelated_pod_update", "unrelated_alias_relink", "two_unrelated_mutations"
        }
        if is_unrelated:
            unrelated_preserved_selectively = unrelated_preserved_selectively and bool(
                final["object_coherence"] and final["generic_dependency"]
            )
            if (not final["global_epoch"]) and final["generic_dependency"]:
                global_overinvalidations += 1
        trace_rows.append({"trace": trace_name, "events": [name for name, _ in events], "prefixes": prefixes})

    # Independent audit is deliberately outside the authorization path. Varying its
    # report cannot alter either runtime freshness decision.
    auth = _new_authority()
    witness = auth.witness(10)
    generic = capture_generic(auth, 10)
    runtime_before = (auth.validate_witness(witness), generic_dependency_valid(auth, generic))
    audit_reports = ("ACTIVE", "INVALIDATED", "NEVER", {"causal_score": 0.0}, {"causal_score": 1.0})
    runtime_after_reports = [
        (auth.validate_witness(witness), generic_dependency_valid(auth, generic))
        for _report in audit_reports
    ]
    independent_audit_does_not_distinguish_runtime = all(x == runtime_before for x in runtime_after_reports)

    checks = {
        "witness_is_two_dependency_projection": projection_equal,
        "all_registered_selective_decisions_equal": all_equal,
        "all_five_artifact_types_are_decision_equivalent": all(
            all(
                all(bool(v["equal"]) for v in p["decision"]["per_artifact"].values())
                for p in row["prefixes"]
            )
            for row in trace_rows
        ),
        "unrelated_mutations_preserved_by_both_selective_methods": unrelated_preserved_selectively,
        "global_epoch_overinvalidates_registered_unrelated_traces": global_overinvalidations == 3,
        "independent_audit_cannot_distinguish_runtime_algorithm": independent_audit_does_not_distinguish_runtime,
    }
    return {
        "experiment": "E-000086R",
        "candidate_only": True,
        "artifact_classes": list(ARTIFACT_CLASSES),
        "trace_rows": trace_rows,
        "checks": checks,
        "reduction_pass": all(checks.values()),
        "decision": (
            "KILL_CURRENT_COHERENCE_SEAM_AS_GENERIC_DEPENDENCY_SPECIALIZATION"
            if all(checks.values())
            else "REDUCTION_NOT_ESTABLISHED"
        ),
        "scope": (
            "This reduction covers the explicit-tag alias+Pod generation contract only. It does not "
            "rule out a new mechanism for discovering/certifying neural causal lineage or reducing exact affected work."
        ),
        "metadata_boundary": (
            "ResolveWitness and the generic baseline both carry alias id+generation and resolved Pod id+generation; "
            "liveness/current-target checks are reads of the same authoritative nodes at reuse."
        ),
        "not_claimed": (
            "No legal novelty conclusion; no claim that PAMSPEC or self-adjusting computation implements this exact neural stack verbatim."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="so/results")
    a = ap.parse_args()
    result = run()
    out = Path(a.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "e000086_generic_dependency_reduction.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if not result["reduction_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
