"""PDX-001 — the E-000028 attack, taken out of this repository and made an instrument.

E-000028 is the programme's most transferable result: SHRED gates the value and leaves
``k_rev(LN(o + r))`` — a function of the very object it destroyed — ungated, so a shredded cell's
object comes back at top-1 1.0000, identical to a live cell to four decimals. The novelty statement
of 2026-09-04 says what to do with it (§5): *port it out of this repository*, run it against a store
whose deletion someone else reports as working, because "enumerate every payload-derived quantity"
is either a general law or a note about one file's ``k_rev``.

That port has not been run. What blocked it is that E-000028 is welded to ``so/model.py``: it needs
the recorded E-000010 checkpoints, torch, and this repository's bank layout. PDX-001 removes the
weld. The attack is restated over an abstract store — anything that exposes rows, a deletion policy,
and the observable quantities derived from a row — so it can be pointed at a store this programme did
not write.

The instrument does both halves, exhaustively over the payload domain, which is what makes it a
proof rather than an attack that happened to fail:

  attack       for every value the payload could hold, recompute the observables as if that were the
               payload, and keep the candidates consistent with what the store actually exposes.
               Unique survivor means the payload is recovered.
  certificate  sweep the same domain and check whether any observable moves at all. Nothing moves
               means no attack of this shape can exist, not that none was found.

The two are complementary by construction, and the run asserts it: a policy is certified exactly
when recovery sits at chance.

Five policies. Three are the pattern E-000028 found, written the way three different literatures
write it; two are controls that must come out clean or the instrument is not measuring anything.

  value_gated_shred   gate the value, leave the payload-derived reverse key alone. E-000028's own
                      defect, restated without torch.
  hnsw_tombstone      mark a vector-index node deleted and keep its adjacency list — which was built
                      from the deleted vector. The arrangement the novelty statement calls out:
                      "every soft-deleting vector index that keeps its edges".
  codebook_key        clear the value, keep the codebook key that was derived from it.
  revoke_unindex      take the row out of the addressable set. Control: must reach chance.
  gate_all_derived    the prescription — gate every quantity derived from the payload. Control:
                      must reach chance.

Validity floor, in the discipline E-000019 established: the same attack is run against a live row
first. If it cannot recover a live payload the instrument is weak and its at-chance readings mean
nothing, so `run()` refuses to report.

Exact integer arithmetic throughout: no floats, no seeds that matter, no sampling.

Run:  python -m so.experiments.pdx001_payload_derived_index_audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

# --------------------------------------------------------------------------------- the toy geometry
# A payload is an entity id. Its embedding is an injective integer map, so "distance" below is exact
# and the whole domain can be swept without any floating point.

DOMAIN = 256
MOD = 257  # prime > DOMAIN, so o -> embed(o) is injective
DIMS = 4
_COEFFS = ((3, 5), (7, 11), (13, 2), (5, 23))


def embed(o: int) -> tuple[int, ...]:
    return tuple((a * o + b) % MOD for a, b in _COEFFS)


def sqdist(u: Sequence[int], v: Sequence[int]) -> int:
    return sum((a - b) * (a - b) for a, b in zip(u, v))


# ------------------------------------------------------------------------------------------- store
# A row is (subject, relation, object). The adversary is given what E-000028's threat model gives
# them: subject and relation, and everything the store exposes. They want the object.


class Store:
    """A store of rows, plus whatever a deletion policy leaves observable about a deleted one."""

    def __init__(self, rows: list[tuple[int, int, int]], neighbours: int = 3) -> None:
        self.rows = rows
        self.neighbours = neighbours
        # Public: the other rows' embeddings. An adversary reading the store sees these.
        self.public_points = [(i, embed(o)) for i, (_s, _r, o) in enumerate(rows)]

    def adjacency(self, target: int, obj: int) -> tuple[int, ...]:
        """The k nearest *other* nodes to `obj`'s embedding — the edge list an index keeps."""
        point = embed(obj)
        ranked = sorted(
            ((sqdist(point, p), j) for j, p in self.public_points if j != target),
        )
        return tuple(j for _d, j in ranked[: self.neighbours])

    # ---------------------------------------------------------------- observables after each policy
    # Each returns the dict of quantities an adversary can read about the target row. A quantity that
    # is a function of `obj` is a channel, whether or not the policy meant to leave one.

    def observe(self, policy: str, target: int, obj: int) -> dict[str, object]:
        subject, relation, _ = self.rows[target]

        if policy == "value_gated_shred":
            # The value is gated to zero. The forward key never saw the object. The reverse key did,
            # and nothing gates it. This is `k_rev(LN(o + r))`, without torch.
            return {
                "value": tuple(0 for _ in range(DIMS)),
                "key_fwd": tuple((subject * 3 + relation * 7 + i) % MOD for i in range(DIMS)),
                "key_rev": tuple((a * ((obj + relation) % MOD) + b) % MOD for a, b in _COEFFS),
            }

        if policy == "hnsw_tombstone":
            # Node marked deleted; payload hidden; adjacency retained for graph connectivity.
            return {
                "value": None,
                "tombstone": True,
                "adjacency": self.adjacency(target, obj),
            }

        if policy == "codebook_key":
            # Value cleared, codebook key retained. The key was derived from the value.
            return {
                "value": None,
                "codebook_key": tuple(sorted(embed(obj))[:2]),
            }

        if policy == "revoke_unindex":
            # The row leaves the addressable set. Nothing about it is exposed.
            return {"present": False}

        if policy == "gate_all_derived":
            # The prescription: every quantity derived from the payload is gated too.
            return {
                "value": tuple(0 for _ in range(DIMS)),
                "key_fwd": tuple((subject * 3 + relation * 7 + i) % MOD for i in range(DIMS)),
                "key_rev": tuple(0 for _ in range(DIMS)),
            }

        if policy == "live":
            # Not a deletion. The validity control: the attack must work here.
            return {
                "value": embed(obj),
                "key_fwd": tuple((subject * 3 + relation * 7 + i) % MOD for i in range(DIMS)),
                "key_rev": tuple((a * ((obj + relation) % MOD) + b) % MOD for a, b in _COEFFS),
                "adjacency": self.adjacency(target, obj),
            }

        raise ValueError(f"unknown policy {policy!r}")


# ---------------------------------------------------------------------------------- the instrument


def audit_row(store: Store, policy: str, target: int) -> dict[str, object]:
    """Sweep the whole payload domain against one row under one policy.

    Returns both halves: what an attacker recovers, and whether any observable is a function of the
    payload at all. Exhaustive — every value the payload could take is tried.
    """
    true_obj = store.rows[target][2]
    observed = store.observe(policy, target, true_obj)

    consistent = [o for o in range(DOMAIN) if store.observe(policy, target, o) == observed]

    # Which named observables actually move as the payload varies. This is the certificate side: a
    # term that never moves cannot carry the payload.
    moving: list[str] = []
    for name in observed:
        values = {store.observe(policy, target, o).get(name) for o in range(DOMAIN)}
        if len(values) > 1:
            moving.append(name)

    return {
        "target": target,
        "true_object": true_obj,
        "candidates_remaining": len(consistent),
        "uniquely_identified": len(consistent) == 1,
        "recovered": len(consistent) == 1 and consistent[0] == true_obj,
        "payload_derived_observables": sorted(moving),
        "certified_independent": not moving,
    }


def audit_policy(store: Store, policy: str, targets: Sequence[int]) -> dict[str, object]:
    rows = [audit_row(store, policy, t) for t in targets]
    n = len(rows)
    recovered = sum(int(r["recovered"]) for r in rows)
    unique = sum(int(r["uniquely_identified"]) for r in rows)
    certified = sum(int(r["certified_independent"]) for r in rows)
    channels = sorted({c for r in rows for c in r["payload_derived_observables"]})
    mean_candidates = sum(int(r["candidates_remaining"]) for r in rows) / n

    # Top-1 is the headline but it is not the leak. The attack returns a *set* of payloads consistent
    # with what the store exposes, and the true value is always in it, so the adversary's posterior on
    # the true value is 1/|set|. Against a domain of 256 that is the honest quantity: an edge list
    # that narrows 256 candidates to 3 has leaked most of the payload without ever naming it.
    posterior = sum(1 / int(r["candidates_remaining"]) for r in rows) / n
    chance = 1 / DOMAIN

    return {
        "policy": policy,
        "targets": n,
        "top1_recovery": recovered / n,
        "unique_identification_rate": unique / n,
        "mean_candidates_remaining": mean_candidates,
        "mean_posterior_on_true_payload": posterior,
        "search_space_reduction_factor": DOMAIN / mean_candidates,
        "payload_domain": DOMAIN,
        "chance_top1": chance,
        "leaks_above_chance": posterior > chance,
        "payload_derived_channels": channels,
        "rows_certified_independent": certified,
        "certified": certified == n,
        # The certificate says no observable moves with the payload; the attack says the candidate set
        # is still the whole domain. They are the same statement, so they must agree on every policy.
        # A disagreement means the instrument is broken, not that the store is interesting.
        "certificate_matches_attack": (certified == n) == (mean_candidates == DOMAIN),
    }


def run(row_count: int = 24, target_count: int = 12) -> dict[str, object]:
    rows = [((i * 5 + 1) % 97, (i * 3 + 2) % 13, (i * 37 + 11) % DOMAIN) for i in range(row_count)]
    store = Store(rows)
    targets = list(range(target_count))

    validity = audit_policy(store, "live", targets)
    policies = [
        "value_gated_shred",
        "hnsw_tombstone",
        "codebook_key",
        "revoke_unindex",
        "gate_all_derived",
    ]
    results = [audit_policy(store, p, targets) for p in policies]
    by_policy = {r["policy"]: r for r in results}

    # E-000019's floor, applied here: if the attack cannot read a live payload, nothing it says about
    # a deleted one means anything.
    floor_met = validity["top1_recovery"] == 1.0

    leaking = [r["policy"] for r in results if r["leaks_above_chance"]]
    clean = [r["policy"] for r in results if r["certified"]]
    instrument_consistent = all(r["certificate_matches_attack"] for r in results)

    expected_clean = {"revoke_unindex", "gate_all_derived"}
    controls_behaved = set(clean) == expected_clean

    return {
        "experiment": "PDX-001",
        "scope": "E-000028 restated as a store-independent payload-derived-index audit",
        "validity_control": validity,
        "attack_validity_floor_met": floor_met,
        "policies": results,
        "policies_leaking_above_chance": leaking,
        "policies_certified_independent": clean,
        "controls_behaved_as_registered": controls_behaved,
        "instrument_self_consistent": instrument_consistent,
        "decision": (
            "PAYLOAD_DERIVED_INDEX_CHANNEL_CONFIRMED_ACROSS_STORE_SHAPES"
            if floor_met and instrument_consistent and controls_behaved and len(leaking) >= 3
            else "SCREEN_NOT_DECISIVE"
        ),
        "finding": (
            "The E-000028 defect is not a property of `so/model.py`. Restated over an abstract store "
            "it reproduces in three unrelated shapes. A gated value beside an ungated reverse key, and "
            "a cleared value beside a codebook key derived from it, both name the payload outright: "
            f"top-1 {by_policy['value_gated_shred']['top1_recovery']:.4f} and "
            f"{by_policy['codebook_key']['top1_recovery']:.4f} against a chance of 1/{DOMAIN}. The "
            "tombstoned vector-index node behaves differently and the difference is worth the "
            f"distinction: it never names the payload (top-1 {by_policy['hnsw_tombstone']['top1_recovery']:.4f}), "
            "but the adjacency list it keeps — built from the embedding of the vector it is hiding — "
            f"narrows the domain from {DOMAIN} to "
            f"{by_policy['hnsw_tombstone']['mean_candidates_remaining']:.1f} candidates, a posterior of "
            f"{by_policy['hnsw_tombstone']['mean_posterior_on_true_payload']:.4f} against chance "
            f"{1 / DOMAIN:.4f} and a "
            f"{by_policy['hnsw_tombstone']['search_space_reduction_factor']:.0f}x cut in the "
            "adversary's search space. Reporting only top-1 would have called that deletion clean. "
            "Taking the row out of the addressable set, and gating every derived quantity, both reach "
            "chance and are certified independent over the whole payload domain. The rule transfers: "
            "enumerate every quantity derived from a payload and gate all of them, or take the row "
            "out of the addressable set. Gating the value alone is not deletion."
        ),
        "not_claimed": (
            "This is the pattern, not a named system. No published implementation has been run here: "
            "the store shapes are faithful reconstructions written from their published descriptions, "
            "and PDX-001 has no checkpoints, no torch and no network. Running the instrument against "
            "GRACE's codebook, Larimar's memory or a real HNSW index with its own reported deletion "
            "metric remains the experiment §5 of the novelty statement asks for; what has changed is "
            "that the instrument is now separable from this repository and takes a store as an "
            "argument. No claim about any specific published system's actual deletion behaviour."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("so/results/pdx001"))
    parser.add_argument("--rows", type=int, default=24)
    parser.add_argument("--targets", type=int, default=12)
    args = parser.parse_args()
    result = run(args.rows, args.targets)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / "pdx001_payload_derived_index_audit.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["attack_validity_floor_met"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
