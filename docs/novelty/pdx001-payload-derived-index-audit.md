# PDX-001 — the E-000028 attack, taken out of this repository

Date: 2026-09-06
Status: **registration + instrument description**

## The experiment this discharges

`docs/so-novelty-2026-09-04.md` §5, "The smallest concrete result that would make this matter outside
the project", names one experiment:

> **Port E-000028 out of this repository.** Take one published external-memory or edit-memory system
> — GRACE's codebook, Larimar's episodic memory, or a soft-deleting HNSW RAG store — and run the same
> attack: after its own deletion or edit-removal operation, sweep candidates through whichever index
> term is a function of the removed payload, and report top-1 recovery against that system's own
> reported deletion metric. No training. Inference hours on existing checkpoints.

It was never run. E-000079 through E-000108 went elsewhere — thirty experiments, all internal to the
transport/lineage theory space, twenty-three consecutive kills.

What blocked the port is mechanical rather than intellectual: E-000028 is welded to `so/model.py`. It
needs the recorded E-000010 checkpoints, torch, this repository's bank layout and its `encode_bank`
signature. There is no way to point it at a store this programme did not write.

PDX-001 removes the weld. It does not finish §5 — see **What this is not**, below — but it converts
the attack from a one-off into an instrument that takes a store as an argument.

## The instrument

An abstract store exposes rows and, for a named deletion policy, the observable quantities an
adversary can read about a deleted row. The audit then does **both halves**, exhaustively over the
payload domain:

- **attack** — for every value the payload could hold, recompute the observables as if that were the
  payload; keep the candidates consistent with what the store actually exposes. A unique survivor
  means the payload is recovered.
- **certificate** — sweep the same domain and check whether any named observable moves at all.
  Nothing moves means no attack of this shape can exist, which is a stronger statement than "no
  attack was found".

The two are the same statement, so the run asserts they agree on every policy; a disagreement means
the instrument is broken rather than the store interesting.

Payload domain is 256 with an injective integer embedding, so the sweep is exact and complete. No
floats, no sampling, no seeds that matter.

## Registered policies

| policy | what it leaves observable | registered expectation |
|---|---|---|
| `value_gated_shred` | value gated; payload-derived reverse key ungated | leaks — E-000028's own defect |
| `hnsw_tombstone` | node marked deleted; adjacency list built from its own embedding retained | leaks |
| `codebook_key` | value cleared; codebook key derived from it retained | leaks |
| `revoke_unindex` | row removed from the addressable set | **control: must reach chance** |
| `gate_all_derived` | every payload-derived quantity gated | **control: must reach chance** |

## Validity floor

E-000019's discipline, applied here: the same attack runs against a **live** row first. If it cannot
recover a live payload, the instrument is weak and its at-chance readings on deleted rows mean
nothing. `run()` reports `attack_validity_floor_met` and `main()` exits non-zero if the floor fails.

There is a second vacuity guard in the tests: a live row must **never** be certified independent. An
instrument that certifies a payload sitting in plain view is not testing anything.

## The metric, and why top-1 is not it

E-000028 reports top-1 recovery, because in its case the sweep returns exactly one candidate. That
generalises badly. The attack returns a *set* of payloads consistent with what the store exposes, and
the true payload is always in that set — the tests check this. So the honest quantity is the
adversary's posterior on the true value, `1/|set|`, against a chance of `1/256`.

This was registered before the results were read and it turned out to matter: one policy scores 0.0000
on top-1 and leaks most of the payload anyway. Reporting only top-1 would have called that deletion
clean.

## What this is not

**No published system has been run here.** There are no checkpoints, no torch and no network in this
environment. The five policies are faithful reconstructions of *shapes* written from published
descriptions, not implementations of GRACE, Larimar or any particular HNSW index, and PDX-001 makes
no claim about any named system's actual deletion behaviour.

So §5 is not discharged. What has changed is that the instrument is now separable from this
repository, takes a store as an argument, and reports a metric that survives the generalisation.
Pointing it at a real index with its own reported deletion metric remains the experiment to run, and
it is now a small one.
