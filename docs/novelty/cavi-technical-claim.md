# CAVI — technical novelty claim candidate

Date: 2026-09-05
Status: **research-level technical novelty candidate; not a legal patentability opinion**

## Name

**CAVI — Causally Attested Versioned Indirection for Neural Memory**

## The narrow claim

A neural-memory system in which:

1. many linguistic access paths are **pointer-only aliases** to one canonical knowledge object;
2. the canonical object has a stable `pod_id` and a monotonic `incarnation`/generation;
3. previously materialized neural-memory tensors are treated as **data, not authority**;
4. immediately before a memory payload is consumed by a neural read/broadcast hook, the runtime must validate a live authorization witness bound to the exact `(pod_id, incarnation)`;
5. UPDATE/REVOKE/SHRED/EVICT/DELETE invalidate all prior incarnation witnesses and therefore invalidate both linguistic aliases **and already-exported neural-memory snapshots** without requiring those snapshots to be found and rewritten;
6. out-of-scope execution uses an explicit `BYPASS` state with exact no-memory injection;
7. lifecycle success is attested against the **same `(pod_id, incarnation)`** in two independent domains:
   - store/control-plane reachability closure; and
   - causal neural accessibility/broadcast audit (e.g. J-space/J-lens), which is never optimized as the deletion objective;
8. output, key-channel, stale-snapshot, reconstruction and adversarial elicitation attacks are additional non-authoritative checks, not substitutes for the two-domain certificate.

## Why this is narrower than known work

The following are explicitly prior art and are **not** claimed:

- external/editable memory — SERAC, WISE, Knowledge Externalization and related systems;
- semantic routing / scope classifiers — SERAC lineage, WISE, DKME, KEDAS/CRAFT;
- canonical records, aliases and pointers — database normalization/indirection and limited-memory systems;
- MVCC / generations / freshness witnesses — database systems and recent agent-memory/authorization work;
- transactional agent memory — MemTX;
- commit-time freshness authorization — Commit-Time Authorization;
- J-space / Jacobian Lens — Anthropic;
- J-space accessibility auditing — J-Access and related mechanistic unlearning audits.

The candidate distinction is the **cross-layer consumption contract**:

> A cached neural-memory materialization cannot authorize its own later consumption. Authority is revalidated atomically at the neural consumption boundary against the current canonical pod incarnation, and deletion is certified for that exact incarnation in both pointer reachability and an independent causal neural-broadcast domain.

## Why the Symlink–Pod idea matters

Without pointer-only aliases, a lifecycle operation can leave duplicated payload carriers and stale copies.
With canonical indirection, all linguistic aliases resolve to one identity. The remaining systems problem is that a previously exported `Bank`/tensor can outlive the source-of-truth transition. CAVI makes this stale neural material non-authoritative by construction.

This changes the deletion unit from "all phrases that can retrieve the fact" to:

`knowledge object = (pod_id, incarnation)`

and the consumption rule from:

`retrieved tensor -> neural injection`

to:

`retrieved tensor + live incarnation witness -> atomic validation -> neural injection OR fail closed`

## Role of J-space

J-space is **not the address bus**. E-000062 falsified that version of the thesis.

J-space is retained only as an independent causal audit surface because interventions there can causally alter downstream model behavior. CAVI does not train against the J-space audit. This separation is important because recent J-Access work shows that optimizing an audit can create audit evasion.

## Falsifiable invariants

A CAVI implementation must satisfy all of the following:

### I1 — Alias identity
Every live alias resolves to exactly one `(pod_id, current_incarnation)`; aliases carry no payload copy.

### I2 — Incarnation monotonicity
RESTORE/ROLLBACK/RESIGN cannot silently resurrect an old incarnation. They create a new authorized incarnation or explicitly activate a version under a new witness.

### I3 — Stale neural snapshots are inert
A `Bank`/tensor exported before UPDATE/REVOKE/SHRED/EVICT/DELETE cannot be consumed after the transition unless re-authorized against the new current incarnation.

### I4 — Atomic consumption validation
No time-of-check/time-of-use race exists between validation and the actual neural read hook. Revocation concurrent with inference must either happen-before consumption or force fail-closed behavior.

### I5 — Exact bypass
`BYPASS` produces the frozen base-model path with zero memory injection, not merely a small soft-gate weight.

### I6 — Same-identity dual attestation
The store certificate and causal neural-access audit both refer to the exact same `(pod_id, incarnation)` and both pass against a never-memory control.

### I7 — Audit independence
The causal audit is not used as a training loss or direct optimization objective.

## Breakthrough experiment sequence

- **E-000066**: stale exported Bank attack — already reproduced the vulnerability.
- **E-000068**: live incarnation authority control — established invalidation of stale capabilities, but using known security primitives.
- **E-000069**: authorized injection boundary — tests validation immediately before injection.
- **E-000070**: real trained symlink consumption attack — tests the contract on the neural adapter rather than only a store simulation.
- **E-000071**: read-hook TOCTOU race — tests atomicity under lifecycle changes concurrent with neural consumption.
- **E-000072**: staged scope state machine — tests exact BYPASS/RESOLVE/UNKNOWN before memory routing.
- **Next**: same-identity composed certificate with store closure + never-memory-controlled J-space causal audit + stale-snapshot/reconstruction attacks across multiple seeds/backbones.

## Research-level claim threshold

Do **not** call this a breakthrough from a single positive experiment.
A defensible technical claim requires:

- >=3 independent seeds;
- >1 public backbone where feasible;
- canonical-pointer baseline, ordinary external-memory baseline, semantic-router baseline and version-only baseline;
- stale snapshot, stale router cache, serialized bank replay, reconstruction and adversarial elicitation attacks;
- concurrent read/revoke race testing;
- exact-bypass locality checks;
- independent J-space/J-lens audit against never-memory controls;
- no direct optimization of the audit;
- a final prior-art search specifically for neural-memory consumption-time incarnation validation and same-object dual-domain attestation.

## Candidate paper claim

> **CAVI introduces version-qualified neural-memory consumption: linguistic aliases resolve to one canonical knowledge identity, but cached neural-memory materializations remain non-authoritative until atomically revalidated against the current pod incarnation at the point of neural consumption. A lifecycle transition therefore invalidates both future alias resolution and previously exported neural-memory snapshots, while deletion is independently attested for the same object incarnation in store reachability and the model's causal broadcast pathway.**

If the composed experiment battery survives, this is the claim to defend. If it fails, the claim is withdrawn rather than weakened into generic external memory, symlink, versioning or J-space language.