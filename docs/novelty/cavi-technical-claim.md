# CAVI-N — technical novelty claim candidate

Date: 2026-09-05
Status: **research-level technical novelty candidate; not a legal patentability opinion**

## Name

**CAVI-N — Neural-Consumption Continuity for Causally Attested Versioned Indirection**

`CAVI` remains the implementation umbrella. `CAVI-N` is the only currently live novelty subclaim.

## Decisive prior-art narrowing

Fresh review of the 19 July 2026 IETF Internet-Draft **PAMSPEC / Architecture and Data Model for Persistent Memory in Agentic Systems** materially falsifies the earlier broad CAVI novelty framing. PAMSPEC already specifies stable canonical Memory Objects, immutable versions, independently versioned Relationship Objects, an authoritative Persistent State Plane, non-authoritative derived representations/caches, scope enforcement during relationship traversal, expected-version concurrency, and propagation of deletion/redaction to derived state.

Therefore none of the following can carry the novelty claim, individually or as a loose composition:

- canonical memory identity;
- versioned objects;
- versioned aliases/relationships/pointers;
- authoritative-vs-derived-state separation;
- freshness/staleness metadata;
- cache invalidation as a general systems requirement;
- lifecycle state, tombstones, event ledgers, provenance or scope enforcement.

The candidate survives only if there is a specifically **neural execution** property not reducible to those established semantics.

## The narrow claim

A neural-memory system in which:

1. linguistic access paths resolve through pointer-only aliases to authoritative knowledge objects, but those object/pointer/version semantics are treated as prior-art infrastructure rather than novelty;
2. every successful resolve produces an **authority-lineage witness** bound to both the exact alias/reference incarnation and the exact referent/pod incarnation;
3. neural material derived from that resolution — including a materialized memory Bank, routing distribution, selected route, resolved payload vector, cached activation or serialized intermediate tensor — is **non-authoritative and cannot become a bearer capability** merely because it was produced while authority was valid;
4. whenever such derived material is later injected into or consumed by the model, the runtime revalidates its original live authority lineage adjacent to that actual neural injection/consumption site;
5. a reference relink, referent lifecycle transition, incarnation change, ABA restore or relevant scope/authorization transition invalidates old lineage even when the old referent remains live and a simple referent-version check would still pass;
6. invalid lineage fails closed to explicit `BYPASS`/`UNKNOWN` semantics, with out-of-scope `BYPASS` equal to the no-memory base-model path rather than a soft residual gate;
7. fresh current-generation state retains the intended memory capability, so safety is not purchased by disabling memory;
8. store/control-plane reachability and causal neural accessibility are measured independently for the same knowledge identity; J-space/J-lens is audit-only and is never optimized, routed through, or treated as authority.

## Candidate differentiator

The remaining distinction is **authority-lineage continuity across the tensor boundary**:

> An authorized memory read does not permanently authorize the neural tensors derived from that read.  If those tensors survive into a later model-consumption event, their originating reference+referent authority must still be live at that event; otherwise they are inert and execution reduces to the correct no-memory path.

This is narrower than saying "derived state is non-authoritative" — PAMSPEC already says that for embeddings and retrieval/ranking caches. CAVI-N only remains interesting if an opaque **neural** derivation can demonstrably resurrect stale knowledge after ordinary memory invalidation, and if preserving authoritative reference lineage through that derivation closes the attack where simpler version or commit-time checks do not.

## Prior-art exclusions

The following are explicitly prior art and are **not** claimed:

- external/editable memory — SERAC, WISE, Knowledge Externalization and related systems;
- semantic routing / scope classifiers — SERAC lineage, WISE, DKME, KEDAS/CRAFT;
- canonical records, stable object IDs, relationships, aliases and pointers — databases, PAMSPEC and systems indirection;
- canonical-vs-derived state — PAMSPEC and cache/index architectures;
- MVCC, expected versions, generations, freshness witnesses, ABA-safe handles and linearizability — database/concurrent systems;
- revocable validity and stale-memory exclusion — STALE, TEPA and related memory lifecycle work;
- freshness-aware cache reuse — FreshCache and caching literature;
- stale KV-cache/association eviction — SleepGate and cache-management work;
- transactional agent memory — MemTX;
- commit-time freshness authorization — Commit-Time Authorization;
- capabilities, epochs, HMACs, leases, locks or one-use tokens — security/systems primitives;
- cryptographic erasure / crypto-shredding — established key-destruction practice;
- J-space / Jacobian Lens — Anthropic;
- J-space accessibility auditing — J-Access and related mechanistic unlearning audits.

## Why the Symlink–Pod idea still matters technically

The symlink/pod structure is no longer itself a novelty claim. It is the experimental substrate that creates a clean **reference-vs-referent invalidation** test.

If alias `A` originally points to live pod `P` and is later relinked to live pod `Q`, then `P` can remain unchanged and current. A simple `pod_id/incarnation(P)` check still approves stale `A -> P` derived state. A full lineage witness bound to the **alias incarnation and binding plus pod incarnation** must reject it.

That counterexample is the key reason to keep aliases explicit: it distinguishes reference freshness from referent freshness and gives a falsifiable baseline against ordinary version checks.

## Role of J-space

J-space is **not the address bus**. E-000062 decisively falsified that thesis.

J-space/J-lens is retained only as an independent causal audit surface. CAVI-N does not train against the audit and does not use it for routing, authorization or gating. This separation is required because J-Access shows that directly optimizing an accessibility audit can induce audit evasion while making later recovery worse.

## Falsifiable invariants

### N1 — Reference + referent identity
Every authorized alias resolve yields a witness for the exact `(alias_id, alias_incarnation, pod_id, pod_incarnation)` tuple. A live old pod is insufficient after alias relink.

### N2 — Derived neural state is non-authoritative
A routing distribution, route selection, resolved payload or hidden activation captured under a valid witness cannot authorize its own later use.

### N3 — Consumption-time lineage validation
If derived neural state is reused at a later adapter/model injection site, its originating witness is validated immediately adjacent to that use under a race-safe authority boundary.

### N4 — ABA safety
DELETE/SHRED/RESTORE/ROLLBACK cannot make an old lineage witness valid merely by reusing a logical ID. Restored/recreated authority is a new incarnation.

### N5 — Reference mutation safety
Relinking or revoking an alias invalidates old state derived through that alias even when the old pod is still live and unchanged.

### N6 — Race safety
A lifecycle/reference mutation concurrent with inference must linearize before or after consumption. Cached pre-check/commit-time authorization cannot bridge an invalidation that occurs before the neural consumption point.

### N7 — Exact bypass
`BYPASS` follows the frozen base-model path with zero neural-memory injection.

### N8 — Fresh-current capability
After relink/update/restore, newly resolved current-generation state still provides the intended memory behavior.

### N9 — Independent audit
Store reachability and J-space/J-lens causal accessibility are evaluated independently against NEVER-memory controls. The audit is not optimized.

## Breakthrough experiment sequence

- **E-000066**: stale exported Bank replay — reproduced the vulnerability 20/20.
- **E-000068**: live-incarnation one-use capability — closed that replay class 5/5, but using known security primitives; not novelty.
- **E-000069**: authorized injection boundary — moved validation toward the effect boundary.
- **E-000070**: real trained symlink consumption attack — compares full alias+pod lineage with pod-only and cached-authorization baselines on actual adapter memory reads.
- **E-000071**: actual read-hook TOCTOU race — moves live validation to the neural consumption hook.
- **E-000072**: staged BYPASS/RESOLVE/UNKNOWN scope — performance/scope line after J-space routing was removed.
- **E-000073**: serialized **post-read hidden activation** replay with `bank=None`; tests whether an authorized neural activation becomes a stale bearer capability after relink/ABA/race.
- **E-000074**: serialized **routing distribution + resolved payload** replay with `bank=None`; tests the earlier neural-derivation boundary and directly compares commit-time, pod-only and full reference+referent lineage checks.
- **E-000063**: independent output/locality + NEVER-controlled J-lens workspace audit; it must remain an audit composition, not a routing objective.

## Research-level claim threshold

Do **not** call this a breakthrough from a single positive experiment. A defensible technical claim requires:

- a demonstrated stale-neural-derived-state resurrection attack, not a hypothetical threat;
- full lineage validation to close that attack while simpler pod/version and cached commit-time baselines fail;
- fresh-current memory capability preserved;
- >=3 independent seeds;
- >1 public backbone or materially different model setting where feasible;
- stale Bank, cached router, cached route, resolved payload, serialized hidden-state, old-alias, ABA, rollback/restore and concurrent race attacks;
- exact-bypass locality checks;
- canonical-pointer/external-memory/version-only/commit-time baselines;
- independent J-space/J-lens audit against NEVER-memory controls with no audit optimization;
- final prior-art search specifically for **lineage-carrying neural intermediate state with consumption-time revocation/revalidation**;
- evidence that the property is not simply ordinary cache invalidation or a direct restatement of PAMSPEC relationship/version semantics.

## Claim-killing tests

Withdraw CAVI-N as a novelty candidate if any of these holds:

1. stale neural derived state does not actually resurrect stale knowledge;
2. a pod/version equality check matches full alias+pod lineage on alias relink and ABA;
3. commit-time authorization is sufficient even when authority changes before neural consumption;
4. ordinary invalidation/version tags on caches provide the same guarantee without preserving the authoritative reference lineage into neural state;
5. the result is semantically covered by PAMSPEC's authoritative/derived separation and relationship traversal rules without a neural-specific correctness property;
6. exact BYPASS perturbs the base model or requires hidden memory injection;
7. the property disappears across seeds/backbones;
8. the independent causal audit cannot separate live-memory influence from query/base-model influence.

## Candidate paper claim — provisional

> **CAVI-N tests authority-lineage continuity for neural memory: memory-derived routing, payload and activation tensors remain revocable after materialization because their originating reference-and-referent witness is revalidated at later neural consumption.  This prevents stale neural intermediates from becoming bearer capabilities across alias relinks and object-incarnation changes, while invalid state collapses to exact no-memory execution and fresh current-generation memory remains usable.**

This sentence is **not yet supported as a novelty claim**. It becomes defensible only if E-000070/071/073/074, multi-seed/backbone replication, E-000063 audit composition and the final prior-art distinction all survive.