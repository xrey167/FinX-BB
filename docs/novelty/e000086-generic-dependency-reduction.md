# E-000086R — decisive reduction of the explicit coherence-domain seam

Date: 2026-09-05
Status: **DECISIVE DIRECTION CHANGE / current explicit-tag coherence seam rejected as a major-invention candidate**

This document records an architecture-level falsification. It does **not** claim that every possible Symlink–Pod–J-Space architecture is old, and it does not assert legal novelty or patent clearance.

## Result

The current E-000086 runtime witness is:

`ResolveWitness(alias_id, alias_incarnation, pod_id, pod_incarnation)`

and `CAVIAuthority.validate_witness` authorizes reuse iff:

1. the captured alias still exists and is live;
2. the captured Pod still exists and is live;
3. the alias incarnation is unchanged;
4. the alias still targets the captured Pod id;
5. the Pod incarnation is unchanged.

That decision is exactly reproduced by an ordinary type-agnostic dependency record over two authoritative nodes:

- alias dependency: `(alias_id, alias_incarnation, current_target_pod_id, live)`;
- Pod dependency: `(pod_id, pod_incarnation, live)`.

No additional runtime freshness information is supplied by calling the pair a canonical knowledge-object generation or a coherence domain. A generic memoized computation that records those two dependencies and validates them before reuse makes the same selective reuse decision.

The equivalence is artifact-type independent. Applying the same decision to Bank state, routing state, resolved payload, post-read hidden activation and KV does not change the algorithm; it applies one generic dependency validator to five derived-value types.

## Registered trace comparison

The preregistered reduction covers:

- no mutation;
- alias RELINK;
- Pod UPDATE;
- alias REVOKE;
- REVOKE -> RESTORE;
- Pod SHRED;
- SHRED -> RESTORE;
- same-id delete/recreate ABA;
- repeated Pod updates;
- RELINK then update the new Pod;
- RELINK then mutate the old Pod;
- unrelated Pod update;
- unrelated alias RELINK;
- two unrelated mutations.

For every registered state transition, the E-000086 object-coherence predicate and the generic two-node dependency predicate are algebraically the same Boolean condition. Target/source mutations reject both; unrelated mutations preserve both. A global epoch is weaker because it rejects unrelated artifacts, but it is not the strongest guarantee-matched baseline.

A local exact reproduction of the current authority semantics produced identical decision sequences for every registered trace. The repository also contains a pinned executable reduction and eight preserved tests. GitHub Actions run `33981787028` was submitted for independent CI reproduction; its completion is not asserted in this record until the artifact exists.

## Why this kills the current seam

The programme's kill rule explicitly says to reject the candidate if it reduces to ordinary cache coherence, self-adjusting computation, PAMSPEC, or generic dependency graphs. The current explicit-tag design does.

Self-adjusting computation already combines memoization with dependency-aware change propagation under mutations. PAMSPEC (Internet-Draft, 19 July 2026) already separates versioned canonical Memory Objects from non-authoritative derived state and states that Derived Indexes become stale after a canonical Update until rebuilt. Invalidation Contracts (31 August 2026) already uses version stamps for selective invalidation while retaining unaffected cached entries. These works do not implement this repository's neural stack verbatim; they occupy the generic mechanism that E-000086 currently instantiates.

References:
- https://arxiv.org/abs/1106.0478
- https://datatracker.ietf.org/doc/draft-infantado-agent-memory-architecture/
- https://arxiv.org/abs/2609.00243

## Why heterogeneous neural artifacts do not rescue it

A dependency validator does not need to understand the derived value. The same captured source-version tuple can guard a serialized Bank row, a route, a payload, a hidden activation, or a KV tensor. Therefore cross-type coverage is useful engineering evidence but not a new coherence mechanism by itself.

Likewise, checking the dependency at the neural consumption boundary closes a TOCTOU window but remains effect-boundary validation. Version counters, ABA protection, exact bypass, source lineage labels, capabilities, symlinks and selective recomputation are independently occupied/prior-art controls and receive zero novelty credit.

## Why J-space does not rescue it

The programme correctly keeps J-space/J-lens independent of runtime authorization. Under that rule, changing an audit report cannot change the reuse predicate. Independent causal attestation can strengthen evidence that a lifecycle effect occurred, but it cannot distinguish the runtime coherence algorithm from generic dependency validation.

If J-space were promoted into the runtime dependency-discovery/authorization mechanism, that would be a **different candidate** and would need a new preregistration and strongest-baseline comparison. It must then demonstrate something generic dependency tracking cannot already obtain.

## Consequence for active E-000083/084/085 lanes

E-000083 has produced a seed-0 strict-capability configuration at `consistency=0.2`, `alt_supervision=0.5`, with all four held-out real-symlink templates at or above 0.95. This remains valuable capability evidence.

E-000084 had a protocol defect: its original positive-interpretation gate checked only template 9. It has been corrected so the exact trained reader must pass **every** held-out template before the CAVI attack can be interpreted. Corrected seed-1/2 and promoted seed-0 runs have been submitted. E-000085 already uses the every-template gate.

Those lanes remain useful to characterize stale Bank/router/payload/hidden/KV failure modes and to validate engineering correctness. However, even a clean heterogeneous closure result cannot revive the rejected major-invention claim if the closure mechanism remains the same explicit alias+Pod dependency tuple.

## New invention target

The programme should now stop spending novelty credit on explicit generation/tag propagation. A surviving candidate must add a mechanism that does **not** collapse to ordinary dependency capture/change propagation. The most defensible remaining directions are:

1. **Neural causal-lineage discovery/certification:** derive the minimal or provably complete causal source set for neural-derived state without requiring the dependency graph to be supplied explicitly, and demonstrate independent correctness against fresh rebuilds/J-space controls.
2. **Exact affected-work compression:** create a model mechanism whose source-conditioned dependency domain is materially smaller than ordinary replay/change propagation and beats the strongest guarantee-matched baseline under the same total memory budget.
3. **Joint nonlinear repair with certified validity:** only if it has a new compact representation/validity mechanism that survives CRR-001/NIC-001 and cannot be reduced to fixed-basis repair, ordinary affine-region caching, generic dependency repair, or exhaustive joint-context tables.

Any new candidate still must meet the programme gates: >=10x mutation-to-ready over the strongest guarantee-matched baseline at scale, <=5% inference overhead, matched memory, >=95% fresh/unseen-paraphrase and lifecycle reading, <=2% deleted-object leakage, >=90% UNKNOWN in declared missing-key scope, exact bypass or <=0.05 nats generic divergence, >=3 seeds and preferably >1 backbone.

## Preserved evidence

- preregistration: `docs/novelty/e000086-generic-dependency-reduction-preregister.md`
- executable: `so/experiments/e000086_generic_dependency_reduction.py`
- tests: `so/tests/test_e000086_generic_dependency_reduction.py`
- workflow: `.github/workflows/e000086-generic-dependency-reduction.yml`
- CI run submitted: `33981787028`

**Decision:** kill the current “canonical knowledge-object generation as a heterogeneous coherence domain” mechanism as a major-invention seam **when implemented solely as the present explicit alias+Pod version/dependency witness**. Continue the programme only with a mechanism that changes the dependency-discovery, causal-certification, or exact-work frontier rather than renaming generic dependency validation.
