# E-000093 result — passive in-band freshness is guarantee-equivalent to a co-located sidecar

Date: 2026-09-05
Status: **DECISIVE DIRECTION CHANGE / scoped family kill**

## Result

E-000093 closes the **passive in-band freshness** family as a major correctness/guarantee novelty seam.

The registered class is any scheme where:

- a cached neural-derived artifact is `T`;
- current lifecycle authority is `A`;
- the reuse decision is deterministic;
- all lifecycle-specific information consumed by that decision is recoverable from `T` through a deterministic statistic `D(T)`;
- lifecycle mutation changes `A` but does not actively transform `T` before the decision;
- the mechanism ultimately accepts or rejects reuse of an otherwise unchanged cached artifact.

For this class, materialize a co-located sidecar `S := D(T)` when `T` is created. Replacing the decode step with the stored statistic yields the same decision for every `A,T` pair by construction.

The executable E-000093 assay instantiated three representative registered schemes:

1. generation identity;
2. generation plus an embedded marker;
3. generation plus a content-bound tensor digest.

Across the finite registered artifact/authority domain and ten fault classes, each scheme produced **17,280 comparisons** and **zero mismatches** between the passive in-band decision and its co-located sidecar shadow: **51,840 total exact comparisons, 0 mismatches**.

Registered faults covered:

- no mutation;
- current-generation update;
- unrelated mutation;
- external metadata deletion;
- external metadata swap;
- serialization/relocation;
- stale replay;
- ABA authority return;
- rollback;
- same-content/different-generation states.

This executable enumeration is a regression witness for the scoped algebraic reduction; the reduction itself is not statistical and does not depend on the three particular decoder examples.

## What is killed

Do not spend major-invention search budget on another passive tensor freshness representation merely because the freshness statistic is stored inside Hidden/KV state. This includes, when used only as a reuse predicate:

- generation watermarks;
- residual generation markers;
- tensor provenance codes;
- freshness ECC/parity/syndromes;
- embedded version tags;
- content-bound freshness digests;
- passive neural-state certificates decoded only to accept/reject reuse.

For correctness guarantees, a correctly co-located sidecar can store the same sufficient statistic and reproduce the same decision. The in-band location receives zero novelty credit.

This is consistent with the programme's E-000090/E-000090B and E-000092 outcomes but is logically stronger: even a *perfect* passive decoder would not create a stronger freshness guarantee solely by putting the statistic in the tensor.

## What is NOT killed

E-000093 is not a universal impossibility theorem for lifecycle-aware neural mechanisms. It explicitly leaves open mechanisms in which the lifecycle transition changes the **computation itself**, for example where:

1. stale state loses a computational capability and cannot be made useful by merely changing an external accept/reject bit;
2. a lifecycle transition induces an algebraic transformation of reusable neural state that is cheaper than guarantee-matched replay/recomputation;
3. exact affected work is represented/updated in a way ordinary dependency tracking cannot reproduce at comparable cost;
4. causal lineage is discovered or certified from model computation rather than supplied as ordinary metadata;
5. a mechanism provides a measured hardware/locality/bandwidth advantage over the co-located sidecar under matched correctness, memory, and fault assumptions.

Any such successor must still survive the full major-break programme gates.

## Strongest-baseline implication

Future in-band candidates must compare against a **co-located, atomically transported sidecar carrying the minimal sufficient statistic used by the candidate**. Deliberately separating the sidecar from the tensor, stripping only the sidecar, or making its transport non-atomic is not guarantee matched.

If a candidate consumes tensor content beyond a named freshness field, the baseline may use a content digest or the minimal verification statistic needed to reproduce the same accept/reject semantics. Systems advantage, if any, must be measured rather than inferred from representation location.

## Prior-art boundary update

Recent 2026 work further crowds ordinary KV reuse/rollback/freshness management. In particular, rollback-consistency work demonstrates that retained stale KV can violate logical rollback and that restoring transaction-local cache state closes the channel; modern serving stacks already implement content-addressed KV reuse and dependency-aware eviction. These are context for the baseline boundary, not the proof of E-000093.

## Decision

**Kill passive in-band freshness as a major correctness novelty route.**

The invention search now moves to the narrower frontier of **active lifecycle-conditioned computation** or **exact affected-work compression / causal-lineage discovery** that can produce a guarantee or systems result a co-located sidecar plus generic dependency machinery cannot match.

Preserved artifacts:

- preregistration: `docs/novelty/e000093-passive-inband-sidecar-reduction.md`
- executable: `so/experiments/e000093_passive_inband_sidecar_reduction.py`
- workflow: `.github/workflows/e000093-passive-inband-sidecar-reduction.yml`

No claim is made for legal novelty or patent clearance, and no novelty is claimed for the reduction itself.
