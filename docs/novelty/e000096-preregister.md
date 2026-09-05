# E-000096 — generation-bound lifecycle frontier utility assay

Date: 2026-09-05
Status: **preregistered before execution**
Classification: research/utility assay. **No novelty or breakthrough credit from dependency tracking, versioning, hard routing, cache invalidation, replay, sparse KV, or ordinary incremental computation.**

## Dependency on E-000095B

E-000096 is executed only if E-000095B first establishes a capable primitive with all of:

1. >=0.95 candidate AND full-vocabulary top-1 correctness on every held-out template for each of >=3 independent training seeds;
2. exact no-memory bypass;
3. exact mutable identity support;
4. byte-identical hidden/KV/logits/continuation after an unrelated canonical payload UPDATE;
5. byte-identical hidden/KV/logits/continuation after an unrelated alias RELINK;
6. fresh-current correctness after relevant target UPDATE and queried-alias RELINK.

If E-000095B fails this prerequisite, E-000096 is NOT interpreted as an LLM result. The architecture direction must be revised instead of weakening the gate.

## Question

Can a capable live LLM reader expose exact `(pod identity, incarnation, generation)` read frontiers during a **multi-read** inference such that a later lifecycle mutation invalidates exactly the neural suffix descended from the changed pod, while earlier/unrelated materialized state remains byte-identically reusable?

The desired object is an **executed neural dependency frontier**, not post-hoc attribution metadata.

Example:

```
H0 --read A:g7--> H1 --ordinary blocks--> H2 --read C:g4--> H3 --ordinary blocks--> H4
```

Required supports:

```
S(H0) = {}
S(H1) = S(H2) = {A:g7}
S(H3) = S(H4) = {A:g7, C:g4}
```

A mutation of unrelated B must leave H0..H4 byte-identical. A mutation of C must preserve H0..H2 and invalidate/recompute H3..H4. A mutation of A must preserve H0 and invalidate/recompute H1..H4.

## Strong baselines

The candidate receives no credit unless compared against all CPU-feasible baselines under identical model/data/quality constraints:

1. **Full replay** from prompt start.
2. **Complete dependency invalidation + clean suffix replay** using exact externally recorded read events.
3. **Fixed/manual frontier placement** at the same read layers.
4. **E-000095-style decoupled semantic-address/exact-payload** without any learned frontier objective.
5. Selective-KV/recompute-style baseline if it can be implemented without changing the model semantics.

Dependency DAGs, MVCC, canonical identity, generations, snapshot isolation and stale-state checks receive ZERO novelty credit.

## Intervention cells

For each qualified seed, construct retained multi-read examples containing >=3 mutable pods and >=2 actual memory reads in one inference. For each example execute:

- unrelated Pod B UPDATE;
- unrelated alias B RELINK;
- second-read Pod C UPDATE;
- first-read Pod A UPDATE;
- second-read REVOKE;
- first-read SHRED;
- rollback of C;
- rollback of A;
- stale Bank replay;
- stale selected-route replay;
- stale resolved-payload replay;
- stale hidden/KV replay from before each mutation;
- generation substitution and incarnation replacement;
- exact-bypass out-of-scope control.

Every mutation is followed by a fresh-current reference execution.

## Correctness requirements

No attack result is interpreted unless the same retained checkpoint satisfies:

- fresh real-symlink correctness >=0.95;
- held-out paraphrase candidate AND full-vocabulary accuracy >=0.95 on every preregistered template;
- REVOKE propagation >=0.95;
- SHRED propagation >=0.95;
- deleted-object leakage <=0.02;
- missing-key UNKNOWN >=0.90;
- generic active-memory KL <=0.05 nats OR exact no-memory bypass on the scoped bypass set;
- independent J-space/J-lens audit passes the preregistered causal criterion;
- >=3 independent training seeds;
- >1 public backbone when CPU-feasible before any breakthrough claim.

## Exact frontier validity

For a mutation outside support, every retained reusable object must be **byte-identical** to the pre-mutation object where deterministic kernels permit it; otherwise compare to a replica-noise floor before interpretation.

For a mutation inside support, no state at or after the first dependent frontier may be reused unless reauthorized and shown equal to fresh-current recomputation.

A support declaration that omits a numerically nonzero executed dependency is invalid even if the final answer is unchanged.

Generated/internal tokens are descendants. If a pod affects an already-generated token, reuse past that token is forbidden unless an exact clean counterfactual is established.

## Utility measurements

Measure, separately from semantic correctness:

- mutation-to-ready latency;
- number of transformer layers recomputed;
- number and bytes of KV tensors recomputed;
- persistent support-metadata bytes;
- inference overhead with no mutation;
- rollback latency;
- alias fan-out scaling at k in {1,2,4,8,16,32,64,128};
- canonical mutation operations vs duplicated-edit operations;
- peak resident memory where available.

A practical success requires the frontier mechanism to beat the strongest semantics-equivalent ordinary baseline by BOTH:

1. >=2.0x median mutation-to-ready speedup at a nontrivial multi-read workload, and
2. >=40% reduction in recomputed KV/layer work,

while adding <=10% median normal-inference latency and <=15% persistent-memory overhead.

These are preregistered utility bars, not novelty bars.

## Research-level novelty bar

Even if all technical/utility gates pass, **do not call a breakthrough** unless a fresh 2025-2026 paper/standards/patent search fails to identify a direct implementation of the following conjunction:

> a capable LLM whose actual executed mutable-knowledge dependency frontier is generation-bound during multi-read neural computation and is optimized/used to reduce lifecycle mutation-to-ready recomputation while preserving exact stale-derived-state safety.

The following are explicitly excluded from novelty claims individually or as loose combinations: pointers/symlinks, aliases, canonical IDs, MVCC/versioning, capabilities, freshness checks, snapshot isolation, dependency DAGs, provenance, cache invalidation, sparse/hard routing, selective KV recomputation, causal probes, J-space/J-lens, external memory, knowledge editing, and counterfactual training.

## Breakthrough decision

Breakthrough = false unless **all** validity, capability, lifecycle, stale-state, J-lens, multi-seed, multi-backbone, utility, and prior-art gates survive on retained checkpoints.

A decisive falsification is recorded if any of these occurs:

- exact frontier support is incompatible with >=0.95 capability across 3 seeds;
- dependency support expands toward the whole Bank/model under real multi-read inference, eliminating selective reuse;
- strongest ordinary baseline matches the candidate's repair cost within 20%;
- generated-history contamination forces near-full replay in typical cases;
- direct prior art implements the same conjunction.
