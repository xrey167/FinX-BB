# Neural-Derived-State Revocation — technical novelty candidate

Date: 2026-09-05
Status: **research-level technical novelty candidate only; not a legal novelty or patentability opinion**

## Why this file exists

The broad CAVI / Symlink–Pod idea has now been narrowed repeatedly against both our own negative results and external prior art.

The following are explicitly **not** claimed as novel:

- canonical records, aliases, pointers or symlinks;
- external/editable memory;
- MVCC, generations, rollback, epochs or stale-handle rejection;
- authority lineage or capabilities in general;
- revalidation at an ordinary software effect boundary;
- cache invalidation or source-version validation;
- J-space / Jacobian Lens;
- J-space accessibility auditing;
- semantic routing, scope classification or exact bypass in isolation.

E-000062 also falsified the original direct Symlink–J-space addressing thesis: raw J-space signatures do not provide a competitive address/scope ABI. J-space remains audit-only.

## External prior art that forces the narrowing

Recent public material makes the broad authorization claim untenable:

- Trace Continuity Labs publicly describes cryptographic authority lineage, material-state admissibility, retrieval-time re-evaluation and a disclosure-time Effect Boundary (provisional filing announced 20 Aug 2026).
- execution-finality / effect-boundary work independently requires authority to remain current at the last preventable effect point and explicitly addresses stale authority, replay and TOCTOU.
- GroundedCache and related RAG/cache work already revalidates source-version validity before reusing cached answers.
- activation/KV caching systems already materialize and later reuse intermediate neural state for performance.

Therefore the candidate must be a specifically **neural execution** property that is not merely ordinary cache invalidation, source-version checking, or action-time authorization.

## Candidate technical novelty: Neural-Derived-State Revocation (NDSR)

### Core invariant

> **A neural intermediate derived from an authorized memory resolution does not inherit permanent authority to influence later model computation. Its originating reference-and-referent lineage remains a live precondition of every later neural consumption of that intermediate.**

The key object is not the cache entry itself. It is the **causal permission for that particular derived neural state to participate in the model's computation**.

### System contract

A system satisfying NDSR has all of the following properties:

1. A linguistic reference / alias resolves to a canonical pod or knowledge object through conventional pointer indirection.
2. The resolve emits a lineage witness binding at least `(reference_id, reference_incarnation, pod_id, pod_incarnation)`.
3. The system may derive arbitrary opaque neural material from that authorized read: routing probabilities, selected routes, projected keys, resolved payload vectors, adapter values, KV fragments, cached hidden activations or serialized intermediate tensors.
4. Those derived tensors are **data, never authority**. Possession of the tensor is insufficient to replay its causal effect.
5. Immediately adjacent to every later injection or consumption of such derived state, the runtime checks whether the originating lineage is still valid.
6. If either the reference binding or referent incarnation has changed, the neural intermediate becomes inert even if the old pod is still live and even if the tensor bytes are unchanged.
7. A cached authorization decision made earlier is insufficient if lineage changes between that decision and the actual neural consumption point.
8. The rejection path is an exact no-memory/BYPASS path where appropriate, rather than a soft gate whose residual effect may remain measurable.
9. Fresh state resolved from the current lineage still works, proving that revocation is selective rather than equivalent to disabling the memory system.
10. J-space/J-lens is used independently after the operation to test whether the invalidated knowledge still participates in the causal broadcast pathway. The audit is never optimized and never used as authority.

## Why explicit Symlink–Pod semantics are load-bearing

The Symlink–Pod structure gives a counterexample that a pod-only version check cannot solve.

Let alias `A` point to live pod `P`. Materialize a neural payload `z = resolve(A -> P)` while the binding is valid. Then relink `A` to live pod `Q`, leaving `P` unchanged and current.

At this point:

- `pod_id=P, pod_incarnation=current` still passes;
- the bytes of `z` are unchanged;
- a commit-time authorization cached before relink still says yes;
- ordinary source-version validation on `P` still says current;
- but `z` is no longer authorized as the consequence of resolving `A`.

NDSR requires the stale `A -> P` neural state to become inert because the **reference lineage changed**, while a fresh `A -> Q` resolve must remain usable.

This is why the Symlink is not decorative: it isolates reference freshness from referent freshness.

## Why J-space is still part of the research programme

J-space is not part of the authorization mechanism. It supplies an independent causal observation plane.

The intended composition is:

`Symlink reference lineage -> canonical Pod -> authorized neural materialization -> neural consumption`

with a separate audit:

`invalidated (reference,pod incarnation) -> output/key/reconstruction attacks + independent J-space causal-access audit`

The scientific role of J-space is to catch a system that successfully suppresses ordinary outputs while stale knowledge remains causally accessible inside the model.

## What would count as the technical break

A defensible research claim requires all of the following, without weakening validity controls:

- a **working real symlink reader** before mutation (`>= 0.95` fresh alias correctness);
- a demonstrated stale-neural-state replay after alias relink or pod lifecycle transition;
- replay succeeds under no check, cached commit-time auth and pod-only version validation;
- full reference+referent lineage validation blocks the same neural replay at the actual consumption site;
- the blocked path is indistinguishable from explicit no-memory/BYPASS under the declared locality tolerance;
- fresh current-generation memory remains functional;
- an in-forward TOCTOU mutation is linearized correctly;
- serialized Bank, router distribution, selected route, resolved payload, hidden activation and KV/activation-cache variants are attacked independently;
- ABA/restore/rollback cannot reactivate stale neural lineage;
- >=3 independent seeds;
- at least one materially different public backbone/model family beyond GPT-2 where feasible;
- independent output, key-channel, reconstruction and J-space/J-lens audits agree on the invalidated knowledge identity;
- prior-art review still finds no system that carries **reference+referent revocation semantics through opaque neural intermediate state to the later model-consumption site**.

## Current evidence status

### Already supportive but not sufficient

E-000066 demonstrated stale exported-memory replay. E-000068 showed incarnation-bound capability invalidation. E-000070 and E-000071 showed the structural distinction we need: alias relink defeats a pod-only check, cached/commit-time authorization can remain stale, and full CAVI lineage checks at the neural consumption boundary close the tested path exactly.

However, E-000070/E-000071 **do not yet count as breakthrough evidence** because their real-symlink validity condition failed: the fresh alias reader was only about 0.40625 correct. Structural controls passing on a weak reader are not enough.

E-000076 is therefore the current validity-restoration experiment: the same CAVI boundary is being rerun at a stronger 3000-step, 3-seed symlink configuration. The breakthrough decision remains blocked until the fresh capability control passes.

### Claim-killing conditions

Withdraw NDSR as a novelty candidate if any of these occurs:

1. stale neural intermediates do not preserve a real stale memory capability once the base memory reader is actually strong;
2. ordinary pod/source versioning matches reference+referent lineage on alias relink;
3. a normal cache-version invalidation scheme provides the same guarantee without carrying origin lineage into neural consumption;
4. existing prior art already specifies revocation of opaque neural intermediate state at later consumption using the originating reference+referent authority;
5. the effect vanishes across seeds/backbones;
6. independent causal audit cannot distinguish memory influence from prompt/base-model influence.

## Provisional research claim

> **Neural-Derived-State Revocation (NDSR) preserves revocability across the tensor boundary: routing, payload, cache and activation tensors produced by an authorized external-memory resolve do not become bearer capabilities. Their originating reference-and-referent lineage remains a live precondition at every later neural consumption, so alias relinks or knowledge-incarnation changes can invalidate already-materialized neural state while fresh current memory remains usable. Independent J-space auditing then tests the same invalidated knowledge identity without participating in routing or authorization.**

This is the current narrow technical novelty candidate built directly from the Symlink–Pod–J-space programme. It becomes a supported research claim only after the validity, multi-seed/backbone, adversarial and final prior-art gates above pass.
