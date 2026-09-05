# E-000097 prior-art tightening — lifecycle frontier claim boundary

Date: 2026-09-05
Status: exclusion note; not a novelty or patentability opinion.

## Newly excluded broad claims

The following broad ideas are NOT candidates for novelty credit in FinX-BB:

1. **Selective preservation/recomputation of unaffected KV/cache regions after a change.**
   - US20250251917A1 / WO2025166377A1 explicitly describe invalidating stale KV entries after model changes, recomputing affected areas, and preserving unaffected portions.
2. **Versioned/provenance-aware cached intermediate thoughts.**
   - US12626167B2 and related persistent-cognitive/thought-cache families describe versioned thoughts, provenance, compatibility markers, and reuse across agents/contexts.
3. **Business/version-aware cache invalidation.**
   - CN121960775A describes cache validity using identity, time windows and business version numbers, including precise invalidation when knowledge changes.
4. **Causal-evidence sparse routing.**
   - Learning What Matters / Causal Evidence Sets (arXiv:2607.21692) already trains sparse routers from intervention-derived causal evidence rather than attention alone.

These join earlier exclusions: aliases, canonical IDs, symlinks/pointers, MVCC, freshness/version checks, hard/sparse routing, semantic-address/storage decoupling, dependency DAGs, cache invalidation, selective KV recomputation, causal tracing, J-space/J-lens probing, and external memory individually.

## Surviving research question

A defensible contribution must be narrower than dependency-aware cache management. The remaining hypothesis is:

> Can an LLM knowledge-access architecture *learn to shape its own executed mutable-knowledge read frontier* so that far more already-materialized neural state is genuinely independent of later knowledge mutations, while preserving high-capability linguistic access and exact generation-bound lifecycle semantics?

The distinction is causal, not metadata-only.

For a materialized neural state H, define the executed mutable support as exact generation identities:

    S(H) = {(PodID, incarnation, generation), ...}

A lifecycle mutation m to Pod P is allowed to preserve H only when the executed computation that produced H had no dependency on the affected generation. Version tags merely verify this fact; they do not create it.

The candidate mechanism earns no research credit unless training/architecture changes reduce the *actual executed support / recomputation frontier* compared with the strongest semantically equivalent ordinary baseline.

## Required utility comparison

A later candidate must retain the previously preregistered system bar:

- >= 2x lower median mutation-to-ready latency than strongest correct baseline;
- >= 40% fewer KV/layer units recomputed;
- <= 10% normal-inference latency overhead;
- <= 15% additional persistent-state memory;
- same >=0.95 held-out reader and REVOKE/SHRED gates;
- <=0.02 deleted-object leakage;
- >=0.90 UNKNOWN on missing key;
- <=0.05 nats generic KL or exact no-memory bypass;
- stale Bank/router/selected-route/resolved-payload/hidden/KV/generated-history replay attacks;
- forward multi-read races;
- independent J-space/J-lens causal audit;
- >=3 genuine training seeds and >1 public backbone where CPU-feasible.

A dependency graph that is correct but not materially cheaper is not a breakthrough.

## E-000097 role

E-000097 receives zero novelty credit. Its only purpose is to determine whether a capable dense semantic reader can be compiled into an exact immutable identity boundary without keeping mutable dense attention active at inference. If this primitive fails, later lifecycle-frontier experiments remain blocked.
