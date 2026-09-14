# R334b Result — Delta World Attention (DWA)

Status: **exact sparse world-attention update gate passed; not DoD and not a standalone novelty claim.**

R334's first run ended before reporting because of a harness variable-name error. R334b is the corrected clean rerun of the mechanism.

## Mechanism

Governed attention separates lifecycle-stable routing keys from generation-scoped mutable value pages:

```text
Q · stable K -> selected PodRefs + fixed softmax weights
                      |
                      v
              current mutable V:g
```

For an ordinary value update, Q and K do not change. Therefore selected addresses and attention weights are invariant. If selected value page `j` changes, every dependent attention output has the exact update law:

`O' = O + alpha_j * (V'_j - V_j)`.

Generation metadata is refreshed even when `V'_j == V_j`, so same-value ABA transitions change lifetime without changing the numeric result.

## Result

Scale:

- Pods: **2,048**
- cached queries: **1,024**
- top-k: **8**
- K dimension: **32**
- V dimension: **48**
- value updates: **5,000**
- same-value generation updates: **1,739**

Exact stable-K path:

- incremental local-audit max error: **6.73e-16**
- final full-recompute output max error: **6.73e-16**
- final route mismatches: **0**
- final attention-weight max error: **0.0**
- final generation mismatches: **0**
- route changes from initial state: **0**
- same-value selected lifetime refreshes: **6,972**
- same-value numeric output changes: **0**

Locality/performance:

- local patches: **19,963**
- global-query invalidation counterfactual: **5,120,000**
- local maintenance reduction: **99.6101%**
- mean queries patched/update: **3.9926**
- p99 queries patched/update: **9**
- median local Python delta patch: **56.1 µs**
- median full attention recompute: **5.443 ms**
- full-recompute/local-patch ratio: **97.05x**

Entangled-key control, where mutable value payload also changes K:

- selected-generation-valid probes: **10,249**
- hidden route false accepts: **2,404**
- hidden output false accepts: **2,404**

Report SHA256: `cdf8074c0c5dfa6404ddbbbb42b35d71870a05dd01c7e58d73d0ca0cb17c5041`.
Artifact ZIP SHA256: `a19895925b724672b37240d29e6b7386cd99ae74fdd89e0e85a4ff200fff9257`.

## Architectural decision

Promote **Delta World Attention** as a new CKCA/WIT design rule:

> Ordinary mutable world payload belongs in V, while governed routing identity belongs in lifecycle-stable K. A payload update must not silently rewrite the route by which that payload is discoverable.

This gives two independent lifetimes:

`routing lifetime != payload lifetime`.

If routing semantics actually change, the Symlink/key/schema generation changes explicitly. If only the current value changes, the route remains reusable and the neural world-attention state can be patched exactly and sparsely.

Attention, K/V separation, top-k routing, reverse indexes and delta propagation all have prior art individually. The candidate research contribution is their use as a temporal neural-world interface together with the rest of CKCA, not these components in isolation.