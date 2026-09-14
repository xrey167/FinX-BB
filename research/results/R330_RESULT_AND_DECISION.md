# R330 Result — Indirection-Stable World Attention (ISWA)

Status: **world-attention routing/lifetime gate passed; not DoD and not a standalone novelty claim.**

## Problem

If a mutable factual value also changes the key used to retrieve/attend to that fact, an update to a previously unselected item can change future top-k routing. A cache that validates only the generations of previously selected values can then look valid while its true fresh attention set has changed. This is a hidden/phantom dependency.

R330 tests a stronger invariant:

> governed routing keys are stable semantic/address state; ordinary mutable values live only on the value page.

Query-to-address selection can therefore be cached independently of ordinary value generations.

## Result

- Pods: **2,048**
- cached queries: **2,048**
- routing dimension: **32**
- top-k: **4**
- value updates: **5,000**
- same-value generation updates: **1,776**
- cache admission probes: **40,000**

### Stable-key architecture

- fresh top-k audit mismatches: **0**
- output false serves: **0**
- selected-generation invalidations: **19,956**
- mean queries selecting the updated Pod: **3.9912**
- p99: **9**

### Entangled-key control

The control allowed current mutable values to alter retrieval keys while cache admission checked only the generations of previously selected values.

- apparently generation-valid serves: **27,223**
- hidden false-route accepts: **3,210**
- semantic output false serves: **3,211**

### Global-predicate counterfactual

A single global routing/predicate generation would be safe but would cause **10,240,000** cache invalidations for these updates.

Stable-key local selected-generation invalidation reduced that by **99.8051%**.

Report SHA256: `17643ba45cd85877bf8d256c33c882c767370f2df159226070ecb8c36061ff14`.
Artifact ZIP SHA256: `bcec75eb0e96e9105202dce1090b8bb456c01b6387f87fc08808a903f9270fb2`.

## Architectural decision

Promote **Indirection-Stable World Attention** as a CKCA rule for governed neural memory/retrieval:

```text
query
  -> stable semantic/address keys
  -> selected PodRefs
  -> current generation capabilities
  -> mutable value pages
  -> J/attention computation
```

Ordinary value writes are not allowed to mutate the routing key. If routing semantics themselves change, that is a separate Symlink/schema/key-generation event with its own lifetime.

This gives CKCA another clean separation:

`routing lifetime != value lifetime`.

It also reduces the need for global predicate invalidation while avoiding the phantom-routing failures seen in the entangled control.

Key-value memories, stable embeddings, top-k attention/retrieval and MVCC are established. The research target is the CKCA lifecycle constraint and its composition with the rest of the Causal Neural Machine, not those components individually.
