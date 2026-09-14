# R336 Result — Epistemic Lifetime Regions (ELR)

Status: **correctness/storage-compression gate passed, but the first Python object implementation failed the latency objective. Keep the architecture idea; reject this representation. Not DoD.**

## Idea

Instead of attaching/copying a complete generation dependency set to every intermediate J tensor, all mutable-derived values inside one CKVM/J transaction carry one compact `RegionID`.

The transaction object owns the growing exact dynamic read-set once. Internal neural operations propagate only the region identity. At the Neural Commit Barrier the read-set is validated; only retained state is converted into a sealed CLFD factor.

Conceptually:

```text
world reads -> Region R owns exact generations
                 |
                 +-> J tensor 1 : RegionID R
                 +-> J tensor 2 : RegionID R
                 +-> J tensor N : RegionID R
                              |
                              v
                      commit(readset R)
                              |
                              v
                        sealed FactorID
```

## Result

Scale:

- Pods: **50,000**
- transactions: **120,000**
- J operations/transaction: **48**
- mean reads/transaction: **4.9947**

Correctness:

- Region vs full-lineage value mismatches: **0**
- final read-set mismatches: **0**
- expected concurrent conflicts: **7,909**
- detected conflicts: **7,909**
- escaped conflicts: **0**
- same-value race conflicts: **3,548**
- post-commit stale serves rejected: **9,378**
- stale serves escaped: **0**

Lineage representation:

- estimated naive dependency-pair materializations: **28,769,472**
- Region final read-set pair storage: **599,364**
- estimated dependency-pair storage reduction: **97.92%**
- median commit validation: **601 ns** in Python
- median sealed-factor creation: **4.256 µs**
- median factor admission: **171 ns**

But the first implementation allocated Python `Borrowed` dataclass objects and did region checks at every scalar operation. Its median synthetic J loop was **36.7 µs** versus **21.1 µs** for the naive control, a **1.74x slowdown**.

Report SHA256: `fd20e3b754a72286bf451ed7fd3c9655a94ccecc073b9828e53ea0e72a1851a1`.
Artifact ZIP SHA256: `597b02869d665e6cdefd85476ff3e2f366958b2814020183c12e3133df92a390`.

## Decision

The important finding is two-sided:

1. **Region-owned lineage is semantically sufficient**: exact reads, same-value generations, commit races and post-commit expiry remained correct.
2. **A Python object per neural operation defeats the purpose.** That representation is rejected.

R336b therefore implements the intended machine representation: the RegionID lives once in execution context/register metadata, not as a newly allocated Python object at every operation. The target is boundary-only lineage work with no set unions inside the neural hot loop.

This means R336 does **not** prove the <5% production overhead target. It provides the correctness argument and a 97.9% lineage-storage reduction, while explicitly exposing the implementation mistake that must be removed.