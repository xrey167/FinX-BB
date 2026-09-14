# R336b Result — Packed Epistemic Lifetime Regions (PELR)

Status: **packed region representation gate passed; strong evidence that exact lifecycle metadata can be boundary-amortized, but production <5% DoD remains open.**

## Repair over R336

R336 proved the RegionID semantics but made a Python `Borrowed` object for every intermediate operation and was slower than the naive control. R336b implements the intended machine representation: the region identity is implicit execution-frame metadata, the exact read-set is constructed once at the World Port boundary, and the interior J hot loop is ordinary arithmetic with no dependency-set allocation/union.

## Result

12,000 programs per depth, depths `{8,32,128,512}`, exact comparison against both a no-lineage baseline and naive per-operation lineage propagation.

Correctness at every depth:

- value mismatches: **0**
- final read-set mismatches: **0**

Median overhead versus no-lineage baseline:

- depth 8: **+34.65%**
- depth 32: **+8.82%**
- depth 128: **+1.18%**
- depth 512: **-0.47%** (measurement noise around parity)

At depth 512, naive per-operation lineage remained **+102.94%** over baseline, while packed regions were statistically at baseline and **2.04× faster** than naive lineage.

Metadata scaling:

- estimated naive dependency-pair materializations: **40,800,000**
- region read-set pair storage: **240,000**
- estimated pair reduction: **99.4118%**

Lifecycle stress, 50,000 trials:

- expected commit conflicts: **5,975**
- detected: **5,975**
- escaped: **0**
- same-value race conflicts: **2,670**
- post-commit stale serves rejected: **4,994**
- stale serves escaped: **0**
- median commit validation: **694 ns**
- median factor admission: **108 ns**

Report SHA256: `7dc147bb4bb5814f6aeb30c3c482136444dd12a5a5b34c8d92826f91dd447fe6`.
Artifact ZIP SHA256: `c9b80cf0318da74585b07adba5a3eccd1dc2f229a9f1685fe08bdf32de3dd109`.

## Architectural decision

Promote **Packed Epistemic Lifetime Regions** as the implementation model for temporal neural typing:

> Exact generation lineage should be paid at World Port / commit / seal / admission boundaries, not as a dependency-set operation at every neural layer or tensor op.

The scaling curve is the important finding. The fixed read-set construction cost is visible at shallow depth, then rapidly amortizes as reusable neural compute grows. This directly attacks the historical ~73% lifecycle-overhead failure without weakening same-value ABA, commit-race or post-commit freshness semantics.

This is **not** yet evidence that a production LLM runtime is below 5% overhead. The next required step is to carry one implicit region handle through a real Transformer execution and measure end-to-end batch/sequence serving overhead.