# R320 Result — CKCA V3 Causal Neural World ABI Reference Runtime

Status: **integrated reference-runtime contract gate passed; not DoD and not a novelty proof.**

## Integrated construction

R320 is the first single executable reference runtime in this branch that composes:

- canonical Symlink/alias -> Pod identity;
- monotonic Pod generations;
- generation capabilities;
- value-independent reusable B plans;
- binding-equivariant typed value execution;
- exact generation read-set collection;
- Neural Commit Barrier;
- CLFD sealing and transitive post-commit lifetime;
- serve-time admission against current authority.

## Corrected run

- Pods: **20,000**
- reusable plans: **5,000**
- transactions: **200,000**
- authority updates: **75,855**
- runtime: **3.3434 s** in the pure-Python reference microbenchmark
- throughput: **59,819 tx/s**

Correctness:

- canonical alias identity errors: **0**
- value-update plan recompiles: **0**
- injected commit conflicts: **36,020**
- commit conflicts detected: **36,020**
- commit conflicts escaped: **0**
- successful J-only retries: **36,020**
- same-value generation conflict witnesses: **18,010**
- post-commit expirations: **23,978**
- stale serves rejected: **23,978**
- stale serves escaped: **0**
- unrelated updates: **15,857**
- unrelated false invalidations: **0**
- successful live serves: **176,022**
- contract result: **PASS**

Corrected report SHA256: `6902939a18509d034c62e477f9f5930da6c84b2b791cee4918b43f3103742e39`.
Corrected artifact ZIP SHA256: `440a0c1a9fcaf4a0a37714e3cf8d158ea91daf23ead87fee3c720b3d11f231fb`.

## Decision

Promote the **Causal Neural World ABI** as the V3 architectural boundary.

The important result is not the Python throughput. It is that the full reference contract can coexist in one executable path without requiring value updates to invalidate/recompile B plans, while generation conflicts and post-commit expiry remain exact.

The research claim remains bounded: this is an executable systems contract, not yet the complete neural-product proof. R312b/R321 and future real-model gates must establish the neural side; RAG/editable-memory baselines, multi-model quality, production overhead and novelty audit remain open DoD items.
