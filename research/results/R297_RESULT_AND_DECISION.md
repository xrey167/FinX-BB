# R297 Result — Causal Lifetime Factor DAG (CLFD)

Status: **runtime scalability gate passed; not DoD and not a standalone novelty claim.**

Executed on GitHub Actions, seed 297.

## Scale

- 50,000 canonical Pods
- 220,000 direct derived states
- 60,000 composed higher-order states
- 330,000 factor nodes before lifecycle updates
- 788,777 factor edges
- 282,250 tracked states after online updates

## Exactness

- 50,000 post-update factor-vs-explicit-dependency audit cases: **0 mismatches**
- ABA generation test: **pass**
- revoke → revive no-resurrection test: **pass**

A new generation creates a new immutable lifetime leaf. Old factors are monotonic-dead and cannot become valid merely because a later generation has the same semantic value.

## Hot-path result

250,000 validity queries:

- factor-bit validation: ~503.7 ns/query
- explicit generation-set scan: ~1,606.5 ns/query
- speedup: **3.19×**

The key scaling change is that dependency work is moved to lifecycle transitions. A J/cache state subsequently pays one `valid` bit lookup on the read path.

## Update locality

750 lifecycle updates:

- median update: ~11.8 µs
- p99 update: ~24.4 µs
- mean invalidated factor nodes/update: ~21.98
- p99 invalidated nodes/update: 40
- mean graph fraction touched/update: **0.00666%**
- p99 graph fraction touched/update: **0.01212%**

This does not prove end-to-end <5% model overhead, but it removes the need to scan each state's full dependency read-set during every inference access.

## Decision

Keep **Causal Lifetime Factor DAG** as the current exact implementation of shared/transitive J-Space lifetime semantics:

`Pod generation leaf -> conjunctive derived factor -> O(1) valid bit`

Invalidation cost is proportional to the causal descendants that actually become invalid, rather than the global cache/workspace.

Next integration gates:

1. attach factor IDs to R296 multi-Pod J-Space outputs;
2. attach factor IDs to assistant/tool segment caches;
3. persist factor/authority journal across restart and replica replay;
4. benchmark the complete verifier + factor lookup + Port read path against the <5% serving-overhead target.

Report SHA256: `8ff21513411181784d68ff66e335b0eb2c1cc6575ba881f9613e096ef6a8acad`
Artifact ZIP SHA256: `287b5f536914422b8456d2ea5f0cad0377838f1f5c8b2c79eac6929eba64e4ac`
