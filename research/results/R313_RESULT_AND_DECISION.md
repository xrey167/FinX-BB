# R313 Result — Neural Read-Set Commit Barrier

Status: **concurrency-coherence gate passed; not DoD and not a standalone novelty claim.**

## Scale

- 25,000 Pods
- 150,000 mutable knowledge transactions

## Result

A transaction performs several generation-scoped reads while concurrent writes may occur between those reads. Without a commit barrier, the runtime would have admitted **32,922 stale or mixed-generation results**.

CKVM's generation read-set barrier rejected all **32,922** conflicting transactions before publication:

- undetected stale commits: **0**;
- barrier rejections: **32,922**;
- retries producing the serial-current result: **32,922 / 32,922**;
- same-value generation conflicts detected: **7,664**;
- B-plane plan recompiles caused by world conflicts: **0**;
- median J-space retry steps: **1**;
- p99 retry steps: **1**;
- median commit-barrier check: **543 ns**;
- p99 commit-barrier check: **837 ns**.

Report SHA256: `1c92d9348dd6ffcbe8308b6ff8530b48b9a2377251394ebd291636e04a9ca0c5`.
Artifact ZIP SHA256: `1d117d0e09e9d51fc996506fafa94b2f61dee32da537b8b9c7acaa227ff5ee8c`.

## Architectural decision

Add a mandatory **Neural Commit Barrier** before a mutable-derived J state or answer becomes externally visible:

`record exact generation read-set -> compute -> validate all generations still current -> commit or retry J transaction`

A conflict retries only the mutable knowledge/J-space execution. The immutable B-plane interpretation/plan remains reusable.

This closes a gap not covered by post-hoc lifetime invalidation alone: an answer must not be allowed to mix generations that stopped being jointly current while the reasoning transaction was still in progress.

Optimistic concurrency control and read-set validation are established, including S-Bus for LLM-agent state. R313 is therefore an integration requirement for CKCA neural coherence, not a standalone invention claim.
