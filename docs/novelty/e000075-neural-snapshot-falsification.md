# E-000075 — neural snapshot-isolation falsification

Date: 2026-09-05
Status: **decisive architecture correction; not a breakthrough or novelty claim**

## What was falsified

The first CAVI-N consume boundary revalidated authority independently at each adapter read layer. That closes a local pre-hook -> memory-hook TOCTOU interval, but it does **not** make a multi-read model forward atomic. A lifecycle/reference mutation can commit after one neural read and before a later read, so one inference can consume a torn mixture of authority generations.

E-000075 exercised that exact inter-read race on the trained symlink GPT-2 adapter. In the valid seed-0 case, per-layer validation allowed the mutation to commit between the two read sites and produced logits different from both the all-old and all-rejected references. A forward-wide authority snapshot blocked the same mutation until the memory-consuming region completed and reproduced the all-old reference exactly.

Seed 0 preregistered checks all passed, including the real-symlink-capability precondition. Seed 1 reproduced every race/snapshot structural check, but its independently sampled snapshot arm failed the fresh-reader capability precondition; therefore seed 1 is not counted as full confirmatory evidence.

## Architecture change

CAVI-N must now require **forward-atomic neural-memory consumption**:

1. take one live alias+referent authority snapshot before the first memory read in an inference;
2. use that same authorization decision for every memory read site participating in that logical forward;
3. prevent a lifecycle/reference mutation from linearizing between those read sites;
4. release the authority boundary after the final memory-consuming read;
5. continue to treat all serialized Bank/router/payload/activation material as non-authoritative data.

The reference implementation is `so/cavi_snapshot.py::ForwardSnapshotConsumptionGuard`. Locks and snapshot isolation are established systems primitives and are explicitly excluded from any novelty claim. The candidate neural-specific property is the requirement that **one inference over one logical knowledge object cannot observe multiple authority incarnations at different neural read sites**.

## Evidence from seed 0

- fresh real symlink capability: PASS for both compared arms;
- mutation committed between read layers under per-layer guard: PASS (attack reproduced);
- torn execution differed from all-old reference: max-abs **10.6732**;
- torn execution differed from all-reject reference: max-abs **0.00294**;
- forward-snapshot guard blocked the inter-read commit: PASS;
- mutation completed only after the guarded forward: PASS;
- snapshot execution matched all-old reference exactly: max-abs **0.0**;
- memory path was material: all-old vs reject max-abs **19.2153**.

## Consequence for the novelty thesis

This narrows CAVI-N further. The claim can no longer be phrased merely as "revalidate lineage adjacent to each neural injection." The stronger correctness contract is:

> A memory-consuming inference executes against one authority-lineage snapshot across all of its neural read sites, while any memory-derived material reused in a later inference must be reauthorized against the then-current reference+referent lineage.

That combines two different temporal boundaries: **snapshot consistency within one inference** and **revocability across later consumption events**. Neither snapshot isolation nor versioning is claimed individually.

## Still required before any breakthrough language

- >=3 valid high-capability seeds with fresh alias correctness >=0.95;
- rerun E-000070/071/073/074 under the corrected forward-atomic guard;
- stale Bank/router/resolved-payload/hidden-state, alias-relink, ABA, rollback/restore and concurrent-race attacks;
- exact BYPASS and locality checks;
- independent J-space/J-lens causal audit against NEVER-memory controls without audit optimization;
- replication on another public backbone where CPU-feasible;
- final prior-art search for neural-memory lineage continuity plus inference-wide snapshot semantics.
