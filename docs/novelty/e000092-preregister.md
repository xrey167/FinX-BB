# E-000092 — exact-support real-reader feasibility

Date: 2026-09-05. **Preregistered negative-or-positive feasibility screen; no novelty claim.**

## Motivation

E-000090 showed synthetically that selected-payload lineage is unsound when shared normalization, routing competition, or control decisions let an apparently unselected source influence persistent neural state. E-000091 is independently testing that issue in the real trained softmax symlink reader.

This experiment asks a narrower engineering question before inventing a new mechanism: **does the already-capable reader retain its held-out capability if its memory routing is evaluated with mathematically exact one-hot support rather than dense softmax support?** If yes, an unrelated nonselected pod can in principle have exactly zero payload influence on the selected read. If no, exact-support lifecycle locality has an immediate capability cost and must be trained, not bolted on.

Hard routing, argmax/top-k, straight-through estimators, sparse attention, selective invalidation and dependency graphs are established techniques and receive zero novelty credit here.

## Frozen preregistered conditions

1. Use the E-000081 consistency-reader training recipe: GPT-2, frozen backbone, BOS enabled, 100 alias groups, 3000 steps, consistency=0.15, alternate supervision=0.5, corrected marker-validity sampling. Three genuine training seeds: 0,1,2.
2. Train exactly with the historical softmax path. Do not use hard-routing evidence during optimization.
3. Evaluate the same retained weights twice on the same independent real-symlink world and held-out templates 8,9,10,11:
   - historical softmax routing;
   - **exact-support evaluation**: replace each resolve and dereference distribution by an exact one-hot argmax over the same masked scores; passthrough/null semantics remain unchanged.
4. Capability validity requires candidate accuracy >=0.95 on **every** held-out template in **every** seed. Full-vocabulary top-1 is reported separately and cannot be substituted by candidate accuracy.
5. For queries whose complete hard route is unchanged after mutation of an unrelated canonical pod B, compare pre/post-B final hidden state, candidate/full logits, each routing slot and captured KV. The support-locality check passes only if those derived tensors are exactly equal under a fixed numerical execution path (maxabs == 0 where bitwise deterministic; otherwise the predeclared tolerance is 1e-7 and must be reported as numerical equality, not bitwise identity).
6. Also record cases where B changes the hard winner. Those are **control dependencies** and must expand lineage or force route recomputation; they may not be called bystanders.
7. Record support size per resolve/deref slot. One-hot support should be one active column by construction, but the complete lineage is the union of all winners and all control dependencies needed to establish those winners. This experiment does not assume that winner-only lineage is already sound.
8. Compare ordinary softmax capability against exact-support capability. A capability loss >0.02 absolute on any held-out template is a decisive negative for post-hoc hardening even if exact locality improves.
9. Exact no-memory bypass must remain maxabs 0.0 because the adapter is not invoked when bank=None.
10. Do not run or interpret the CAVI adversarial battery unless the exact-support arm itself satisfies the >=0.95 three-seed capability prerequisite.

## Interpretation boundaries

A positive E-000092 would establish only that an exact-support execution is *feasible on this reader*. It would not establish novelty, deletion, REVOKE/SHRED, UNKNOWN, generic KL, J-space completeness, multi-backbone replication, production speedup, or a breakthrough.

A later novelty candidate must add a lifecycle-specific learned objective or execution property and beat matched hard-routing/sparse-attention baselines on **mutation-to-ready cost**, inference overhead and preserved capability. Merely naming hard routing `lifecycle locality` is explicitly forbidden.

A negative E-000092 changes direction: lifecycle-local reuse cannot be obtained by post-hoc exact support on this reader; support locality has to be co-trained or the project must accept conservative global recomputation.
