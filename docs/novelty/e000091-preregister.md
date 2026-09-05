# E-000091 — real-reader exact-lineage density audit

Status at registration: **UNRUN**. Parent: `research/e000082-revocation-local-persistence` at `1ce899bd110bf83581d394a6272c0d8fcb06a28a`. This is a falsification/architecture-contract experiment, not a novelty claim.

## Trigger

E-000080 established a useful controlled systems substrate for object-scoped derived-state lineage, but its payload injection deliberately bypassed learned real-symlink routing. E-000090 separately showed that globally normalized soft routing can make an apparently unrelated source part of the exact dependency cone. The unresolved question is whether that issue occurs in the actual strong real-symlink reader.

Before interpreting any attack, the reader must use E-000081's BOS + consistency `0.15` + alternate supervision `0.5`, 3000 steps, 100 groups, and the existing marker-validity radius `0.35` with proposals conditioned on that predicate. For each seed 0/1/2, candidate accuracy must be >=0.95 on **every** held-out template 8..11; template 9 is not allowed to substitute for the four-template gate.

## Primary falsification

For a fresh symlink world, query alias A while canonical pod B belongs to a different alias group. Record the full live neural computation for A (routing, final hidden state, full logits and causal-LM KV cache). Record an object-scoped lineage containing A's alias+pod witness only. Then update B's canonical payload to a different object, advance B's independent authority incarnation, leave A and its authority state unchanged, rebuild the Bank, and recompute the same A prompt.

The old A-only lineage will still validate by construction. The universal exact-reuse claim is falsified if the fresh post-B-update computation differs at any neural-derived state that would be reused under that lineage. Report byte equality plus max-absolute differences. A descriptive materiality floor may be reported but does not authorize reuse.

Because the current reader uses finite-score softmax over all allowed cells, also report the number of active real rows with strictly positive routing probability at each resolve/dereference slot. Do not call this a causal proof by itself: the intervention above is the falsification witness.

## KV continuation

Capture `past_key_values` from the same actual memory-augmented forward. After the unrelated B update, compare continuation logits from stale pre-update KV with continuation logits from a freshly rebuilt post-update KV. Also compare the stored KV tensors themselves where the backend exposes them. The continuation token is fixed before looking at the result.

## Controls

1. No-op rebuild from the unchanged pre-update store must reproduce logits/hidden/routing/KV within the deterministic execution.
2. A-only authority witness must remain current after B's update; B's old witness must be stale.
3. The queried alias A, its canonical pod, and A's object value must be unchanged by the intervention.
4. Exact full recomputation after the update is the counterfactual reference. No approximate tolerance turns stale state into exact state.
5. Preserve all failed seeds. The three reader seeds are training seeds; a second backbone remains a later requirement.

## Decision rule

- If all three strict reader seeds pass and an unrelated B payload update changes reusable A neural-derived state while A-only lineage still validates, then E-000080-style **selected-object-only lineage is decisively unsound for the current real softmax reader**. The architecture must not claim exact object-local reuse unless the neural operator enforces an exact zero-influence support boundary or the lineage expands to the full actual dependency set.
- If complete lineage expands to essentially every active row under ordinary softmax, that collapses exact selective reuse toward global invalidation and receives no novelty credit. The next mechanism search must therefore create verifiable support locality rather than merely tag caches.
- Hard/top-k routing, reverse-MIPS, cache invalidation, dependency graphs, generations, MVCC and recomputation remain prior-art baselines. A sparse-router fix is not automatically an invention.
- If the strict reader gate fails on any seed, do not interpret the lineage attack positively; preserve the capability failure instead.

No >=10x system utility, <=5% inference overhead, <=2% leakage, >=90% UNKNOWN, J-space completeness, patent novelty or second-backbone claim may be promoted from this screen alone.
