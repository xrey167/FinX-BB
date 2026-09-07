# E-000097 — Associative Exact-Transport Reduction

Date: 2026-09-05
Status: **PREREGISTERED KILL SCREEN / scoped reduction; not a novelty claim**

## Trigger

E-000095 and E-000096 close calibration-fitted cross-context revision receipts as the current exact lifecycle-transport route. A tempting successor is to redesign the mutable computation as an associative recurrence/state-space process so that a local Pod edit can be propagated by recomposing cached suffix summaries instead of replaying every downstream token.

This screen asks whether **associativity / segment-tree recomposition itself** can carry invention credit.

## Candidate class

Consider an ordered computation over states `x` with transition operators

`x_i = f_i(x_{i-1})`.

Assume there exists an exact stored summary `S(f)` and an associative binary composition operator `⊗` such that

`S(g ∘ f) = S(g) ⊗ S(f)`

and applying the composed summary to an input state is exact.

A Pod lifecycle mutation changes one or more local transition operators while the remaining operators are unchanged. A candidate caches exact composable summaries and updates/recombines them instead of replaying the full suffix.

The class includes affine/state-space recurrences, exact scan summaries, tree-structured function composition, and any nonlinear finite-state transition representation that is closed under exact composition. It does **not** include a mechanism whose claimed advance is a new compact exact neural-specific summary representation that generic machinery does not already possess.

## Reduction

If the candidate exposes exact summaries `S` and associative composition `⊗`, then a generic dynamic segment tree can store the same summaries at leaves/internal nodes. Replacing a local transition requires updating the affected leaf and recomputing only its ancestors. The root is exactly the composition of the full current transition sequence.

Therefore:

1. exactness is inherited from the candidate's own summary algebra;
2. local update work is `O(log n)` summary compositions for a balanced tree;
3. the generic baseline uses the **same summary representation, same composition primitive, and same persistent summary bytes** as the candidate, up to ordinary tree-index overhead;
4. associativity, scan, prefix products, tree recomposition, or the phrase "neural segment tree" cannot by themselves create a stronger lifecycle guarantee or a unique systems frontier.

If the candidate instead requires state-dependent recomputation inside a summary after an edit, that work must be charged. If a summary is not compact for the real transformer operator, the compactness problem remains unsolved rather than being solved by the tree.

## Executable witness

The registered finite assay uses two exact transition families over a prime field / finite state domain:

- compact affine transitions `x -> a*x+b (mod p)`;
- arbitrary nonlinear finite-state maps represented by exact lookup tables.

For multiple sequence lengths, seeds, mutation positions, initial states, and multi-edit traces it compares:

- dense fresh replay of every current transition;
- generic balanced-tree recomposition using the candidate's exact summary algebra.

It reports exact output mismatches, root-summary mismatches, update composition counts, and summary footprint. The nonlinear lookup-table arm is intentionally included to show that the reduction is not restricted to linear recurrences; its large summary footprint locates the real challenge.

## Kill rule

**Kill associative recurrence / exact scan / dynamic segment-tree recomposition as a standalone major-invention seam** if the generic tree exactly reproduces every registered current-state output and composed summary while using the same summary algebra and asymptotically the same update work the candidate would claim.

This does not kill a future mechanism that provides a genuinely new compact exact representation for the nonlinear neural suffix and demonstrates that its representation/composition is materially cheaper than exact suffix recomputation and all guarantee-matched generic baselines.

## Promotion boundary after a kill

A successor exact transport mechanism must earn credit in the representation itself, not in ordinary dynamic recomposition. It must show at least one of:

1. a compact exact neural-specific sufficient state closed under lifecycle transport for real transformer suffixes;
2. exact affected-work compression not reproducible by generic dependency/change propagation or dynamic function composition at comparable memory;
3. a hardware/locality/bandwidth frontier that survives a baseline using the identical summary algebra;
4. independently discovered causal structure that reduces exact work without being supplied as ordinary metadata.

All programme gates remain unchanged: interpreted real-reader jobs remeasure >=0.95 on every held-out Symlink template; >=3 genuine seeds; >=2 backbone families; <=2% old/deleted leakage; >=90% UNKNOWN in declared missing-key scope; exact bypass or <=0.05 nats generic divergence; stale Bank/router/resolved-payload/Hidden/KV attacks; UPDATE/RELINK/REVOKE/SHRED/DELETE/RESTORE/ABA/rollback/TOCTOU; key/reconstruction attacks; independent J-space/J-lens audit only; <=5% steady-state inference overhead; matched memory; and material fleet-level mutation-to-ready advantage over the strongest guarantee-matched baseline.

## Explicit non-claims

No novelty is claimed for segment trees, monoids, scans, associative recurrences, state-space models, function composition, affine transforms, lookup-table transitions, or this reduction. No lower bound for arbitrary transformer repair is claimed.