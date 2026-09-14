# R314 Result — Causal Knowledge Transaction (CKT)

Status: **bounded transaction+lifetime composition gate passed; not an unbounded proof, not DoD and not a standalone novelty claim.**

## Exhaustive bounded result

- reachable states: **1,378**
- explored transitions: **4,337**
- in-flight transaction commit checks: **794**
- retained-artifact serve checks: **1,119**

Counterfactual policies fail in two different phases:

1. **No commit barrier:** 600 mixed/stale generation read-sets could be committed.
2. **Commit barrier but no post-commit lifetime:** 1,044 retained artifacts became stale after later source updates and would still have been served.

The composed CKT protocol had:

- validated-protocol mixed-generation commits: **0**
- full CKT stale serves: **0**
- semantic-value-only false serves: **366**
- explicit same-value ABA witnesses: **5** sample counterexamples retained.

Artifact ZIP SHA256: `4f69c384f3f104a40376928c0f2d4e8003bddb4e5644e4a66bc2c0ba00c2b3f5`.

## Architectural decision

CKCA needs **both** transaction-time and post-transaction coherence:

1. compile/reuse immutable B-plan;
2. execute lazy verified generation reads;
3. record exact dynamic read-set;
4. validate all generations at the Neural Commit Barrier;
5. abort/retry only J/knowledge execution on conflict;
6. after successful commit, attach the transitive CLFD generation lifetime to the derived neural artifact;
7. serve it only while that lifetime remains valid.

The commit barrier prevents a mixed world from being published while computation is running. CLFD prevents a once-valid artifact from remaining live after a later update. Neither mechanism alone is sufficient.

OCC/read-set validation and versioned derived-state invalidation are established systems techniques. R314 formalizes their required composition for CKCA neural artifacts rather than claiming either one as the invention.
