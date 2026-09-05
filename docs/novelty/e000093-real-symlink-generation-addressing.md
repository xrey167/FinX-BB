# E-000093 — Real Symlink Generation-Keyed Addressing

Status: preregistered Phase-B falsification experiment; **not a novelty claim**.
Date: 2026-09-05

## Trigger

E-000092 Phase A passed on DistilGPT-2 and Pythia-70M under controlled payloads: stale and current rows coexisted physically, the current generation was selected in every registered case, stale selection was zero, metadata swaps did not change selection, unrelated rows remained routable, stale payloads were behaviorally material, and the current-generation injected output matched current-only gold. This establishes a mechanism substrate, not novelty.

## Question

Does generation-keyed addressability survive when the rows are **real LINK->Pod rows** produced by the trained Symlink reader, and does it prevent stale-generation resurrection under the lifecycle operations the project actually cares about?

## Mechanism under test

A real canonical Pod has identity `P` and generation `g`. Every live memory row derived from that generation receives an address component `C(P,g)`. A query resolved through an alias to current authority `P@g_current` receives the matching current component.

Stale rows are not deleted for the primary arm. They remain physically present but should be unreachable from current addressing.

The experiment must bind both `pod_id` and generation. Generation-only addressing is invalid because two Pods can share the same generation number.

## Reader validity

Use only a reader configuration that remeasures >=0.95 on **every** held-out real-Symlink template in the exact run. The known valid configurations are starting points, not assumed passes:

- seed0: consistency=0.20, alt_supervision=0.50;
- seeds1/2: consistency=0.15, alt_supervision=0.50.

If the exact run misses the reader gate, its lifecycle result is VOID.

## Registered lifecycle attacks

For each valid seed:

1. ACTIVE `P@g` through target and aliases;
2. UPDATE to `P@g+1`, retaining stale `P@g` row;
3. RELINK alias A from `P` to `Q`, retaining old alias/P address material;
4. REVOKE current Pod;
5. RESTORE with a new generation, never resurrecting the old generation address;
6. SHRED / RESIGN;
7. DELETE;
8. ABA-style restore/rollback attempt;
9. stale Bank replay;
10. stale cached router/resolved-payload replay;
11. stale post-read Hidden replay;
12. stale KV replay where the reader/backbone exposes it;
13. in-forward generation change between read sites.

## Strong baselines

A. ordinary co-located sidecar generation check + mask;
B. forwarder/capability indirection;
C. physical stale-row deletion;
D. global epoch flush;
E. object-scoped dependency/version tags;
F. generation-keyed addressing with no external mask (candidate);
G. generation-keyed addressing + sidecar defense-in-depth.

Key rotation, capability revocation, associative memory and attention addressing are prior art. The candidate survives only if the neural addressing arm has a practical guarantee/availability property that the correctly implemented sidecar/forwarder does not provide at equal or lower cost.

## Primary measurements

Per seed and operation:

- every held-out alias template active/current correctness;
- current-row top-1 selection rate;
- stale-row top-1 selection rate;
- **stale routing mass**, not only argmax;
- old-generation answer leakage;
- UNKNOWN rate when operation requires absence/refusal;
- current-only gold KL and top-1 agreement;
- unrelated-Pod correctness/locality;
- generic-text KL / exact bypass;
- mutation-to-ready latency;
- normal inference overhead;
- retained derived-state fraction versus global epoch;
- metadata bytes / address bytes.

## Validity requirements

V1 exact run reader >=0.95 on every held-out template.
V2 stale row is behaviorally material when forced/unguarded.
V3 current row remains behaviorally useful.
V4 stale and current rows coexist physically in the no-mask candidate arm.
V5 no J-space/J-lens signal is used for routing, training or authorization.
V6 unrelated Pod control passes.
V7 stale routing mass is reported; a stale argmax rate of zero with large stale softmax mass is not sufficient for strong revocation.

## Phase-B decision rule

### KILL

Kill the generation-addressing invention seam if any is true:

- stale generation obtains >0.02 old-object answer leakage under a registered lifecycle attack;
- stale routing mass remains materially high at the alpha/locality point required for normal operation and no hard/exact addressing rule closes it;
- stale router/payload/Hidden/KV bypasses the generation address because it is already downstream of the addressing boundary;
- current capability falls below 0.95 on any held-out template;
- generic locality exceeds the existing programme bar without a declared exact-bypass mechanism;
- a correctly co-located sidecar/forwarder catches every registered stale-state attack with equal or lower latency/metadata and no larger collateral invalidation;
- the mechanism requires per-Pod retraining after lifecycle transitions.

### SURVIVE

Survive only if across >=3 seeds:

- current generation is read correctly through unseen alias phrasings;
- one Pod transition makes stale rows unreachable without physical deletion;
- stale Bank/router/payload/Hidden/KV attacks are either intrinsically unreachable or rejected at a clearly defined downstream boundary;
- unrelated state remains reusable;
- current-only gold is preserved;
- at least one important fault/availability case is handled better than the strong sidecar/forwarder baseline.

Still not a novelty conclusion.

## Independent audit after survival

Only after the above passes, run J-space/J-lens as an **independent content-accessibility audit** comparing ACTIVE, stale-forced, current, deleted and NEVER-memory states. It must not be optimized against.
