# E-000089 — Cross-Model Causal Generation Attestation

Status: preregistered falsification experiment; **not a novelty claim**.
Date: 2026-09-05

## Why this experiment exists

E-000088 killed Certified Semantic Dependency Elision on all three seeds: every ACTIVE payload mutation changed routing, hidden state, logits and answer somewhere in the registered artifact, `semantic_extra_over_field = 0`, and a field-sensitive dependency baseline was strictly stronger than the semantic-elision proposal. We therefore stop pursuing semantic non-influence certificates as the invention.

The obvious next ideas are also heavily occupied by prior art:

- dynamic slicing / self-adjusting computation and selective recomputation cover affected-cone discovery and replay;
- MindBridge (ACL 2025) covers cross-model knowledge editing using an LLM-agnostic memory modality;
- Cross-Model Memory Transfer (arXiv:2608.17050, 2026) covers frozen external memory transferred between backbones via target-side readers;
- Knowledge Objects (arXiv:2603.17781) cover model-independent first-class factual objects;
- shared-memory / governed-memory systems cover fleet-wide supersession, provenance and stale propagation;
- cross-model steering and concept alignment cover transferable internal directions;
- J-space / Jacobian-lens and J-Access cover internal causal/accessibility auditing.

Therefore **none** of these ingredients, individually or as a loose bundle, is our candidate contribution.

## Narrow remaining question

Can one canonical Symlink-Pod generation be consumed by heterogeneous LLM backbones and carry a **common lifecycle identity into the model-internal causal pathway**, such that one Pod transition is both:

1. operationally observed by every participating model without per-Pod re-editing/retraining, and
2. independently attested inside each model as a causal transition attributable to the same `(pod_id, generation)` rather than merely a changed external lookup result?

The important distinction is between **shared memory consistency** and **cross-model causal generation consistency**.

A generic shared memory can prove that every client fetched generation `g+1`. It does not, by that fact alone, prove that each model's internal neural computation stopped using generation `g`, especially when stale Bank/KV/router/payload/activation material or model-specific readers can persist.

## Candidate object: Generation Attestation Record (GAR)

For one canonical Pod generation `G=(pod_id, incarnation)`, each model produces an attestation record:

- `model_id` / backbone family;
- exact Pod generation consumed;
- reader/interface version;
- external resolution result;
- stale-derived-state rejection result;
- fresh current-generation answer/result;
- independent causal-audit result measured without optimizing against the audit;
- locality / unrelated-state control.

The fleet-level transition is accepted only when all required models attest the same current generation and all registered stale-generation attacks fail.

This record format itself is **not** claimed as novel. The experiment asks whether the composed neural property exists and produces a useful systems guarantee beyond ordinary shared-memory version checks.

## Phase A — structural kill screen

Use at least two heterogeneous frozen backbones already exercised by this repository where possible (initially DistilGPT-2 and Pythia-70M controlled-memory arms; later real LINK->Pod readers once available).

For one canonical Pod shared across the two backbones:

1. establish a live generation `g` that materially changes each model's output relative to BYPASS/NEVER;
2. capture model-specific derived state for `g` (at minimum the memory representation consumed by the reader; preferably routing/resolved payload/hidden/KV where exposed);
3. mutate the Pod exactly once to generation `g+1`;
4. show a generic shared-memory version check can establish `g+1` externally;
5. deliberately replay stale `g` derived state separately into each model;
6. require the model-side generation guard to reject stale state at the consumption boundary;
7. recompute from `g+1` and require agreement with the fresh current-generation reference;
8. independently audit the model's internal causal response for `g`, stale `g`, `g+1`, and NEVER/BYPASS controls;
9. preserve unrelated Pod/model-derived state and base behavior.

## Strong baselines

The candidate must be compared against:

- ordinary shared-memory generation/version checking;
- global fleet epoch invalidation;
- per-model dependency/version tags;
- full per-model recomputation;
- MindBridge/XMemTransfer-style cross-model memory integration where reproducible;
- external Knowledge Object / retrieval-only use;
- any model-local J-space/J-Access audit without generation binding.

If external version checking plus ordinary per-model dependency tags gives the same acceptance/rejection decisions and the same practical guarantee, this seam is killed.

## Validity controls

V1. **Material causal capability:** on every participating model, ACTIVE generation `g` must measurably change the registered output relative to BYPASS/NEVER. A no-effect memory cannot certify anything.

V2. **Cross-model current-generation utility:** after the one canonical update, every model must consume `g+1` correctly without per-Pod model retraining.

V3. **Stale replay attack:** previously materialized `g` state must still be capable of changing output if injected without the guard; otherwise stale-state rejection is vacuous.

V4. **Consumption-boundary rejection:** guarded stale replay must inject/consume zero invalid state and match BYPASS or the declared safe fallback exactly.

V5. **Fresh-current capability:** fresh `g+1` state remains usable and matches its gold/current reference.

V6. **Independent audit:** J-space/J-lens or another causal internal audit is measurement-only and not used for training, routing, generation checks or the pass decision of V2-V5.

V7. **Unrelated locality:** unrelated Pods and generic prompts stay within their preregistered locality bounds.

V8. **No cross-model hidden retraining loophole:** any reader/linker training must be one-time per backbone/interface, not one training job per Pod edit.

## Decision rule

### KILL

Kill this seam if any of the following holds:

- cross-model lifecycle behavior requires re-editing/retraining each backbone for each Pod mutation;
- stale model-derived state cannot materially resurrect old behavior in the unguarded control;
- ordinary shared-memory versioning + standard dependency tags is decision-equivalent to the candidate at every registered boundary;
- the independent causal audit adds no information beyond output/current-generation checks across the registered attacks;
- model capability/locality fails materially;
- the same property cannot be reproduced on at least two different backbone families.

### SURVIVE AS RESEARCH DIRECTION

The seam survives only if:

- one Pod mutation updates every participating model without per-Pod retraining;
- stale derived state can resurrect old behavior unguarded and is rejected guarded;
- fresh current-generation state remains functional;
- the independent audit detects/attests the same generation transition inside each model;
- unrelated state remains reusable;
- at least one registered failure mode is **not** captured by external generation/version checking alone;
- the result reproduces across >=3 seeds and >=2 backbone families.

This still does not establish global novelty.

## Phase B — only if A survives

Build a practical fleet runtime and measure:

- one-update-to-all-models propagation latency;
- per-model inference overhead;
- retained cache fraction versus global epoch invalidation;
- memory footprint of attestation metadata;
- rollback / ABA / restore behavior;
- stale Bank/KV/router/payload/hidden replay resistance;
- concurrent/in-forward race behavior;
- alias fan-out 1..10,000;
- model fan-out 2..N backbones.

Major-technology target before any breakthrough language:

- >=95% real LINK->Pod reading/lifecycle propagation on the registered tasks;
- <=2% deleted/old-generation object leakage;
- >=90% UNKNOWN in declared missing-key scope;
- exact BYPASS or <=0.05 nats generic divergence;
- <=5% normal inference throughput penalty;
- >=10x rollout/mutation-to-ready advantage **or** an equivalently large retained-cache/availability advantage over the strongest guarantee-matched baseline;
- >=3 seeds and >=2 backbone families;
- independent causal audit agrees with the lifecycle transition while being held out from optimization.

## Potential technical claim if the seam survives

The only candidate claim worth pursuing would be narrowly stated:

> A canonical, versioned neural knowledge object can be hot-updated once and consumed coherently by heterogeneous language-model backbones, while generation-qualified guards reject stale model-specific derived state at the neural consumption boundary and independent per-model causal audits attest that the same knowledge generation transition occurred inside each model's computation.

This is intentionally narrower than cross-model memory, knowledge editing, shared memory, cache invalidation or J-space auditing individually.
