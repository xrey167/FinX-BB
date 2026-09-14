# CKCA V5 Evidence Addendum — R329b / R330 / R331

Date: 2026-09-14
Status: **evidence update only; DoD remains open.**

## R329b — strong parametric-world counterfactual

R329's weak query-only monolith was replaced with a deliberately strong parameterized world table. Both systems use the same neural language compiler, trusted source-pointer linker and identity-blind J Transformer. The only difference is the value source.

The parametric table fit all home entity->value bindings at 100%. Both paths reached the same 97.70% home task accuracy. After complete world derangement with no model update, the parameter-owned world path fell to 31.42%, while the CKCA external-current-world path retained 97.15%. Future + held-out language-template accuracy was 31.23% vs 94.62%. The controlled future gap is +65.73 percentage points.

This supports the V5 separation hypothesis in the bounded neural gate: **when mutable state is outside reusable neural skill state, future rebinding is a runtime data transition rather than a parameter-generalization problem.** It does not yet compare against model editing or RAG.

## R330 — Indirection-Stable World Attention

R330 exposed a hidden lifecycle dependency in mutable-key retrieval. If the mutable factual value changes the key used for routing, an update to an unselected item can change future top-k while a cache checking only previously selected generations still appears valid.

The entangled-key control produced 3,210 hidden false-route accepts and 3,211 semantic false serves over 40,000 probes. The stable-key architecture produced zero route/output false serves. Local selected-generation invalidation reduced the counterfactual global-predicate invalidation surface by ~99.805%.

V5 therefore separates:

`routing lifetime != value lifetime`.

Ordinary value updates must not mutate governed routing keys. Routing-key/schema/Symlink changes are separate lifecycle events.

## R331 — Lifetime-Carrying Semantic Weave

R331 attacks free-form output granularity. A monolithic autoregressive response conservatively makes every later token dependent on an early mutable read. V5 instead allows explicit semantic reset boundaries: independently regenerable clauses whose lifetimes are the transitive union of direct world inputs and explicit parent clauses.

Across 20,000 documents / 160,000 clauses / 12,000 updates, full-current oracle mismatches and stale-generation survivors were both zero. The semantic weave reduced repair surface by ~71.53% versus conservative monolithic suffix regeneration and ~86.56% versus whole-document regeneration.

This supports the V5 principle that free-form governed language should be a **lifetime-carrying semantic DAG**, not necessarily one indivisible autoregressive persistence object.

## Current strongest integrated architecture hypothesis

The emerging system is now:

```text
language B-Transformer
  -> Epistemic Plan Firewall
  -> Trusted Span/Symlink Linker
  -> Versioned Symlink capabilities
  -> Indirection-Stable World Attention
  -> current Pod / Predicate generations
  -> identity-blind J-Transformer
  -> Neural Commit Barrier
  -> CLFD temporal seal
  -> GANPT neural page admission
  -> Causal Semantic Airlock / Typed World Slots
  -> CSOG / CLSV / Lifetime-Carrying Semantic Weave
  -> GCRD minimal causal repair
```

The next decisive work is not another isolated synthetic mechanism. It is an integrated **real pretrained World Port**: route current typed world values into a late Transformer layer/side rail, keep earlier B states reusable and world-independent, carry generation lifetimes through KV/page admission, and compare free-form quality/latency against strong RAG and editable-memory baselines.
