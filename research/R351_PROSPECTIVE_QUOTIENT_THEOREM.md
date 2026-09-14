# R351 — Prospective Quotient Theorem / Future-Bindable Cache Normal Form

Date: 2026-09-14
Status: **new project theorem candidate; proof checked mechanically in finite/linear cases, but novelty is not claimed.**

## 1. Question

R340–R350 motivate a stronger theoretical question:

> If a cached neural state must remain byte-unchanged across arbitrary future world rewrites while every serve is exactly current, what information is a *useful* retained cache allowed to contain about the old mutable world?

The answer is stricter than “late binding is convenient.” Under universal exactness, any old-world dependence that remains observable by the current-world decoder is impossible.

This yields a normal-form result for write-maintenance-free mutable-knowledge caches.

---

## 2. Setup

Let:

- `q ∈ Q` be immutable query/language/skill state;
- `w_old ∈ W` be the mutable world at cache-construction time;
- `w ∈ W` be an arbitrary future/current authoritative world;
- `F : Q × W -> Y` be the desired current-world semantics;
- `K : Q × W -> C` be a cache builder;
- `S : C × W -> Y` be a serve/materialization function.

A cache protocol has **Universal Future Freshness (UFF)** when

`S(K(q, w_old), w) = F(q, w)`

for **every** `q`, every historical `w_old`, and every current/future `w`.

There is no cache rewrite in this equation. The exact same retained cache object built under any historical world must be valid under all future worlds.

---

## 3. Observational equivalence

Define an equivalence relation on cache states:

`c1 ≡ c2  iff  for all w ∈ W: S(c1,w) = S(c2,w)`.

Two cached byte strings may be physically different but are semantically identical if no possible current world can make the serve function distinguish them.

Let `[c]` denote the equivalence class of `c`.

---

# 4. Prospective Quotient Theorem

### Theorem

If `(K,S)` satisfies Universal Future Freshness for `F`, then for every fixed query `q` and every pair of historical worlds `u,v`:

`K(q,u) ≡ K(q,v)`.

Therefore the quotient cache

`K*(q) = [K(q,w_old)]`

is well-defined and **independent of historical mutable world state**.

### Proof

Take arbitrary fixed `q` and arbitrary historical worlds `u,v`.

For every possible current world `w`, UFF gives

`S(K(q,u), w) = F(q,w)`

and

`S(K(q,v), w) = F(q,w)`.

Thus for every `w`:

`S(K(q,u),w) = S(K(q,v),w)`.

By definition of `≡`:

`K(q,u) ≡ K(q,v)`.

Since this holds for every pair `u,v`, all historical-world variants of the cache for `q` occupy one observational equivalence class. Hence `K*(q)` does not depend on `w_old`. QED.

---

## 5. Meaning

A universally fresh, zero-write-maintenance cache **may physically contain bits derived from an old world**, but those bits must be decoder-observationally irrelevant to all future current-world outputs.

Consequently:

> Any *useful* retained information in a minimal UFF cache can be represented without historical mutable-value dependence.

This is stronger than the engineering statement “external values are easier to update.” It says that if we demand all three simultaneously:

1. arbitrary future world rewrites;
2. exact current-world semantics;
3. no cache patch/invalidation on ordinary value writes;

then historical-world dependence cannot remain in the decoder-observable retained state.

A retrospective hidden state can satisfy the contract only if its old-world component lies entirely in a semantic null-space—or if the cache is repaired/invalidated.

---

# 6. Referential corollary

Many governed tasks factor as

`F(q,w) = G(h(q), w[R(q)])`

where:

- `R(q)` identifies the canonical world cells required by the query;
- `h(q)` is world-independent language/operator continuation state;
- `w[R(q)]` denotes current dereference of those cells.

Then one UFF normal form is

`K_PNS(q) = (h(q), R(q))`

and

`S_PNS((h,R),w) = G(h, w[R])`.

This is exactly the semantic shape of a **Prospective Neural State**:

`κ_q = (continuation_state, live_references)`.

The theorem does **not** prove that references are the unique implementation. It says the useful quotient cache cannot depend on the historical payload. References are a compact way to represent which future world arguments remain unresolved.

---

# 7. Linear neural-cache corollary

Consider a finite-dimensional linear cache:

`k = A q + B w_old`

and serve function

`y = C k + D w`.

Suppose the target is

`F(q,w) = T q + U w`.

UFF for all `q,w_old,w` requires:

`C A = T`

`C B = 0`

`D = U`.

The important equation is

`C B = 0`.

Every historical-world direction retained through `B` must fall in the serve decoder's null-space. If `rank(CB) > 0`, an old-world value is observably contaminating the cache and universal freshness is impossible without repair.

This gives a simple measurable proxy for conventional neural hidden-state contamination.

---

# 8. Write-fanout trilemma

For a cache whose output semantically depends on mutable world value `x`, at least one of these must hold after arbitrary updates to `x`:

1. **rewrite/invalidate** every retained numeric state whose semantics absorbed `x`;
2. **recompute** from an earlier state that did not absorb `x`;
3. retain a **prospective state** in which `x` remains unresolved and dereference current `x` at use time.

This is not claimed as a universal lower bound on every possible computer architecture; it is the direct operational consequence of the UFF contract and the quotient theorem for retained serve state.

CKCA/PNS deliberately chooses option 3 for reusable state, then uses option 2 only after explicit materialization boundaries when a generation changes.

---

# 9. Generations are a second semantic dimension

Payload equality is not temporal equality.

Let a cell transition

`(pod, g17, value=42) -> (pod, g18, value=42)`.

For a value-only function `F`, the visible answer may be unchanged. But a lifecycle-safe artifact carries an admissibility predicate over generation identity.

Therefore the authoritative world argument is better written as

`w = (payload_state, temporal_generation_state)`.

PNS can remain payload-unbound before materialization, while the materialized J state records the exact generation read-set. A same-value ABA rewrite changes the temporal component and therefore invalidates old J state even if the output bytes are coincidentally equal.

The quotient theorem concerns the retained prospective cache. Commit/lifetime semantics govern materialized descendants.

---

# 10. Relation to prior ideas

This result must be audited against:

- memoization with mutable state;
- self-adjusting / incremental computation;
- partial evaluation and binding-time analysis;
- closures and environment passing;
- database query plans and MVCC;
- cache coherence / invalidation;
- pointer/reference machines;
- external neural memories.

The elementary proof is not presented as mathematically deep. Its potential value is architectural: it gives an exact criterion for when a neural cache can be both permanently reusable across arbitrary world revisions and semantically current.

The research question is whether the **neural activation consequence**—reference-valued / future-bindable Transformer residual state—is a useful and sufficiently non-obvious architecture contribution after direct prior-art review.

---

# 11. Falsification implications

For every candidate PNS implementation we should now test:

1. **World-write byte invariance** of the retained prospective cache.
2. **Full-current oracle equivalence** after arbitrary future worlds.
3. **Old-world ablation:** no old payload byte is required to produce current output.
4. **Null-space audit:** if an implementation retains historical-world numerical components, perturb them within the old-world subspace and verify they cannot affect serving.
5. **Generation race validation** at the materialization/publication barrier.
6. **Matched retrospective cache control** showing that an actually decoder-visible old-world component requires invalidation or becomes stale.

R347 already satisfies 1, 2, 5 and a matched version of 6 in its bounded learned gate.

R349/R350 move the same criterion into a pinned pretrained Transformer.
