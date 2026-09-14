# R345 hypothesis — Prospective Neural State (PNS)

## The distinction

A conventional retained hidden state is **retrospective**: it is a numeric tensor computed from one past world snapshot.

`h_t = F(q, W_t)`

If `W_t` changes, `h_t` is potentially stale and must be invalidated, patched, or recomputed.

A **Prospective Neural State** is different. It is a retained neural continuation whose unresolved operands are live canonical world references:

`κ_q : W -> h`

The cached object is `κ_q`, not `κ_q(W_t)`.

It may contain reusable neural computation, routing, coefficients, a compact latent continuation code, and canonical reference bindings. It must not contain ordinary mutable world payload bytes as though they were timeless activations.

At use time:

1. dereference the live references under current generations;
2. evaluate the remaining continuation;
3. validate the exact generations at a commit barrier;
4. if retained as a numeric result, attach the resulting generation lifetime to that materialization.

Thus the prospective descriptor itself may survive arbitrary ordinary world rewrites; a *materialized numeric result* remains temporal.

## Safety proposition

Let `K` be a deterministic continuation over references `r_1..r_n`. A materialization transaction reads `(v_i,g_i)` from each current `r_i`, computes `y=K(v_1..v_n)`, then commits only if every current generation still equals `g_i`.

Then every committed `y` equals the evaluation of `K` over one simultaneously current authority snapshot at commit time. Same-value ABA is safe because generation identity, not payload equality, is validated.

If a committed numeric `y` is retained beyond the transaction, it receives a post-commit lifetime factor over `{(r_i,g_i)}`. The prospective descriptor `K` itself does not receive those value-generation lifetimes because it never copied those values.

## Novelty target

Partial evaluation, closures, lazy tensors, computation graphs and external-memory pointers are all prior art. Therefore PNS cannot claim novelty for any of those ideas individually.

The narrower architecture hypothesis is:

> **Neural hidden state / attention output may be a generation-safe live-reference continuation over future mutable world state, retained across world revisions as the model's native activation form, with explicit dereference/commit semantics separating immortal prospective computation from temporal numeric materializations.**

R340–R343 test progressively stronger instances of this semantics. R345 model-checks the temporal contract itself independently of any one numeric representation.
