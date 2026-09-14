# PNS Novelty Audit V3 — Belief-Lookback Pressure

Date: 2026-09-14
Status: **candidate narrowed again; no novelty/breakthrough claim.**

## New high-overlap prior art

A 2025/2026 interpretability line, *Language Models use Lookbacks to Track Beliefs* (Prakash et al., arXiv:2505.14685, under review ICLR 2026), provides substantially closer prior art than generic pointer networks.

The paper reports that Llama residual streams can contain low-rank reference information which acts as an **address** alongside a payload at one token and as a **pointer** at a later token. Later attention dereferences the pointer against the addresses and brings the corresponding payload into the pointer token residual stream.

The authors explicitly describe a layer interval in which the final-token residual state carries a pointer but not yet the payload, followed by later layers that dereference it and retrieve the payload.

This kills any broad PNS claim of the form:

> “Transformers have not previously been shown to carry reference-like/pointer state in the residual stream across layers and dereference it later.”

That statement would now be false.

It also weakens any novelty framing around “reference-valued residual state” by itself. Internal reference information and delayed payload retrieval clearly have direct Transformer prior art.

---

# Exact overlap with PNS

Belief-lookback and PNS now share several mechanisms:

1. **Reference information exists inside the residual stream.**
2. It can persist across multiple layers before its payload is retrieved.
3. A later attention operation performs a genuine dereference.
4. The pointer state and payload state are temporally/layer-wise separable.
5. The reference may be low-rank and compact relative to the payload.

These points can no longer be used as novelty arguments for PNS.

---

# Remaining semantic distinction

The lookback payload is part of the already-encoded in-context Transformer computation. The pointer/address mechanism selects among payloads represented in the current prompt/residual context.

The surviving PNS research question is narrower:

> Can the retained reference-bearing Transformer state be made **authoritative-world prospective**: byte-stable while an external governed payload changes after cache construction, with the later dereference resolving against that future/current external version rather than the historical payload represented during the original forward pass?

This adds requirements that the belief-lookback work does not, in the material reviewed here, claim to provide:

- payload can be changed/revoked **after** the reference-bearing neural state was cached;
- the same retained reference state is reused without value-derived patching;
- dereference resolves against an external canonical world cell's **current generation**;
- a same-value generation transition remains a lifecycle conflict;
- every materialized descendant carries a generation read-set/lifetime;
- publication is rejected if a referenced generation races between dereference and commit;
- a stale historical payload in earlier neural/cache state is not allowed to reassert authority.

Thus the candidate is no longer “pointer-like Transformer residual state.” It is a **temporal authority semantics for such state**.

---

# Stronger candidate wording

The best currently defensible candidate to test is:

> **Authoritative Prospective Neural State (A-PNS): a retained Transformer reference state whose denotation is intentionally late-bound to a versioned external authoritative world after cache construction; the retained state excludes decoder-observable historical governed payload, ordinary world writes require no cache patch, and the later materialization produces generation-scoped neural descendants validated at commit.**

This wording deliberately concedes the major pointer/residual-stream prior art.

---

# Distinguishing experiment required

A decisive benchmark against lookback-like historical state should use this temporal sequence:

```text
T0  compile / cache reference-bearing Transformer state
T1  mutate or revoke authoritative external payload
T2  reuse exactly the same cached reference state
T3  dereference / materialize
T4  prove output equals fresh T2-world recomputation
T5  prove old payload cannot be served via the governed path
```

Matched retrospective control:

```text
T0  encode pointer + historical payload into numeric Transformer state
T1  external payload changes
T2  reuse historical numeric state
T3  show divergence or need for repair/recompute
```

R347 already implements this temporal distinction synthetically. R349/R350 are intended to establish it inside a frozen pretrained Transformer path.

---

# Why R351 still matters

The Prospective Quotient result now provides the strongest conceptual support for the distinction.

If the same retained cache must be exactly correct for **every future world** with zero ordinary-value cache maintenance, then all historical-world variants of that cache must be observationally equivalent under every future current world. Any decoder-observable historical payload component violates the contract unless it is repaired.

A conventional in-context lookback can legitimately retain a historical payload because its semantic task is to reason over that fixed context. A-PNS imposes a different contract: the governed answer must follow an authority state that can change independently after the reference state was created.

So the research question is not whether pointers can exist in Transformers. They clearly can. It is whether **mutable-world authority can become a native temporal type of Transformer reference/cache state**.

---

# Current novelty verdict after this pressure test

**Broad reference-valued Transformer novelty: rejected.**

**Pointer/residual-state/delayed-dereference novelty: rejected.**

**Authoritative future-world temporal semantics for retained Transformer references: still unresolved and worth testing.**

No claim should be made until direct literature + patent searching finds no equivalent architecture and the pretrained internal experiments succeed.

The project should now use `A-PNS` when discussing the candidate novelty, reserving generic `PNS` for the broader architecture family.
