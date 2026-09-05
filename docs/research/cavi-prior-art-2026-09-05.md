# CAVI prior-art boundary — 2026-09-05

This note is a novelty **falsification map**, not a novelty or patent claim.  The architecture must remain narrower than the union of established ingredients below.

## Current candidate property

**Causally Attested Versioned Indirection (CAVI)** is currently tested as an execution invariant, not as a new primitive:

> A canonical knowledge object is reached only through pointer-only aliases; BYPASS / RESOLVE / UNKNOWN are explicit; both the reference binding and referent incarnation are live-revalidated adjacent to *actual neural memory consumption*; any memory-derived state that survives beyond that boundary remains dependent on the same live referential witness and must be revalidated at later neural injection/consumption; no-memory scope is an exact bypass; store reachability and causal neural accessibility are audited independently.

The proposed differentiator is therefore **end-to-end dependency preservation from canonical referential identity to neural consumption**, including post-authorization derived neural material.  Epochs, HMACs, capabilities, pointers, transactions, authorization, freshness, audit probes and cryptographic erasure are not individually claimed.

## Prior art that removes broad claims

| Prior art | What is already established | Consequence for CAVI |
|---|---|---|
| SERAC (Mitchell et al., 2022, `arXiv:2206.06520`) | explicit edit memory + learned scope/routing + counterfactual model | external memory and routing are not novel |
| WISE (Wang et al., 2024, `arXiv:2405.14768`) | side memory, router, lifelong edit sharding/merging | dual memory, routing and sharding are not novel |
| DKME (Zheng et al., ACL Findings 2026) | decoupled semantic addressing and partitioned knowledge storage | address/storage decoupling is not novel |
| Knowledge Externalization (Li et al., ICLR 2026) | modular external memory tokens, reversible forgetting/restoration and post-externalization edits | removable/modular external knowledge is not novel |
| STALE (Chao et al., 2026, `arXiv:2605.06527`) | stale/invalid memory is a first-class agent-memory failure mode | freshness as a problem statement is not novel |
| MemTX (Li et al., 2026, `arXiv:2607.23929`) | snapshot-isolated transactional belief commit, provenance/validity, cascade repair | transactions, validity metadata and commit discipline are not novel |
| Commit-Time Authorization (Santos-Grueiro, 2026, `arXiv:2607.10487`) | earlier authority evidence cannot authorize a later durable effect after invalidation | revalidation at a commit boundary is not novel |
| J-Access / *Measure, Don't Optimize* (Song et al., 2026, `arXiv:2608.11408`) | Jacobian-lens accessibility is useful as an independent recovery-risk audit; optimizing it causes audit evasion / worse recovery | J-space/J-lens stays audit-only; never a CAVI router or training objective |
| canonical pointers / indirection / MVCC / ABA prevention | long-established systems semantics | pointer identity, generations/incarnations and ABA-safe handles are not novel |
| crypto-shredding | long-established cryptographic erasure by key destruction | keys/HMAC/key destruction cannot carry novelty |

## The current differentiating test

E-000066 showed that an old exported memory Bank can replay deleted knowledge after SHRED/DELETE.  E-000068 showed that a live-incarnation one-use capability can close that replay class, but that mechanism is established systems/security prior art.  E-000070 and E-000071 therefore move validation to real symlink neural consumption and compare full alias+pod referential validation against pod-only version checks and cached commit-time authorization.

The next stricter falsification is E-000073: **derived neural-state replay**.  It asks whether a post-authorized serialized activation can bypass row-level freshness entirely.  The attack captures the real post-symlink-read hidden state, changes alias/pod authority, then injects the stale tensor with `bank=None`.  Baselines are no guard, cached commit-time authorization and pod-only version validation.  Full CAVI only passes if it rejects stale derived state at the actual replay/injection hook while the fresh current incarnation remains usable.

If E-000073 exposes replay and full referential revalidation closes it, that still does not establish novelty.  Required next attacks include cached routing scores, cached resolved payload vectors, old aliases, delete/restore ABA, concurrent mutation, rollback/restore and replay races; followed by multi-seed/model replication and the E-000063 independent J-lens audit.

## Claim-killing conditions

Do **not** claim a CAVI novelty if any of the following is true:

1. A simple pod/version equality check performs identically on alias-relink and ABA attacks.
2. Commit-time authorization alone is sufficient when authority changes after resolution but before neural consumption.
3. Stale post-authorization derived neural state can be replayed without fresh dependency validation.
4. The boundary can be reduced to ordinary cache invalidation with no neural-consumption-specific invariant.
5. Exact BYPASS requires hidden memory injection or materially perturbs the base model.
6. J-space/J-lens must be optimized or used as a router for the architecture to work.
7. The composed benefit disappears across seeds/backbones or under fresh-current controls.

## Sources checked on 2026-09-05

- https://arxiv.org/abs/2206.06520 — SERAC / Memory-Based Model Editing at Scale
- https://arxiv.org/abs/2405.14768 — WISE
- https://aclanthology.org/2026.findings-acl.792/ — DKME
- https://proceedings.iclr.cc/paper_files/paper/2026/hash/7e9c2053258b1bdd32ff2654802cd594-Abstract-Conference.html — Knowledge Externalization
- https://arxiv.org/abs/2605.06527 — STALE
- https://arxiv.org/abs/2607.23929 — MemTX
- https://arxiv.org/abs/2607.10487 — Commit-Time Authorization
- https://arxiv.org/abs/2608.11408 — J-Access / Measure, Don't Optimize
