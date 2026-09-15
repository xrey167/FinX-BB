# Symlink J-Space Pod — novelty thesis and falsification plan

Status: research hypothesis, **not** a novelty or patentability claim.
Date: 2026-09-05

## Decisive update: J-space is NOT the address bus

`E-000062` directly falsified the first version of this thesis. On frozen GPT-2, a country-token J-lens signature used as the scope/address representation failed every preregistered criterion across five subject/template splits:

- worst J-space joint route/abstain accuracy: **0.3235** vs required 0.75;
- worst positive correct-route rate: **0.25** vs required 0.70;
- worst specificity: **0.20** vs required 0.80;
- worst J-space advantage over semantic embedding keys: **-0.1029** (required +0.05);
- worst advantage over matched random projection: **-0.1618** (required +0.05);
- worst scope-balanced advantage over raw residual: **-0.0571** (required +0.02).

This is not a tuning miss. The tested idea — **use a raw sparse J-space signature itself as the memory scope/address ABI** — does not earn an architectural role. It is removed from the novelty claim unless a materially different mechanism later supplies new evidence.

That negative result improves the architecture: the neural workspace and the memory control plane should be **independent**, not coupled. J-space is retained only as an external causal audit surface.

## The prior-art boundary is explicit

The broad ingredients are **not** ours and must never be claimed individually:

- **Canonical records + aliases/pointers:** database normalization/indirection is old; Raeesi & Roed (2026) explicitly propose canonicalization at write time with aliases/paraphrases stored as pointers into one canonical record for limited-memory LMs. `so/closure.py` therefore already states: *the pod is not a new idea*.
- **MVCC/versioning:** established database technology. Our `MVCCStore` is an application of that idea to a knowledge layer, not a claim to MVCC.
- **External/editable memories:** SERAC, WISE, MindBridge, iReVa, MECA and Knowledge Externalization all establish modular or external knowledge/edit memory as prior art.
- **Address/storage decoupling:** DKME (ACL Findings 2026) explicitly separates semantic addressing from partitioned memory storage. Decoupling by itself is not novel.
- **Scope classification/routing:** SERAC-lineage editors and WISE already route queries to edit memories. The August 2026 INLAY negative result further shows that many editing benchmarks cannot even measure abstention without explicit negative queries; our evaluation therefore includes withheld-edit and generic negatives.
- **J-space/Jacobian Lens:** Gurnee et al. (Anthropic, 6 July 2026) introduce the Jacobian lens and show a sparse verbalizable/broadcast workspace. J-space itself is not ours.
- **J-lens as an unlearning audit:** Song et al., *Measure, Don't Optimize: Forecasting Recovery in LLM Unlearning*, arXiv:2608.11408 (11 Aug 2026), introduce **J-Access**, explicitly using the Jacobian lens as an independent internal audit of residual accessibility. They also show directly optimizing the audit can create audit evasion. Therefore a J-lens audit alone is not novel and must never be optimized as our deletion objective.
- **Representation-level unlearning audits:** 2025–2026 work on superficial editing/unlearning, latent recovery, information decomposition and mechanistic unlearning already shows that output forgetting is insufficient. A hidden-state audit alone is not our contribution.

References fixing this boundary:

- SERAC: Mitchell et al., ICML 2022, *Memory-Based Model Editing at Scale*.
- WISE: Wang et al., arXiv:2405.14768.
- DKME: Zheng et al., Findings ACL 2026, `https://aclanthology.org/2026.findings-acl.792/`.
- Knowledge Externalization: Li et al., ICLR 2026, `https://proceedings.iclr.cc/paper_files/paper/2026/hash/7e9c2053258b1bdd32ff2654802cd594-Abstract-Conference.html`.
- Anthropic workspace/J-lens: Gurnee et al., 2026, `https://transformer-circuits.pub/2026/workspace/`.
- Scope-benchmark negative result / INLAY: Singh, arXiv:2608.26292 (26 Aug 2026).
- J-Access: Song et al., arXiv:2608.11408 (11 Aug 2026).
- Illusion of Erasure in KE: Basani & Chhabra, arXiv:2606.23276 (22 Jun 2026).
- Mechanistic Unlearning: Guo et al., ICML 2025.

## Revised candidate technical novelty: Causally Attested Versioned Indirection (CAVI)

The candidate is now a **split-plane architecture**, not “J-space routing.”

### Plane A — memory control plane

1. **One canonical knowledge identity.** A pod has a stable identity and owns payload + lifecycle/version state. Linguistic aliases do not own payload copies.
2. **Pointer-only alias graph.** Surface forms and dependent references resolve to pod identity / generation rather than duplicating payload.
3. **Explicit scope state machine.** A conventional learned/semantic/key router may decide whether memory is in scope, but it must expose a hard three-way contract: `BYPASS`, `RESOLVE(pod)`, or `UNKNOWN`. `E-000060` tests this separation. Scope is evaluated with true negatives.
4. **Version/generation semantics.** Aliases resolve the current committed generation. Revocation, rollback, evict, shred and delete have distinct semantics; stale generations cannot silently re-enter service.

### Plane B — neural data/broadcast plane

5. **Resolve then broadcast.** The model receives a payload only after scope and pointer resolution. Address, identity, storage and injection are distinct objects.
6. **Exact bypass target.** Out-of-scope text should execute the frozen base path with no memory injection, rather than relying on a soft router whose weights merely become small.

### Plane C — independent attestation plane

7. **Reachability certificate.** Store-side closure proves that no live pointer/path reaches the invalid pod generation under a declared workload.
8. **Independent causal workspace audit.** J-space/J-lens is used only to measure whether the removed payload remains accessible/broadcast after the operation. It is never an optimization target. Post-operation measurements are compared against a **never-memory** control so prompt identity cannot masquerade as residual knowledge.
9. **Attack composition.** Output, key-channel, reconstruction, stale-generation, adversarial elicitation and J-space audits must agree. A pass on one surface cannot certify deletion.

### What may actually be distinct

The narrow research target is the **composed lifecycle contract**:

`scope state machine -> pointer identity -> versioned canonical pod -> controlled broadcast`

paired with an **independent two-domain certificate**:

`pointer/reachability closure AND causal neural-access audit`,

where both certificates refer to the **same pod identity/generation**.

None of pointer sharing, MVCC, external memory, semantic routing, J-lens auditing or adversarial unlearning auditing is individually new. The open question is whether prior work has already specified and demonstrated this exact cross-layer contract: a version-qualified external knowledge object with pointer aliases, one-operation lifecycle propagation, exact bypass, stale-generation resistance, and an independent causal audit tied to the same object identity. That is now the only novelty seam worth defending.

## Evidence already established inside this repository

These are substrate facts, not novelty proofs:

- `E-000020`: explicit symlink cells and dereference reads work in a frozen-GPT2 adapter and can propagate one canonical update through aliases.
- `E-000032` / `so.closure`: fact-level deletion resilience/closure distinguishes duplicated storage from a canonical pod and carries a certified lower bound when derivations permit it.
- `E-000037`: store closure and workspace closure are different quantities; a canonical record does not imply one neural carrier.
- `E-000042`: J-lens support can be subjected to explicit causal ablation and post-ablation probe checks; readout blinding is not accepted as deletion.
- `E-000044`: the pod objective explores allocation of access paths toward shared representational carriers, but is not CAVI.
- `E-000050..060`: scope/locality is the hard performance seam. `E-000060` tests a scope-before-routing state machine.
- `E-000061`: **passed 5 seeds × 8 alias fan-outs (1..128)**. Duplicate fact closure grows `k+1` while canonical pod closure remains exactly 1; one canonical eviction closes all aliases. This validates the pointer substrate but is known indirection behavior, not novelty.
- `E-000064`: **passed 20 lifecycle runs** (5 seeds × alias counts 1,4,16,64). UPDATE, rollback, revoke/restore, shred/resign, evict/restore and irreversible delete were observed consistently by every pointer alias with zero payload copies. Again: substrate, not novelty.
- `E-000062`: **negative**. J-space is rejected as the direct address/scope ABI.

## Active breakthrough experiment family

### E-000063 — composed Workspace-Pod deletion certificate

The explicit symlink GPT-2 adapter is tested with one canonical SHRED. At the first broadcast site, J-lens directions for the pod objects audit ACTIVE, SHRED and NEVER-memory states. Crucially, a probe is trained on ACTIVE states only and its same weights are applied to SHRED and NEVER, preventing the alias text itself from being misread as memory recovery. Output deletion, J-space residual, final-state residual and unrelated-pod locality are preregistered together.

A pass would support the **attestation composition**, not J-space routing and not legal novelty.

### E-000065 — first end-to-end CAVI prototype

Only after `E-000060` identifies a robust scope mechanism: connect the explicit `BYPASS / RESOLVE / UNKNOWN` state machine to an epoch-qualified pointer/pod resolver, and force exact zero injection on BYPASS. Compare the same memory under soft routing and semantic-router baselines. Do **not** optimize J-space audit scores.

### E-000066 — stale capability / generation attack

Attempt to revive a superseded or deleted pod generation using saved alias addresses, old link versions, rollback paths, replayed bank tensors, cached router outputs, serialized banks and crafted prompts. Passing means stale handles cannot cause the model to consume a non-current generation unless an explicit authorized rollback/restore operation reactivates it.

### E-000067 — audit independence / Goodhart control

Because J-Access shows an audit can be gamed if optimized, compare two systems: one never trained on the J-space audit and one explicitly penalized on it. The certificate is credible only if the unoptimized CAVI lifecycle passes independent recovery attacks, and direct J-space optimization does not receive credit merely for lowering the audit.

## Strong claim only if the composition survives

A defensible research claim would require, on multiple seeds and more than one public backbone where feasible:

- canonical pointer closure and temporal semantics pass;
- explicit scope negatives and exact bypass pass;
- one pod lifecycle operation propagates through unseen linguistic access paths;
- store reachability and independent causal workspace audits both pass against never-memory controls;
- key/reconstruction/adversarial-elicitation/stale-generation/recovery attacks fail;
- unrelated pods and generic text stay within fixed locality bounds;
- strong semantic/external-memory baselines show the gain is from generation-safe indirection + composed attestation, not generic memory routing.

Only then is the intended research-level claim:

> **A causally attested, versioned-indirection memory can bind many linguistic access paths to one canonical mutable knowledge identity, propagate one lifecycle operation across those pointer paths, enforce an exact bypass for out-of-scope computation, and certify the same pod generation independently in both pointer reachability and the model's causal broadcast pathway.**

This is the current technical-novelty target. It is still a research hypothesis, not a claim of legal patent novelty or “first ever,” until the composed experiments and professional prior-art search survive.
