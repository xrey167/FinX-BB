# Symlink J-Space Pod — novelty thesis and falsification plan

Status: research hypothesis, **not** a novelty or patentability claim.
Date: 2026-09-05

## The prior-art boundary is now explicit

The broad ingredients are **not** ours and must never be claimed individually:

- **Canonical records + aliases/pointers:** database normalization/indirection is old; Raeesi & Roed (2026) explicitly propose canonicalization at write time with aliases/paraphrases stored as pointers into one canonical record for limited-memory LMs. `so/closure.py` therefore already states: *the pod is not a new idea*.
- **MVCC/versioning:** established database technology. Our `MVCCStore` is an application of that idea to a knowledge layer, not a claim to MVCC.
- **External/editable memories:** SERAC, WISE, MindBridge, iReVa, MECA and Knowledge Externalization all establish modular or external knowledge/edit memory as prior art.
- **Address/storage decoupling:** DKME (ACL Findings 2026) explicitly separates semantic addressing from partitioned memory storage. Decoupling by itself is not novel.
- **Scope classification/routing:** SERAC-lineage editors and WISE already route queries to edit memories. The August 2026 INLAY negative result further shows that many editing benchmarks cannot even measure abstention without explicit negative queries; our evaluation must include withheld-edit and generic negatives.
- **J-space/Jacobian Lens:** Gurnee et al. (Anthropic, 6 July 2026) introduce the Jacobian lens and show a sparse verbalizable/broadcast workspace. J-space itself is not ours.
- **Representation deletion/closure:** this repository already studies workspace closure and J-lens support certificates (`E-000037`, `E-000042`). A J-space deletion probe alone is therefore not the new claim either.

References used to fix this boundary:

- SERAC: Mitchell et al., ICML 2022, *Memory-Based Model Editing at Scale*.
- WISE: Wang et al., arXiv:2405.14768 / NeurIPS-era lifelong editing work.
- DKME: Zheng et al., Findings ACL 2026, `https://aclanthology.org/2026.findings-acl.792/`.
- Knowledge Externalization: Li et al., ICLR 2026, `https://proceedings.iclr.cc/paper_files/paper/2026/hash/7e9c2053258b1bdd32ff2654802cd594-Abstract-Conference.html`.
- Anthropic workspace/J-lens: Gurnee et al., 2026, `https://transformer-circuits.pub/2026/workspace/`.
- Scope-benchmark negative result / INLAY: Singh, arXiv:2608.26292 (26 Aug 2026).

## Narrow candidate technical novelty: Workspace-Native Versioned Indirection (WNVI)

The candidate is no longer “pods + symlinks + J-space.” That wording is too broad. The narrower architecture under test is a **causal workspace address ABI connected to an identity-preserving, versioned indirection graph, with a two-surface deletion certificate**:

1. **One canonical knowledge identity.** A pod has a stable identity and owns the payload plus lifecycle/version state. Linguistic aliases do not own payload copies.
2. **Pointer-only access graph.** Surface forms/aliases resolve to pod identity (or an epoch-qualified capability), so UPDATE/REVOKE/ROLLBACK act on identity rather than on every phrase-specific storage location.
3. **Workspace-native scope/address ABI.** The *permission and address signal* is derived from a causal J-space/workspace signature (or a distilled surrogate proven equivalent), not merely a generic semantic embedding. This signal is trained/evaluated on explicit positive **and negative** scope queries.
4. **Resolve, then broadcast.** The architecture separates: `(workspace scope/address) -> (pod identity/epoch) -> (canonical payload) -> (controlled broadcast/injection)`. Address, identity, storage, and broadcast are independently testable objects.
5. **Version/generation safety.** Aliases resolve the currently committed generation; stale capabilities cannot revive a revoked/deleted generation. Rollback is explicit version selection, not implicit stale-pointer behavior.
6. **Reachability certificate.** Store-side closure proves no live pointer/path resolves to the invalid generation under a declared workload.
7. **Causal workspace certificate.** Model-side J-space/J-lens interventions test whether the deleted pod can still be *summoned/broadcast* through aliases or paraphrases. Output refusal alone is insufficient; a refitted/probe or counterfactual workspace attack must also fail.
8. **Composition theorem/contract target.** The eventual technical claim must be compositional: if the pointer graph has closure 1 for the target pod and the workspace scope/broadcast certificate passes, then one lifecycle operation changes every alias while unrelated pods remain invariant within preregistered locality bounds.

The possible research contribution is therefore the **cross-layer contract** between a causal neural workspace and a database-like versioned object identity — not either layer by itself.

## Evidence already established inside this repository

These are substrate facts, not novelty proofs:

- `E-000020`: explicit symlink cells and dereference reads work in a frozen-GPT2 adapter and can propagate one canonical update through aliases.
- `E-000032` / `so.closure`: fact-level deletion resilience/closure distinguishes duplicated storage from a canonical pod and carries a certified lower bound when derivations permit it.
- `E-000037`: store closure and workspace closure are different quantities; a canonical record does not imply one neural carrier.
- `E-000042`: J-lens support can be subjected to explicit causal ablation and post-ablation probe checks; readout blinding is not accepted as deletion.
- `E-000044`: a “pod objective” explores allocation of one fact’s access paths toward shared representational carriers; this is related but does not implement WNVI.
- `E-000050..060`: the current performance line isolates scope/locality as the hard seam. `E-000060` tests an explicit scope-before-routing state machine rather than soft post-hoc gating.

### Newly registered substrate controls

The earlier draft reserved E-000055..058, but those IDs were consumed by the performance/locality line. The novelty track therefore starts at **E-000061**.

- **E-000061 — pod closure scaling control.** Duplicated payload rows vs one canonical FACT + pointer-only LINK aliases across growing fan-out. Must show `O(k)` logical edit/erasure operations for duplication versus `O(1)` canonical payload mutation/eviction while every alias loses reachability. This is a control for a known indirection benefit, not novelty.
- **E-000064 — versioned symlink lifecycle audit.** UPDATE/ROLLBACK/REVOKE/SHRED/EVICT/RESTORE/DELETE must be observed consistently by every alias without relinking, and aliases must remain payload-free. Again, a substrate contract, not novelty.

## Breakthrough experiment family

### E-000062 — causal J-space address ABI

On an open model with the repository’s Jacobian Lens implementation, compare scope/address keys built from:

A. raw residual state,
B. matched-dimensional random/residual projection,
C. conventional semantic/output-aligned controls,
D. a sparse J-space signature.

Training sees only a subset of paraphrases. Evaluation includes disjoint held-out paraphrases **and explicit negatives**: withheld-edit questions of the same form plus generic prose. Thresholds are selected on training/validation negatives only. A J-space ABI claim requires a preregistered improvement in the worst-split robustness-specificity frontier, not merely higher mean classification accuracy. A representation that only predicts the answer token but does not reject out-of-scope facts does not pass.

### E-000063 — composed Workspace-Pod deletion certificate

Take the explicit symlink GPT-2 adapter (`E-000020`) rather than a standalone natural-fact probe. For one canonical pod with multiple aliases, perform one lifecycle operation and attack through direct keys, unseen paraphrases, aliases, multi-hop prompts, prefix/suffix perturbations, key-channel reconstruction, and causal J-space “summon”/support probes. Compare:

- output-only deletion,
- store reachability closure,
- workspace/J-space reactivation,
- unrelated-pod collateral.

The interesting result is a **composition** result: one canonical lifecycle operation closes every pointer path and every certified workspace broadcast route without editing unrelated model weights. If J-space adds no information beyond the existing output/key battery, the J-space part of WNVI is weakened.

### E-000065 — first end-to-end WNVI prototype

Only if E-000062 supports J-space as a useful scope/address coordinate system: route an external edit with a J-space-derived address/scope signal to an epoch-qualified symlink, resolve one canonical pod, then broadcast the payload through a controlled injection. Compare against the same memory with raw-residual/semantic routing and against the existing soft router. The benchmark must contain positive and negative scope examples.

### E-000066 — stale capability / generation attack

Attempt to revive a superseded or deleted pod generation using saved alias addresses, old link versions, rollback paths, replayed bank tensors, cached workspace signatures and crafted prompts. Passing means stale handles cannot cause the model to consume a non-current generation unless an explicit authorized rollback/restore operation reactivates it.

## Strong claim only if the composition survives

A defensible research claim would require all of the following, on multiple seeds and more than one public backbone where feasible:

- canonical pointer closure and temporal semantics pass;
- J-space scope/addressing beats strong matched controls on held-out positives **and negatives**;
- one pod lifecycle operation propagates through unseen linguistic access paths;
- store reachability and causal workspace/J-space certificates both pass;
- key/reconstruction/stale-generation attacks fail;
- unrelated pods and generic text stay within fixed locality bounds;
- comparisons against SERAC/WISE/DKME-style semantic routing establish that the gain is from the workspace-native/versioned-indirection contract rather than generic external memory.

Only then would the research-level claim be:

> **A workspace-native versioned-indirection architecture can bind many linguistic access paths to one canonical mutable knowledge identity, use a causal workspace signature as the scope/address contract, and compose pointer-reachability with workspace-level causal verification so that a single lifecycle operation propagates across aliases without leaving a live stale generation or broadcast route.**

That is the novelty target. It is still a hypothesis, not a claim of legal patent novelty or “first ever,” until the experiments and a professional prior-art search support it.
