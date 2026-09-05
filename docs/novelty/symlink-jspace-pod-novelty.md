# Symlink J-Space Pod — novelty thesis and falsification plan

Status: research hypothesis, not a novelty or patentability claim.
Date: 2026-09-05

## Problem
Current editing systems increasingly separate addressing and storage (SERAC, WISE, REP, DKME), use modular memories/adapters (MECA, MindBridge, iReVa), or externalize knowledge into detachable memory tokens (Knowledge Externalization, ICLR 2026). Anthropic's J-space work identifies a sparse, vocabulary-linked, broadcast workspace in intermediate residual representations. None of those facts alone makes our current pod/symlink design novel.

## Candidate technical novelty: Workspace-Native Indirection (WNI)

Treat a knowledge edit as a versioned object graph rather than as a parameter update or a retrieved text record:

1. **Canonical Pod** — one mutable/deletable canonical knowledge object owns the payload and lifecycle state.
2. **Symlink aliases** — surface forms, paraphrases, relations and dependent references carry no duplicate payload; they resolve to a stable pod identity through an explicit indirection layer.
3. **J-space address contract** — the router is trained/constructed against a sparse J-space/workspace signature (or a distilled approximation), not merely a raw hidden-state semantic key. The signature defines the *scope* in which the pod may activate.
4. **Two-stage resolve/broadcast** — query -> workspace signature -> symlink/pod ID -> payload -> controlled residual/J-space injection. Addressing, identity, storage, and broadcast are distinct objects.
5. **Deletion by reachability closure** — DELETE/SHRED invalidates the canonical pod and its capability/epoch; aliases cannot retain payload. A certificate enumerates every reachable alias/pointer and proves that no live route resolves to the deleted epoch.
6. **Counterfactual workspace certificate** — before/after deletion, J-lens/J-space probes test whether the deleted concept can still be summoned into the broadcast workspace, in addition to output-level refusal and key/reconstruction attacks.
7. **MVCC/version semantics** — edits create pod versions/epochs. Symlinks resolve only to the current committed epoch; rollback and revocation do not rewrite unrelated model weights.

The potentially publishable claim is therefore not 'external memory', 'routing', 'pods', 'symlinks', or 'J-space' individually. It is the **combination of explicit canonical identity + pointer-only aliases + workspace-native scope/addressing + versioned deletion closure + workspace-level deletion verification**.

## Why this might be technically distinct

- Parameter editors (ROME/MEMIT) mutate weights and do not provide canonical object identity/pointer closure.
- Memory editors (SERAC/WISE/DKME/REP) address edits but generally treat memory entries/partitions as the edit substrate rather than a versioned object graph with pointer-only aliases and deletion reachability certificates.
- Modular/externalized approaches (MECA, MindBridge, iReVa, Knowledge Externalization) provide detachable/editable modules or tokens, but that is not the same as one canonical payload with explicit alias indirection and closure-based deletion semantics.
- J-space work is primarily a representation/readout/broadcast result; using a J-space signature as an edit scope/address ABI and as a deletion-verification surface is a separate hypothesis that must be demonstrated.

These distinctions require a formal prior-art search before any novelty/patentability claim.

## Breakthrough experiment family

### E-000055 — Symlink closure advantage
Compare duplicated alias records vs canonical pod + pointer-only aliases at equal fact coverage. Perform UPDATE, REVOKE and SHRED on the canonical fact. Measure unseen-paraphrase propagation, stale-alias leakage, deleted-object recovery, unrelated KL, and edit cost as alias count grows. Required signature: deletion/update cost approximately independent of alias count for the symlink design while duplicated-record baselines grow with alias count, with no accuracy/locality loss.

### E-000056 — J-space address ABI
On an open model supported by Jacobian Lens, compare routing keys from raw residual state, conventional semantic embeddings, and sparse J-space signatures. Train only on a subset of paraphrases. Required signature: J-space routing improves the robustness-specificity frontier on unseen paraphrases and generic prose, not merely mean accuracy.

### E-000057 — Workspace deletion certificate
After SHRED, attack the deleted concept through direct queries, unseen paraphrases, aliases, multi-hop prompts, prefix/suffix perturbations, key-channel reconstruction, and J-space 'summon' probes. Required signature: output leakage and workspace reactivation both fall to preregistered null/control levels while neighboring pods remain stable.

### E-000058 — Versioned symlink semantics
Create fact v1 -> v2 -> revoked -> restored sequences with concurrent aliases. Required signature: every alias observes the committed epoch, no stale payload survives SHRED, rollback restores only an explicitly retained version, and unrelated pods remain invariant.

## Strong claim only if all four survive
If E-000055..058 survive multiple seeds/models and strong baselines, the defensible research claim becomes:

> A workspace-native, versioned knowledge-object architecture can give an LLM one canonical mutable knowledge identity whose many linguistic access paths are pointer-like rather than payload-bearing, allowing edits and deletion to propagate through an explicit closure while J-space defines and audits the neural broadcast scope.

That is the novelty target. Until the experiments and prior-art search are complete, label it a candidate technical novelty, not a proven first or a patentable invention.
