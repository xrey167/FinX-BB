# R317 Result — Binding-Equivariant CKVM

Status: **architectural binding-generalization gate passed; not DoD and novelty of the complete architecture is not yet established.**

## Core hypothesis

Do not ask a generic network to learn that mutable facts should be independent of entity identity. Remove the shortcut structurally.

After canonical address resolution, the value operator obeys:

`Exec(op, a, b, W) = R(op, W[a], W[b])`

and `R` has **no access to entity identities `a` or `b`**.

Entity identity exists only on the address path. Current world values exist only on the data path.

## Result

Three seeds, 64 entities, 64 values, four operators. Every future world was a derangement of the training binding, so **no entity retained its training value**. Entity/operator combinations were also deliberately removed from training.

Monolithic shortcut model:

- home accuracy: **99.41%**
- deranged future-world accuracy: **28.76%**

Binding-equivariant CKVM model:

- home accuracy: **99.71%**
- deranged future-world accuracy: **99.61%**
- held-out entity/operator accuracy: **99.62%**
- max logit change when post-resolution entity identities were randomized: **0.0**

Typed CKVM ALU path:

- deranged future-world exact accuracy: **100%**

Binding-equivariant gain over monolithic future accuracy: **+70.85 percentage points**.

Report SHA256: `21774a45fda97e2f2012d234de159b455b2d3fd8c5cceacf4003c74d3c1d115d`.
Artifact ZIP SHA256: `4d77cdb5a51f1b5f167396e1b3d5c7fc7d346dd07deb360777b2ed1245e78b7a`.

## Decision

Promote **Binding Equivariance** to a core CKCA invariant:

1. natural language / aliases may identify canonical Pods;
2. identity may participate in address resolution;
3. after current Pod values are read, the value-execution path must not receive entity identity again unless the operation semantically requires identity itself;
4. mutable facts therefore cannot be recovered from an identity shortcut inside the operator network;
5. future world rebinding becomes a runtime data change rather than a learned pair-generalization task.

R315 showed that curriculum randomization alone did not create this behavior. R317 shows that the stronger structural information-flow constraint does.

## Why this matters to the main vision

This is a direct answer to the old held-out binding failure: **the architecture removes the degree of freedom that caused the failure instead of hoping training will suppress it.**

It also aligns with lifecycle control. If mutable value bytes are not represented in the reusable identity/skill path, edits and revocations need only affect the current data/J path.

Permutation-equivariant architectures, semantic parsing, typed execution and tool use all have prior art. The candidate novelty is therefore not “equivariance” by itself; the research target remains the complete composition with canonical Symlink/Pod generations, neural lifetime typing, transactional commit validation, post-commit CLFD, and late-bound neural materialization.
