# R332b Result — Frozen-Qwen Neural ISA + Typed World ALU

Status: **strong partial result; strict held-out-language gate failed. Do not lower the gate. Not DoD.**

## Why R332b exists

R332 showed that a generic terminal MLP asked to infer language semantics and world arithmetic jointly was too weak. R332b factorizes the interface:

```text
frozen Qwen semantic B features
    -> compact neural ISA decoder
    -> typed current-world ALU
```

The frozen Qwen side never receives mutable world values. The ISA decoder predicts only the allowlisted operation. Current entity values are read from the World ABI after compilation and executed by a typed operator.

## Result

- neural ISA probe parameters: **881,924**
- train operation accuracy: **100%**
- ordinary eval operation accuracy: **100%**
- held-out language-template operation accuracy: **88.0208%**
- B feature delta across world rewrite: **0.0**

End-to-end typed execution:

- home-world accuracy: **100%**
- fully rebound future-world accuracy: **100%**
- future + held-out language-template answer accuracy: **91.875%**
- stale-parametric future control: **32.7083%**
- future gain current-world vs stale world: **+67.2917 percentage points**
- world-rebinding gradient steps: **0**

The microbenchmark cached-ISA/current-world execution path was roughly **182,302x** cheaper than rerunning the frozen Qwen language compile, but this ratio is dominated by the asymmetry between a full CPU Qwen forward and a tiny typed ALU and should not be interpreted as a production serving speedup.

Strict gate outcome: **FAIL**, because held-out-language operation accuracy `88.02% < 90%`.

Report SHA256: `d11288266fc9c2748468ed973df63b520518e60eb7846e75111b6a0ade4ec6c3`.
Artifact ZIP SHA256: `87c46b3f7cd813f237127639b39abdc6d5bfd6bc2f0b46a04a6ae0bc0ddc8f0c`.

## Decision

The factorized B -> compact ISA -> current-world execution interface is substantially stronger than R332's generic fusion head and clearly preserves future rebinding. However, **the compiler still needs better linguistic generalization**.

Do not weaken the 90% held-out-language criterion. R332c should improve the frozen-Qwen compiler by increasing paraphrase/word-order diversity and using an operation-prototype/contrastive objective or stronger pooling, while keeping the mutable world and typed ALU unchanged.

The important research separation remains:

> failures of language compilation and failures of mutable-world lifecycle should be independently measurable. Current-world execution is already exact when the compiler emits the correct ISA.