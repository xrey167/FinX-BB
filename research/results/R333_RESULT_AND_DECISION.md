# R333 Result — First Mid-Layer Qwen World Port Attempt

Status: **inconclusive / killed by wall-time; no quality conclusion may be drawn from this run.**

## Attempt

The frozen Qwen2.5-0.5B decoder was split into an early reusable B prefix and four frozen late J layers. A trainable typed residual World Port was injected at the split and optimized by backpropagating through the four frozen late decoder layers.

## Outcome

The CI job reached the experiment itself but exceeded the 30-minute job budget before producing an evaluation report. The runner cancelled the process at approximately 29m44s of experiment time.

No accuracy, exactness or future-rebinding result was emitted. Therefore R333 is **not a negative mechanism result**; it is a negative implementation/experiment-design result.

Workflow run: `34857330155`.
Artifact ZIP SHA256: `b4ce6601084d41a6c835ee50d8962c764b2c480df450ef22c2c89f037ca57520` (stdout only; no completed report).

## Decision

Do not spend the CPU budget repeatedly backpropagating through several frozen pretrained layers merely to learn a tiny World Port.

R333b should separate the problems more aggressively:

1. typed/compiled current-world execution determines a compact result/control code;
2. a very small **mid-layer neural materializer** maps that code to a residual;
3. only one frozen late Qwen layer is traversed during the materializer training/evaluation;
4. training optimizes a tiny set of class/residual vectors rather than a large B+value fusion network;
5. later integration composes this materializer with the independently tested frozen-Qwen Neural ISA compiler.

The machine principle remains: B stays world-independent; the failed piece here is the computationally wasteful training procedure.