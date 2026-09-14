# R332 Result — Frozen-Qwen Terminal World Port

Status: **negative quality result; terminal side-rail formulation is rejected as the V5 integration architecture.**

## Attempt

A frozen pretrained Qwen2.5-0.5B produced reusable language/query features. A small terminal J side rail received those B features plus two current typed values. The world itself was never written into Qwen parameters and world rebinding required zero gradient steps.

## What worked structurally

- frozen backbone: **true**
- B feature delta across a complete world rewrite: **0.0**
- frozen-backbone control-logit delta after J training: **0.0**
- current-world path beat the stale-parametric control after rebinding by **+22.01 percentage points**
- cached-B J-only update path was roughly **10,198x** cheaper than recomputing the frozen B representation in this microbenchmark

## What failed

The architecture did not preserve enough task quality:

- home accuracy: **57.42%**
- fully rebound future-world accuracy: **53.78%**
- future + held-out language-template accuracy: **34.77%**
- held-out value-pair accuracy: **38.15%**

The strict quality gate therefore failed.

Report SHA256: `5d5bd49e7ae2cece3d62f845a4d0b399f4181ce36d3ccbcb060024a2efb07005`.
Artifact ZIP SHA256: `8a5d00c1dc3e405564f5381e5645787b5aeaecda0e7c05e6a0415ca5f8b34cfd`.

## Decision

Do **not** promote “final frozen hidden state + generic MLP World Port” as the architecture.

The failure suggests that language control and current-world computation should not be compressed into one weak terminal fusion head. Two stronger paths are now being tested:

1. **R333:** inject the typed World Port *inside* the pretrained decoder and let frozen late Transformer layers perform the J transformation;
2. **R332b/R334 direction:** explicitly decode a compact neural ISA/control state from frozen B, then execute current values through a typed/factorized J operator rather than asking a single MLP to rediscover both language semantics and arithmetic.

This is a useful kill result: B/J separation remains structurally sound, but the interface must expose the right computation structure rather than merely concatenate features.