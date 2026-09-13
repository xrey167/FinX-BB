# R294 Novelty Kill Matrix — 2026-09-13

Purpose: aggressively remove claims that are already occupied by prior work. This document narrows the research claim; it does **not** assert novelty.

## 1. Editable / composable ordinary KV is occupied

### Models Take Notes at Prefill: KV Cache Can Be Editable and Composable — arXiv:2606.17107

Reported result: overwriting only a changed field's own K/V can fail because downstream prefill states already contain field-conditioned conclusions; the paper calls these downstream states notes. It uses append-only errata and RoPE-repositioned KV composition, reporting strong latency gains and high logit similarity.

**Killed claims:**

- "we discovered that KV is editable";
- "we can move/reposition cached neural state";
- "we can cheaply append a correction to stale KV";
- "we can compose precompiled KV skills".

**Pressure on So:** source-only edit is not enough for lifecycle integrity because downstream state can retain the old generation. R292 therefore tests transitive lifetime closure.

## 2. Policy-directed KV span editing is occupied

### Leyline: KV Cache Directives for Agentic Inference — arXiv:2606.01065

Reported mechanism: an external policy issues remove/replace directives to a serving primitive; per-architecture kernels preserve positional correctness, including RoPE correction and selective re-prefill modes.

**Killed claims:**

- "external lifecycle policy can edit KV";
- "remove/replace a cached span without global reprefill";
- "policy and serving kernel can be separated".

**Pressure on So:** the claim must be about canonical neural knowledge lifetime and transitive causal admissibility, not merely cache editing APIs.

## 3. Rollback consistency is an identified cache-integrity problem

### Aborted but Not Forgotten: KV-Cache Retention Breaks Rollback Consistency in Language Agents — arXiv:2608.15939

Reported result: logical transcript rollback can diverge from the state still attended through retained KV. Rebuilding/restoring the correct cache closes the tested channel.

**Killed claims:**

- "we are first to notice stale attended state after logical rollback";
- "transcript state alone defines model state".

**Pressure on So:** authority/lifecycle must cover the state actually admitted to model computation, including restart, rollback, replica, tool and transcript caches.

## 4. Exact deletion depends on memory representation

### Subtract or Replay? Exact Deletion from Language-Model Memory — arXiv:2607.27539

Reported result: addressable record influence can sometimes be algebraically removed; influence transformed by later recurrent/shared-state writes can require replay/rebuild for exact deletion.

**Killed claims:**

- "exact deletion is a universal local subtraction";
- "masking the original record proves counterfactual never-ingested equivalence".

**Pressure on So:** CTPT structurally isolates mutable Port state from persistent Base-KV so that exact deletion does not require replaying the entire ordinary linguistic cache. Intentionally persistent derived states must carry exact lifetime dependencies.

## 5. KV compression is deeply occupied

### KV Cache Transform Coding for Compact Storage in LLM Inference — arXiv:2511.01815

Reported direction: transform coding combining PCA-like decorrelation, adaptive quantization and entropy coding, with large compression factors while retaining benchmark utility.

### MixKVQ and related mixed-precision KV work

Keys and values have unequal error sensitivity; mixed-precision, channel/groupwise quantization and query-aware allocation are established techniques.

**Killed claims:**

- "our novelty is low-bit KV";
- "shared anchors/residual KV compression are the invention";
- "our novelty is that keys and values need different precision".

**Pressure on So:** R290's K3/V3 result is only feasibility evidence that a Port payload need not remain BF16-size. It is not a novelty claim.

## 6. KV itself as a representation / reasoning substrate is occupied

### Beyond Speedup — Utilizing KV Cache for Sampling and Reasoning — arXiv:2601.20326

Reported direction: KV-derived representations can be reused for downstream reasoning/sampling tasks rather than only decoding acceleration.

**Killed claims:**

- "KV can be more than an inference cache";
- "KV is a model-native representation".

## 7. Existing external/editable neural memory families remain close prior art

Claim space also overlaps with Larimar, Memory^3, MEMORYLLM, TransMem, Prometheus Mind, TF-Engram, support-vector/Forgetful Attention memories, TEPA-style revocable evidence, DKME-style external memory model editing, and retrieval/KV hybrids.

**Killed claim:** "an LLM can have an external neural memory".

## 8. Candidate claim that still needs to survive audit

The current research candidate is not any component above. It is the joint invariant enforced by the **Causal Twin-Plane Transformer**:

1. mutable world-state bindings are canonical data, not retrained identity/value skills;
2. one canonical Symlink/Pod identity owns a monotonic generation history;
3. a Causal Page Table independently verifies and materializes only current authority;
4. mutable Port state is structurally excluded from reusable Base-plane KV;
5. model-native Port payloads are compact and query-independent;
6. J-Space derived neural states carry the exact union of source generation lifetimes;
7. changing/revoking any source makes every dependent neural state causally inadmissible while unrelated Base/J state stays reusable;
8. previous generations, stale caches, replicas and old assistant/tool segments may remain physically present but cannot regain authority;
9. an independent verifier checks the same lifecycle transition before admission;
10. the system must beat strong RAG/KV baselines on a joint quality + TTFT + write/edit/delete + context-cost frontier with no material degradation when the Port plane is unused.

This may still collide with undiscovered papers or patents. No novelty claim until the full literature/code/patent search and empirical gates are complete.

## 9. Falsification rules

Abandon or materially narrow the architecture claim if any prior system demonstrates the same combination of:

- canonical mutable neural identity;
- generation-specific authority rather than freshness-by-retrieval;
- structural separation from reusable base cache;
- exact transitive lifetime closure of derived neural state;
- active-only model execution despite physically retained stale generations;
- independently verified lifecycle capability;
- zero-optimization future entity/value binding;
- compact model-native payloads;
- and competitive real-model serving benchmarks.

A paper possessing only one or several ingredients is prior art pressure, not automatically identity with the joint architecture.
