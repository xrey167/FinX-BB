# E-000079 — cross-forward neural-derived-state falsification

Date: 2026-09-05
Status: **decisive architecture correction; not a breakthrough or novelty claim**

## What was falsified

E-000078 made the memory-consuming region of one model forward atomic by holding one authority snapshot from the first configured neural memory read through the last configured neural memory read. E-000079 shows that this boundary is still too short for autoregressive generation.

A memory write at an intermediate transformer layer changes the residual stream entering later layers. Those later layers can therefore materialise K/V state whose causal ancestry includes the memory write. Under the E-000078 boundary, the authority lock is released after the final configured memory read, while downstream transformer computation and cache materialisation can still be in progress. A REVOKE/UPDATE/RELINK can consequently commit before the forward returns, and the returned `past_key_values` can encode the now-invalid generation. A later decode forward can reuse that derived state without performing another memory read.

E-000079 exercises this exact race with real HuggingFace autoregressive `past_key_values` on two public causal-LM backbones. It uses a controlled residual memory payload rather than a trained symlink reader so the result is independent of reader capability. This makes E-000079 a contract falsification only; it does **not** bypass the >=0.95 real-symlink reader validity gate required for positive CAVI evidence.

## Corrected run

Workflow: `E000079 generation KV lineage race`
Run: `33964913121`
Head commit: `1147f6357f4e8eac3540f7e85e69e1060d21ff0b`
Backbones: `distilgpt2`, `EleutherAI/pythia-70m-deduped`
Seeds per backbone: `0,1,2`

For every run:

1. two controlled residual memory reads execute under one `ForwardSnapshotConsumptionGuard`;
2. immediately after the final read hook releases the E-000078 lock, a lifecycle mutation commits under the same authority lock;
3. that mutation commits before the next downstream transformer block and before the prefill forward returns;
4. the prefill returns `past_key_values` derived from the old memory-bearing forward;
5. all subsequent comparison forwards have no memory context;
6. reusing the stale prefill cache is compared with full recomputation under the current revoked state;
7. a cache built cleanly under the current state is compared with the same full recomputation as a numerical control.

### Results

| Backbone | Seeds | stale KV vs current max-abs | stale KV vs current KL | clean current KV vs full recompute | Behavioral effect |
|---|---:|---:|---:|---:|---|
| `distilgpt2` | 3/3 | **14.230518** | **1.259328 nats** | **9.16e-05 max-abs** | stale top-1 **23304**, current top-1 **12** on all seeds |
| `EleutherAI/pythia-70m-deduped` | 3/3 | **21.0** | **1.600688 nats** | **0.0 max-abs** | material distribution shift; top-1 remained 282 |

Timing witnesses also show that the mutation linearized before downstream computation and well before the prefill returned. On `distilgpt2`, mutation-to-downstream was approximately **97–103 us** and mutation-to-prefill-return approximately **27.3–29.7 ms**. On Pythia-70m, the corresponding ranges were approximately **61–82 us** and **14.9–16.1 ms**.

The clean-current-cache control matching full current recomputation rules out ordinary cache-vs-recompute numerical drift as the explanation. The stale cache alone carries the material discrepancy.

## Preserved failed harness attempt

The first workflow run, `33964783049`, was not valid evidence because the mutation trigger also fired during later no-memory control forwards, incrementing the synthetic authority generation repeatedly and overwriting the timing witness. That attempt is preserved as a harness failure rather than silently discarded. The corrected experiment makes the lifecycle mutation one-shot and only eligible on the initial memory-bearing prefill.

## Architecture change

E-000078 remains a necessary intra-forward consistency primitive, but it must **not** be promoted as the complete live-generation transaction contract.

The required boundary is now cross-forward:

1. **Creation:** any reusable neural-derived state whose causal ancestry includes a pod read must be associated with the exact authority/pod incarnation(s) that contributed to it.
2. **Commit:** lifecycle mutations may linearize only under a defined generation boundary; extending the lock merely to model return is sufficient for creation consistency but is not sufficient for later reuse.
3. **Reuse:** before KV, hidden, activation, selected-route, resolved-payload, Bank, or other reusable derived state is consumed in a later inference/token step, its derivation lineage must still be current.
4. **Repair:** a stale derivation is not authoritative. The system must reject it and obtain current state by recomputation, selective refresh, or another semantically equivalent mechanism.
5. **Freshness:** UPDATE/RELINK must not merely suppress the old derivation. The next read must be able to resolve the current alias -> current pod generation and preserve fresh-current capability.
6. **Locality:** derivation tracking should be object-scoped rather than a single global epoch where possible, otherwise one unrelated pod update destroys all reusable neural state and forfeits practical utility.

This is stronger than ordinary forward snapshot isolation because the object being protected is not only the read itself: the model creates latent derived state that can survive the transaction and become a future causal input.

## What is explicitly NOT claimed as novel

E-000079 is a falsification, not a novelty result. The following are established mechanisms or directly crowded by current work and remain excluded as standalone claims:

- generations/epochs, MVCC, snapshot isolation, locks, capabilities;
- generic cache invalidation, cache tags, dependency/version checks, recomputation;
- KV editing, erasure, selective refresh or reprefill;
- pointers/symlinks or external memory by themselves.

Current direct prior-art pressure includes, among others:

- **Leyline: KV Cache Directives for Agentic Inference**, arXiv:2606.01065 — serving-side remove/replace cache directives and selective preservation/re-prefill;
- **KVEraser**, arXiv:2606.17034 — localized learned erasure of KV influence after prefill;
- **Models Take Notes at Prefill**, arXiv:2606.17107 — direct evidence that downstream KV states can retain conclusions after a local field edit, plus editable/composable caches;
- **AgentKVShift**, arXiv:2607.21604 — per-memory-unit KV residual correction for agentic memory reuse;
- **WO2026087278A1 (IBM)** — direct knowledge injection into LLMs through a KV-cache network layer.

These references make a bare claim around editable KV, cache refresh, invalidation or injected cache memory indefensible.

## Remaining candidate research seam

The potentially research-worthy seam is therefore narrower and remains **unproven**:

> object-scoped causal derivation closure for mutable neural knowledge: many linguistic aliases resolve to one versioned pod; one pod lifecycle operation changes authority once; the model records a compact, independently auditable causal dependency from that pod generation into reusable neural-derived state; only state whose causal dependency intersects a changed pod is made stale; all other inference state remains reusable; fresh-current reads remain capable and out-of-scope inference remains exact/near-exact.

This is not yet a novelty claim. It only identifies what must be differentiated from generic invalidation and current editable-KV work.

## Next falsification/validation sequence

Before any breakthrough language:

- restore a >=0.95 fresh real-symlink reader on **every** one of >=3 seeds;
- implement and test object-scoped derived-state lineage rather than a global generation epoch;
- measure selective invalidation/refresh against full global cache invalidation and full recomputation;
- attack stale KV, hidden, activation, selected-route, resolved-payload and Bank state after UPDATE/REVOKE/SHRED/RELINK/ROLLBACK;
- prove fresh-current re-resolution rather than only stale rejection;
- run exact no-memory bypass/locality and generic KL gates;
- run independent J-space/J-lens causal audit without training the audit target;
- measure alias fan-out, mutation cost, cache footprint, inference overhead, rollback latency and avoided O(k) duplicate edits;
- repeat on >1 public backbone where CPU-feasible;
- continue paper/patent search specifically for object-scoped provenance or causal dependency labels attached to neural runtime state.
