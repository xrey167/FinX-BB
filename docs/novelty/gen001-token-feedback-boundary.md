# GEN-001: token feedback beyond fixed-token cache purity

2026-09-05. Structural counterexample and ordinary conservative reference repair. No breakthrough or novelty claim.

## E-000084: completed, artifact-verified

Run 33970654975, commit `7431619988d1f441a62c700e52e8a3590cd15388`. All five downloaded ZIP digests matched Actions metadata.

| Arm | Training seed | Candidate held-out mean | Full-vocabulary held-out mean | Worst full-vocabulary template | Strict pass |
|---|---:|---:|---:|---:|---|
| A in-place | 0 | 0.95500 | 0.95500 | 0.935 | No |
| A in-place | 1 | 0.99000 | 0.99000 | 0.960 | Yes |
| C deferred | 0 | 0.66375 | 0.61625 | 0.285 | No |
| C deferred | 1 | 0.64500 | 0.60750 | 0.270 | No |
| C deferred | 2 | 0.62125 | 0.59625 | 0.290 | No |

All C rows have `kv_maxabs=0.0` and `block_input_maxabs=0.0` on the 16 fixed-prompt exposure checks; output changes are material. C fails every unchanged >=0.95 capability gate. A0 also fails, so no multi-seed anchor is promoted.

The workflow's capability step has `continue-on-error: true`: green CI is not a scientific pass. The JSON is authoritative.

This does not establish that downstream payload processing is universally necessary. Removing the first in-place write also changes the second read's input/routing feedback, as the existing first-read-only equality test acknowledges. Optimization and normalization also differ.

## Missing channel

A cache can be pod-independent conditional on fixed input token IDs, yet become pod-dependent after consuming a token chosen using the pod:

```
pod(g1) -> final-block write -> generated token
                                    |
                                    v
                        next embedding -> new K/V -> later output
```

Rebuilding K/V from the same old internally generated tokens retains that channel. Token history is part of the state being revoked, not an automatically clean input to replay.

## Executed GEN-001

The actual repository `KnowledgeAdapterLM` is unchanged. The diagnostic uses a four-block, 64-dimensional frozen random GPT-2, one controlled pod, constant routing and a disabled marker gate. Only the first sampling decision reads the pod; all six subsequent token-consumption steps use the frozen core without a new pod read. This is NOT learned symlink routing, trained NLP capability, or a pretrained-model generalization result.

Reference: apply REVOKE or UPDATE before selecting the first internally buffered token, then regenerate under the same prompt, model and greedy schedule. All generated tokens remain in an uncommitted internal buffer. Previously published external text is out of scope.

Run **33974856390**, executed commit **cd28ad7b0d5dda2ff35f1e4512878ae70a5ac642**: **10/10 diagnostic cases passed**.

| Initialization seed | Transition | Prefill K/V delta | Later K/V delta vs fresh | Later logit delta vs fresh | Repair logit delta |
|---:|---|---:|---:|---:|---:|
| 101 | revoke | 0.0 | 0.532810628 | 0.612012982 | 0.0 |
| 101 | update | 0.0 | 0.662739694 | 0.657258153 | 0.0 |
| 102 | revoke | 0.0 | 0.539302707 | 0.954592407 | 0.0 |
| 102 | update | 0.0 | 0.620636940 | 0.831117630 | 0.0 |
| 103 | revoke | 0.0 | 0.552488923 | 0.793798864 | 0.0 |
| 103 | update | 0.0 | 0.657480240 | 1.140162826 | 0.0 |
| 104 | revoke | 0.0 | 0.538272917 | 0.823598623 | 0.0 |
| 104 | update | 0.0 | 0.462127656 | 0.759481490 | 0.0 |
| 105 | revoke | 0.0 | 0.606594682 | 0.757237911 | 0.0 |
| 105 | update | 0.0 | 0.568535209 | 0.715723872 | 0.0 |

Deltas are maximum absolute tensor-coordinate differences, not rates/probabilities. Every old trajectory repeats the old payload token six times; revoked references produce UNKNOWN and updated references the new payload token. Deliberately controlled repetition is not a language benchmark.

In all 10 cases, full K/V rebuilding from the retained old token sequence still differs from the fresh reference, while numerically agreeing with the old continuation.

## Conservative reference repair

`Deps(new_token) = union(Deps(prefix), fresh read stamps)`; a stamp is `(pod identity, generation)`.

Ordinary transitive lineage rejects the final descendant even though it made no fresh pod read. The buffer is cut before its first invalid token and regenerated under the current state. The repaired token sequence, final logits and every K/V tensor match the independently computed fresh reference bit for bit in all 10 cases in the same deterministic CPU schedule. This is an independent recomputation, not copying reference output. ABA restores do not revalidate old stamps. Independent bystander sessions remain bit-identical; mixed-pod same-session locality is NOT established.

Lineage and regeneration are by construction and receive no novelty credit. This reference does not authenticate arbitrary serialized state or implement concurrent publication.

## Verification

Actions: **38 passed, 1 warning in 4.20 seconds**, comprising existing write-layer and adapter tests plus ten new lineage/feedback tests. The ten-case diagnostic matrix ran separately.

Local sandbox: eight pure-lineage tests passed; two model tests deliberately deselected because Transformers was absent and outbound package installation failed. Model execution is from Actions, not a claimed local reproduction.

Environment: Python 3.11.16, PyTorch 2.14.0+cpu, Transformers 5.16.1, one CPU thread, eager attention and deterministic algorithms.

Artifact ID `9972013359`, independently verified ZIP SHA-256:
`f7863aeb153d4e608dcf9d144c352421713aa625059825e8a1098d83932dab7d`.

Executed experiment SHA-256:
`d2b2441f1880b2c0de49128511216bd9a70b1d9e20d1109bd3477a81d2d200ae`.

Unchanged adapter SHA-256:
`6b2cb98f01e32fa2884fa6b88bd3d07c5a1a7714719b9de1e1a338d0ffea87ad`.

Both archived source hashes were verified after download; the local authored experiment matches the executed source hash.

## Prior-art boundaries and next decision

Ramesh (2026), *Subtract or Replay? Exact Deletion from Language-Model Memory*, already distinguishes retained-contextualized-key certificates from complete rebuilds and uses checkpointed replay. Storek et al. (2026), *GIF: Locally Sound Geometric Information Flow Control for LLMs*, studies Jacobian-based information flow under local assumptions and the overtainting problem. Girrens and Wang (2026), *SPA*, studies label-preserving persistent artifacts. None permits claiming ordinary replay or inherited lineage as our novelty. Local geometric soundness must not be restated as a global finite-revocation guarantee.

Primary sources consulted:
- https://arxiv.org/html/2607.27539v1
- https://arxiv.org/abs/2606.23277
- https://arxiv.org/abs/2608.27234
- https://huggingface.co/docs/transformers/main/cache_explanation

The next model comparison should retain the in-place reader as an anchor and use ordinary token-aware invalidation/regeneration as a strong safety baseline. Do not use failed C as the only baseline or promote fixed-token cache purity to whole-generation lifecycle closure. No J-space, multi-backbone, concurrency, latency, minimal-lineage or mixed-context-locality claim is established here. Historical results and >=0.95 gates remain unchanged; no protected confirmation programme is executed.
