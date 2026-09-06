# E-000087 — a KV-cache erasure primitive does not erase, but the residue does not reach the answer

Date: 2026-09-06
Status: **audit result on two public backbones. Not a novelty claim** — the prior-art search is still
running, and the threat model is the weakest one the field accepts. Read the limitation first.

## The one-line result

Rewriting a chunk's own span in a KV cache — the cheap erasure every serving system does instead of
recomputing — leaves the erased entity **perfectly recoverable from the retained suffix**, because the
suffix tokens attended to the chunk before it was rewritten. Full recomputation is exactly at chance.
**But the residue is a tensor-level fingerprint, not a served answer**: the model's own output does not
surface the erased entity.

## Numbers, 64 candidate entities, chance 0.0156

| Arm | GPT-2 | Pythia-70m | what it is |
|---|---:|---:|---|
| P_IDEAL — recompute the suffix without the chunk | **0.0156** | **0.0156** | erasure done properly |
| P_STUB — replace the chunk's span, keep the suffix | **1.0000** | **1.0000** | the primitive under test |
| P_ZERO — zero the chunk's span, keep the suffix | **1.0000** | **1.0000** | the cruder variant |
| P_NOOP — no erasure at all | 1.0000 | 1.0000 | sanity |

Behavioural, decoding the query **against the erased cache**:

| | GPT-2 | Pythia-70m |
|---|---:|---:|
| the erased entity is the top-1 answer | **0.0** | **0.0** |
| its mean rank | 2820 | 916 |
| log-prob gain over the ideal arm | +0.66 | +1.66 |

## Controls, and they can fail

This session produced four measurement errors, every one a comparison that was not like-for-like or a
control that could not fail. These are stated before the result rather than after:

- **C1** the ideal arm's retained tensors are **bit-identical** across all 64 entities — asserted in
  code, since the entity never enters that forward. Passes.
- **C2** the same attack run against the ideal arm lands at **exactly chance** on both backbones. If a
  matcher could identify the entity from state that provably does not depend on it, every other number
  here would be void. Passes.
- **C3** every arm is matched against references built the same way, retained-suffix against
  retained-suffix. No arm is compared against a differently normalised quantity — the specific error
  that produced two wrong conclusions earlier in this session.
- **C4** behavioural leakage is measured from the **erased** cache and reported as a gain over the
  ideal arm, so a prompt that merely makes an entity likely cannot be read as leakage. The first
  version of this measurement compared the un-erased forward and was wrong; it is corrected here.
- Every chunk is verified to be the same token length, so removing one shifts no positions and the
  arms stay comparable. The check fired on the first run and was fixed rather than relaxed.

## The limitation, stated plainly

The attacker reads the retained K/V tensors and knows the candidate set. That is a **storage-layer
adversary** — the same one Ghost Vectors (arXiv:2606.18497) used for soft-deleted embeddings, and the
weakest model the field accepts. The behavioural rows say the leak does **not** reach a served answer,
so this is not "the model will tell you what you deleted". An earlier design review of this very claim
preregistered exactly this outcome as a downgrade condition: *"residue is a fingerprint only — downgrade
to a storage-layer finding equal in kind to Ghost Vectors."* That is what happened, and the claim is
recorded at that strength and no higher.

## Why it was worth running anyway

It is the shape of this programme's one transferable result. E-000028 gated a payload, passed four
attacks at chance over 750 trials, and gave the object up at 1.0000 through a derived index the
primitive never touched. Here a primitive rewrites a span, reports the chunk removed, and gives it up
at 1.0000 through suffix state it never touched. The generalisable rule is the same one:
**enumerate every quantity derived from the thing you are deleting, or do not call it deleted.**

For the Symlink/Pod line specifically: a pod DELETE cascades over pointer aliases in the store, and the
store-side cascade is exact. The neural frontier is a different set — every cached entry whose attention
window contained the pod. **ON DELETE CASCADE over pointers is necessary and never sufficient for
attention caches.**

## What is not established

- Novelty. The prior-art search is running; the nearest known neighbour is *Models Take Notes at
  Prefill* (arXiv:2606.17107), which reports downstream KV retaining conclusions after a local field
  edit — the same phenomenon, possibly not framed as an erasure audit. Until that is fetched and read,
  nothing here is claimed as new.
- Any served-output leak. Measured and absent.
- Any statement about production systems. This is a controlled two-backbone harness, not vLLM or
  SGLang, and the primitives are idealised versions of what Leyline and KVEraser do.
- Positional re-anchoring. Every chunk here is the same length, which is the case most favourable to
  the erasing system; a real removal that shortens the sequence was not tested.

Reproduce: `python -m so.experiments.e000087_cache_erasure_residue --model gpt2 --n-entities 64`
