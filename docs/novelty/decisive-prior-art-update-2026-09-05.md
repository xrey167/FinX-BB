# Decisive prior-art update — mutable/deletable neural KV is not the novelty target

Date: 2026-09-05

Status: **decisive direction change, not a breakthrough.**

## New direct collision: IBM US20260119893A1

A targeted patent search found `US20260119893A1`, **Direct knowledge injection
into large language models using a key-value cache network layer**, assigned to
IBM. Google Patents reports priority/filing date 2024-10-24 and publication date
2026-04-30; legal status is listed as pending with the usual Google legal-status
disclaimer.

The disclosure is directly material to the broad FinX-BB / CAVI idea of an
editable neural knowledge pod:

- add a KV-cache network layer to an LLM;
- train that layer with key/value encoders and a joint contrastive + instruction
  tuning procedure;
- position the KV-cache network layer as part of the LLM, including behind the
  original network layers;
- inject new knowledge through the layer;
- perform real-time insertion, modification **and deletion** of existing data
  points;
- propagate those updates back into the KV-cache network layer;
- use the updated neural layer to answer questions without changing the original
  LLM data points for each small knowledge update.

Claim 6 explicitly recites real-time insertion, modification and deletion of
existing data points, and claim 7 positions the KV-cache network layer as part
of the LLM. Parallel product/system claims repeat the same update/delete
semantics.

Primary source:
https://patents.google.com/patent/US20260119893A1/en

## Consequence

The project must **stop treating any of the following as a defensible novelty
target**:

- a trainable KV-like neural memory attached to / inside an LLM;
- direct natural-language key/value knowledge injection into such a layer;
- real-time add / modify / delete of knowledge in that layer;
- a mutable/deletable neural knowledge pod merely because it avoids full-model
  retraining.

This collision is neural-specific and substantially closer than the earlier
PAMSPEC/MemTX systems prior art.

## Independent architecture collision: ReCache

ReCache (arXiv:2608.19662, 2026-08-20) independently makes fixed
resource-local/composition-invariant KV a strong prior-art baseline:

- independently cached resource representations;
- resource-wise attention removes cross-resource interactions;
- resource-local positions yield composition-invariant KV blocks;
- selective layer/KV-head-group resource visibility;
- reported 82.3% vs 82.4% dense invocation F1, 3.655x TTFT speedup, 92.43%
  allocated KV-tensor memory reduction and 1.423x attention speedup.

Primary source:
https://arxiv.org/abs/2608.19662

Therefore fixed resource compartmentalization / composition-invariant KV is
also not a novelty target.

## What remains live

The narrow surviving question is **not how to insert/edit/delete knowledge**.
It is whether live LLM execution can preserve useful cross-source neural
composition while making *derived neural state* lifecycle-correct and cheap to
repair:

1. many aliases may resolve to canonical mutable/versioned pods;
2. an update/revoke/delete/rollback changes authority once, not O(k) alias
   copies;
3. every persistent neural derivative that depends on revoked material becomes
   unusable even if replayed after a valid earlier resolve;
4. unaffected persistent state remains reusable;
5. cross-source interactions needed for capability are preserved rather than
   eliminated merely to make invalidation easy;
6. exact BYPASS, UNKNOWN, deletion leakage, held-out reading, REVOKE/SHRED and
   independent J-space/J-lens gates remain unchanged;
7. the mechanism must beat both dense recomputation and ReCache-style fixed
   source isolation at matched capability/safety.

## Current structural evidence for that narrower question

RL-MIX-001 is only a controlled cache-interaction assay, not a semantic symlink
reader. It nevertheless shows why simply assuming source independence is too
weak as a universal baseline:

- on GPT-2 at the strongest controlled intervention, order-1/source-additive
  reconstruction reached max KL up to about 0.158 versus independently
  recomputed dense current state;
- adding ordinary pairwise interaction terms reduced the corresponding max KL
  below about 0.0084 in the tested 4-source cells;
- Pythia-70m was much more nearly additive under the same style of intervention.

This does **not** make pairwise/Möbius interaction decomposition novel. It only
motivates testing whether real symlink readers require cross-pod interactions
that ReCache-style zero-cross-resource designs cannot preserve.

RL-MIX-002 now tests a polynomial `1 + n + C(n,2)` pairwise source-set baseline
at n=4 and n=8 with independent dense recomputation controls. It is explicitly
marked `breakthrough=false` until real-symlink capability, lifecycle security,
scaling, locality and prior-art gates are satisfied.

## Promotion boundary

No claim is promoted unless it survives the unchanged project requirements,
including >=0.95 real-symlink fresh correctness across >=3 seeds before CAVI
attack interpretation, held-out/full-vocabulary capability, REVOKE/SHRED,
deleted-object leakage, UNKNOWN, exact/no-damage BYPASS, stale-state replay and
race attacks, independent J-space/J-lens audit, scaling/performance, >1 public
backbone where feasible, and a final targeted prior-art/patent search.

The current research direction is therefore narrowed to:

> **interaction-preserving, lifecycle-selective repair/revocation of persistent
> neural-derived state at matched semantic capability**, not mutable KV memory
> itself.
