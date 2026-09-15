# CAVI prior-art differentiation matrix

Date: 2026-09-05
Status: research scoping only; **not a legal novelty/patentability opinion**.

This matrix exists to stop the project from accidentally renaming known ideas. Every ingredient below
must be treated as prior art unless the composed experiments establish a narrower distinction.

| prior work / established mechanism | what it already covers | consequence for this project |
|---|---|---|
| SERAC — Mitchell et al., ICML 2022 | scope classifier + external counterfactual memory | external edit memory and learned scope are not novel |
| WISE — Wang et al., arXiv:2405.14768 | lifelong editing with side-memory / routing | routing to a separate memory is not novel |
| DKME — Zheng et al., Findings ACL 2026 | explicitly decouples semantic addressing from partitioned memory storage | address/storage separation is not novel |
| Knowledge Externalization — Li et al., ICLR 2026 | modular/reversible externalized knowledge | reversible external memory is not novel |
| Limited Memory LM / Raeesi & Roed audit — arXiv:2607.00605 | deletion boundary is largely the external retrieval graph; evaluates alias-closure deletions and residual retrieval paths | database-side deletion and alias closure must be treated as established; our evidence must go beyond output deletion |
| database normalization / pointers / symlinks | one canonical record with multiple references | pod + pointer fan-out is not novel |
| MVCC / monotonic generations / ABA prevention | version histories, rollback, freshness/version tags | versioning a pod is not novel |
| STALE — Chao et al., arXiv:2605.06527 | benchmark for recognizing outdated agent memories and resisting stale premises | “stale memory is a problem” is not novel |
| MemTX — Li et al., arXiv:2607.23929 | snapshot-isolated transactional belief commit, permissions/provenance/validity, cascade repair | transactional governance of agent memory is not novel |
| Commit-Time Authorization — Santos-Grueiro, arXiv:2607.10487 | stale authority evidence, approval epochs/version witnesses, fail-closed freshness checks at a durable action boundary | fresh version witnesses/capabilities are not novel; CAVI must distinguish **memory consumption** and the composed neural lifecycle, not freshness authorization itself |
| Hindsight freshness-aware memory (2026) | marks consolidated memories stale and falls back to raw facts | staleness metadata / revalidation on read is not novel |
| crypto-shredding / revocable capabilities / HMAC / nonces | established security mechanisms | E-000068 primitives are controls, not novelty |
| Anthropic Jacobian Lens / J-space (Gurnee et al., 2026) | causal/verbalizable workspace coordinates | J-space is not novel |
| J-Access — Song et al., arXiv:2608.11408 | Jacobian-lens accessibility audit for unlearning; warns that optimizing the audit can cause evasion | J-lens deletion auditing is not novel and must remain an independent measurement |
| mechanistic / representation-level unlearning audits | output forgetting can hide latent recoverability | hidden-state probing alone is not novel |

## What E-000062 removed

The first thesis proposed using a sparse J-space signature as the memory address/scope ABI. `E-000062`
falsified it across five preregistered splits: worst J-space joint route/abstain accuracy 0.3235, positive
correct-route 0.25, specificity 0.20, and negative margins versus semantic, random and raw-residual
controls. **J-space is therefore not the router.** It remains only an independent audit plane.

## What E-000061 / E-000064 established — and why it is not the novelty

- `E-000061` passed 5 seeds × alias fan-outs 1..128: duplicated fact closure grows `k+1`, canonical pod
  closure stays 1, and one canonical eviction closes every alias.
- `E-000064` passed 20 lifecycle runs: UPDATE/ROLLBACK/REVOKE/SHRED/EVICT/RESTORE/DELETE are observed
  consistently through pointer aliases without relinking or payload copies.

These are valuable substrate properties, but they are expected consequences of canonical indirection +
MVCC and are explicitly excluded from the novelty claim.

## New falsification: exported neural memory is stale authority

`E-000066` reproduced the stale-snapshot attack in **20/20** runs: a fresh store export correctly closes
after SHRED/DELETE, but a pre-operation exported `Bank` remains byte-stable and still resolves the old
object because the current Bank interface carries no live incarnation/generation authority.

This is the key systems seam: **deleting the source of truth is insufficient if a previously authorized
neural-memory materialization can still be replayed at inference.** This parallels the general stale-
authority problem in Commit-Time Authorization, so the problem and the generic freshness solution are
not themselves novel.

`E-000068` then passed **5/5** control runs with a live monotonic incarnation authority and one-use,
nonce-bound HMAC capability: pre-transition capabilities fail after UPDATE/REVOKE/SHRED/EVICT/DELETE;
RESTORE/RESIGN mint a new incarnation rather than reviving an old one; unrelated pods stay authorized.
Again, the primitives are established prior art.

## Narrow remaining novelty seam

The only defensible research target left is a **specific cross-layer composition**, provisionally CAVI:

1. many linguistic aliases are pointer-only references to one stable pod identity;
2. lifecycle changes create a new monotonic pod incarnation (rollback/restore does not resurrect an old
   incarnation);
3. inference has an explicit `BYPASS / RESOLVE(pod) / UNKNOWN` scope state and exact no-memory bypass;
4. a serialized memory snapshot is *data, not authority*: the Bank can be consumed only through a live,
   incarnation-qualified authorization boundary;
5. only after live authorization does the adapter broadcast the canonical payload into the frozen model;
6. deletion/revocation is attested against the SAME `(pod_id, incarnation)` on two independent domains:
   pointer/reachability closure and causal neural accessibility (J-space/J-lens), plus output/key/recovery
   attacks;
7. the J-space audit is not optimized, preventing the most obvious Goodhart/audit-evasion failure.

### Candidate technical statement — still unproven

> A version-qualified neural-memory consumption boundary treats cached external-memory tensors as
> untrusted stale material and requires live authorization of the canonical pod incarnation before
> broadcast; combined with pointer-only aliasing, this makes one lifecycle transition invalidate all
> linguistic access paths **and previously materialized neural-memory snapshots**, while an independent
> causal audit certifies that the same invalidated pod generation is no longer accessible in the model's
> broadcast pathway.

This is materially narrower than “editable LLM memory”, “symlinks”, “versioned memory”, “freshness
checks”, or “J-space deletion”. It is the claim that must survive `E-000060`, `E-000063`, `E-000069`, a
trained symlink replay test, stale-router/cache races, multiple backbones, strong semantic/external-memory
baselines, and a professional prior-art search before any stronger novelty language is justified.

## Primary sources

- SERAC: https://proceedings.mlr.press/v162/mitchell22a.html
- WISE: https://arxiv.org/abs/2405.14768
- DKME: https://aclanthology.org/2026.findings-acl.792/
- Knowledge Externalization: https://proceedings.iclr.cc/paper_files/paper/2026/hash/7e9c2053258b1bdd32ff2654802cd594-Abstract-Conference.html
- LMLM forgetting audit: https://arxiv.org/abs/2607.00605
- STALE: https://arxiv.org/abs/2605.06527
- MemTX: https://arxiv.org/abs/2607.23929
- Commit-Time Authorization: https://arxiv.org/abs/2607.10487
- Anthropic workspace/Jacobian Lens: https://transformer-circuits.pub/2026/workspace/
- J-Access: https://arxiv.org/abs/2608.11408
