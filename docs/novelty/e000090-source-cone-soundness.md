# E-000090 — source-local payload lineage is not a complete repair certificate

**Architecture-contract falsification; not a major invention.** Date: 2026-09-05.
Parent E89: `0bfb98c3c644816d5a623bf0808174c609d28e0c`.
Preregistration: `d7f9df756c989e800dfb16e5d1054a5032d7825d`, committed before numerical execution. Historical evidence and main are unchanged.

## Executed evidence

Five random parameter/intervention seeds (0–4), width 128, source block 16, eight nonlinear persistent layers. Separate adversarial routing cases: 64 queries, three persistent layers. Local selected suite: **71 passed in 1.31s**, no skips. Fifteen tests independently check analytic properties with PyTorch autograd/normalization. This is not external reproduction or a trained-language-model experiment.

Exact equality means identical shape, dtype and bytes within the deterministic execution. The 1e-10 threshold describes effect support only; it never authorizes reuse.

| Normalization | First-layer changed coordinates (>1e-10), every seed | Max error outside claimed source cone, range across seeds | Local oracle patch equals full rebuild |
|---|---:|---:|---|
| Identity | 16 / 128 | 0 | Yes |
| Per-block LayerNorm | 16 / 128 | 0 | Yes |
| Per-block RMSNorm | 16 / 128 | 0 | Yes |
| Global LayerNorm | 128 / 128 | 0.100763–0.401144 | No |
| Global RMSNorm | 128 / 128 | 0.025038–0.083647 | No |

The oracle is deliberately stronger than a real repair: exact rebuilt values are supplied inside the proposed 16-coordinate cone, and old values reused outside. Even this fails with global normalization. Block-diagonal weights do not isolate shared normalization statistics. This is not a claim that every edit changes those statistics; invariant-statistic edits require a separate proof.

**Dormant routing source:** before a routing-key edit, source 0 is selected by 0/64 queries; afterward by 64/64. All 192/192 affected persistent vectors are missed by old selected-payload lineage in every seed. Stale max-absolute errors: 0.625903–1.185141. A conventional decision-aware full-batch reference equals rebuild byte-for-byte. This is a constructed mutable-key counterexample, not a real-model failure-frequency estimate or a payload-only update result.

**Unchanged winner, stale coefficient:** editing an unselected score changes the selected value's global-softmax coefficient. The winner stays unchanged, but downstream error is 0.006928–0.007765. Selection followed by local renormalization is an exact negative control. No novelty is claimed for tracking this denominator dependency.

**Averaged-Jacobian blind spot:** for `F_c(h)=a*h[0]+c*b*tanh(h[1])`, contexts `c=+1,-1` give opposite source derivatives. The balanced average has an exactly zero source column. Stale and NEVER states have identical averaged-lens readouts, yet context-conditioned output effects are 0.711486–2.468173 in constructed units. Independent autograd checks pass. This is not an attack result on the full Anthropic J-space methodology or on an LLM, and does not assume every real J-lens is low rank.

**Exact positive controls:** with identity or per-block normalization, ordinary source-block replay equals every full-rebuild write for UPDATE and deletion-to-NEVER, all five seeds. Candidate and ordinary reference get identical state/operators and perform the same eight affected matvecs. The implementation copies the old trajectory; no speed, minimality or system-memory advantage is inferred.

## Architecture change

The earlier proposal that a learned cone plus a J-lens can *prove* completeness is too strong. A learned component may propose work. Deterministic operator/guard verification or conservative exact recomputation must authorize reuse. J-space remains an independent challenge to the result, not a substitute for that authorization.

A sound proposal must cover payload ancestry, score inputs, normalization domains, control conditions and relevant membership/generation state. Mutable-key edits can create routes absent from the old payload graph. Immutable-key designs can exclude that mutation, but must declare their scope. Insertions, relinks, absence, ties and concurrency require separate contracts; E90 did not implement them all.

A conservative sound dependency superset is not an exactly minimal cone. False positives cost performance; false negatives invalidate equality. Canonical root revision alone is not mutation-to-ready, and a generation check alone does not repair descendants.

These are correctness requirements and conventional baselines, not an invention. A useful neural representation that learns small, verifiably stable domains without losing nonlinear capability remains an unverified target.

## Prior-art boundary checked this turn

- Xiao et al., arXiv:2608.30198 (2026): memory-update error propagation and pathway repair motivate the problem, not novelty.
- MemoRepair, arXiv:2605.07242 (2026): withdraw affected descendants, construct successors using retained support/repaired predecessors, validate before predecessor-closed republication. Complete influence provenance is a premise. Its exact min-cut concerns repair selection, not exact neural tensors; neural skills use parametric repair. Barrier-plus-lineage repair is already described.
- Yu et al., arXiv:2608.10502 (2026): memory/execution dependencies and selective rollback/replay, preserving independent support. Ordinary typed-graph repair remains excluded.
- AgentKVShift, arXiv:2607.21604 (2026): layer-specific memory-level residual offsets estimated from probes correct reused K/V. Its target is approximate fresh state and near-full task performance, not every-write equality. Bounded probes/correction vectors across layers are not new by themselves.
- Oliver Zahn's author essay, published February 27, 2026, explicitly describes knowledge objects with sparse/one-hot lineage vectors propagated with attention alongside activations. This is conceptual prior disclosure, not validated counterfactual repair. The linked ANML paper (2602.11690) evaluates contributor-quality weighting and must not be misrepresented as implementing that entire lineage architecture.
- Gurnee et al.'s 2026 workspace paper describes context-averaged, token-indexed J-lens directions and limitations beyond single-token concepts and a bag of concepts. Its interpretability findings do not provide a universal lifecycle certificate.
- Sharir and Anandkumar, arXiv:2307.14988: architecture changes to improve incremental neural reuse already exist.
- Patent screening: IBM US20260119893A1 covers a specific additional KV-layer/twin-Gaussian-mixture mechanism including updates/deletions; ETRI US20260105279A1 covers side-channel neural knowledge control. Neither reviewed claim set establishes this full descendant-counterfactual contract. This is not legal clearance or an exhaustive novelty search.
- W3C PROV-DM already specifies derivation, revision and invalidation vocabulary. Metadata terminology receives no novelty credit.

Primary sources:
https://arxiv.org/abs/2608.30198
https://arxiv.org/html/2605.07242v1
https://arxiv.org/html/2608.10502v1
https://arxiv.org/html/2607.21604v1
https://www.linkedin.com/pulse/beliefs-knowledge-oliver-zahn-7gduc
https://arxiv.org/html/2602.11690v1
https://transformer-circuits.pub/2026/workspace/
https://arxiv.org/abs/2307.14988
https://patents.google.com/patent/US20260119893A1/en
https://patents.google.com/patent/US20260105279A1/en
https://www.w3.org/TR/prov-dm/

## Gates and provenance

The >=10x mutation-to-ready, <=5% throughput loss, matched complete memory budget, >=95% fresh/unseen-paraphrase and lifecycle reading, >=90% scoped UNKNOWN and <=0.05 nats generic divergence/exact bypass gates are **not evaluated by E90**. Five operator seeds are not trained-reader seeds or a second backbone. No threshold was lowered.

Reproduce from repository root (NumPy, pytest; PyTorch required for all 71 tests):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m so.experiments.e000090_source_cone_soundness \
  --seeds 0 1 2 3 4 --output results/e000090-rerun.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest so/tests/test_e000090_source_cone_soundness.py -q
```

Without PyTorch, 15 tests skip; report that instead of calling it a 71-test replication. This is not the full repository suite. Local environment: Python 3.13.5, NumPy 2.3.5, PyTorch 2.10.0+cpu, pytest 9.0.2, Linux x86-64.

Original results: `results/e000090/e000090-results.json.gz` (lossless gzip; decompress with Python gzip or `gzip -dc`).

- Experiment SHA-256: `30fd396e07f00926d21b3b153dbc613d5666d5cebd1baaeb6424ce15020b561f`.
- Tests SHA-256: `8a42ab1a518cd9c9b9191bf378fc1224dc8b3d488057e96a33f791d8ef212dc0`.
- Raw JSON SHA-256: `c2dea79d8c73ca01cde9557837c8090bd01ddf87d371cc44a77acb59e56bbd8d`.
- Gzip SHA-256: `70cdd1ae852eefe07af9c8a805623130426e8bd7e11d35e741aa0bc7e9cf988e`.

Publication corrections: an initial shell invocation failed before execution because its work directory did not yet exist. A compressed-result upload failed the local Git-blob hash check, remained unreferenced and was replaced by a hash-matching blob before publication. No scientific arm was retuned; all expected negative arms remain in raw data.

Separate static E89 audit: `_full(..., heavy=True)` changes `u`, not just its computation cost. Its timing is not guarantee-matched utility evidence. The matched exact-delta/cached-replay comparison is separate and unaffected. No historical file was overwritten and no additional E89 execution is claimed here.
