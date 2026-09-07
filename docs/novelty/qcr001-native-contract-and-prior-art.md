# QCR-001 — native-state exactness and primary-source boundaries

Date: 2026-09-05. No new invention, patent clearance or completed pretrained result is asserted by this note.

## Exactness target

Let F_pi name the actual prescribed execution: fixed model/source data, exogenous inputs, arithmetic order, dtypes and kernel/backend configuration. The repair target is every persistent write of F_pi(revised sources). Evaluating the model more precisely then casting once is not automatically the same target. A successful real-valued proof may therefore need an execution-conformance condition.

For target float q, let C(q) be its rounding cell. An approximate pre-cast response can still produce the exact stored value if a proven enclosure of the required value lies inside a single correct rounding cell. Error magnitude without boundary distance is insufficient. QCR's scalar controls prove their own enclosures; they are not certificates for arbitrary LLM outputs. A deployable repair certificate cannot use the fresh result it is meant to avoid computing.

## Restricted fixed-basis separation witness

For response z=old+U*c followed by one final native rounding, exact reconstruction requires l-old <= U*c <= h-old, where the closed box [l,h] contains every target rounding cell. Closing tie endpoints enlarges the feasible region, so strict disjointness still proves impossibility.

If an extra row U_j equals an exact linear combination a^T*U_A of r anchor rows, its response lies between sum_i min(a_i*l_i,a_i*h_i) and sum_i max(a_i*l_i,a_i*h_i). A disjoint required interval at row j certifies infeasibility for this fixed represented-real affine basis, for any coefficient vector. QCR computes coefficients and old-value endpoint shifts with exact Fractions. Numerical QR/LP merely proposes rows; optimizer status alone is not proof. Failure to find a witness is inconclusive.

The independent standard-library checker validates exact row relations, interval bounds and shifts and keeps validation active under python -O. It does not authenticate a full model trace, source lineage, or the origin of target arrays. It does not rule out another basis, a nonlinear decoder, finer factorization, fallback, or a finite program exploiting intermediate rounding.

## Direct primary prior art

**Kamera, arXiv2606.23581v1,22 June2026.** Inspected main mechanism and appendices C.6/D. A clean canonical multimodal KV chunk is combined with a low-rank context-conditioning patch and separate RoPE relocation. Patch formation uses a conditioned forward, amortized across reuse. The paper reports reconstruction within bf16 rounding and near-ceiling behavior, not a blanket every-write native-bit-equality guarantee; its residual logit divergence is nonzero. The patch depends on antecedent content and may require refresh when that changes. Canonical clean storage plus a correction patch is not new when renamed Pod repair. QCR's scalar text-latent source-response rank axis differs from Kamera's token-by-feature multimodal patch rank: QCR is not a replication or refutation of that system.
https://arxiv.org/html/2606.23581v1

**RLibm-MultiRound, arXiv2504.07409v1,10 April2025.** The RLibm approach targets admissible intervals around correctly rounded outputs, with polynomial inequality constraints and arithmetic-aware bounds; MultiRound handles application rounding modes. Thus a nonzero approximation error yielding exact stored outputs, or interval-based rounding authorization, is already prior art. It does not automatically provide a cheap complete neural-memory repair certificate.
https://arxiv.org/html/2504.07409v1
https://people.cs.rutgers.edu/~sn349/rlibm/

**Lipschitz-Based Robustness Certification Under Floating-Point Execution,2603.13334v1.** Distinguishes real-valued network guarantees from actual floating execution and states runtime/arithmetic premises. It concerns robustness, not source-lifecycle repair; generic arithmetic-aware certification is not a novel differentiator.
https://arxiv.org/html/2603.13334v1

**Incremental Neural Network Verification via Learned Conflicts,2603.12232v1.** Solver-learned clauses are reused under same-network query-refinement conditions. They are not neural predictions of causal dependency cones. Reusing verification evidence itself receives no novelty credit.
https://arxiv.org/html/2603.12232v1

**LiveMem,2608.02515v1,3 August2026.** A per-layer GDN2 recurrent side path is trained to carry useful historical influence across bounded KV turnover. Sections3 and E.1 explicitly distinguish its lossy fixed-capacity state from an exact archive. This is direct architectural prior art for intrinsic persistent memory, not a demonstrated source-edit counterfactual repair protocol. Its lifecycle terminology concerns ongoing inference/context turnover, not automatically canonical-source revocation.
https://arxiv.org/html/2608.02515v1

Xiao2608.30198 remains motivation, not novelty. TEPA2608.07429v2 supplies keyed precedent revocation/active-retrieval governance and does not by itself repair all materialized neural tensors.
https://arxiv.org/html/2608.30198v1
https://arxiv.org/html/2608.07429v2

## Standards and patent screening

PyTorch2.10's official numerical-accuracy note addresses non-associativity and the absence of general bitwise equivalence across mathematically equivalent evaluations/backends. The public IEEE754-2019 page was inspected, not the full paid standard; a format alone does not specify an entire transformer runtime.
https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html
https://standards.ieee.org/ieee/754/6210/

**US20260228135A1**,published6 August2026, Neural Network Cache Eviction to Avoid Restore: published claims1–20 and the token-tree/priority-queue description were read. They address eviction choices that avoid restoration of retained dependent cache data. This occupies conventional dependency-aware cache management, not the full canonical-source numerical rebuild contract.
https://patents.justia.com/patent/20260228135

**US12682211B2**,issued14 July2026, Optimizing Key Value Cache for Large Language Model Inference: issued claims1–20 were read. They combine hybrid local/global attention, multi-query attention and adjacent/non-adjacent cross-layer KV sharing. The reviewed wording does not specify our complete source-revision rebuild contract. Cross-layer sharing itself receives no novelty credit.
https://patents.justia.com/patent/12682211
https://patents.google.com/patent/US20250390703A1/en

The direct Google endpoints for the two patent numbers were unavailable; the Justia reproductions supplied their published/issued claim text. Top-page filing dates were not confused with publication/grant dates. Secondary AI-generated patent summaries and irrelevant broad-search results were not used as technical proof. No unpublished claims, full legal analysis, exhaustive novelty search or clearance is asserted. An absence of search matches is not evidence of novelty.

## Remaining utility contract

Any actual candidate must still pay for cache compilation, source-context conditioning, storage, failed proposals, verification, repaired-state materialization and publication. None of the source disclosures, scalar controls, or oracle feasibility tests qualifies >=10x total mutation-to-ready, <=5% throughput loss, matched total memory, fresh/paraphrase/lifecycle reading, UNKNOWN, generic divergence, independent J-space, or trained second-backbone validation.
