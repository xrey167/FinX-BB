# CAT-001 — exact editable sufficient state exists, but dynamic tree aggregation owns this construction

**Constructive baseline result, not a major invention.** Date: 2026-09-05.
Parent FIR001: `b3cfa14069787893e0e934c490ca795f84978498`.
Preregistered before numerical execution at `2dfca78af0079b4698d72c0dfd4d2c65becb4586`.
Executed source/run: `7d927a4a5d6b87963eace6e500833edd56c46a68` / **33989932774**.
Main and historical experiments remain unchanged.

## Decision

FIR001 showed that exact finite source lifetime loses persistent source recall unless some other channel retains/reintroduces the source. CAT001 constructs the complementary case: persistent long-term source information CAN live in a compact compiled sufficient state that is exactly editable without query-time raw-Pod rereads.

However, the mechanism is an associative ordered fold stored in a balanced dynamic product tree. A conventional product/segment tree with the identical leaves, internal summaries, arithmetic and memory performs exactly the same update. The full-rebuild advantage is real but receives zero novelty credit.

This result therefore changes the programme boundary in a useful way: **persistent editable sufficient state is not impossible**, but associative compilation by itself is already a mandatory baseline. A surviving invention must outperform or escape that baseline under matched semantics/resources rather than claim novelty from source-addressable tree aggregation.

## Exact construction

There are N canonical sources in a fixed order. Each `(source_id,payload)` maps deterministically to an invertible 2x2 transform over the prime field `p=2^31-1`. The encoder is a controlled algebraic stand-in, not a trained semantic reader.

For source transforms `M_0,...,M_(N-1)`, the persistent compiled state is the ordered product

`G = M_(N-1) ... M_1 M_0`.

The matrices generally do NOT commute. Query `x` consumes only G through the projective readout

`y = (G00*x + G01) / (G10*x + G11) mod p`.

Thus long-term source information is present in the compiled root itself; no raw Pod payload is fetched at query time. The readout is nonlinear in ordinary projective coordinates, and changing another source changes the effect of an edit. This is still a restricted algebraic model, not evidence of language-level expressivity.

The balanced product tree stores exact products for subranges. Replacing one source leaf recomputes only the `log2(N)` ancestors. Every resulting root and 64 query outputs are compared with a clean full ordered rebuild.

## Executed evidence

Five seeds x four source counts `{64,256,1024,4096}`. Each cell performs 32 sequential random one-source updates: **640 exact update events** total. Every updated root and every tested query output equals the clean rebuild exactly in the finite field.

| Sources N | Full rebuild compositions | Tree update compositions | Full/tree operation ratio | CI wall full/tree range across 5 seeds |
|---:|---:|---:|---:|---:|
| 64 | 63 | 6 | **10.5x** | 8.27–9.15x |
| 256 | 255 | 8 | **31.875x** | 21.15–24.31x |
| 1024 | 1023 | 10 | **102.3x** | 72.38–72.90x |
| 4096 | 4095 | 12 | **341.25x** | 144.00–150.61x |

The primary metric is operation count. Python CPU wall timing is secondary and not an LLM/system latency claim. Building the tree costs N-1 compositions and stores `2N` matrix slots in the simple array implementation (including unused index0). This is not a matched total-system memory measurement.

The guarantee-matched conventional dynamic-product-tree baseline performs exactly the same `log2(N)` composition calls using exactly the same internal products. Therefore

`T_strong_baseline_operations / T_candidate_operations = 1.0`

in every cell. The programme's >=10x gate versus the strongest guarantee-matched baseline FAILS even though the weak full-rebuild comparison exceeds10x from N=64 onward.

## Interaction and lifecycle controls

The construction is not secretly a commutative sum. In every one of the20 seed/size cells:

- swapping two adjacent source transforms changes the root and query outputs;
- all16 tested projective inclusion/exclusion interaction values are nonzero;
- the naive context-free O(1) patch `G * M_old^-1 * M_new` fails for the tested interior edit.

The naive-patch failure is only a counterexample to that formula, not a dynamic-product lower bound. Prefix/suffix context or other data structures change the algorithm.

Revocation-to-identity matches a clean rebuild in the CI tests. ABA control updates a payload away and back: the numerical root returns to its old value, but a stale compiled snapshot is rejected by a separate mutation epoch. That epoch mechanism is ordinary generation metadata and receives zero novelty credit; it is not the full publication/concurrency protocol.

## Prior-art boundary is stronger than expected

### DeepMind/Google hierarchical neural external-memory patent

US11010664B2, priority **2016-02-05**, is a direct boundary for broad claims around a neural tree whose leaf edit updates only the ancestor path. The specification describes a hierarchical external memory arranged as a binary tree, leaf updates, then reverse-order updates of the nodes on that path using a **join neural network**. It also states logarithmic access complexity. CAT001's exact projective join differs algebraically and in lifecycle semantics, but `neural memory + binary tree + update leaf + recompute ancestors/root` is plainly not new.

Primary: https://patents.google.com/patent/US11010664B2/en

### Memoroids

Morad et al., **Recurrent Reinforcement Learning with Memoroids**, NeurIPS2024, explicitly formulate efficient recurrent memory using monoids and emphasize associative operators/parallel scan. CAT001's ordered transformation fold sits squarely inside the general associative-memory algebra boundary. It is not a replication of their RL model, but monoidal neural-memory composition is established.

Primary: https://proceedings.neurips.cc/paper_files/paper/2024/hash/19f7f755908372efb25826d61959cdf9-Abstract-Conference.html

### MonoidReduce

MonoidReduce, ICLR2026 submission, further treats neural computations as monoidal folds. It concerns memory-efficient neural layers rather than editable source lifecycle, but prevents broad novelty claims around using monoid algebra as the neural-computation insight.

Primary: https://openreview.net/forum?id=Fopv5Hpm1C

### Segment-tree agent memory

SegTreeMem, arXiv:2606.04555, uses a segment tree for long-horizon agent memory. Its nodes are temporal memory summaries and retrieval differs from CAT001's exact transform aggregate, but segment-tree memory organization itself is active 2026 prior art.

Primary: https://arxiv.org/abs/2606.04555

No exhaustive patentability or freedom-to-operate conclusion is asserted. Classical segment trees, dynamic range products, monoids and projective/Möbius matrix composition predate all of these neural-memory uses and receive zero novelty credit.

## Architecture consequence

The surviving target is no longer simply `persistent editable sufficient state`; CAT001 proves a restricted version is easy once the state lies in a closed associative algebra.

A major-invention candidate must now satisfy at least one genuinely differentiating condition:

1. retain useful nonlinear cross-source reasoning that cannot be represented by a conventional associative fold without material blowup, yet preserve exact source-local revisionability; or
2. update a richer persistent representation materially cheaper than the strongest dynamic aggregation/dependency baseline under matched memory; or
3. introduce a sound neural representation/certificate that changes the achievable editability-versus-expressivity frontier, rather than merely changing the data-structure schedule.

If we choose an associative closed operator family, the dynamic product tree (and applicable scan/monoid methods) is the baseline, not the invention. If we break associativity to gain expressivity, NIC001/CRR001/E89-style interaction and replay costs return unless a new exact mechanism is found. This is the current technical bottleneck.

## Gates and provenance

Completed Actions run **33989932774**: **17 tests passed in0.11s**, no skips. Artifact **9976308662** downloaded and ZIP SHA-256 verified against GitHub metadata:
`63aae585d7f947d3ab99a5b90698339f13f5eaa92eac845f5b84817776bf82bf`.

Archived hashes:
- experiment: `ef6cbc77acaeabc51074562a9514ca2ab54ebca4946264e91d271e80deef4eef`
- tests: `4fa4c11b629334b15e8703b8c451ee193db8f69599c4a831414539cf2b6310f7`
- CI results JSON: `193c266550976891a21a891bcf448bb9315b81834fcc1fa39cfc24846440e9c9`

Five algebraic seeds are zero trained-reader seeds and zero language backbones. No semantic Pod reader, J-space audit, UNKNOWN, generic divergence, publication race, total memory/throughput or trained capability gate is evaluated. All full programme gates remain unqualified, especially the >=10x strongest-baseline gate, which CAT001 explicitly fails at1.0x.
