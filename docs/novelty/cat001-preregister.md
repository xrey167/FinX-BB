# CAT-001 — compiled source-transform sufficient state

Registered 2026-09-05 BEFORE numerical execution. Parent FIR001: b3cfa14069787893e0e934c490ca795f84978498. Status UNRUN. This is a constructive baseline-discrimination experiment, not a novelty claim.

## Question

Can a persistent state retain long-term source-dependent behavior, avoid query-time raw-Pod rereads, and admit exact sublinear one-Pod updates? Yes in principle if sources compile into an associative transformation algebra. Test the strongest simple instance, then compare against the conventional dynamic-product-tree baseline with IDENTICAL source transforms, internal summaries, memory budget and arithmetic. If the baseline matches, this family gets zero mechanism novelty even if it beats full rebuilding by >10x.

## Fixed construction

- Five seeds; N in {64,256,1024,4096} canonical Pods.
- Each (source id,payload,generation) deterministically maps to an invertible 2x2 matrix over the prime field p=2^31-1. This encoder is a controlled algebraic stand-in, NOT a trained semantic reader.
- Source order is canonical and transformations are generally noncommutative. The persistent sufficient state is their ordered product stored in a balanced product tree.
- Queries consume only the root transformation and a query coordinate. No raw Pod payload is reread at query time. The projective readout y=(a*x+b)/(c*x+d) mod p is nonlinear in ordinary coordinates.
- One Pod UPDATE replaces one leaf and recomputes only its root path. Compare every internal node on that path, root transform and 64 query outputs against a clean full rebuild.
- REVOKE maps the source to the identity transform; UPDATE->REVOKE->UPDATE and ABA-generation controls must not reuse stale leaves.
- Adjacent-swap and context-interaction controls must demonstrate genuinely order-sensitive/nonadditive behavior rather than a commutative sum.

## Strong baselines fixed in advance

1. Full ordered rebuild: N-1 composition calls after leaf materialization.
2. Conventional balanced segment/product tree with the SAME leaf transforms and stored internal nodes. It receives exactly the same update algorithm and memory. Candidate/strong-baseline operation ratio must therefore be reported, not hidden behind the full-rebuild speedup.
3. Naive O(1) global inverse patch `G * M_old^-1 * M_new` is tested only as a negative control for an interior noncommutative edit. Failure does NOT prove an O(1) dynamic-product lower bound; no such theorem is claimed.

Primary cost metric is exact composition-call count. CPU medians are secondary. Tree construction cost and stored matrix counts are reported. The >=10x programme gate applies versus the STRONGEST guarantee-matched baseline, not versus full rebuilding.

## Claim limits

This experiment is intentionally close to classical dynamic monoid aggregation. It does not test trained language capability, semantic Pod addressing, J-space, UNKNOWN, generic divergence, publication races or full system throughput. Five algebraic seeds are zero trained-reader seeds and zero backbones. Projective transformations are not claimed to be sufficiently expressive for language memory.

A positive result only establishes existence of an exact editable sufficient state in a restricted associative algebra. Associative scans, monoids, segment trees, self-adjusting computation, incremental view maintenance and Memoroids are mandatory prior-art/baseline boundaries and receive zero novelty credit. A surviving invention must add a technically distinct representation or update primitive that beats this conventional dynamic aggregate under matched semantics/resources, not merely rename it neural memory.

## Prior art located before execution

- Morad et al., Recurrent Reinforcement Learning with Memoroids, NeurIPS 2024 / arXiv:2402.09900: efficient memory recurrences are formulated as monoids; associative operators admit parallel scan.
- MonoidReduce, ICLR 2026 submission: neural kernels represented as monoidal folds; further evidence that algebraic neural aggregation is established.
- SegTreeMem, arXiv:2606.04555 (2026): segment-tree organization is already used for long-horizon agent memory, although for retrieval summaries rather than exact neural source transforms.
- Classical segment trees/dynamic range products and Möbius/projective matrix composition predate this programme. No novelty is assigned to these ingredients.
