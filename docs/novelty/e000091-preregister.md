# E-000091 — certified finite-response domains

Registered 2026-09-05 BEFORE numerical execution. Parent E90: 57997fc351b65bc74ca795f92e7bb35bbd6fd737. Status: UNRUN. This is a candidate/baseline discrimination experiment, not a novelty claim.

## Question

E89's statement that sparse dependency cones are the only remaining way to avoid dense downstream work was too broad unless finite-response structure is excluded. A ReLU persistence stack can have dense descendants but an exactly affine response to a low-dimensional source revision while its entire activation pattern remains unchanged. Test this restricted escape and whether it reduces to ordinary activation-region caching/affine folding. Literature search BEFORE this registration already surfaced Neural/Hypersphere Caching (LongevIoT 2025), Incremental Neural Network Verification (2023/2025), and a July 2026 public Jacobian-Causal Memory Control research preview. No novelty is assigned to any of those ingredients.

## Fixed execution

Five random parameter/intervention seeds 0..4; 256 channels, eight persistent ReLU layers, two source coordinates. Integer-valued network and source inputs to separate exact algebra from floating-point reassociation. Dense random weights in {-1,0,1}; immutable exogenous input/weights/biases during each revision. Retain full original persistent states in every arm.

Compile source response maps to every PREACTIVATION. For a finite edit, predict all preactivations and authorize the fast path only when every activation predicate agrees with the compiled region. Compare every stored layer coordinate, not final output, against a full rebuilt integer trajectory. Independent arbitrary-precision execution must check no int64 wraparound. Unsafe arithmetic must reject, not silently wrap.

Revisions: 16 random integer edits for each magnitude 1,16,256,4096 per seed, plus deletion-to-source-zero/NEVER. Report acceptance and exactness for each magnitude, never pool them into a language-model success rate. Preserve all rejected cases. Compare an UNSAFE unverified fixed-response arm to expose finite-edit failures.

Hybrid: reuse a verified prefix; at the first failed predicate replay that layer and the remaining suffix, then compare every persistent write to full rebuild. This is conventional fallback, not novelty.

Strongest matched baseline: ordinary cached activation-region affine maps evaluated at the revised source, using identical immutable context, state, operator family, response matrices and memory budget. Distinguish it from weaker dense cone replay. If ordinary folding reproduces the candidate's work, a speedup over dense replay gets no mechanism-novelty credit. Giving this baseline the same compiled representation is a mechanism ablation, not a universal argument against possible future novel representations.

Measure CPU medians for warmed certified evaluation, ordinary affine baseline, full/cone replay, hybrid rejected edits, and construction of the response cache. Include all layers in outputs. Record array bytes and compilation work, not just mutation acknowledgement. These are operator microbenchmarks, NOT language-model throughput or complete system mutation-to-ready gates.

Smooth-nonlinearity negative control: four-layer tanh stacks, width 64, two source coordinates, five seeds. Compare frozen source-Jacobian responses with actual finite-edited persistent writes and a no-op control. A derivative is not a finite-response certificate for tanh. This control is not a second language backbone.

## Gate and claim limits

- Exact byte identity within deterministic int64 execution, checked against arbitrary-precision values. No tolerance relaxation or cross-platform timing promise.
- No trained reader, fresh/paraphrase score, scoped UNKNOWN, independent J-space validity, generational publication-race closure, overall memory-budget/throughput or >=10x full-system gate is established here.
- Expected mathematical exactness inside a verified ReLU region does not establish novelty. Region crossing must not use stale maps without fallback.
- Keep A/B/C research open for a demonstrably new finite-response representation, but ordinary affine caching and graph replay remain excluded.
