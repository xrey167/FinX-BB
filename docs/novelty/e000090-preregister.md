# E-000090: source-cone soundness boundary

Status at registration: UNRUN. Negative-result / architecture-contract experiment, not an invention claim. Parent: E-000089 commit 0bfb98c3c644816d5a623bf0808174c609d28e0c. Historical evidence is not modified.

## Questions fixed before numerical execution

1. Does block-sparse source injection remain block-local through persistent nonlinear computation when global LayerNorm/RMSNorm is present? Compare identity, per-block normalization and global normalization with the same block-diagonal weights. Five random parameter seeds (0..4), width 128, source block 16, eight persistent layers. Compare every persistent coordinate with a clean full rebuild. An oracle patch that is allowed to take exact rebuilt source-block coordinates but must reuse all other coordinates is a deliberately strong test of the proposed support, not a timed implementation.
2. Can lineage containing only previously selected payloads safely handle edits to mutable routing keys? Construct a dormant-key activation case with 64 queries and three persistent nonlinear layers. Verify the previously unselected source enters the selected set and that reuse based on the old payload cone disagrees with full rebuild. Include an exact decision-aware recomputation control. These are adversarial constructions, not estimated real-model failure frequencies.
3. Can an unselected source influence a selected persistent write without changing the winning route? Compare top-1 weighting after global softmax with top-1 selection followed by local renormalization. The former must account for denominator lineage; the latter is a control where the unselected source should have no effect while the route is unchanged.
4. Can a context-averaged Jacobian lens prove the absence of all source influence? Construct a context-dependent sign channel whose Jacobians cancel under balanced context averaging while interventions have nonzero effects in each context. Compare repaired/NEVER states, the averaged lens and context-conditioned interventions. This is an analytic instrument counterexample, NOT an implementation or evaluation of Anthropic's full J-space method on a language model.
5. Implement an exact block-local replay reference, compare UPDATE and deletion-to-NEVER at every persistent write, and permit the conventional dependency-aware replay baseline identical state, weights, lineage and operators. Do not award speedup or novelty when both execute the same affected operations.

## Fixed evidence contract

- Equality for reference repair means byte-identical floating-point arrays in this deterministic execution, not final logits or a tuned tolerance. Numerical max-absolute differences are also reported.
- Register exact-coordinate support separately from a descriptive >1e-10 effect threshold. Neither a zero derivative nor a below-threshold effect establishes nondependence.
- At least five parameter/intervention seeds; these are not five trained-language-model seeds and these operator families do not satisfy the second-language-backbone gate.
- A single explicit valid counterexample falsifies a universal soundness claim for that certificate design. No claim that all learned certificates, all sparse architectures, or all neural repair mechanisms are impossible.
- No fresh-paraphrase, UNKNOWN, generic-KL, throughput, matched-system-memory or >=10x utility claim from these numerical screens. Those gates remain unpassed/unmeasured.
- Preserve failures, implementation corrections and all per-seed results.

## Candidate architecture consequence (conditional on results)

Payload lineage alone is not a complete persistent-state dependency type. Routing decisions, normalization domains, missing-key decisions and changed control flow must either carry sound lineage or be proved source-independent. Learned certificates may propose work; a deterministic verifier/replay fallback must authorize reuse. Independent J-space auditing remains corroborative, not a completeness proof. These are correctness requirements and conventional baselines, not novelty.
