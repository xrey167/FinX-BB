# E-000094 — straight-through exact-support real-symlink reader

Status: preregistered before implementation/results. **Baseline mechanism test only; no novelty credit for hard routing, straight-through estimators, sparse attention, or top-k selection.**

## Motivation

E-000091 produced an interpretable qualified counterexample on seed 1: the current softmax reader passed every held-out template at >=0.975, yet an update to an unrelated canonical Pod B changed routing, hidden state, KV and full logits for a query whose selected Pod A lineage and witness remained current. Every real Bank row had strictly positive routing mass in every inspected slot. Therefore selected-object-only lineage is unsound for exact derived-state reuse under the current softmax reader.

E-000092 asks whether post-hoc hard routing preserves capability. E-000094 asks the stronger feasibility question: can the reader be **trained with exact discrete forward support from the start**, while keeping a differentiable surrogate only for backward optimization?

## Mechanism

For each resolve/dereference routing score vector s:

    p_soft = softmax(s)
    p_hard = one_hot(argmax(s))
    p_ST   = p_hard + p_soft - stop_gradient(p_soft)

Forward execution is exactly one-hot. Backward gradients follow the soft distribution. The same rule is applied at both canonical resolve and dereference. The frozen backbone remains unchanged.

The straight-through estimator is established prior art and receives zero novelty credit. A positive E94 only establishes compatibility of exact finite forward support with the real-symlink reader.

## Fixed training/evaluation contract

- Backbone: GPT-2 CPU, same adapter family as E-000088/E-000091/E-000092.
- `SO_BOS=1`, `status_gated=True`, `use_links=True`, `n_deref=1`.
- 3000 training steps unless a smoke run is explicitly labeled otherwise.
- 100 alias groups minimum.
- Seeds: 0,1,2 independent training seeds.
- Same consistency=0.15 and alternate-supervision=0.5 recipe as the current strongest reader; no extra mutation-locality loss in E94.
- Existing marker radius and lifecycle semantics remain unchanged.

## Preregistered primary gates

A seed is feasible only if all hold on the retained E94 checkpoint:

1. Every held-out paraphrase template 8..11 has candidate correctness >=0.95.
2. Full-vocabulary top-1 is reported separately and must not be hidden by candidate-only scoring.
3. Exact no-memory bypass max-abs == 0.0.
4. Exact routing support contains one real/null choice per routing slot in forward execution; there may be no positive off-support probabilities in the executed routing tensor.
5. Under the E-000091 unrelated-B intervention, when the chosen hard route and its A lineage remain unchanged, hidden/KV/full-logit state must be byte-identical. Any nonzero difference falsifies the claim that the explicit support is complete for the current adapter execution.

All three seeds must pass before E94 is considered a viable reader substrate. A mean >=0.95 does not substitute for per-seed/per-template gates.

## Negative controls / interpretation

- If capability falls below 0.95, post-hoc or trained exact single-route support is insufficient; do not relax to top-k merely to obtain a positive result. A separate preregistration would be required for structured multi-edge support.
- If the route is unchanged but neural state changes after unrelated-B mutation, the recorded route is not the full computational dependency and E94 is falsified.
- If capability passes and state identity holds, E94 still is NOT a breakthrough. It becomes a baseline substrate for generation-bound support, lifecycle replay attacks, J-space/J-lens audit, REVOKE/SHRED/leakage/UNKNOWN/locality/performance tests.
- No CAVI attack is interpreted unless >=3 retained real-reader seeds satisfy the >=0.95 capability prerequisite.

## Novelty boundary

Do not claim hard routing, straight-through estimation, sparse support, cache invalidation, version tags, or selective recomputation as novel. The only later research question that could earn novelty credit is whether lifecycle-counterfactual training plus exact generation-bound neural dependency support creates a materially better mutation-to-ready frontier than strong ordinary invalidation/replay and current selective-KV baselines while preserving the full live mutable-memory contract.
