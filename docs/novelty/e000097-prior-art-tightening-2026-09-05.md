# E-000097 prior-art tightening — causal evidence routing is not the novelty

Date: 2026-09-05

## Direct collision

Allchin, *Learning What Matters: Supervising Global Context Pruning with Causal Evidence Sets*, arXiv:2607.21692 (July 2026), trains sparse selectors from causally recovered evidence sets rather than dense-teacher attention. The work explicitly shows attention and causal dependence can disagree, uses interventions/masking to recover causal evidence, and reports selectors that match or exceed dense-teacher performance on its retrieval tasks.

Therefore FinX-BB must NOT claim any of the following as novel:

- training sparse/hard routing from causal evidence rather than attention;
- recovering relevant support by masking/intervention;
- distilling a dense teacher into a causally supervised selector;
- exact/sparse context support as such;
- attention-vs-causality disagreement as a discovery.

Primary source: https://arxiv.org/abs/2607.21692

## Surviving narrower question

The remaining SO/CAVI research question is lifecycle-specific and stateful:

> Can a capable live LLM bind each actually executed mutable-memory read to `(canonical Pod identity, incarnation, generation)` and use the resulting generation-bound neural dependency frontier to preserve already-materialized Hidden/KV/routing/generated state across unrelated lifecycle mutations while invalidating/recomputing only true descendants after UPDATE/REVOKE/SHRED/RELINK/rollback, with a material mutation-to-ready advantage over strongest ordinary incremental/recompute baselines?

The following still receive zero novelty credit individually or as loose combinations: causal evidence sets, sparse routing, hard routing, dependency DAGs, MVCC, generations, canonical IDs, pointers/symlinks, cache invalidation, selective KV recomputation, distillation, causal probes, external memory, knowledge editing, and snapshot isolation.

## Consequence for E-000097

E-000097 remains a capability-baseline experiment only. Success would establish that dense semantic competence can be compiled into an exact immutable identity boundary; it would not establish novelty.

A later research-level claim must show the *lifecycle use* of generation-bound executed neural support on already-materialized state, including stale-state resurrection attacks and measurable mutation-to-ready savings, while satisfying all existing capability/security/J-lens gates.

Breakthrough = false for E-000097 by construction.
