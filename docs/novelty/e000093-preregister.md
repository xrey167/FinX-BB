# E-000093 — Mutation-locality objective

Status: preregistered before implementation/results. This experiment is a falsification/mechanism test, not a novelty award.

## Question

Can the real frozen-GPT-2 symlink reader be trained so that changing an unrelated canonical Pod has negligible effect on the neural state of a query whose true dependency set excludes that Pod, while preserving the strict held-out reader capability gate?

The target property is not generic sparse attention. It is **mutation-local causal support**: a canonical knowledge mutation should perturb only neural computation whose verified dependency set intersects the mutated Pod.

## Baseline

Use the E-000081 strong real-symlink reader family with BOS, strict marker contract, 100 symlink groups, four held-out templates (8..11), and the same answer/routing/gate/paraphrase-consistency losses. E-000091 supplies the unrelated-Pod intervention and exact hidden/logit/KV comparison. E-000092 remains the hard-routing baseline and receives no novelty credit.

## Intervention-pair training objective

During the full-loss phase, sample a small paired sub-batch of one-hop queries A. For each query, clone the current bank and mutate the payload of one active canonical Pod B chosen so that B is not the query target and not in the reference resolution path for A. Run the same textual query against the original and counterfactual bank.

Add a locality loss over quantities that should be invariant to B:

1. routing distribution at every resolve/dereference slot;
2. final candidate-logit distribution;
3. final hidden representation.

Use symmetric KL for routing/candidate distributions and normalized MSE for the final hidden state. The counterfactual arm is not given a changed answer target because its mutation is reference-irrelevant to A.

This objective is explicitly different from paraphrase consistency: the text is held fixed and the *knowledge state* is intervened on.

## Arms

- control: historical E-000081 objective;
- locality: same objective + mutation-locality loss.

No thresholds may be changed after results are observed.

## Primary capability gate

For every requested seed independently:

- candidate correctness >= 0.95 on each held-out template 8,9,10,11;
- no-memory bypass max absolute logit difference == 0.

A seed failing the capability gate is not interpretable for the locality claim.

## Primary locality metrics

On an independent E-000091-style unrelated-B canonical payload mutation:

- maxabs hidden change;
- maxabs full-logit change;
- routing maxabs change;
- KV-cache maxabs change when available;
- stale-vs-fresh continuation logit maxabs;
- A-only authority witness remains current;
- old B witness becomes stale.

The control and locality arms use the same evaluation world family and intervention selection rule.

## Success criterion

The locality arm must, on all interpretable seeds:

1. preserve the strict capability gate;
2. reduce hidden-state intervention maxabs by at least 10x versus its matched control seed;
3. reduce full-logit intervention maxabs by at least 10x versus control;
4. not increase stale-vs-fresh continuation divergence;
5. preserve the authority controls (A current, B-old stale).

Exact byte identity is recorded but is not required for E-000093 success because the training objective is continuous. Exact finite support remains the later target.

## Falsification

E-000093 is falsified as a useful route if locality training either:

- destroys the strict reader capability gate on >=2/3 requested seeds, or
- fails to produce the 10x hidden/logit locality improvement on >=2/3 interpretable seeds.

## Novelty boundary

Even a positive result does **not** establish novelty. Sparse/hard attention, selective KV recomputation, consistency regularization and counterfactual representation learning all have prior art. The narrower candidate claim requiring separate prior-art clearance is the combination of:

> canonical editable neural knowledge identities + training-time counterfactual mutation-locality objective + versioned lifecycle authority + selective validity/reuse of materialized neural state.

E-000093 only tests whether the central trainability premise of that combination is empirically plausible.
