# E-000042 — a certified lower bound for deletion in a representation, from the J-lens support

Frozen gpt2, layer 7, no training. Directions are J-lens vectors -- rows of
`W_U J_l`, one vector-Jacobian product each (`so/jlens.py`) -- and the pool never contains a
token in the clean top-10 output, which is the workspace paper's own guard
against ablating the readout instead of the fact. Every subset of the pool is ablated, so
the true optimum comes from enumeration and the bound is compared against the answer.

## The pool cannot silence these facts

the paper-faithful J-lens ablation, with the paper's own guard against touching tokens in the clean output, does not silence these facts at this pool size -- so there is no closure to bound, and that is the result.

| measure | value |
|---|---|
| facts the model answers at >= 0.75 | 17 |
| facts attempted | 6 |
| facts any subset of the pool could silence | 0.0000 |

This is a result, not a failure to get one: a J-lens ablation that respects the
paper's guard does not remove a capital fact from GPT-2 small at this pool size. The
eight-direction ablation that DID silence these facts in an earlier run removed the
unembedding rows of the candidate answers themselves, which blinds the readout rather
than deleting anything.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| held/answer_before | >= 0.75 | 0.9338 | PASS |
| control_bound_can_exceed_one | >= 1.0 | 1.0000 | PASS |
