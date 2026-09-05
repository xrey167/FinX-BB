# E-000040 — the traceless price in a representation: a deletion that announces itself

Frozen gpt2, layer 7, no training. The deletion is each fact's workspace
closure (E-000037), and the ablation is that closure made continuous: the state's component
in the closure span is scaled by `a` at every layer from the read layer up and at every
position. `a = 0` is E-000037's projection exactly; `a = 1` is the identity; `a` is the size
of the hole an adversary measures.

## How much of a fact's carrier is its own

Before any closure is searched for, each direction in the fact's basis is removed on
its own and the bystander facts are re-read. A direction whose removal alone takes them
down is not this fact's to delete: it is the shared carrier the workspace paper
describes, and removing it is the activation-space form of deleting the object a pod's
other aliases still point at. The search then runs on the rest.

| measure | mean over seeds | worst seed |
|---|---|---|
| the model answers the fact, before anything | 0.9338 | 0.9338 |
| share of the fact's basis that is SHARED with other facts | 0.0784 | 0.0809 |
| facts silenceable using only their own directions | 0.5098 | 0.4706 |
| bystander accuracy under those deletions | 0.6708 | 0.6167 |

Read these two rows together, because separately each one misleads. Only a small share
of a fact's basis is shared, so the carrier is mostly the fact's OWN -- and yet only
about half of the facts can be silenced using nothing but their own directions, and the
bystanders still lose ground when they are. Privacy of the carrier is therefore not
sufficient for a clean deletion: the directions a fact does not share with any single
other fact are still directions the rest of the model is using. A store can name the
aliases that would be left dangling; a model cannot, and this is what that costs.

## The deletion, and what it costs bystanders

| measure | mean over seeds | worst seed |
|---|---|---|
| directions in the closure | 3.33 | 3.67 |
| the model answers, before | 0.9444 | 0.9375 |
| the model answers, closure removed | 0.0000 | 0.0000 |
| bystander facts under the same ablation | 0.8611 | 0.8056 |
| bystander facts with nothing removed | 1.0000 | 1.0000 |

## The statistic, on the deviation from the population mean

A raw residual is dominated by what every state at this layer and position has in
common, so an uncentred fraction along any unit direction is near the same for a state
that holds the fact and one that does not: a first pass measured 0.813 for bystanders
against the holder's 1.000, which is a property of the common mode, not of the fact.
Every ratio built on it is inflated toward one, and that inflation biases the result
TOWARD the claim under test. The statistic below is therefore the component along the
fact direction of the state's DEVIATION from the population mean, and the never-held
level it is compared against is a distribution, not zero.

| measure | mean over seeds | worst seed |
|---|---|---|
| a state that holds the fact | +0.1877 | +0.1819 |
| a state that never held it | -0.0162 | -0.0204 |
| spread of the never-held level (1 sd) | 0.0518 | 0.0568 |
| the deleted state | -0.0780 | -0.0470 |
| how far below never-held the deletion sits, in sd (z) | +2.61 | +2.50 |
| facts where that z reaches 2 | 0.6944 | 0.6667 |

## The price of hiding it

| measure | mean over seeds | worst seed |
|---|---|---|
| smallest scale whose statistic looks natural (a_hide) | 0.5431 | 0.6417 |
| largest scale at which the fact stays silenced (a_answer) | 0.4194 | 0.4000 |
| the gap | +0.1236 | +0.0375 |
| facts where the gap is positive | 0.3611 | 0.2500 |

Both quantities come from ONE sweep: the snapshot is re-read at every scale and the
statistic recomputed from it, so no linearity between the scale and the statistic is
assumed. The adversary holds one snapshot and a hypothesis about which fact was
deleted, which is the setting of Chen et al. (CCS 2021) and Gao et al.
(arXiv:2202.03460) minus the two model versions they both require.

## The rule, fixed before the run

traceless_gap = a_hide - a_answer, over facts admitted by the collateral rule. POSITIVE: the smallest hole that looks natural is bigger than the largest hole the deletion can afford, so unreachable and traceless are not jointly achievable by scaling the fact's component and the store's law (E-000041, T > U) carries into the representation. NEGATIVE: there is a scale that both silences the fact and leaves a natural-sized component, so the representation admits a traceless erasure the store does not, and the reason would be that the closure span has dimensions the fact does not need. Both readings, and this rule, are fixed before the run.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| held/answer_before | >= 0.75 | 0.9338 | PASS |
| answer/deleted | <= 0.25 | 0.0000 | PASS |
| collateral | >= 0.6 | 0.8056 | PASS |
| hole_detectable | >= 0.75 | 0.6667 | FAIL |
