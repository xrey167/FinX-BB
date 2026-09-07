# PDX-003 — §13(b), by a separately registered run

**E-000033 failed its own control** on 2026-09-07: both arms must answer before any deletion at
>= 0.80, and mean-pooled `gpt2` read **0.0467** mean, **0.0250** worst seed. Its docstring says what
that means — "the comparison is between a working store and a broken one" — so its two data columns
are not interpretable and §13(b) was not discharged.

**This does not replace or amend E-000033.** Its record, criteria and source are untouched, and a
test asserts it. PDX-003 is a separate registration of the same protocol with **exactly one declared
change**: the encoder.

## The one change

`sentence-transformers/all-MiniLM-L6-v2` instead of mean-pooled `gpt2`. A causal LM's mean-pooled
hidden state is not a retrieval representation and was never trained to be one; a sentence encoder
is. Everything else — the facts, the twelve templates, canonical-vs-duplicated index construction,
the read test, the closure search, `retrieve`, `resolve` — is **E-000033's own code, imported and
called**. A test asserts those imports, so "same protocol, one swap" is true of the code and not
only of this sentence.

## Result — 3 seeds, 150 facts, 4 chunks each, 600 chunks per arm

| | canonical | duplicated |
|---|---|---|
| answers before deletion | 0.8967 | 0.9350 |
| chunks holding the fact | **1.00** | **4.00** |
| still retrievable after one deletion | **0.1333** | **0.9867** |

| criterion | required | observed | |
|---|---|---|---|
| `control/read_before_deletion` | >= 0.80 | **0.8800** (worst seed) | PASS |
| `control/read_gap` | <= 0.15 | 0.0500 | PASS |
| `canonical/fact_closure_max` | <= 1.0 | 1.0000 | PASS |
| `duplicated/fact_closure_mean` | >= 4.0 | 4.0000 | PASS |

**Decision: `CLOSURE_REPRODUCED_IN_A_RETRIEVAL_STORE`.**

The control E-000033 missed by a factor of thirty is met on every seed. §6's closure holds in the
substrate §13(b) asked for: deleting the chunk a fact's own question retrieves removes the fact in
the canonical store and leaves the duplicated store answering **0.9867** of the time. That is Codd's
modification anomaly applied to a delete, measured where practitioners meet it.

## What this does not show

Swapping a component of a pre-registered experiment until it passes is precisely the move this
programme distrusts. So: the change is one function, it is declared in the record, and this run
carries its own registration rather than inheriting E-000033's. **E-000033's failing record stands
as recorded.**

It shows the closure survives in a retrieval store *when the store can actually be read*. It does
**not** show that E-000033's own configuration was sound — that configuration is still, on its own
evidence, unable to answer.
