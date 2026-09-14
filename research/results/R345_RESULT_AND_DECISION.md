# R345 Result — Prospective Neural State (PNS)

Status: **bounded temporal-semantics proof gate passed; this formalizes the new activation semantics but does not prove novelty.**

## Core distinction

A conventional retained hidden state is retrospective:

`h_t = F(q, W_t)`

It is a numeric result from one historical world snapshot.

A Prospective Neural State caches instead:

`κ_q : W -> h`

`κ_q` is a neural continuation over live references. It is not patched when ordinary world values change because it never copied those values as timeless hidden state. A numeric materialization of `κ_q(W)` is transactionally generation-validated; if retained, that numeric materialization receives a temporal lifetime.

## Exhaustive bounded result

The model checker enumerated **2,500** schedules over two mutable cells, including writes before/between/after reads, value-changing writes and same-value generation rewrites.

Safe PNS semantics:

- committed output mismatches: **0**
- post-commit stale numeric serves accepted: **0**
- prospective rematerialization mismatches: **0**
- prospective descriptor survival checks: **2,500 / 2,500**
- prospective descriptor invalidations required: **0**
- prospective descriptor patches required: **0**

Counterexamples to weaker semantics:

- naive no-commit false publications: **1,500**
- value-only commit false accepts: **940**
- same-value ABA false accepts: **940**
- old numeric materialization invalidation witnesses: **2,000**

Report SHA256: `776c9165503f8532d5b5aa6c6465516dff7979aa21a0e8fe74d62479dda46aca`.
Artifact ZIP SHA256: `5e6332f534e30a85c57868178bc4bb8f0393444ccae476e08c432c792ebb3b99`.

## Decision

Use **Prospective Neural State** as the semantic umbrella for R340–R344:

> A retained neural state may denote a computation over future current-world references, rather than a numeric tensor from the past.

The prospective state itself is not tied to value generations; only each numeric materialization is. Generation validation prevents mixed-snapshot publication and post-commit lifetime prevents later stale serving.

Partial evaluation, closures and MVCC are known. The open novelty question is narrower: whether this prospective semantics has been used as a native Transformer attention/hidden-state representation over mutable generation-scoped world references. No novelty claim is made yet.
