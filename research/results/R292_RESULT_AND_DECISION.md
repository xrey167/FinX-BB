# R292 Result — Transitive Lifetime Closure

Status: **real-model anti-resurrection mechanism gate passed; not DoD.**

Executed on frozen `Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`, checkpoint SHA256 `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.

## Result

The experiment deliberately created downstream memo K/V while an old Pod generation was authoritative, then replaced only the source field K/V.

Across 12 transition/memo cases:

- oracle full-recompute accuracy: **100%**;
- naive source-only field swap top-match: **100%**, but this hides substantial logit contamination;
- maximum naive-vs-full-recompute full-vocabulary delta: **0.40625**;
- cases with detectable stale downstream effect: **9 / 12**;
- lifetime-closure top-match vs fresh masked oracle: **100%**;
- maximum closure-vs-fresh-masked-oracle full-vocabulary delta: **0.0**;
- maximum delta after arbitrarily randomizing masked stale downstream K/V: **0.0**.

Artifact ZIP SHA256: `aea42bee17c9240f40e2f5c715466777f63ec51ddf810914fc321ca67c6bf1f5`.

## Interpretation

Replacing or revoking only the source Port is insufficient as a general lifecycle contract. Downstream neural state produced while an old generation was live can still carry its influence even when the final discrete answer happens not to flip.

R292 therefore upgrades the architecture invariant:

`valid(derived_state) = AND over all authoritative source generations in its causal read-set`

When any source generation changes or is revoked, all dependent downstream neural state becomes inadmissible. Physical bytes may remain present, but the stale state is masked from causal execution.

The zero-delta stale-byte randomization result is the strongest part of the gate: after lifetime closure, stale downstream K/V contents are causally irrelevant to full-vocabulary logits for the tested cases.

## Decision

- Keep **transitive lifetime closure** as a mandatory architecture primitive.
- Reject architectures that only replace/delete the original fact representation.
- Combine this with the R297 Causal Lifetime Factor DAG so the full causal read-set does not need to be scanned on every inference read.
- Next integrated gate must combine canonical Pod authority, independent verification, multi-Pod J-Space, transitive lifetime factors and a frozen real backbone.

This remains a mechanism result, not a breakthrough claim or proof of RAG superiority.
