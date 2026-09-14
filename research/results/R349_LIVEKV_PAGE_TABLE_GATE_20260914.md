# R349 — LiveKV Page-Table Attention gate

Executed in GitHub Actions on 2026-09-14, run `34898941683`, commit `97e89385a676d917af8b33474d73db17e35b1b80`.

## Passing result

- Contract: **PASS**
- 4,096 canonical value pages, 4,000 cached queries, 32-d K/V, hard top-k = 8
- Exact current-world equivalence to matched full sparse-attention recomputation: maximum absolute error **0.0**, class mismatches **0**
- Mean future-world LiveKV class accuracy: **1.0**
- Mean stale numeric-cache class accuracy after complete future rewrites: **0.1286770833**
- Gain over stale numeric cache: **+0.8713229167**
- 50,000 world writes required **0** LiveKV descriptor patches/invalidations
- Counterfactual eager numeric-cache repairs: **391,403**, mean fanout **7.82806** cached entries/write
- LiveKV descriptor / FP16 numeric attention-output cache size: **0.75×**
- Controlled Python CPU warm-path full retrieval / LiveKV materialization: **43.4217×**
- Estimated reuse crossover in that controlled benchmark: **1.1025 queries**
- Generation races expected/detected: **4,001 / 4,001**
- Escaped races: **0**; race semantic mismatches: **0**
- Report SHA256: `d329bade0cbcb49e1609234e557c04f8d263eaf2b55d589e471322c07c8b8a89`
- Evidence ZIP SHA256: `45f9bb2363a355e2de93dcf0d028bd7a5c58d1da12e297b934532ed349dc024d`

## What this establishes

The gate implements attention cache indirection as `(canonical page IDs, FP16 attention weights)` rather than a copied numeric V-derived hidden vector. Because semantic K is stable and the mutable V payload lives only in authoritative pages, rewriting one page changes the next materialization for every cached query referencing that page without touching the query descriptors. Exact generation validation prevents publishing across same-value ABA and value-changing races.

## What it does not establish

This is **not** a novelty proof and not a real-LLM/RAG benchmark. PagedAttention already establishes page-table indirection for KV storage, and external-memory/DNC systems establish mutable addressed memory. The narrower unresolved claim is a cache whose V semantics deliberately refer to canonical mutable world pages shared across cached inference artifacts, rather than pages containing immutable historical KV tensors. A real pretrained-attention intervention and direct comparison against prefix/KV caching, RAG, editable KV, and model-integrated memory are still required.
