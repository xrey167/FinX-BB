# E-000091 — 2026 prior-art screen while the real-reader audit runs

Status: prior-art boundary only. No legal opinion, no exhaustive patent clearance, no novelty claim.

## ReCache (arXiv:2608.19662, submitted 20 Aug 2026)

ReCache is a direct collision with any broad claim that the invention is simply to alter transformer computation so independently named resources have composition-invariant reusable KV. It introduces **resource-wise attention** that removes cross-resource interactions and uses resource-local positions, specifically to make each resource's KV block reusable independently of order/composition. It further selects layer × KV-head-group visibility routes using contribution scores and semantically prunes resource fields. The paper reports resource-wise attention at 82.3% Inv-F1 versus 82.4% dense with 3.655x TTFT speedup; its full system reports 92.43% allocated KV-tensor memory reduction and 1.423x attention acceleration.

Primary/public sources:
- https://arxiv.org/abs/2608.19662
- https://github.com/EIT-NLP/ReCache

**Boundary:** "remove cross-resource attention / localize positions / cache each resource independently" receives zero novelty credit here. ReCache does not, from the material screened here, establish our exact canonical Symlink-Pod lifecycle semantics, every-dependent-write full-rebuild counterfactual equality after UPDATE/RELINK/REVOKE/SHRED, or independent J-space audit. Those differences are requirements to test, not assumed novelty.

## KV-Direct / residual-state reconstruction (arXiv:2603.19664, 20 Mar 2026)

*The Residual Stream Is All You Need: On the Redundancy of the KV Cache in Transformer Inference* argues that transformer K/V tensors can be reconstructed exactly from residual stream checkpoints and reports bit-identical reconstruction over several model families in its tested setup. It therefore narrows any claim that exact KV reconstruction from smaller neural checkpoints is itself novel.

Primary source: https://arxiv.org/abs/2603.19664

**Boundary:** storing a residual checkpoint and exactly reconstructing KV is a baseline candidate, not the invention. Our utility benchmark must include it wherever its assumptions match the state being repaired.

## Huawei selective KV segment recomputation patent

WO2026086089A1, published 30 Apr 2026 (priority shown by Google Patents as 23 Oct 2024), describes selecting target segments/key tokens and updating/recomputing KV cache using related preceding/other-segment tokens for sequence inference.

Source: https://patents.google.com/patent/WO2026086089A1/en

**Boundary:** selective KV segment recomputation based on dependencies is not a defensible standalone novelty claim.

## LinearKV (arXiv:2608.11231, 31 Jul 2026)

LinearKV addresses position-independent caching in hybrid LLMs and explicitly reuses matched local states while selectively recomputing where required. Exact state composition is discussed against alternative initialization strategies.

Source: https://arxiv.org/abs/2608.11231

**Boundary:** selective state reuse/recompute in hybrid attention/recurrent models is prior-art territory.

## MemLineage (arXiv:2605.14421, 14 May 2026)

MemLineage attaches cryptographic provenance plus derivation lineage to persistent LLM-agent memory entries and applies lineage-based policy propagation/gating.

Source: https://arxiv.org/abs/2605.14421

**Boundary:** attaching provenance/derivation DAG metadata to mutable LLM memory and enforcing policy from that lineage is not new. This work is at the agent-memory entry level rather than exact neural-derived-state counterfactual repair, but broad "lineage-guided memory enforcement" language is occupied.

## Consequence for the surviving seam

If E-000091 confirms that the current dense-softmax real reader makes selected-object-only lineage unsound, the next architecture cannot claim novelty merely by replacing dense interaction with resource-local attention or by tagging/recomputing caches: ReCache and the broader incremental-computation/cache-repair field already occupy those moves.

A surviving candidate would need a technically narrower property, for example a lifecycle-stable neural support construction whose canonical-pod transitions have **provably exact source support**, whose reconstruction is counterfactually exact through downstream persistent writes, and whose mutation-to-ready advantage survives ReCache-style composition-invariant caching, KV-Direct-style residual reconstruction, dependency-aware recomputation and ordinary invalidation under matched memory. This is a target, not a claim.
