# CKCA Novelty Audit 5 — Cache-Coherence Analogy and Identity Aliasing

Date: 2026-09-13. Status: adversarial claim narrowing, **not novelty certification**.

## Token-coherence analogy is occupied

`Token Coherence: Adapting MESI Cache Protocols to Minimize Synchronization Overhead in Multi-Agent LLM Systems` (arXiv:2603.15183) explicitly maps LLM multi-agent artifact synchronization onto MESI-style cache coherence, constructs an Artifact Coherence System, proves a token-savings theorem and reports TLA+-verified single-writer/monotonic-versioning/bounded-staleness properties.

Therefore CKCA must not claim novelty for:

- recognizing an LLM system as a cache-coherence problem;
- importing MESI/coherence terminology;
- lazy invalidation of shared LLM artifacts;
- monotonic artifact versions as a coherence mechanism.

CKCA's surviving question is narrower: whether **live neural execution artifacts whose hidden/KV content has causally absorbed mutable knowledge** can obey exact generation-lifetime coherence while reusable base-neural state remains valid.

## Entity aliasing / anonymization is occupied

Privacy-preserving RAG work such as `Privacy-Preserving RAG by Concealing Sensitive Information from External LLMs` (SEAG, arXiv:2608.12675) explicitly replaces sensitive entities with generated aliases before sending queries/documents to an external generator. Counterfactual relation-extraction work has likewise used entity replacement to reduce factual/entity bias.

Therefore CKCA must not claim generic entity aliasing, anonymization, opaque identifiers or entity replacement as novel.

The R302 `Symlink Namespace Firewall` should be treated as an **architecture-specific safety primitive**: when a fact is governed by CKCA lifecycle authority, the generative path should not receive lexical identity cues that allow pretrained parametric memory to bypass Port revocation. Its contribution, if any, can only be in the joint lifecycle contract, not the replacement operation itself.

## Knowledge-conflict literature reinforces the need, not novelty

Long-standing and recent work on parametric-vs-external knowledge conflict shows that LMs do not reliably prioritize current external/tool knowledge over internal memory. Recent `Tool-Memory Conflict` work reports that prompting and RAG-based conflict resolution remain unreliable in tested settings.

This strengthens CKCA's engineering requirement:

> A governed mutable relation should not rely on instruction-following to suppress obsolete parametric knowledge.

But the observation that parametric and external knowledge conflict is itself established prior art.

## Surviving claim hypothesis after Audit 5

The candidate major contribution is now strictly the joint **neural generation-coherence protocol**:

1. natural aliases resolve to one canonical lifecycle identity;
2. governed execution receives only an independently verified current generation capability;
3. entity-specific mutable value enters through a model-native Port rather than by rewriting B-plane skill state;
4. all neural derivatives inherit the exact transitive lifetime of source generations;
5. expired neural derivatives remain physically present yet become causally inadmissible;
6. same-value ABA/revive cycles cannot reactivate old state;
7. serving admission is O(1) after dependency compilation;
8. restart/replica recovery cannot recreate authority by possession of stale bytes;
9. governed factual execution can prevent lexical parametric-memory bypass without disabling the base model's reusable language/reasoning ability;
10. this joint system must beat strong RAG/dedicated-memory baselines on the declared quality/latency/mutation/lifecycle frontier.

If prior art is found that already combines these properties over real transformer hidden/KV execution, the claim must narrow again.
