# CKCA Novelty Audit 4 — Read Sets and Derived-State Repair

Date: 2026-09-13. Status: adversarial prior-art narrowing, not novelty certification.

## New closest mechanisms found

### S-Bus occupies automatic LLM read-set reconstruction for shared mutable state

`S-Bus: Automatic Read-Set Reconstruction for Multi-Agent LLM State Coordination` (arXiv:2605.17076) reconstructs per-agent read sets from observed HTTP GET operations, applies optimistic concurrency control, defines Observable-Read Isolation, and includes TLAPS/TLC/Dafny evidence. Its exhaustive TLC study reports more than 20 million states with zero violations for the modeled property.

Consequences for CKCA:

- “LLM systems should have read sets” is not novel;
- automatic read-set reconstruction is not novel;
- using read sets to reject stale writes/commits is not novel;
- formal consistency properties for LLM shared state are not novel.

S-Bus is aimed at HTTP-observable multi-agent mutable state and commit safety, not at transformer hidden/KV derivatives or neural-logit anti-resurrection. That distinction must be experimentally and formally maintained; it cannot be assumed to create novelty.

### MemoRepair and dependency-guided rollback occupy descendant repair

MemoRepair (arXiv:2605.07242) explicitly models stale derived artifacts after source changes and withdraws affected descendants before validated repair/republication. Dependency-Guided Rollback Repair (arXiv:2608.10502) builds a typed memory-to-action dependency graph, deactivates unsupported downstream memory state and selectively replays affected computation.

Consequences:

- cascade invalidation of derived memory is not novel;
- dependency graphs for memory repair are not novel;
- selective preservation of unaffected descendants is not novel;
- selective replay rather than global reset is not novel.

### Runtime KV lifecycle metadata is an active infrastructure direction

Recent LLM serving/runtime work and engineering roadmaps increasingly associate KV artifacts with semantic/program lifecycle metadata, global addressability and context types. Therefore attaching lifecycle metadata to KV blocks, by itself, cannot carry a major claim.

## Stronger surviving distinction

The remaining candidate is specifically **generation-coherent neural execution**:

`authoritative mutable generation -> admitted model-native Port -> neural derived state -> transitive generation lifetime -> constant-time serve gate`

The differentiating target is not reconstructing which external records an agent read. It is ensuring that a neural artifact which has already absorbed mutable knowledge cannot remain causally admissible after any source generation expires, while unaffected base-model state stays reusable.

Required evidence must therefore show all of the following jointly:

1. real transformer hidden/KV/J state, not only structured memory records;
2. current-generation capability verified before neural materialization;
3. source updates while stale neural bytes remain physically present;
4. measurable stale-logit influence without closure and zero stale-logit influence with closure;
5. transitive lifetime through composed neural states;
6. same-value rewrite / ABA and revoke→revive non-resurrection;
7. O(1) neural-artifact admission after dependency compilation;
8. unchanged reusable B-plane under mutable world writes;
9. restart/replica stale artifacts cannot regain admission;
10. no full replay required merely to reject old neural derivatives.

R292, R297, R291b, R295b and R299 now cover narrow pieces of this list. The integrated R298b and later generative gates must show the pieces still hold when combined.

## Claim language to avoid

Do not use any of the following as a novelty statement:

- read-set tracking for LLM agents;
- observable causal consistency for agent state;
- dependency-guided memory invalidation;
- descendant cascade repair;
- provenance-preserving memory;
- lifecycle-tagged KV caching;
- immutable/versioned memory alone.

If a major contribution survives, it must be framed as a **neural knowledge coherence protocol**, not as generic agent-memory lifecycle management.
