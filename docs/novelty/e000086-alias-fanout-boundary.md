# E-000086 — alias fan-out is a utility claim, not a novelty claim

Date: 2026-09-05.

This lane measures one practical engineering property of the Symlink architecture: whether one canonical target lifecycle operation replaces target+alias duplicated mutations as alias fan-out grows. It does **not** promote aliases, pointers, canonical IDs, alias portability, or lifecycle propagation as novel.

## Direct prior-art exclusions

The broad alias story is already occupied by multiple lines of work:

- Tang et al., **Aligning Language Models with Real-time Knowledge Editing**, ACL 2026, introduces CRAFT with explicit **alias portability** and KEDAS for evolving real-time edits. https://aclanthology.org/2026.acl-long.14/
- Green et al., **BabelEdits**, Findings ACL 2025, evaluates rich entity aliases across 60 languages; BabelReFT learns modular entity-scope representation interventions applying across multilingual aliases. https://aclanthology.org/2025.findings-acl.1113/
- Cohen et al., **Evaluating the Ripple Effects of Knowledge Editing in Language Models**, TACL 2024, includes subject aliasing as an edit-propagation criterion. https://aclanthology.org/2024.tacl-1.16/
- US patent application **20260052096** describes a dynamic alias system resolving semantic aliases to canonical identifiers before policy/memory access. https://patents.justia.com/patent/20260052096

Therefore `many aliases -> one canonical identity` and `one semantic update should be visible through aliases` receive **zero novelty credit** here.

A second decisive collision is Ramesh, **Subtract or Replay? Exact Deletion from Language-Model Memory**, arXiv:2607.27539 (2026-07-30). It demonstrates on pretrained LLMs that exact deletion is representation-dependent: addressable influence can be algebraically removed; entangled later writes require checkpointed replay. https://arxiv.org/abs/2607.27539

Therefore addressable deletion, exact counterfactual deletion, decrement-vs-replay, or designing a memory representation to be deletable also receive **zero novelty credit**.

## What E-000086 may establish

Only the following engineering statement is under test:

> Given the existing E-000015 same-world symlink and duplicate stores, one canonical target UPDATE/SHRED has operation count O(1) in alias fan-out while implementing the same visible semantics that require O(k) copied target+alias mutations in the duplicate arm.

This is a useful systems property if measured, but it is expected from canonical indirection and cannot itself be a breakthrough.

The timing screen excludes model inference, routing, authorization, I/O, generated-history repair and J-space. It reports raw paired CPU measurements but the operation-count result is the primary invariant.

## Remaining novelty target

A successor must contribute something beyond canonical indirection and beyond exact deletion of addressable memory. The open technical question is whether a **qualified real LLM reader** can give a mutable canonical identity a neural effect that is simultaneously:

1. robustly reachable from held-out linguistic formulations;
2. current under alias relink/update/revoke/delete/rollback;
3. impossible to resurrect through any previously materialized neural-derived state;
4. exact-BYPASS outside scope;
5. independently corroborated by a never-optimized J-space/J-lens audit; and
6. materially cheaper or more capable than strong ordinary alternatives such as canonical retrieval + cache invalidation, addressable decrement, source isolation, or suffix replay.

Even this conjunction is not assumed novel. A final claim requires a direct 2025-2026 paper/standards/patent search around the **specific surviving mechanism** and a measured systems advantage.

## Qualification boundaries

E-000086 is store-level only. It must not be used to interpret E-000070/071/073/074 or other CAVI attack results. Positive neural interpretation still requires, on the same retained candidates:

- >=0.95 fresh real-symlink correctness across >=3 genuine training seeds;
- >=0.95 held-out paraphrase reading;
- >=0.95 REVOKE and SHRED propagation;
- <=0.02 deleted-object leakage;
- >=0.90 UNKNOWN on missing key;
- <=0.05 nats generic KL or exact no-memory bypass;
- stale Bank/router/selected-route/payload/hidden/KV/activation replay and race closure;
- independent J-space/J-lens audit; and
- preferably more than one public backbone.

Negative results remain evidence and are not rewritten as supporting claims.
