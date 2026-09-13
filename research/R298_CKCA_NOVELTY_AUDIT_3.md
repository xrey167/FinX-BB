# CKCA Novelty Audit 3 — Provenance, Authority and Cascade Repair

Date: 2026-09-13. Status: claim-narrowing audit, **not a novelty certification**.

The architecture claim is narrowed again after explicitly searching recent 2026 work on persistent-memory authority, provenance laundering, derived-memory invalidation and cascade repair.

## Prior art that kills broad claims

### Origin-bound memory authority is occupied

`Securing LLM-Agent Long-Term Memory Against Poisoning: Non-Malleable, Origin-Bound Authority with Machine-Checked Guarantees` (arXiv:2606.24322) formalizes non-malleable origin-bound authority for persistent agent memory and reports machine-checked guarantees. Therefore CKCA cannot claim novelty for attaching cryptographic/IFC-style authority to memory records or for failing closed on insufficient authority.

### Provenance-preserving memory firewalls are occupied

`Memory Provenance Laundering in LLM Agents: A Non-Amplification Firewall for Persistent Memory` (arXiv:2607.29167) targets authority laundering during memory consolidation and preserves platform-maintained provenance through a memory firewall. Therefore source-authority preservation across textual memory transformation is occupied.

### Fine-grained provenance/capability contracts are occupied

PACT (`The Granularity Mismatch in Agent Security...`, arXiv:2605.11039) tracks provenance through replanning and enforces role-specific capability contracts on tool arguments. Therefore fine-grained runtime provenance plus capability-style authorization is not CKCA novelty.

### Dedicated memory readers on frozen LMs are occupied

TokenMem (arXiv:2607.22625) attaches a dedicated cross-attention knowledge path to frozen LMs with a thin gating adapter. `Trained Persistent Memory for Frozen Decoder-Only LLMs` (arXiv:2603.22329) evaluates several persistent-memory interfaces, including parallel cross-attention and KV extension. Therefore B-plane + generic separate memory reader is not sufficient novelty.

### Derived-memory cascade invalidation/repair is occupied

MemoRepair (`MEMOREPAIR: Barrier-First Cascade Repair in Agentic Memory`, arXiv:2605.07242) formalizes the cascade update problem over durable derived artifacts, withdraws affected descendants before repair and only republishes validated predecessor-closed successors. With complete influence provenance it reports zero invalidated-memory exposure in its evaluated setting.

`From Faulty Memories to Corrected Actions: Dependency-Guided Rollback Repair for Memory-Augmented Agents` (arXiv:2608.10502) builds a typed memory-to-action dependency graph, traces downstream effects, deactivates unsupported memory state and selectively replays affected computation.

Therefore **“source invalidation propagates to derived memories” is already occupied**. CKCA cannot claim that as the invention.

Public agent-memory implementations/specs also explicitly require source revocation/supersession to force dependent derived state into revalidation/non-current posture. Versioned/tombstoned memory references and append-only event ledgers are likewise established engineering patterns.

## Surviving research gap is narrower

The remaining candidate is not provenance over records. It is **coherence over live neural execution artifacts**.

CKCA must demonstrate a joint property stronger than durable-memory cascade repair:

1. a canonical mutable identity has monotonic neural generations;
2. only an independently verified current generation can enter the neural read path;
3. the immutable language/skill cache is kept separate from mutable world state;
4. every *neural* derivative that may later influence logits — KV segments, hidden/J states, speculative branches, assistant prefixes and cached reasoning artifacts — carries the exact transitive lifetime of the generations that causally created it;
5. invalidation converts those derivatives into causally inadmissible state without requiring full replay or global cache destruction;
6. stale bytes may remain physically resident and may be arbitrarily mutated without altering current logits once their lifetime has expired;
7. same-value rewrites and revoke→revive cycles cannot validate a previous neural artifact;
8. the hot serving path reduces the lifetime check to a constant-time admission primitive;
9. restart/replica recovery cannot re-authorize an expired neural generation;
10. the complete mechanism is competitive with strong RAG and dedicated-memory systems on quality, TTFT, context cost and online mutation cost.

R292 and R297 are specifically relevant to points 4–8: R292 shows that downstream stale neural KV can retain measurable influence even after source-only replacement, while lifetime masking removes that influence to zero full-vocabulary delta in the executed probes; R297 moves transitive validity from repeated source-set scanning to an O(1) factor-bit read.

R291b and R299 address points 2 and 9: current-only neural page materialization plus authenticated recovered authority.

## Claim discipline

No CKCA novelty claim should use phrases equivalent to:

- provenance-aware memory;
- revocable memory;
- cascading deletion;
- dependency-guided memory invalidation;
- capability-protected agent memory;
- separate memory attention;
- versioned/tombstoned memory;
- append-only memory ledger.

Those are components or neighboring prior art.

The only potentially defensible major contribution is the **neural coherence protocol as a whole**, if future real-model and literature/patent gates show that exact generation authority + transitive lifetime over derived neural state + constant-time causal admission + immutable-base reuse has not already been disclosed.
