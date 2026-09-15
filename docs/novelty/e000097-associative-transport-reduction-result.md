# E-000097 result — associative exact recomposition is a generic dynamic baseline

Date: 2026-09-05
Status: **DECISIVE DIRECTION CHANGE / scoped family kill**

## Decision

**Kill associative recurrence / exact scan / dynamic segment-tree recomposition as a standalone major-invention seam.**

This does not kill a future compact exact neural-specific state representation. It kills the idea that, once exact composable summaries already exist, organizing them into an associative scan/tree and updating only the changed path is itself the invention.

## Registered reduction

The candidate class has ordered transitions `x_i = f_i(x_{i-1})`, an exact stored summary `S(f)`, and an associative composition rule `⊗` satisfying

`S(g ∘ f) = S(g) ⊗ S(f)`.

A generic balanced dynamic segment tree can store the candidate's exact summaries unchanged at leaves and internal nodes. After a lifecycle edit changes a local transition, the baseline replaces the affected leaf and recomputes only the path to the root. It therefore uses the same summary representation, the same composition primitive, the same exact current computation, and `O(log n)` summary compositions per local update.

The reduction is elementary dynamic function composition, not a new theorem or mechanism.

## CI execution

GitHub Actions run `33992162818`, head `6e221ac284d135a32b41df3e290cdf530aba3502`, completed successfully. The registered experiment used 32 deterministic seeds, sequence lengths `8, 31, 64, 127`, 24 mutations per family/length/seed, and two exact transition families:

1. compact affine maps over the prime field `p=65537`;
2. arbitrary nonlinear maps over a 31-state finite domain, represented by exact lookup tables.

The nonlinear arm is deliberate: the generic reduction is not restricted to linear recurrences. It also exposes the real problem—arbitrary nonlinear exact summaries can be large.

### Exact result

- mutation traces: **6,144**;
- root-summary comparisons: **6,144**, mismatches **0**;
- current-output comparisons: **116,736**, mismatches **0**;
- generic tree update compositions: **32,256**;
- dense transition evaluations used for the checked fresh-replay outputs: **6,712,320**;
- accumulated affine-tree summary words: **29,696**;
- accumulated arbitrary-nonlinear lookup-tree summary words: **460,288**.

The executable returned `KILL_ASSOCIATIVE_RECOMPOSITION_ALONE_AS_NOVELTY_SEAM`. Source/result hashes and artifact `9976950841` were emitted by the successful workflow.

The absolute dense-evaluation count is descriptive rather than a claimed speedup: the output checker deliberately evaluates many initial states, while each tree mutation updates one path. The scientific conclusion is equality under the same summary algebra, not a benchmark ratio.

## Consequence for the active invention programme

E-000095 and E-000096 already show that statistically learned cross-context correction receipts do not supply exact held-out transport for the tested transformer states. E-000097 now prevents the programme from escaping that falsification by relabeling ordinary associative recomposition as a neural lifecycle invention.

Do not allocate major-invention credit to:

- associative scan itself;
- prefix products / affine recurrence composition;
- dynamic segment trees over cached transition summaries;
- `O(log n)` recomposition after an edit when the exact composable summary was already assumed;
- a "neural segment tree" whose nodes store the same sufficient transition state available to a generic implementation.

The next useful question is no longer **how to recombine exact summaries**. It is whether a real nonlinear transformer suffix admits a **compact exact lifecycle-transport summary** that is materially smaller/cheaper than the strongest exact recomputation baseline and cannot simply be handed to generic dynamic function-composition machinery with the same benefit.

## Fresh 2025–2026 boundary

The targeted literature/patent screen this turn further crowds broad recurrence/composition/cache-transform claims:

- **ScanWeaver** (arXiv:2606.00601, 2026) explicitly lowers affine/selective recurrences to associative scans. Associative affine scan is therefore ordinary compiler/runtime machinery, not a novelty anchor.
- **LinearKV** (arXiv:2608.11231, 2026) explicitly discusses algebraically composing cached states for hybrid recurrent/attention models, while finding cheaper approximate initialization preferable on its evaluated tasks. This is not lifecycle mutation or exact transformer-state repair, but it further removes broad credit for composing recurrent cached states.
- **KVCOMM** (NeurIPS 2025) already estimates cross-context KV corrections from an online anchor pool and reports >70% reuse and up to 7.8x prefill speedup. This remains an approximate/cross-context serving baseline rather than an exact lifecycle guarantee.
- **KVEraser** (arXiv:2606.17034, 2026) treats exact localized context erasure as suffix recomputation and learns approximate KV steering to avoid it; it remains a direct repair baseline.
- Huawei **WO2026086089A1**, published 2026-04-30, covers KV-cache segment recomputation after selecting target/key-token regions.
- Intel **US20260080217A1**, published 2026-03-19, covers KV-cache transformation/compression using gauge transformations and explicitly references a provisional titled "COMPOSABLE EXACT KEY-VALUE CACHE COMPRESSION". Its exact mechanism is representation compression, not knowledge-mutation repair, but broad exact/composable KV transformation claims are crowded.
- IBM **WO2026087278A1 / US20260119893A1**, published 2026-04-30, covers direct insertion/modification/deletion of knowledge through a KV-cache network layer.

These references are boundary evidence, not an assertion that any one anticipates the narrow surviving target and not patentability/infringement advice.

Primary sources inspected:
- https://arxiv.org/abs/2606.00601
- https://arxiv.org/abs/2608.11231
- https://proceedings.nips.cc/paper_files/paper/2025/hash/1a074a28c3a6f2056562d00649ae6416-Abstract-Conference.html
- https://arxiv.org/abs/2606.17034
- https://patents.google.com/patent/WO2026086089A1/en
- https://patents.google.com/patent/US20260080217A1/en
- https://patents.google.com/patent/WO2026087278A1/en

## Surviving frontier

A successor may remain interesting only if the **representation itself** creates the advantage. Examples of admissible questions:

1. Can a transformer suffix expose a bounded, compact exact sufficient state for one memory-edit influence that composes without replaying the ordinary nonlinear suffix?
2. Can the model discover an exact causal quotient of the suffix so only a substantially smaller algebra must be updated, without ordinary dependency metadata supplying that quotient?
3. Can a lifecycle transition destroy or transform a stale-state computational capability in a way that remains exact under downstream nonlinear mixing and is cheaper than last-read-site patch + minimal suffix recomputation / exact KV reconstruction?
4. Is there a hardware-local representation whose exact update cost/bandwidth advantage persists when the baseline is allowed to use the identical representation and composition rule?

If the answer reduces to "store an exact transition summary and update it in a tree," E-000097 kills it immediately.

## Major-break gates remain unchanged

No major useful invention is promoted. Before any future mechanism can be interpreted as a major break it still requires, at minimum: real LINK->Pod reader >=0.95 on every held-out template in every interpreted job; >=3 genuine seeds; >=2 backbone families; <=2% old/deleted leakage; >=90% UNKNOWN in declared missing-key scope; exact bypass or <=0.05 nats generic divergence; stale Bank/router/resolved-payload/Hidden/KV attacks; UPDATE/RELINK/REVOKE/SHRED/DELETE/RESTORE/ABA/rollback/TOCTOU; key/reconstruction attacks; independent J-space/J-lens audit only; <=5% steady-state inference overhead; matched memory; and a material fleet-level mutation-to-ready advantage over the strongest guarantee-matched per-session suffix-recompute/KV-repair baseline.
