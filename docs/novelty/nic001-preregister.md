# NIC-001 — joint nonlinear repair and chart-context validity

Registered BEFORE numerical execution, 2026-09-05. Parent CRR001: 5996a0074831f152e84f57ad1635ec06f747007b. Status: UNRUN. This is a restricted-family falsification and control, not an invention claim.

## Candidate discrimination

CRR001 rejects fixed low-rank output carriers but leaves nonlinear decoders open. Test a different restriction: independent source-local nonlinear finite responses cached in a common original context, and the stronger extension including ALL exact pairwise interaction responses. Give those candidates oracle fresh tensors, no rank limit and no learned approximation error. Does isolated or pairwise exactness imply exact joint lifecycle repair?

Denote source-removal subset S by H(S), with three canonical synthetic sources initially present. Compute anchored finite interactions I(S)=sum_{T subset S}(-1)^(|S|-|T|) H(T). Independent nonlinear source responses reconstruct H(empty)+sum I({i}); pairwise chart composition also includes all I({i,j}). Check the all-three-removal state against full rebuild. Preserve the full third-order remainder, per-layer and per-persistence-stage values. Boolean Möbius/ANOVA decomposition is classical and excluded from novelty; this is not F-IVM or a new graph repair mechanism.

## Exact structural control

Use rational-valued nonlinear maps on three Boolean revision flags, with (1) separable nonlinear output, (2) only pairwise interactions, (3) a pure three-source interaction invisible on all proper subsets, and (4) a compact nonlinear joint decoder with nonzero higher-order interactions. All exact controls use Python Fraction. In the pure triple case all single and pair tests must pass while the joint repair fails. Show that adding source-local response increments in different orders can commute while being wrong. Do not infer impossibility of a compact joint nonlinear decoder or an exponential general lower bound from an interaction table.

## Frozen pretrained memory-update screen

Pinned models, CPU float64, eager attention, deterministic eval, one thread, no remote code or training:
- distilbert/distilgpt2: 2290a62682d06624634c1f46a6ad5be0f47f38aa
- EleutherAI/pythia-70m: a39f36b100fe8a5377810d56c3f4789b9c53ac42

Two fixed initial prompts (same as CRR001):
1. The engineer checked the updated reference before making a decision.
2. A researcher compared the measurements with the previous laboratory notes.

Three source-direction seeds 0,1,2. Sources have separate token positions 3,6,9 after block0. Generate each direction with seed 1000*seed+source_index and scale its RMS to half the corresponding unmodified hidden activation RMS. Assert prompts contain all positions. Source amplitude is 1 if present and 0 if removed. Evaluate all eight presence/removal worlds independently; no retuning if interaction is weak.

For each world, collect actual persistent K/V after prefill and after three fixed EXOGENOUS continuation chunks, carrying that world's past_key_values into each next forward. The chunks are:
- Then the assistant reviewed the record.
- Next it compared the evidence again.
- Finally it prepared a short conclusion.

Chunks will be tokenized with leading spaces, without truncation. The source-injection hook applies to prefill only. Fresh continuation inputs are identical in all worlds; generated endogenous tokens, semantic reading, and external user behavior are not modeled. Continuation isolates the internal persistent K/V pathway and is not a full MemoryBank experiment.

## Controls and metrics

Check every (layer,key/value) tensor at every stage, not final logits. Full rerun of original and all-absent worlds must be byte-identical. All-absent with a zero hook must match no hook. Block0 K/V must be source-independent. Prior cached tensor prefixes must remain byte-identical when continuation appends state; record effects on newly written continuation slots separately, so merely retaining old differences is not counted as propagated NEW writes.

Report per-tensor independent/pairwise residual maxabs and norms; all-order reconstruction arithmetic floor; exact byte counts; per-stage new-write effects; cached single-source chart drift after another source is removed. Repeat-order agreement is not correctness. Near-roundoff mismatch alone does not establish meaningful interaction. Use descriptive floor max(1e-10,100*full-order-reconstruction maxabs) to classify material residuals, NEVER as authorization for approximate reuse. Exactness remains byte identity to full rebuild. A table lookup of every fresh world is the exact positive reference; it is ordinary memorization and not a compressed/timed repair method.

At the synthetic control, increment arithmetic is exact. In floating models, delta addition can itself cause roundoff: do not misattribute cancellation/rounding to higher-order interactions. Full snapshots are temporary for computation; artifacts retain complete per-stage/layer statistics and source hashes, and scripts regenerate snapshots.

## Scope and architecture outcome

A failure rejects context-frozen source-local additive chart composition, and (if measured) the pairwise-only extension. It does NOT reject arbitrary joint nonlinear decoders, adaptive reconditioning, revision-dependent carriers or all exact repair. Source-indexed does not imply independent of other source versions: chart dependencies may need a joint revision context. Generation metadata can prevent stale use but cannot calculate the missing numerical interaction.

Xiao2608.30198 explicitly includes interaction effects and motivates internal-memory propagation; no novelty is claimed for nonadditivity. Search before registration surfaced Path Dependence in Sequential Engram Editing2607.24805, MemoRepair2605.07242, ABFT/checksum neural correction and 2025 error-correcting matrix multiplication. Independent checksums or generic source-response provenance are not promoted as inventions.

All user full-system gates remain NOT EVALUATED. These are three direction seeds and two frozen architectures, not trained-reader/second-backbone qualification. No fresh paraphrase, UNKNOWN, generic divergence, real alias fanout, publication-race safety, independent J-space audit, matched system memory, <=5% throughput or >=10x mutation-to-ready claim.
