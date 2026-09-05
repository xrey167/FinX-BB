# E-000089 — generic nonlinear correction carrier is not the invention

**Status:** decisive negative result; no breakthrough; no novelty claim.

## Motivation

Xiao et al. (arXiv:2608.30198, 2026) show that memory-update pathways can propagate errors into later memory states and that pathway-guided repair sharply reduces residual propagation. That motivates source-lineage repair, but it is not novelty for this programme.

## Question

Can a bounded correction carrier propagate a canonical pod revision through later nonlinear persistent neural writes, remain exactly equal to a full rebuild at every dependent write, and deliver a material systems advantage over the strongest guarantee-matched replay baseline?

## E-000089 result

The screen uses a nonlinear recurrence with a finite pod edit. The strongest generic exact delta carrier retains old preactivations and propagates the exact nonlinear difference. The matched replay baseline retains source-independent terms and replays only the affected recurrence. Both are given one cached vector per affected step in addition to the persistent old state.

Across five numerical seeds, both exact mechanisms reproduce every dependent write of the full rebuild. The exact carrier's worst dependent-write error is <= 6.67e-16; cached replay is exactly 0.0 in the checked floating-point execution. Both perform 40 dependency matvecs and 40 nonlinear evaluations for the 40 affected steps. The cached replay was slightly faster in every recorded seed: replay/carrier wall-time ratio 0.910–0.923. A deliberately weak raw rebuild that repeats irrelevant source-independent work is only about 2.22–2.26x slower than the carrier, far below the programme's >=10x major-utility gate.

A frozen-Jacobian/stale-coefficient carrier appears deceptively accurate at the final state because recurrence contraction erases error later, but it fails the required counterfactual equality at intermediate persistent writes by 1.54–1.86 max-abs across the five seeds. Final-state-only evaluation is therefore invalid for this lane.

A structurally block-local control does permit exact source-local repair: only 16 of 128 channels are descendants of the edited source and the repaired trajectory equals the full rebuild exactly. The nominal dense matmul work ratio is 64x. However, the strongest cone-aware exact replay baseline can exploit the same cone. Sparse dependency itself is useful, but the carrier algebra does not create the advantage.

## Prior-art boundary

This result also closes an attractive but non-novel framing. Sharir & Anandkumar's *Incrementally-Computable Neural Networks: Efficient Inference for Dynamic Inputs* (arXiv:2307.14988) already formulates neural incremental computing through computational dependency graphs and modifies transformer computation to increase reusable unaffected work. 2026 work such as *Subtract or Replay?* already separates algebraically removable addressable influence from replay-required entangled recurrent state. KVEraser and selective KV-recomputation systems further occupy localized cache repair/recomputation territory. Therefore "dependency-aware partial neural recomputation" is not a defensible novelty claim.

## Architecture decision

Stop pursuing generic bounded correction carriers as a breakthrough lane. Keep exact delta propagation only as a baseline/reference implementation. Future candidates must create a **genuinely smaller source-lineage dependency cone in persistent neural state** and must beat a **cone-aware exact replay** baseline under matched memory.

The only still-interesting technical target is narrower:

1. A neural mechanism that causes persistent writes to carry explicit, source-local lineage in a way that remains sparse under useful nonlinear composition.
2. A learned dependency certificate may accelerate mutation only if an independent J-space/J-lens causal audit verifies that it omitted no causal descendant.
3. Affected writes must be exactly equal to a clean full-rebuild counterfactual; stale/Jacobian approximations do not qualify.
4. The certificate/representation itself, not generic graph traversal or replay, must produce the >=10x mutation-to-ready benefit against cone-aware exact replay while preserving reader capability and <=5% inference-throughput loss.
5. Canonical alias fan-out remains an engineering utility primitive, not novelty.

This keeps lanes A/C alive only under a much stronger standard: **learn or architect neural sparsity of source influence, then independently prove the learned cone is complete.** Generic dependency repair, generic incremental computation and delta propagation receive zero novelty credit.

## Evidence

- GitHub Actions run: `33973769943`
- Commit: `bafc29e1a2db3af657e70a321429c0696ebbed39`
- Five seeds, `d=128`, 48 recurrence steps, edit at step 8, 40 affected writes.
- Workflow artifact SHA-256 is recorded by GitHub Actions.
