# E-000101 result — first downstream nonlinear MLP is an ordinary-coordinate exact-transport barrier

Date: 2026-09-06
Status: **DECISIVE DIRECTION CHANGE / scoped transport-family kill**

Preregistered source head: `29ad34e108b9848f4da9acda85b9ac5213f1266a`
GitHub Actions run: `34000560078`
Backbones: `distilgpt2`, `EleutherAI/pythia-70m`
Fresh intervention seeds: `20,21,22`
Contexts: 128 per seed, sequence length 16, payload RMS 2.0.

## Decision

**Kill ordinary-coordinate operator-local exact delta propagation as the source of a major mutation-to-ready advantage in the registered frozen standard-transformer regime.**

E-000100 showed that, across the whole two-block suffix, even a free oracle revealing exactly unchanged dense-projection output rows leaves at most ~1.08% arithmetic headroom. E-000101 asks a sharper question: suppose every exact update before the first downstream nonlinear MLP-down is free. Does the edit still reach that first large dense matrix in a compact exact form that ordinary sparse or fleet-low-rank delta propagation can exploit?

Across all six backbone×seed cells, the answer is no under the preregistered screen. By the first downstream MLP-down, the expanded activation correction is essentially dense in ordinary coordinates, its output correction is essentially dense, oracle Cartesian row/column skipping exposes less than 1% coefficient headroom even in the best tested context, and the 128-session activation-delta matrix requires full row rank 128 under the necessary `1e-6` Frobenius exactness bound.

This is a scoped falsification, not an information-theoretic lower bound. It does not exclude a genuinely new exact fast representation/algorithm for the frozen dense matrix, a nonlinear symbolic quotient that never materializes the ordinary MLP activation, or a retrained lifecycle-native architecture.

## Registered protocol

The candidate receives unusually favorable assumptions:

- all exact LayerNorm/attention/residual update work before the first downstream MLP-down is free;
- a free oracle reveals unchanged expanded MLP input coordinates and unchanged MLP-down output rows;
- detection, indexing, memory movement and all other update costs are ignored;
- across the 128 sessions, the candidate may exploit a compact exact linear correction rank if one exists.

The read site is `max(0,n_blocks-3)`, leaving two complete transformer blocks. We capture the final-token input/output of the first downstream MLP-down, repeat the old forward for exact determinism, and compare one fixed old->new payload edit across 128 independently generated contexts per seed.

The preregistered family kill required all six valid cells to have:

1. median exact changed fraction of MLP-down inputs >= 0.99;
2. median exact changed fraction of MLP-down outputs >= 0.99;
3. maximum oracle Cartesian coefficient skip <= 0.05;
4. minimum rank not excluded by the `1e-6` necessary Frobenius bound >= 0.90*128.

Every cell passes every kill condition and every validity control.

## DistilGPT-2

Architecture at the barrier: hidden width 768, expanded MLP input width 3072, output width 768, MLP-down weight size 2,359,296 coefficients. Two full transformer blocks remain after the read site. Final-logit edit materiality is 100% for every seed.

| Seed | Median exact input changed | Median exact output changed | Max oracle exact coefficient skip | Min exact-compatible rank / 128 |
|---:|---:|---:|---:|---:|
| 20 | 100.0000% | 100.0000% | 0.000000% | 128 |
| 21 | 100.0000% | 100.0000% | 0.032550% | 128 |
| 22 | 100.0000% | 100.0000% | 0.032550% | 128 |

The material (`abs(delta)>1e-6`) input correction is also essentially total: means are 99.9944%, 99.9959%, and 99.9942%. MLP-down outputs are 100% materially changed in every context of all three seeds.

For the fleet rank screen, the 128x3072 activation-delta matrices have necessary Frobenius bounds of only `0.000627069`. At rank 115 the optimal residual is still `14.6825`, `14.6864`, and `13.2218` respectively; only rank 128 reaches zero in the measured matrices. The smallest singular values remain material (`3.5654`, `3.6776`, `3.3151`).

## Pythia-70M

Architecture at the barrier: hidden width 512, expanded MLP input width 2048, output width 512, MLP-down weight size 1,048,576 coefficients. Two full transformer blocks remain after the read site. Final-logit edit materiality is 100% for every seed.

| Seed | Median exact input changed | Median exact output changed | Max oracle exact coefficient skip | Min exact-compatible rank / 128 |
|---:|---:|---:|---:|---:|
| 20 | 99.7559% | 100.0000% | 0.682640% | 128 |
| 21 | 99.8047% | 100.0000% | 0.828648% | 128 |
| 22 | 100.0000% | 100.0000% | 0.244045% | 128 |

The best oracle context in the entire Pythia screen can skip only 0.828648% of the first MLP-down coefficient rectangle even when unchanged input coordinates and output rows are known for free. The corresponding material-threshold ceiling is the same maximum.

For the 128x2048 fleet activation-delta matrices, the necessary Frobenius bound is `0.000512`. At rank 115 the optimal residual remains `1.12578`, `1.21053`, and `3.85787`; only rank 128 reaches zero in the measured matrices. Smallest singular values are `0.170416`, `0.194804`, and `0.571513`, all vastly above the lifecycle exactness scale.

## Validity controls

All six cells pass:

- final-logit material edit rate = 1.0;
- two complete nonlinear suffix blocks remain;
- repeated old forward is byte-identical for captured MLP input, MLP output and logits;
- prefix positions before the edited final token remain byte-identical at the first downstream block;
- captured MLP-down input is the expanded nonlinear activation (`3072>768` for DistilGPT-2; `2048>512` for Pythia), not a residual tensor.

The two model-family jobs completed successfully and archived the registered source/result hashes as Actions artifacts.

## Interpretation

E-000101 localizes the standard-transformer failure more tightly than E-000100. Even granting a hypothetical exact compact update through all operations before the first downstream MLP-down, the canonical Pod edit arrives at the first large post-nonlinearity dense matrix as a session-specific, almost fully dense, full-row-rank ordinary-coordinate correction.

Therefore do not spend major-invention budget on carrying ordinary exact deltas layer-by-layer through the existing frozen transformer and expecting sparsity or a fleet-shared linear basis to remain useful after the first nonlinear MLP. A successor must provide at least one genuinely different mechanism:

- a demonstrable exact structural representation of the frozen MLP matrix/state pair that evaluates the changed dense product materially faster than the strongest exact baseline;
- a nonlinear symbolic/compressed sufficient state that avoids materializing the ordinary expanded activation and is not equivalent to generic incremental computation;
- or a lifecycle-native architecture trained so useful knowledge composition lives in an exact update algebra with materially lower mutation cost while retaining normal capability.

## Prior-art / strongest-baseline boundary

Fresh literature reinforces that generic incremental computation and generic fast dense multiplication cannot supply novelty by themselves:

- Anand, van den Brand & McCarty, NeurIPS 2025, *The Structural Complexity of Matrix-Vector Multiplication* studies preprocessing a fixed matrix for repeated exact matrix-vector queries. General matrices retain worst/average-case barriers; speedups require demonstrable matrix structure such as bounded VC/pseudodimension. A future exact fast-MLP route therefore has to identify and exploit a specific structure, and that structure must be compared against the same optimization applied to ordinary inference.
- Sharir & Anandkumar, *Incrementally-Computable Neural Networks* (2023), already identifies dense connectivity as causing small changes to cascade and uses vector quantization to recover approximate reuse. Generic delta propagation through cached activations is prior art and its demonstrated transformer savings are not exact-state guarantees.
- KV-Direct / *The Residual Stream Is All You Need* (2026) reports bit-identical KV reconstruction from residual-stream checkpoints across tested architectures, preserving residual-checkpoint suffix recomputation as a strong exact baseline.
- Synaptics US Patent 12,561,965, issued 2026-02-24, covers cached neural activations plus difference detection and selective recomputation in a CNN setting. Selective cached-activation recomputation receives no broad novelty credit.
- ULD-Net, ICLR 2026, scales fully polynomial neural networks to ViT/ImageNet scale. Merely replacing nonlinearities with polynomial operators to enable algebraic manipulation would not itself constitute the lifecycle invention.

These are broad-claim exclusions and baselines, not patent-clearance or infringement opinions. The targeted search did not identify a reference that supplies the remaining narrow mechanism: exact lifecycle transport of a mixed standard-transformer session state after one canonical mutable knowledge edit with a material fleet-wide advantage over exact suffix repair.

## Programme consequence

The frozen-standard-transformer search has now eliminated passive freshness, exact static/mutable segregation, cross-context learned receipts, generic associative recomposition, shared exact linear correction spans, ordinary coordinate sparsity, and ordinary operator-local exact delta propagation as standalone breakthrough mechanisms.

The remaining technical frontier is no longer "how do we propagate a delta?" It is:

`Can a real neural architecture expose an exact lifecycle-update algebra whose sufficient state remains compact through useful nonlinear knowledge composition, and whose update cost is materially below guarantee-matched suffix recomputation without reducing to late binding or generic incremental computation?`

No major invention is promoted by E-000101. Every full-system gate remains required: real LINK->Pod reader >=0.95 on every held-out template; >=3 genuine seeds; >=2 backbone families; <=2% old/deleted leakage; >=90% UNKNOWN in declared missing-key scope; exact bypass or <=0.05 nats generic divergence; stale Bank/router/resolved-payload/Hidden/KV attacks; UPDATE/RELINK/REVOKE/SHRED/DELETE/RESTORE/ABA/rollback/TOCTOU; key/reconstruction attacks; independent J-space/J-lens audit only; <=5% steady-state inference overhead; matched memory; and a material fleet-level mutation-to-ready advantage over the strongest guarantee-matched baseline.
