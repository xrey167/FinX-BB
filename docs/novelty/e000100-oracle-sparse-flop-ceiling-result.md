# E-000100 result — oracle sparse-coordinate exact repair has no material dense-projection FLOP headroom

Date: 2026-09-06
Status: **DECISIVE DIRECTION CHANGE / scoped systems-family kill**

Source / CI head: `72eefb47e4ed468aa10e184840a29ef0d8345beb`
GitHub Actions run: `33998950072`
Backbones: `distilgpt2`, `EleutherAI/pythia-70m`
Fresh intervention seeds: `10,11,12`
Contexts: 64 per seed, sequence length 16, controlled payload RMS 2.0.

## Decision

**Kill ordinary output-coordinate sparse exact repair as a major systems-advantage seam for the registered standard-transformer regime.**

E-000099 deliberately used a severe coordinate-density rule and therefore did not kill the family on Pythia-70M: a small fraction of QKV output coordinates remained exactly unchanged. E-000100 registered a fresh independent screen before new numerical execution and asks the systems question directly.

The candidate is given an unrealistically strong oracle advantage: for every session and every suffix dense projection it receives, at zero cost, the exact set of output rows whose values change after the Pod edit. It may evaluate only those rows. Detection, indexing, memory traffic, layer norms, nonlinearities, softmax, residual operations and all other repair costs are ignored. The reported `oracle_skip_fraction` is therefore an **upper bound** on the dense-projection arithmetic this ordinary row-selective repair class could save.

The preregistered family kill requires, in all six backbone×seed cells, that the maximum oracle exact skip fraction over 64 contexts is <=5% and the corresponding `abs(delta)>1e-6` skip maximum is <=5%, with all controls passing.

All six cells pass the kill rule.

## Registered result

| Backbone | Seed | Exact skip mean | Exact skip maximum | >1e-6 skip maximum | Material final-logit edit rate | Decision |
|---|---:|---:|---:|---:|---:|---|
| DistilGPT-2 | 10 | 0.000000% | 0.000000% | 0.005425% | 100% | NO_MATERIAL_SPARSE_FLOP_HEADROOM |
| DistilGPT-2 | 11 | 0.000000% | 0.000000% | 0.000000% | 100% | NO_MATERIAL_SPARSE_FLOP_HEADROOM |
| DistilGPT-2 | 12 | 0.000000% | 0.000000% | 0.005425% | 100% | NO_MATERIAL_SPARSE_FLOP_HEADROOM |
| Pythia-70M | 10 | 0.513204% | **1.082357%** | **1.082357%** | 100% | NO_MATERIAL_SPARSE_FLOP_HEADROOM |
| Pythia-70M | 11 | 0.490443% | 0.805664% | 0.805664% | 100% | NO_MATERIAL_SPARSE_FLOP_HEADROOM |
| Pythia-70M | 12 | 0.447845% | 0.773112% | 0.773112% | 100% | NO_MATERIAL_SPARSE_FLOP_HEADROOM |

The best tested session for the sparse candidate is Pythia-70M seed 10, and even with perfect free changed-row knowledge it can skip only **1.082357%** of the registered suffix dense-projection coefficient work. DistilGPT-2 has zero exact row-skipping headroom in every session in all three fresh seeds.

All registered controls pass in every cell:

- final-logit material edit rate = 1.0;
- two complete nonlinear transformer blocks remain after the read site;
- repeated old forward reproduces every captured projection output and final logits exactly;
- first downstream block residual positions before the edited final token remain exactly unchanged, confirming causal locality of the hook.

The registered dense-projection weight totals are 14,155,776 coefficients for the two DistilGPT-2 suffix blocks and 6,291,456 for the two Pythia-70M suffix blocks. The measured operators include attention QKV, attention output, MLP up and MLP down projections in every suffix block.

## Why E-000099 did not justify this conclusion by itself

E-000099 found nearly total downstream coordinate influence but its preregistered density criterion required >=99% fifth-percentile exact change in every QKV/MLP/residual family. Pythia QKV p05 values were roughly 96–98%, so that exact screen correctly returned `DO_NOT_KILL_SPARSE_COORDINATE_REPAIR`.

E-000100 does not reinterpret E-000099 post hoc. It uses fresh seeds, a fresh context RNG, different old/new payload directions and a separately preregistered work-based decision rule. It shows that the residual coordinate sparsity visible in Pythia does not translate into a material dense-projection arithmetic advantage.

## Scope of the kill

The result closes a specific successor route:

> Keep the ordinary dense pretrained transformer operators, discover which ordinary output coordinates changed after a Pod mutation, and obtain the major mutation-to-ready advantage by evaluating only those rows.

If a perfect free oracle cannot expose >~1.1% arithmetic headroom in the best tested session, a deployable detector, router or dependency mechanism cannot turn this ordinary-coordinate sparsity into the programme's material systems advantage without another new mechanism.

This result does **not** prove that suffix recomputation is information-theoretically necessary. It does not kill:

- a compact exact nonlinear coordinate system in which the affected computation has a smaller representation;
- an algebraic dense-state transform whose evaluation cost is substantially below ordinary row dot-products;
- a retrained architecture with exact lifecycle-structured sparsity and retained capability;
- a session-specific exact causal quotient / sufficient statistic that is not ordinary channel or neuron sparsity.

Those are now the relevant remaining invention classes.

## Current prior-art boundary

Fresh 2026 literature reinforces the baseline rather than supplying the surviving mechanism:

- KVEraser (`arXiv:2606.17034`) explicitly notes that exact localized context erasure requires suffix recomputation and uses learned approximate KV steering to avoid it.
- KV-Direct / *The Residual Stream Is All You Need* (`arXiv:2603.19664`) shows K/V are deterministic residual-stream projections and reports bit-identical reconstruction in its evaluated architectures, strengthening exact residual-checkpoint recomputation as a baseline.
- KVBoost (`arXiv:2608.21362`) uses deviation-guided selective recomputation for cache reuse but optimizes task accuracy rather than exact fresh-state lifecycle equality.
- Intel `US20260080217A1` covers gauge-transformed / rank-r KV representations and claims priority to a provisional titled “COMPOSABLE EXACT KEY-VALUE CACHE COMPRESSION”.
- Huawei `WO2026086089A1` covers KV-cache segment recomputation.
- IBM `WO2026087278A1` covers direct insertion/modification/deletion through a KV-cache network layer.

These references are broad-claim exclusions and baselines, not an assertion that they anticipate the narrow surviving lifecycle problem or a patent-clearance opinion.

## Programme consequence

Do not spend major-invention budget on ordinary output-row/channel/neuron selective recomputation in the existing DistilGPT-2/Pythia-style suffix as the source of the speed advantage. The next mechanism must change the **representation or algebra of exact affected work**, not merely its index set.

The strongest remaining technical seam is therefore:

`actual session-specific mixed neural computation + Pod mutation -> compact exact nonlinear lifecycle transform / causal quotient -> fresh-current state`

where the quotient or transform must be cheaper than exact read-site patch + minimal suffix recomputation / residual-to-KV reconstruction and must not reduce to generic dependency propagation, row sparsity, late binding, reusable correction receipts, shared linear spans or generic associative recomposition.

No major invention is promoted by E-000100. All full-system gates remain required: real LINK->Pod reader >=0.95 on every held-out template; >=3 genuine seeds; >=2 backbone families; <=2% old/deleted leakage; >=90% UNKNOWN in declared missing-key scope; exact bypass or <=0.05 nats generic divergence; stale Bank/router/resolved-payload/Hidden/KV attacks; UPDATE/RELINK/REVOKE/SHRED/DELETE/RESTORE/ABA/rollback/TOCTOU; key/reconstruction attacks; independent J-space/J-lens audit only; <=5% steady-state inference overhead; matched memory; and a material fleet-level mutation-to-ready advantage over the strongest guarantee-matched baseline.
