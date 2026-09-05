# E-000091 — dense descendants can have cheap exact responses, but affine caching owns this example

**Architecture correction and negative novelty result, not a major invention.** Date: 2026-09-05.
Preregistered before numerical execution at `6ab2c50836f4aa0c9fb2858595ecbd34746e67ae`, on E90 parent `57997fc351b65bc74ca795f92e7bb35bbd6fd737`. Main and historical records are unchanged.

## Decision

E89's generic tanh carrier/replay result remains intact. Its extrapolation that only small dependency cones could avoid affected dense work was too broad. Dense descendants can share a cheap, exactly computable finite response to a low-dimensional source revision. Dependency support and response complexity are different quantities.

E91 constructs that restricted escape, then rejects its novelty: ordinary activation-region affine caching reproduces its exactness and speed. The local 25–27x gain over dense replay is approximately 1x against the strongest matched baseline; in CI the ordinary baseline is slightly faster.

## Exact mechanism and authority boundary

Let `h0=c+Bp`, `zl=Wl h(l-1)+bl`, `hl=ReLU(zl)`. With fixed context, basis and model, compile preactivation responses `U0=B`, `Vl=Wl U(l-1)`, `Ul=Dl Vl`, where `Dl=diag(zl>0)`.

For finite `p'=p+delta`, propose `zl'=zl+Vl delta`, `hl'=Dl zl'`. If every predicted activation predicate matches the compiled pattern, induction proves equality at every preactivation and hidden write. This is a finite-domain proof, not a small-delta Jacobian approximation. A boundary can be conservatively rejected even when particular values coincide: the guard is sufficient, not exactly minimal.

At the first failed predicate, keep only the verified prefix and replay that layer and the remaining suffix. This is ordinary fallback. The conventional baseline instead stores `al=zl-Vl p` and evaluates `zl'=al+Vl p'`, with identical maps and checks. This is a mechanism ablation, not a universal argument that any future new representation can be donated to a baseline and declared non-novel.

Trusted compilation and immutable context are essential premises. Tests deliberately show that forged maps or a changed context can pass all gate checks while disagreeing with fresh states. Gate agreement does not authenticate learned maps. No production generation binding or independent J-space audit is implemented here.

## Executed evidence

Five random parameter/intervention seeds; width 256, eight ReLU layers, two source coordinates, dense ternary integer weights. Independent Python arbitrary-precision execution verifies all compiled maps and all 325 rebuilt trajectories, ruling out hidden int64 wraparound. Every preactivation and hidden write is checked; no final-output-only scoring.

| Revision magnitude | Fast-path accepted | Exact with verified-prefix/replay fallback |
|---|---:|---:|
| 1 | 80/80 | 80/80 |
| 16 | 80/80 | 80/80 |
| 256 | 30/80 | 80/80 |
| 4096 | 0/80 | 80/80 |
| Deletion to zero/NEVER | 0/5 | 5/5 |

190 accepted proposals include 9 no-op draws. The 181 nontrivial accepted edits change 933–1,030 of 2,048 hidden-state coordinates and 1,935–2,021 preactivation coordinates. This is not repair of only two output coordinates. All 135 rejected unverified proposals fail every-write equality; exact fallback repairs all 325. NEVER here means no source injection with identical exogenous inputs, not an LLM missing-key test.

Local selected suite: **47 passed in 1.04s**, no skips. Five tests independently execute PyTorch integer stacks. Tests also cover overflowing arithmetic, source dtype/shape, boundaries, sequential recompilation and the two trust-premise counterexamples. Sequential source restoration is not ABA-safe lifecycle validation. This is not the full repository suite or five trained-reader seeds.

Four-layer tanh controls (width 64) fail frozen-Jacobian finite-response equality on every seed, with max errors 4.0967e-6–1.3811e-4; no-op controls are exact. This is not a second language backbone.

## Strong baseline, costs and timing correction

Warmed CPU integer medians, materializing all layers and checking predicates:

| Environment | Tight dense replay / candidate | Ordinary affine / candidate | Compile / tight dense replay |
|---|---:|---:|---:|
| Local, 5 seeds | 25.040–26.621x | 1.022–1.031x | 3.733–3.942x |
| CI, 5 seeds | 18.478–18.980x | 0.970–0.982x | 4.199–4.372x |

The small candidate/ordinary timing difference reverses between environments. No material strongest-baseline advantage survives. These are not end-to-end mutation-to-ready or language-model throughput results.

The initial run timed a dense reference that repeated conservative weight-array overflow-bound scans. Its original JSON is preserved. The final run adds a tighter raw dense kernel, permitted only for tested inputs independently checked against arbitrary precision before timing. That kernel is not a general certifier. Neither the event protocol nor equality bar was relaxed. Source shape/dtype checks were also added before publication; initial and final source hashes differ.

Both alternatives can use 67,600 bytes of cache arrays for one source/context, including 32,768 response-map bytes. Original preactivation anchors and affine intercepts are alternative encodings; hidden states are retained. The harness holds both anchors for comparison. Python overhead, temporary buffers, aggregate source/context caching and lineage metadata are not included: this is **not** the matched complete system-memory gate. Model arrays occupy 4,216,832 bytes. Compilation costs are explicitly reported, not hidden in mutation timing or mislabelled as measured LLM throughput loss.

## Prior-art differentiation

**Sallinger et al., Neural Caching, IoT Workshops 2025, DOI 10.34749/3061-1008.2025.1.** Full PDF pages 3–4 and figures inspected. Cached affine maps and conservative hypersphere containment based on activation boundaries already give exact region reuse. Certification itself is therefore prior art, not a missing novelty component. Source-restricted intermediate maps, endpoint guard checks and prefix fallback differ in implementation, but are not a demonstrated major invention. This is not a replication of the authors' full system or its empirical results.

**Achiral / Marvin Danig, July 30, 2026, Jacobian-Causal Memory Control.** Public author disclosure combines a behavioural-effect J-space abstraction, observed memory interventions, lineage and recovery. It explicitly reports no benchmark or finished implementation. Its J-space is not automatically Anthropic's hidden-state workspace. The author says a provisional was filed July 28; unpublished claims were not inspected. Broad naming overlap does not establish the specific exact neural-tensor repair mechanism.

**Xiao et al., arXiv:2608.30198:** memory-update error propagation and pathway repair are motivation, not novelty. **MemoRepair, arXiv:2605.07242**, and **Sharir and Anandkumar, arXiv:2307.14988**, remain boundaries for generic repair and neural incremental computing.

**Patent/standard screen:** reviewed IBM US20260119893A1 claims 1 and 6 concern an added KV layer, twin/Gaussian-mixture updates and insertion/modification/deletion. No exact all-descendant finite-response requirement was identified in those reviewed claims. W3C PROV-DM already supplies revision, derivation and invalidation terminology. This is a limited technical screen, not exhaustive novelty search, patentability advice or legal clearance.

Primary sources:
https://iotw-proceedings.dsg.tuwien.ac.at/article/749/galley/780/download/
https://achiral.ai/research/jacobian-causal-memory-control
https://arxiv.org/html/2608.30198v1
https://arxiv.org/html/2605.07242v1
https://arxiv.org/html/2307.14988
https://patents.google.com/patent/US20260119893A1/en
https://www.w3.org/TR/prov-dm/

## Architecture and unchanged gates

Keep both small sound dependency domains and cheap exact finite-response representations in the search space. Neither is claimed exhaustive. Ordinary affine caching is now a mandatory baseline for the second family. A new learned representation remains an unverified research target, not a result of this experiment.

All full-system gates remain unmeasured by E91: >=10x complete mutation-to-ready against the strongest baseline, <=5% inference-throughput loss, matched total memory, >=95% fresh/unseen-paraphrase and lifecycle reading, >=90% scoped UNKNOWN, <=0.05 nats generic divergence/exact bypass, >=3 trained seeds and second backbone. No independent J-space, alias fan-out or generation-publication guarantee is promoted.

## Reproduce and provenance

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m so.experiments.e000091_certified_response_domains \
  --seeds 0 1 2 3 4 --output results/e000091-rerun.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest so/tests/test_e000091_certified_response_domains.py -q -ra
```

Dependencies pinned in the workflow: NumPy 2.3.5, pytest 9.0.2, PyTorch 2.10.0 CPU. Without PyTorch, five tests skip; do not call that a 47-test replication. No paid compute purchased.

SHA-256 experiment: `52ca7f44ae3c6c08bf91208144a318114d5ccfa64b078c9293dded0a68205fc8`; tests: `31077b1a51cb11f4a0771f262dd9ef8f61e76dab45cd8b8284dcc386755e6df6`.
Final local JSON: `a0afd8b2fa9ff6f9413f995e172a8bfcf6f1aba934cdfbfb21fca0877302620b`.
Initial local JSON: `fd7c36b64d6557bc083e5f775669980000f261bd492025247dff8eb61eec102e`.

Full original local JSON and logs are preserved in the downloadable research bundle. Complete per-event CI JSON is in verified artifact 9972689019; details in `e000091-ci-evidence.md`. The CI run is a separate same-source CPU execution, not independent laboratory replication.
