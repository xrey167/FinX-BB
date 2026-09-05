# QCR-001 — execution status and restricted conclusions

Date: 2026-09-05. **No major invention or pretrained-model verdict is promoted.**

Preregistered before numerical execution at0ab1004f7d6cc70a90047e6e91a27049a99657bc. Original scientific code/workflow anchor959441c95aaf9a08d41a2e04e8d280d090730664. The protocol tests native BF16/FP32 rounding, not a cast of earlier FP64 experiments.

## Completed locally

Original selected suite: **29 passed in3.35s**, no skips. After adding the independent standard-library checker and its adversarial regressions: **45 passed in4.72s**, no skips. This is29 original tests plus16 checker tests, not45 pretrained-model tests or the full repository suite. The scheduled original workflow selects only the29 original tests.

The scalar controls establish:
-64 nonzero approximation-error cases round exactly when a proven enclosure stays inside one rounding cell.
-Midpoint-crossing controls with differences1/65536,1/67108864 and1/70368744177664 round differently. Error size alone is insufficient.
-Explicit staged BF16 arithmetic produces1 where evaluating the ideal expression and rounding only at the end produces129/128. PyTorch agrees with the staged exact-rational control. Thus ideal/high-precision final casting is not automatically the correct native execution reference.

These are elementary correct-rounding facts, not a new theorem. The independently implemented witness checker accepts a synthetic exact separation and rejects malformed, inconsistent, reversed or tampered cases, including under python -O. Its original draft used assert statements; these were replaced before publication so optimization cannot disable verification. No native LLM separation witness has been obtained locally. The checker validates algebra/interval disjointness, not authentication of an entire model trace.

## Native model arms not counted yet

At the latest observed GitHub read before this status file, all five jobs of run **33983919322** were **queued**, with no runner and no scientific steps. The artifact list was empty. There are therefore zero completed QCR pretrained arms and no native BF16/FP32 success/failure rates to report at this point. A queued workflow is not an execution or a promise of future evidence.

The pinned code implements two backbones x two native precisions x two prompts x three direction seeds, actual persistent K/V continuation, all-tensor and new-slot equality, an all-fresh-state oracle, fixed-basis exact rounding-box separation attempts and same-quantized-parameter/native-versus-lifted arithmetic controls. Those implemented measurements must remain UNMEASURED until executed artifacts are inspected. Rank16 failure is not assumed in advance, and unsuccessful separation search is inconclusive.

A local transformers installation attempt returned no matching package from the configured source. A subsequent pinned config download could not proceed through the available fetch path. No access restriction was bypassed, and neither weights nor local pretrained results are claimed. This is not diagnosed as a particular network fault without evidence.

## Source/data hashes

-Experiment SHA256:05429f6e961b9198ab8a885ad927c50fcbd477508332479044a299c7c1166da3
-Original tests SHA256:760c93068a1316f3e491a2751eae2fcebc3891bc920f3a1704fcbe680a2e3d4f
-Independent checker SHA256:e4961cf4cfb85d6a658ebfa9e5f3572eda3f2185d5ac59eff67863728d972da1
-Checker tests SHA256:a756c517045f69d82f24b2fbfc61614f72a26639d7832e984f66357da74f21f4
-Original local scalar JSON SHA256:1ddd1f7c87da1683c7b1bd5a988a96c6cff792aed563ac66dd30e8ebb7630db4
-Combined local pytest log SHA256:9f2c4c39da4ee41580bd40adb756f2d58e9c6359771be8f1ba1013b621025d2f

Python3.13.5, NumPy2.3.5, SciPy1.17.0, PyTorch2.10.0+cpu. Code upload blob hashes are checked against local files separately. Later documentation/checker commits do not change the executed-source anchor or imply a new workflow run.

## Research decisions at their actual scope

The exactness contract must bind native arithmetic. Earlier FP64 fixed-basis failures cannot simply be promoted to BF16 failures, while a BF16-sized error is not itself exact equality. Rounding-cell correctness is a possible escape but already standard correct-rounding methodology. Kamera already discloses canonical clean KV plus context-conditioned low-rank patches; that broad architecture is not a new differentiator. See qcr001-native-contract-and-prior-art.md for primary sources and precise distinctions.

RSI001's formerly queued run33981893332 completed and its artifact9974226922 was verified this turn:36 tests pass and all scientific fields match the original local result. Its supplement is on the RSI branch; it is not QCR evidence. A separate read-only E88 audit combined three verified historical seed artifacts (candidate accuracy0.975/0.965/0.950 on one template). That clears only its narrow candidate-space prerequisite, not full-vocabulary/paraphrase/lifecycle or J-space gates. No new E88 training was performed.

All complete application gates remain unqualified: >=10x mutation-to-ready, <=5% throughput loss, matched total memory, >=95% fresh/unseen-paraphrase/lifecycle reading, >=90% scoped UNKNOWN, <=0.05 nats generic divergence/exact bypass, independent J-space, generation/publication safety and trained second-backbone validation.
