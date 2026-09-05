# CRR-001 / CRR-001-L — completed execution evidence

Status: restricted fixed-output-basis falsification; not an invention or application-gate pass.

## Executions

Initial run **33978377217**, executed source **3210ba5ca8b525f8b351f8a231ad7d83c8d3eda2**: all three jobs completed successfully (exact, DistilGPT2, Pythia-70M). The exact job reported **28 passed in 0.08s**, without skips. Local exact suite: **28 passed in 0.18s**. All exact matrix fields and descriptive spectra match across those two executions. Frozen pretrained models ran in Actions, not locally.

Layerwise extension run **33978751122**, executed source **df41fb1a640ca5b78dfb131ade7b5eadfbc6b8f9**: both model jobs completed successfully. Each reran the entire pinned source-amplitude sweep and allowed independent bases/coefficients for every layer's key and value tensor. Combined suite: **31 passed in 1.26s** in the DistilGPT2 job; **31 passed in 1.24s** in the Pythia job; **31 passed in 3.33s** locally. No skips. This is the original 28 tests plus three extension controls, not the full repository suite.

Every aggregate K/V result field in all 12 model/prompt/direction cells, including spectra, matches between the initial and layerwise runs. This is same-program measurement repeatability; raw-tensor cross-platform byte identity and independent laboratory replication are not claimed.

No continue-on-error is used for scientific execution. Pipefail is enabled. Original preregistration precedes initial execution. The stronger layerwise extension is explicitly registered after seeing the aggregate results and before running the extension.

## Verified archives

All five ZIPs were downloaded and their SHA-256 digests checked against GitHub metadata. Their source and result JSON file digests were also independently checked in the local container.

| Run | Artifact ID | Arm | ZIP SHA-256 |
|---|---:|---|---|
| 33978377217 | 9973004152 | exact | a9eaf17fca0664e0b102fd2ac2c8f95277a822b52346e917f59371ba34e6809c |
| 33978377217 | 9973010450 | DistilGPT2 | 8807cce9b497d7e6c1faddba4d1a06494feadebd90a70c9a0163be157f07664c |
| 33978377217 | 9973027591 | Pythia-70M | 814695965e975ee41460069ddbc0e5f60b13e14009f7c6a2f80f9b27faa3c44e |
| 33978751122 | 9973122047 | independent-layer DistilGPT2 | 5e324f763afe31aced9677c3fb59a75f07222107874495279e7db87d73490a8e |
| 33978751122 | 9973150236 | independent-layer Pythia-70M | e0d1bd9d2fb45ccb1ed54141fdb4cb18bd1245ed21c96355a21510e4f2d94025 |

## Source SHA-256

- so/experiments/crr001_finite_response_rank.py: `2831e9c3401baa5b74f22b5b77e61b3b03951c51c37c5b7439592f691a02de2e`
- so/experiments/crr001_layerwise_extension.py: `e37f5b1ca86a9d771a279be394a746e6e50c612a5a2a9cd7e5ac3154ec9097af`
- so/tests/test_crr001_finite_response_rank.py: `056c8e4c19611654c1b0e8255bf1e1a2ae1f9927a15455f0bb10700c4894c9fc`
- so/tests/test_crr001_layerwise_extension.py: `02e9c4953131630a5d524f9faa30752c3470e1dca15c72b8a7b0948d79711460`

## Result SHA-256

- Initial exact JSON: `7149c71fedd0b8638426c5332c163d442b0d284c79747b9389729b142f869272`
- Initial DistilGPT2 JSON: `fa4878c531601f6e1beb421909543c82fce89dda0067523c8f43b6ad3dac56fa`
- Initial Pythia JSON: `c3d7b074d416f662e74015dcdcab85d59fa5702fb3590c3c1a0b90e2055bb533`
- Layerwise DistilGPT2 JSON: `2ed79caa2235328651eb04bb72f10b9d53863903ee332a093bf73f90f1a8a6b7`
- Layerwise Pythia JSON: `17cb977afb2c55a19b9a0dddd71b623ae278b21f47ea6d679c036f17ffc4249e`

Backbone revisions are fixed in the layerwise entrypoint and both sets of result files:
- distilbert/distilgpt2: `2290a62682d06624634c1f46a6ad5be0f47f38aa`
- EleutherAI/pythia-70m: `a39f36b100fe8a5377810d56c3f4789b9c53ac42`

Pinned libraries: NumPy2.3.5, transformers4.57.6, PyTorch2.10.0+cpu, pytest9.0.2. Workflows request Python3.13 and capture pip freeze. No paid compute or model API was purchased. Models/weights are not copied into the research bundle.

## Boundaries

The downloadable bundle preserves the exact original ZIPs, extracted JSON/error spectra/control records, local exact results, scripts, tests and evidence manifest. Raw K/V snapshots were not archived; the pinned scripts regenerate them. The initial --model entrypoint resolves a current SHA then pins that run, whereas the extension hardcodes the original recorded SHA and is the recommended exact-checkpoint reproducer.

Three direction seeds are not trained-reader seeds. Two frozen architectures do not qualify the proposed semantic memory system on two backbones. No full mutation-to-ready, throughput, complete memory-budget, fresh/paraphrase, UNKNOWN, generic divergence, generation-safety or J-space gate is promoted.

Runs:
https://github.com/xrey167/FinX-BB/actions/runs/33978377217
https://github.com/xrey167/FinX-BB/actions/runs/33978751122
