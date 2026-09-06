# E-000091 — completed CI replication

Run **33977262876** completed successfully at executed commit `ea294305809681864211f9850aa02533cbb1446b`. The artifact was downloaded and verified, rather than inferring scientific success from a green workflow.

**47 tests passed in 2.11s**, without skips. Local: 47 passed in 1.04s. Every field in all **325 integer event records** agrees across local and CI executions. Both environments report 190 accepted proposals (181 nontrivial and 9 no-ops), 135 rejected/inexact unverified proposals, and all 325 hybrid repairs exactly matching every full-rebuild write. All five tanh controls fail the frozen finite-response proposal and pass their no-op controls.

The one cross-platform smooth numerical summary difference is seed 3: local `3.454111332928944e-05` versus CI `3.454111332923393e-05`. It does not change any equality result or claim. This is a same-source rerun on a separate CPU environment, not independent laboratory replication or a trained second backbone.

| Environment | Tight dense replay / candidate | Ordinary affine / candidate | Compile / tight dense replay |
|---|---:|---:|---:|
| Local, five seeds | 25.040–26.621x | 1.022–1.031x | 3.733–3.942x |
| CI, five seeds | 18.478–18.980x | 0.970–0.982x | 4.199–4.372x |

The ordinary affine baseline is slightly faster in every CI seed. The small local advantage reverses between environments. No statistically robust candidate advantage or >=10x strongest-baseline advantage is established. These are CPU integer microbenchmarks, not total mutation-to-ready or language-model inference-throughput measurements.

Artifact: **9972689019**, name `e000091-certified-response-domains`.
ZIP SHA-256 verified against GitHub metadata:
`b60057e9c18432f36708bf4a8f9b769ca4a0f33d8cab45a78ea6aaf5008f45df`.

The downloaded `sha256.txt` matches the locally tested sources:
- Experiment: `52ca7f44ae3c6c08bf91208144a318114d5ccfa64b078c9293dded0a68205fc8`.
- Tests: `31077b1a51cb11f4a0771f262dd9ef8f61e76dab45cd8b8284dcc386755e6df6`.
- CI JSON: `bb2f06280897da0131b195219dcb447c9d78364599f0a721f5fd86405ad7c6dd`.

CI: Python 3.13.15, NumPy 2.3.5, PyTorch 2.10.0+cpu, Linux 6.17.0-1022-azure/glibc2.39. The workflow uses pipefail and no continue-on-error in the experiment/test step.

Run: https://github.com/xrey167/FinX-BB/actions/runs/33977262876

No main or historical experiment was modified. A documentation-only follow-up may advance the branch without re-executing the unchanged code; the executed SHA above is the evidence anchor.
