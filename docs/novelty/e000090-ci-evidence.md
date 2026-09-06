# E-000090 — completed CPU replication evidence

Status: negative-result countermodels replicated; no invention or language-model capability claim.

GitHub Actions run **33975331836** completed successfully on executed commit `5c39e2f68a986fbc6a1e13db95469a3e4968b878`. The five-seed experiment completed, the original archived data digest check passed, and the selected suite reported **71 passed in 2.00s**, without skips. The local execution had reported **71 passed in 1.31s**.

Artifact **9972143115** was downloaded after completion and its ZIP SHA-256 verified against GitHub's reported digest:
`dfb2be999e9647e5d447b30bfa3d51e97fd5197e549345b5253ad58b26bcc080`.

The downloaded `sha256.txt` matches the local experiment and test sources exactly:
- Experiment: `30fd396e07f00926d21b3b153dbc613d5666d5cebd1baaeb6424ce15020b561f`.
- Tests: `8a42ab1a518cd9c9b9191bf378fc1224dc8b3d488057e96a33f791d8ef212dc0`.
- CI result JSON: `0e7a426aba9b28c6b55029c59edd6f25c8b83ce408b482e18873a196b5299a2b`.

All exact-reference equality and falsification assertions pass in both environments. Among per-seed result fields, a single floating-point summary differs across environments: seed 4's denominator-effect maxabs is `0.007212812878360397` locally and `0.007212812878360508` in CI, a difference of `1.1102230246251565e-16`. All other per-seed fields match. No cross-platform byte-identity claim is made. Reference repairs are byte-identical to their full rebuilds within each execution.

CI environment: Python 3.13.15, NumPy 2.3.5, PyTorch 2.10.0+cpu, Ubuntu runner on Linux 6.17.0-1022-azure/glibc 2.39. Local environment: Python 3.13.5, NumPy 2.3.5, PyTorch 2.10.0+cpu, Linux 6.18.35/glibc 2.41. The workflow enables pipefail and does not use continue-on-error for the experiment or tests.

This is a separate execution using the same source in a second CPU environment, not independent laboratory replication, five trained-reader seeds, or a second language backbone. All real utility and capability gates remain unmeasured by E90.

Repository context inspected after the run: PR #9, GEN-001, separately reports generated-token feedback into persistent K/V. That experiment was not authored or re-executed as part of E90. Its recorded boundary is consistent with treating generated internal tokens as endogenous descendants rather than immutable inputs to a deletion counterfactual. E90 uses fixed exogenous inputs and does not claim autoregressive/publication-race coverage.

PR: https://github.com/xrey167/FinX-BB/pull/10
Run: https://github.com/xrey167/FinX-BB/actions/runs/33975331836
Parallel record: https://github.com/xrey167/FinX-BB/pull/9
