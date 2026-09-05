# BHC-001 execution status at report publication

Observed 2026-09-05. This is a record of completed work and observed status, not a promise of later execution.

**Completed locally:** all four exact scalar witnesses, all20 native nonlinear parameter/format cells, even-bias controls,512-step common-reset exact repair and unsafe partial-state controls. **52 selected tests passed in8.04s**, no skips; not the full repository suite. Independent Fraction checker under python -O validated all four scalar fixed-point instances.

**CI not counted:** run33986978675, job101362272480, scientific sourceb92facf3c640d4e46b95df9d3f809414445eacc4 remained **queued** at the latest observed read, with no runner or scientific steps. There is no claimed completed CI replication or verified CI result artifact. A scheduled workflow is not evidence that its tests ran.

The three uploaded experiment/test/checker Git blob hashes exactly match the locally executed source bytes. Original result JSON and test/environment records are archived at results/bhc001. Their SHA-256 values are recorded in the report and results README.

The downloadable bundle additionally exports every registered case's old and NEVER trajectories, common-reset old/fresh/repaired/unsafe trajectories, weights and scalar parameters as native tensor bytes with explicit shape/dtype metadata. Export reruns the unchanged functions and checks the recorded gap and repair transition count before saving. This is post-result reproducibility packaging, not a new assay, trained seed, external replication or pretrained experiment.

The bundle's standard-library verifier checks hashes, gzip equivalence, exact scalar arithmetic and archived byte-equality controls, including complete join35 and unsafe residual48. It does not re-execute the nonlinear operator or authenticate an untrusted model producer. Those scientific computations are separately reproducible with the pinned experiment/tests. All application utility and lifecycle gates remain unmeasured by BHC001.

Run: https://github.com/xrey167/FinX-BB/actions/runs/33986978675
