# RBC-001 — consumption-boundary completeness

Status at registration: UNRUN. This is a source-level correctness/identifiability screen, not a major-invention candidate or a learned-reader attack result.

Parent: `947206b72a75dc3616ca44dc5434a0a293f4bca7` on `research/e000051-clean-bystanders`.
Frozen implementation: `so/cavi.py`, Git blob `c0cc8326660f480034547836c9f47f88dd16ccbb`, SHA-256 `df9ca7ad21d7b783a56f2148009385a2ec42a6ac0f6ca9dc54102fcb728632a6`.

## Why this screen

E-000086R reduces current explicit-generation coherence to ordinary dependency validation. Before searching for a neural mechanism beyond that baseline, test whether the current consumption and attestation contracts are actually sufficient. E-000071's in-forward mutation is registered BEFORE the guard pre-hook; this does not cover mutation AFTER the live mask was computed but BEFORE the adapter consumes it.

## Preregistered hypotheses

### A. Same-thread mutation after validation

Run the unmodified `CAVIAuthority` and `NeuralConsumptionGuard` with actual PyTorch module hooks and a one-row numerical read fixture. The fixture is deliberately not GPT-2 and carries no language capability claim. Register a permitted authority lifecycle callback (1) before the guard, (2) after its pre-hook, and (3) in the block body. Compare the captured live mask with authoritative witness validity at actual consumption.

Mutation cases: alias relink, pod update, alias revoke, pod shred, alias revoke/restore, same-ID pod delete/recreate, plus unrelated-pod update control. Hypothesis: the existing RLock serializes other threads but permits same-thread reentrant mutations, so cases 2/3 can consume a stale row. Before-check mutations must be rejected; unrelated mutations must preserve reuse. A barrier-controlled second-thread test must confirm the existing lock correctly delays the competing mutation until after consumption.

A late check performed within the numerical consumption hook is a conventional comparator for this exact schedule, not a proof against arbitrary callbacks inside a larger operator. Do not substitute a non-reentrant Lock and silently accept deadlock as rejection.

### B. Witness/payload binding

For a cache interface whose serialized tensor and claimed witness are separately replaceable, pair old numerical state with a current legitimate witness. Hypothesis: `validate_witness` accepts the witness independently of the tensor bytes. This is a failure only of an untrusted-envelope interpretation; it is not an exploit claim against a trusted in-process closure or a network service.

Compare conventional producer-authenticated envelopes that bind tensor bytes, shape, dtype and witness. Retain the countercontrol where a trusted producer signs an incorrect lineage: authentication must not be described as proving source-lineage completeness.

### C. Activation-only generation attestation

Compare an old-generation artifact and a fresh-generation artifact after a same-value Pod update. Hold the numerical computation and input constant, and provide the auditor no authenticated production history or generation channel. The old witness must be invalid and the fresh witness valid, while hidden state, continuation values and Jacobian observations may be identical. This establishes an indistinguishability boundary, not a new J-lens implementation and not a claim about every possible provenance-aware audit.

State the all-only historical-lineage lower bound: if all k-subsets of N source-generation identities can produce the same observed numerical state, a separate exact certificate distinguishing all singleton invalidation queries needs at least ceil(log2(binomial(N,k))) bits. This is an elementary counting argument, not claimed mathematical novelty. It does not apply unchanged to a semantic-equivalence reuse contract.

## Interpretation rules

A failed universal synchronization claim can be refuted by a source-level counterexample independently of language accuracy. Nevertheless, NONE of these fixture results may count as a GPT-2 stale-state attack pass/fail, deleted-object leakage measurement, or achievement of the user's trained-reader gates. Any learned integration must remeasure all templates 8..11 >=0.95 in that exact checkpoint/job first. Seeds in this screen randomize numeric fixtures only; they are NOT three trained-reader seeds or additional backbones.

Do not weaken E-000083/084/085 thresholds or modify their recorded results. Do not claim new cryptography, locking, effect-boundary authorization, lineage tags, cache invalidation, generations, dependency graphs or J-space. No >=10x performance, <=5% overhead, memory parity, UNKNOWN, generic divergence or reconstruction-security claim follows from this screen.

## Evidence

Preserve source hashes, full result JSON, environment, tests and failed hypotheses. Commit the report only after execution. A later CI run is separate execution, not independent laboratory replication. No production fix is preregistered: first localize the contract violation and distinguish the appropriate known correctness baseline from an invention.
