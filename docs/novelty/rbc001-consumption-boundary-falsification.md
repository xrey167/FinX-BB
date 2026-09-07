# RBC-001 — consumption-boundary and attestation falsification

Date: 2026-09-05. Status: source-level countermodels executed locally; no invention claim and no trained-reader attack result. Core implementation unchanged.

Parent: `research/e000051-clean-bystanders` at `947206b72a75dc3616ca44dc5434a0a293f4bca7`. Work branch: `research/rbc001-consumption-boundary-completeness`. Preregistered in commit `5cfce09614480d04b70a6c13f6e04785cf087824`; experiment/test/workflow source frozen at `b4d5f642ec6a6584d9095c1a6f4759bd91d67db5`.

## Decision

Do not promote the current pre-hook guard as a universal atomic neural-consumption boundary. Under a contract permitting a lifecycle callback on the same thread after its live-mask check, it consumes stale state. Also, do not treat a current witness as proof of tensor provenance, or a history-free numerical audit as a complete certificate of historical generation identity. These are correctness and identifiability boundaries, not new mechanisms.

E-000086R's existing coherence-to-dependency reduction remains the novelty kill. Fixing RBC-001 using standard checks, transaction rules or authenticated envelopes does not reopen that candidate.

## A. The untested in-forward interval

The frozen `NeuralConsumptionGuard` takes the authority RLock in a block pre-hook, calculates a row-validity mask, retains the lock through the adapter read, and releases it afterwards. Python RLock excludes other threads but deliberately permits acquisition again by its owning thread. All authority lifecycle methods use that same RLock.

E-000071 registers its mutation BEFORE the guard pre-hook. RBC-001 additionally places the mutation AFTER that pre-hook and in the block body. A permitted same-thread lifecycle call can commit in either interval. At consumption, the old witness is invalid but the precomputed mask remains true.

This is conditional on permitting reentrant lifecycle callbacks. A runtime contract that explicitly forbids or defers them would exclude these schedules; the present guard itself does not enforce that restriction. No claim is made that other-thread exclusion is broken or that an external service was exploited.

The test uses the unmodified authority/guard source, actual PyTorch module hooks, and a one-row numerical fixture. Only transformer-block discovery is adapted to the fixture. This is not GPT-2, a J-lens experiment, or a language-capability benchmark.

The grid contains six relevant transitions: alias relink, Pod update, alias revoke, Pod shred, alias revoke/restore, and same-ID Pod delete/recreate; plus unrelated-Pod update as a reuse control. It crosses three hook placements, two guard modes and three numerical fixture seeds.

| Observation | Local result |
|---|---:|
| Total schedule cases | 126 |
| Original guard: stale consumption in targeted after-check cases | 36 |
| Conventional late-check comparator: stale consumption | 0 |
| Relevant before-check cases rejected by original guard | 18/18 |
| Unrelated-update cases preserved by original guard | 9/9 |
| Barrier-controlled other-thread serialization controls | 3/3 |
| All selected unit tests | 138 passed in 2.12 s |

These counts are deterministic countermodel coverage, not leakage estimates or trained-reader seeds. The late-check comparator performs a final check inside this fixture's consumption hook with no intervening callback. It proves only the tested schedule. A larger consumer still needs an explicit atomicity/commit contract. Replacing RLock with a plain lock and causing deadlock is not a valid repair. No production fix was implemented here.

## B. Witness currentness is not artifact binding

Under an explicitly untrusted, separately replaceable cache-envelope interface, an old tensor paired with a current legitimate ResolveWitness passes `validate_witness`: that function sees no tensor bytes. This is not a failure claim against a trusted in-process closure that keeps the original witness bound to the original tensor.

A conventional producer-authenticated test envelope binding bytes, shape, dtype and witness rejects tensor/witness substitutions and authentic stale generations. The countercontrol deliberately signs an incorrect lineage using the trusted issuer: authentication still accepts it. Therefore authenticated binding proves a producer claim was preserved, not that the claim's causal source set was complete or correct. No cryptographic novelty or reconstruction-security result is claimed.

## C. History-free activation audit cannot certify arbitrary generation history

In each numerical fixture, update a Pod to a new generation with identical numerical content. An old artifact and a fresh recomputation then have byte-identical hidden state, K, V, logits and downstream Jacobian. The old witness is invalid and the fresh witness valid.

Let O be everything supplied to a history-free numerical auditor. If O(H_old) = O(H_fresh) but the required historical-validity labels differ, no decision function of O alone can always label both correctly. Randomization cannot supply the missing information. This is an elementary indistinguishability argument, not a new mathematical result or an actual J-lens run.

It kills only unconditional historical-generation certification from those observations alone. It does not kill provenance-aware intervention audits, semantic-equivalence reuse, or J-space's role in independently checking numerical causal effects. A same-value transition need not produce a numerical effect. Such a case must not be falsely counted as positive numerical attestation of the generation transition.

Related counting control: if all k-subsets of N source-generation identities can produce the same observable tensor, a separate exact certificate answering every singleton historical invalidation query needs at least ceil(log2(binomial(N,k))) bits. The N=8, k=3 fixture has 56 identical-output histories requiring 56 distinct membership vectors, hence at least 6 bits. N=1,000,000, k=8 gives 145 bits. This bound is conditional on arbitrary historical source sets and identical observations; it is not a general memory lower bound for every possible architecture or semantic contract.

## D. Exact-job capability ledger: E-000085 is not an all-pass

Downloaded all four completed artifacts from run `33979783563`, executed SHA `7f05f16fab8e8e9eee09e2b49f76b49931d95470`, and verified each ZIP against GitHub's reported SHA-256 digest.

| Exact job | T8 | T9 | T10 | T11 | Interpretation |
|---|---:|---:|---:|---:|---|
| hidden, seed 1 | 1.000 | .960 | 1.000 | 1.000 | Strict capability and structural screen pass |
| hidden, seed 2 | .985 | .960 | .995 | .995 | Strict capability and structural screen pass |
| router/payload, seed 1 | .990 | .975 | .990 | .990 | Strict capability and structural screen pass |
| router/payload, seed 2 | 1.000 | .945 | .995 | 1.000 | Capability fails; attack interpretation excluded |

The last job's structural flags are true, but .945 is below .95. No rounding, averaging, substitution of seed 2's hidden-job score, or transfer from another run is allowed. The numerical source-level counterexamples above are not promoted as learned-reader attack failures either.

All three interpretable E85 jobs report zero max-absolute logit difference between guarded rejected replay and bypass in their tested cases. This does not certify other intervals, all stale-state types, all prompts, or the full utility/security gates.

Artifact hashes:

- `9974536896` hidden s1: `1b15a6f14b50ace8e7443d9ca2506bdff57d7745866859c2a44d9781398699b2`.
- `9974570263` hidden s2: `0d7bac54da970dcb40c9c18941cf312905b4a81e7d3eb315ade7d76b6bd7d8ab`.
- `9974110380` router s1: `c3d8a9b5fbb65f98a7b50d652114f44e0c45f736be96364babd9a42ecfb02767`.
- `9974006311` router s2: `eb91c42d996ea06eba6640ae3d2644636dae2da108ff74cbe613ef5afb15ffe8`.

The existing E83 seed-0 promotion is independently confirmed by artifact `9973770429` from run `33979228980`: consistency .2, alternate supervision .5, BOS, 3000 steps, 100 groups; held-out scores .99/.97/1.00/1.00. ZIP digest `bf40565d54122d46dbbf1f615d98efafa9b0a553bb265215753b3ddba2f7af8d`. This does not substitute for seed-0 capability measurement in E84/E85 integration jobs.

E86R CI artifact `9974221001` from run `33981787028` is now downloaded and digest-verified: `d9002d14dd1a3fea0dfd72d374fb2f8f151a15b8cfe21a90071dcd97680078f6`. It records 14 traces, 34 prefixes, five artifact labels, identical selective-validation decisions, and 8 tests passed in 0.87 s. Those labels do not mean five actual neural artifact implementations were benchmarked. This is a second execution environment, not independent laboratory replication.

## Reproducibility and remaining gates

Frozen `so/cavi.py`: Git blob `c0cc8326660f480034547836c9f47f88dd16ccbb`; SHA-256 `df9ca7ad21d7b783a56f2148009385a2ec42a6ac0f6ca9dc54102fcb728632a6`.
Experiment SHA-256: `18faccba08fb4b7061a2bd8e51bbebcb8b18d5b0f77f83c17edd4d281baa4f34`.
Tests SHA-256: `9542e4325c32e5a7c33156ea15c9158d9ff678890a67bfc5f63060351ea86145`.
Full compressed local JSON: `results/rbc001/rbc001-results.json.gz`; SHA-256 `3ec123c72f2a7025192ea50d6c6be16d14aab20b50b4ac5ec08b9ca12f0c489c`.

Local environment: Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, Linux 6.18.35 x86_64/glibc 2.41.

```bash
python -m research_screens.rbc001.experiment --results-dir so/results/rbc001
python -m pytest -q research_screens/rbc001/test_experiment.py
```

RBC-001 CI run `33984317369` was triggered on `b4d5f642ec6a6584d9095c1a6f4759bd91d67db5`; no completed CI artifact had been inspected when this report was written. Its result is NOT counted here.

No >=10x mutation gain, <=5% inference overhead, matched memory, <=2% leakage, >=90% UNKNOWN, >=95% lifecycle propagation, second backbone, or complete J-space attestation has been established by RBC-001. No shared-checkpoint all-state battery or production repair was implemented in this screen.

The next research admission requirement is an explicit separation of historical lineage, artifact binding, synchronization semantics and independent causal-effect audit. Existing version checks, locks, authenticated envelopes and generic dependency propagation remain baseline engineering. They must not be renamed as the major invention.

## Primary boundary references

Python RLock semantics: https://docs.python.org/3/library/threading.html#rlock-objects
Anthropic J-space description: https://www.anthropic.com/research/global-workspace
Repository source: `so/cavi.py` and `so/experiments/e000071_cavi_read_hook_race.py` at the frozen parent commit above. The identifiability argument and numerical results in this report are our explicit countermodels, not claims that either external reference proves this project's lifecycle guarantees.
