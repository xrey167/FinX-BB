# Qualified-reader provenance falsification

Date: 2026-09-05. Classification: **decisive architecture correction, not a breakthrough**.

This record supersedes the *unconfirmed* status of the three provenance hypotheses in
`CAVI_CONTINUATION_2026-09-05.md`. It does not supersede or erase any negative reader,
training, optimization, locality, UNKNOWN, or lifecycle result.

## Evidence and fixed capability prerequisite

Training continuation run: `33963051896`, source
`59afe5aead6b9e8b961bc9fdfdf1d10114abbbbc`.
Provenance audit run: `33963491759`, source
`6f902bd7e0bef05e1e32dbe2c67bf44e7b4eaf85`.
Independent archive-digest collector: `33964155807`, successful; retained JSON and
member hashes under `research/evidence/cavi-confirmed-33964155807-1/`.

The GPT-2 candidates continued immutable E79 parents for a fixed additional 1500
fresh-world training steps at learning rate .0002. The parent marker centre was
preserved. This uses a new AdamW optimizer, not an exact optimizer-state resume.
No held-out template was introduced into training or selected out of evaluation.
The audit reloads the checkpoint into the original full-sequence LM-head execution.

Each candidate first had to pass **every** cell of three fixed worlds times four
held-out templates, 200 real aliases per cell. Candidate-set accuracy remains the
unchanged >=.95 gate; unrestricted full-vocabulary accuracy is separately recorded.

| Seed | Minimum candidate accuracy | Minimum full-vocabulary accuracy | Every one of 12 cells >=.95 |
|---|---:|---:|---|
| 1 | .965 | .965 | yes |
| 2 | .955 | .955 | yes |

These are empirical rates on a finite, correlated synthetic evaluation, not a
confidence-bound guarantee for arbitrary natural language. The original E79 parents
had failed the expanded gate at .945/.935/.915 across seeds 0/1/2 and their attacks
were skipped. Those failures remain in the ledger. Seed zero's first continuation
failed operationally while waiting for its still-running parent; the separate
same-budget recovery must not be silently counted as completed or successful.

Checkpoints:
- seed 1: `d0c78cf63b1232749500f45e57616d3584e48ef1341a1ddf9346a7f9c51c6faf`
- seed 2: `4a8d3a8480f37083aaa89a2606d702e7e71b898538348bff6352b6f2b5a2c5b5`

Audit result files:
- `33963491759/9968855339/1-result.json` (seed 1)
- `33963491759/9968827607/1-result.json` (seed 2)

The corresponding original ZIP SHA-256 digests, verified against GitHub metadata:
- seed 1: `382a667a4b04768dc90701b8fa76c651f96250a4f35719c8e46123b1acf334a3`
- seed 2: `2b759e0779482e8218f91118d4ae67c1525828c9f47f3a5b4beb1499efc0d4fc`

## Threat model

The caller may recombine serialized Bank, row-manifest, resolved-value and witness
fields. The independent live authority is not rolled back. Lifecycle changes are
legitimate, recorded operations. The attack does not remove the authority check.
The Bank attack uses the actual forward-atomic consumption guard. The resolved-value
attacks exercise the existing E74 replay construction with its witness check intact.
This is not a claim about a production consumer that already cryptographically binds
trusted derivation contents to complete lineage; such a consumer is not implemented
by the tested interface. Nor is this a hostile-process isolation or arbitrary-code
execution claim.

## Counterexample A: current manifest does not authenticate older consumed contents

All canonical pods receive a legitimate UPDATE while retaining their canonical IDs.
The attack supplies the old Bank and the new, genuinely current row manifest.

| Measurement | Seed 1 | Seed 2 |
|---|---:|---:|
| Fresh before UPDATE correctness | .965 | .955 |
| Fresh after UPDATE correctness | .975 | .960 |
| Old-answer rate, old Bank + original manifest | .005 | .000 |
| Old-answer rate, old Bank + current manifest | .965 | .955 |
| New-answer rate under splice | .000 | .000 |
| Spliced vs original full-logit max-abs difference | 0.0 | 0.0 |
| Spliced vs current full-logit max-abs difference | 55.914058685302734 | 38.289337158203125 |

Both independent reader controls pass. The stale-generation rejection works on the
ordinary old manifest but does not survive recombination with unrelated fresh
metadata. The full old neural result is restored exactly, not merely a stale flag.

## Counterexample B: a fresh witness can authorize an unrelated cached resolved value

A legitimate alias RELINK changes the correct answer. The original witness rejects
replay at every read layer. Pairing the same old resolved values with a new genuinely
valid alias witness authorizes both read layers and restores the original logits.

| Measurement | Seed 1 | Seed 2 |
|---|---:|---:|
| Correct old answer ID | 4 | 32 |
| Correct new answer ID; actual fresh answer | 20 | 73 |
| Replayed answer with new witness + old values | 4 | 32 |
| Injected layers with original stale witness | 0 | 0 |
| Injected layers with spliced current witness | 2 | 2 |
| Replay vs original full-logit max-abs difference | 0.0 | 0.0 |

Both attack-specific before/after controls pass. A fresh permission is not evidence
that a caller-supplied cached vector was derived under that permission.

## Counterexample C: an intended-pod witness omits real dense-mixture dependencies

The cached value is a dense neural mixture. For each read layer its direct
coefficient on row j is `p_deref[j] + p_deref[null] * p_first[j]`. The witness names
the intended alias and canonical pod, but another actual contributor may be absent
from that witness. The audit selects the largest pre-mutation coefficient of another
eligible canonical pod and SHREDs it. The queried witness remains current, and fresh
row masking rejects the deleted pod, yet replay still injects its old contribution.

A paired intervention removes precisely that deleted row's linear value contribution
from the cached mixture. All other values, coefficients, prompt and authority state
are held fixed.

| Measurement | Seed 1 | Seed 2 |
|---|---:|---:|
| Deleted row coefficient summed across reads | .011798259802162647 | .001118337269872427 |
| Deleted-value removal: full-logit max-abs change | .2251434326171875 | .015064239501953125 |
| Replay vs original full-logit max-abs difference | 0.0 | 0.0 |
| Queried witness remains valid | yes | yes |
| Queried answer correct before and fresh after SHRED | yes | yes |

This proves the tested cached mixture retains a **causal dependence** on a deleted
other pod. It is not, by itself, an answer-level deleted-object leakage rate, an
independent J-space audit, or a claim that the deleted object became the top-1 answer.

## Architecture decision

**Forward-atomic freshness is necessary but not sufficient for the target.**
The candidate consumption contract must require all three:

1. One coherent authority generation throughout memory consumption in the forward.
2. Trusted binding of the exact consumed contents and derivation context to lineage.
3. Complete actual dependency closure, rather than only the intended/argmax pod.

A MAC over cached bytes and an incomplete single witness addresses the splice but
still fails the dependency-omission problem. A public `sign(arbitrary_payload,
fresh_metadata)` entry point would simply recreate the splice behind a signing API.
The trusted producer must establish provenance while performing the export or
computation, not accept the caller's asserted association.

A conservative reference may invalidate a derived state whenever any genuinely
consumed source changes. That can greatly enlarge invalidation sets. A more local
candidate would need causally bounded reads; merely retaining top-k value rows does
not prove locality when routing, normalization, key/marker channels or earlier
activations depended on other rows. These are design requirements inferred from the
counterexamples, **not implemented or validated countermeasures**. Their inference,
cache-hit, bystander and lifecycle costs still need measurement.

## Separate performance pilot: useful but not a joint pass

Run `33963938909` fitted an isolated write-only scope head on the ORIGINAL E79
checkpoints, freezing and hashing the reader weights. Forced-ON and forced-OFF
preflights checked exact reader/base equality before training. Scope training used
4096 fixed examples, templates 0..7 only, 500 head-only optimizer steps and logit
threshold zero; no held-out threshold selection.

On the twelve preregistered generic evaluation prompts, active-memory KL to the
clean base model and full-logit max-abs difference were **exactly 0.0 for all three
seeds**, compared with original generic KL 3.6472/4.0374/5.2269 nats. Observed
four-template fresh and lifecycle answer rates were preserved. The expanded reader
minima stayed .945/.935/.915 and the stale UNKNOWN failures remained; all three joint
screens therefore still fail. This is a small fixed-corpus result, not universal
semantic scope or a measured latency guarantee. It is NOT a joint result on the
stronger continuation checkpoints used in the provenance attacks.

The earlier E72 negative concerned a different transplanted scope-before-routing
execution path. It must remain negative, but it does not justify ruling out every
isolated post-hoc write-only scope design. This pilot establishes the narrower
mechanism distinction, not novelty for scope gating.

## Unmet breakthrough conditions

No three-seed/two-backbone joint candidate is certified. The second-backbone pilot
has not established the required reader capability. The full REVOKE/SHRED,
missing-key, deleted-object leakage, read-hook race, stale generation, cache replay,
exact bypass, reconstruction/key-channel and independently calibrated J-space causal
battery has not been passed by one corrected retained candidate. Generic locality
on twelve prompts cannot replace that battery.

Pointers, MVCC, capabilities, freshness, external memory, J-space, HMAC/content
binding, dependency tracking and ordinary scope routing remain excluded as individual
novelties. The result here is an experimentally qualified falsification of the
current composition, not an invented novelty claim and not an achieved breakthrough.
