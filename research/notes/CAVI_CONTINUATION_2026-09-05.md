# CAVI continuation: evidence and exclusions, 2026-09-05

Status: **no breakthrough certificate**. This note is a research log, not a positive abstract.

## Immutable evidence

Base source: `f31493e7bbb9b88ea5888a671fe282268c8466ca`.
E79 training source: `30315e7d3e76a32952b82dc79141f74d4f0fa512`.
All modifications in this continuation are on `research/cavi-continuation-audit`, not main.

The initial Actions/source snapshot enumerated all available runs and artifact metadata.
A separate collector verified ZIP digests against GitHub artifact metadata, then hashed their
members and retained original result JSON under `research/evidence/cavi-33963854769-1/`.
The collector does not infer scientific passes from workflow success.

### E79: actual 3000-step retained readers, three seeds

| Seed | Original four-template fresh minimum | Active-memory generic KL, nats | Stale UPDATE UNKNOWN | Stale RELINK UNKNOWN | SHRED UNKNOWN |
|---|---:|---:|---:|---:|---:|
| 0 | .975 | 3.6471917629241943 | .88 | .84 | .975 |
| 1 | .950 | 4.037354946136475 | .84 | .80 | .950 |
| 2 | .945 | 5.226874828338623 | .77 | .80 | .965 |

All joint screens fail. The explicit no-memory repeat is exactly equal, but does not establish
locality with memory enabled. Ordinary stale UPDATE old-answer rates are zero in this screen;
that does not establish all deleted-object leakage or a complete REVOKE/missing-key battery.

### Stronger prospective provenance-audit precondition

Same checkpoints, original full-sequence LM head, **three fixed worlds x four held-out templates**,
200 aliases per cell, no world/template selection. The minimum candidate accuracies are
**.945, .935, .915** for seeds 0, 1, 2. Full-vocabulary minima are .945, .935, .910.
All three artifact-splice audits on these parents were correctly **SKIPPED_INVALID_READER**.
Do not describe these skipped attacks as successful defenses or successful neural counterexamples.

### Recorded continuation and second backbone

Fixed GPT-2 continuation: 1500 additional fresh-world steps, learning rate .0002, same parent
marker centre, independent recorded RNG, new AdamW optimizer. This is not exact optimizer resume.
The seed-zero job timed out waiting for its still-running E79 parent; the parent later appeared.
A separate recovery workflow preserves that operational failure instead of replacing its artifact.

Pythia-70m, original 3000-step recipe, read layers 3 and 5: the optimized-head seed 0 and 1 readers
failed the stronger gate at .72 and .68 candidate accuracy (.49 and .485 full vocabulary).
Seed 2 failed optimization gradient equivalence before training: maximum absolute difference
.0197546482 against .0002 absolute tolerance. The control was **not relaxed**. All three seeds
were separately scheduled on the original full LM head; those results are not inferred here.

## Preregistered, not-yet-certified falsification hypotheses

1. A current row manifest may be spliced onto an older serialized Bank if the consumer checks
   supplied identity/generation fields but does not bind those fields to the consumed contents.
2. A new alias witness may be spliced onto old cached resolved values if witness validation is
   independent of the values' provenance.
3. A dense resolved-value mixture may depend on multiple canonical pods even when its stored
   witness mentions only the intended queried pod. Removing another actual contributor can leave
   that single witness current while the cached value still contains the removed contribution.

The third audit intervenes on the exact linear contribution of the selected deleted value,
keeping other values, routing coefficients, prompt and authority fixed. This is a test of stale
causal dependence, **not by itself an answer-level deleted-object leakage metric**.
All neural interpretations require the unchanged >=.95 reader gate and attack-specific before/
after capability controls. The code does not grant arbitrary authority rollback or hook removal.

Potential corrective requirement, conditional on confirmation: a forward-atomic consumption
boundary also needs trusted content-to-lineage binding and the complete actual dependency closure
of a derived artifact. Hashing, authenticators, dependency tracking and capability binding are
established techniques; this statement alone is not a novelty claim or an implemented fix.

## Parallel performance experiment

A clean-prefix, write-only scope pilot keeps the entire learned reader frozen. It classifies a
question before the first memory write and reuses that decision across read sites. Accepted
queries run the original resolver/dereference/write path; rejected queries suppress every write.
The fixed training set uses templates 0..7, independent generic training text, 4096 examples,
500 head-only optimizer steps, and logit threshold zero. Held-out templates do not calibrate the
threshold. Forced-ON equality with the reader and forced-OFF equality with the clean model must
pass before training. These controls do not substitute for learned scope accuracy or generic KL.

E72 should remain negative, but its source transplanted weights into a different scope-before-
routing execution class. It does not establish that every isolated write-only scope mechanism
must fail. The new pilot tests that narrower distinction without relaxing CAVI authorization.

## Prior-art exclusions expanded

- Gurnee et al., *Verbalizable Representations Form a Global Workspace in Language Models*,
  Anthropic, 6 July 2026: https://transformer-circuits.pub/2026/workspace/index.html .
  J-lens/J-space are their prior art. An independently calibrated average-Jacobian lens and
  causal interventions are not interchangeable with authority bits, learned write directions,
  an arbitrary projector, or an output logit lens. A late output-stage signal alone is not a
  demonstration of all functional workspace properties.
- Parakhin, *Token Coherence*, arXiv:2603.15183, 16 March 2026:
  https://arxiv.org/abs/2603.15183 . Artifact-cache coherence, monotonic versioning, invalidation,
  and MESI-style agent-memory integration cannot be claimed individually as new here. Its
  simulations and protocol analysis are not this programme's empirical neural-activation audit.
- Wu and Canedo, *Invalidation Contracts for Cross-Episode Agent Memory*, arXiv:2609.00243,
  surfaced in the 5 September search: https://arxiv.org/abs/2609.00243 . The indexed primary
  abstract describes version-stamped memory invalidation and separates protocol validity from
  model compliance. Full text was not retrievable in this session; numerical claims have not
  been independently verified. Treat this as a concrete exclusion/review item, not absent art.
- WISE, arXiv:2405.14768: https://arxiv.org/abs/2405.14768 . Dual-memory routing and the
  reliability/generalization/locality tradeoff are prior-art areas, not novelty established
  by adding a scope classifier.

Pointers, symlinks, external memory, MVCC, freshness, cryptographic capabilities, HMACs, J-space,
ordinary scope routing and cache invalidation remain explicitly excluded as individual novelties.
A defensible contribution would have to survive the complete coupled neural lifecycle, adversarial,
independent causal-audit and performance contract on repeated retained public-backbone readers.
