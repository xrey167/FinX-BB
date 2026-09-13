# R301 Result — Bounded CKCA Coherence Model

Status: **bounded protocol-model gate passed; not an unbounded formal proof and not DoD.**

The finite-state model explores two canonical Pods, two semantic values, monotonic generations up to generation 3, reusable capability slots, derived J-state slots and transitive J-state composition through traces of depth up to 8.

## Exhaustive bounded result

- reachable states: **35,614**;
- explored transitions: **153,066**;
- retained-artifact serve checks: **104,564**;
- CKCA generation-aware false accepts: **0**;
- semantic-value-only false accepts: **22,616**;
- Pod-ID-only false accepts: **48,040**;
- explicit same-value ABA counterexamples found: **5** sample witnesses retained in the report.

Artifact ZIP SHA256: `50b1b7a364329a407a020d032a0787b9f4916c4e1ae01c3294166b339ee3be17`.

## Interpretation

A semantic-value-only freshness policy is unsafe even if the current value equals the value stored in an old artifact. A same-value rewrite is a new generation: an old capability or derived J state must remain dead. Pod-ID-only freshness is weaker still.

The bounded model supports the CKCA rule:

`admissible(artifact) = AND over exact current (pod_id, generation) source leaves`

J-state composition takes the transitive union of source-generation leaves. Generation monotonicity then makes old dependencies monotonic-dead under same-value rewrites, ABA-style cycles and revoke/revive sequences.

This result does not establish novelty. S-Bus and other distributed-state work already show read-set based consistency for LLM systems; R301 is specifically a bounded model of CKCA's generation-lifetime admission semantics.
