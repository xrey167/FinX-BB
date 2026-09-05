# LCC-001 — completed and digest-verified CI evidence

Date: 2026-09-05. Status: **completed second CPU execution; two restricted structural falsifications, NOT a technical novelty**.

This addendum supersedes only the queued-CI status in `lcc001-result.md`. It changes no assay source, experimental decision, threshold or result. Read the scope restriction in `lcc001-oracle-scope-clarification.md` with all results.

## Executed run

- Actions run: `33985611376`.
- Job: `101358438552` (`exact-screen`), completed successfully.
- Executed source/workflow commit: `27b31ee52b429339c60eb3811e7ebb30846055e2`, confirmed in the checkout log.
- Preregistration commit: `b0d25b651ff02c5b847f30c30778e51febd8ce90`.
- Ubuntu 24.04.4; Python 3.13.15.
- All nine test methods passed, no skips; unittest reports 1.140 seconds.
- The complete assay then emitted its full nine-scenario result and uploaded the evidence.

## Artifact verification actually performed

Artifact `9975101455` (`lcc001-exact-certificate-boundary`) was downloaded after completion. Its 4,504 ZIP bytes were hashed locally and matched the digest reported by the upload log:

`2dae8c598cbad2806b6e7a954e8fccfbcd587685963fc5c3362f61e43065e2a4`

The archive contains `result.json`, `stdout.txt`, and `sha256.txt`. Both recorded file hashes were checked against the actual bytes:

- Executed source SHA-256: `8f6ba1186c24a9e464725c3dd953c0f3a5edd374476ac83841b0cf7e10eb1698`.
- CI full result SHA-256: `f5d75008c95e6e6b4a199c37b823b33b3bbd86717c4dc724a5d76e73e66753d6`.
- Local full result SHA-256: `d60693ab500a71b22b84ca7d78ba5c00e766fee12e54b9ac76a3237ee082a813`.

Parsed local and CI result objects were compared in full, including every rational probe location, knot, counterexample witness and interval statistic. The **only differing field is Python version**, 3.13.5 locally versus 3.13.15 in CI. All experimental values agree exactly; result-file byte identity is not claimed because the version field differs.

## Confirmed results in both environments

| Check | Result |
|---|---:|
| Equal numerical-state history pairs with opposite validity | 48/48 |
| Correct individual decisions with supplied provenance | 96/96 |
| Equal audited output vectors | 1,008 |
| Equal shared forward numeric traces | 1,008 |
| Equal mixed derivative vectors with respect to context/source inputs through order four | 15,120 |
| Audited revocations with zero source effect | 504 |
| Constructed unprobed revocations with nonzero effect | 9/9 |
| Scoped interval certificates agreeing with exact range reference | 1,890/1,890 |
| Intervals certified independent | 949 |

Arithmetic is exact Python Fraction, without numerical tolerances. This is a repeat execution of the same source in another CPU environment, not an independent laboratory replication.

## Claim boundary remains unchanged

Arm A excludes recovering arbitrary missing historical origin from an unlabelled post-merge numerical state alone. Arm B excludes unrestricted whole-domain support certification based only on the registered finite source/context-response observations. Its derivative oracle is with respect to x and m, not all internal edges or model parameters; richer white-box information can distinguish the constructed pair. Arm C positively verifies a restricted white-box scope and carries no novelty credit.

These are three synthetic construction seeds, zero trained-reader seeds, zero language backbones and no actual J-lens execution. No real-symlink attack, lifecycle closure, UNKNOWN, deleted-object leakage, generic divergence, matched-memory, overhead or speedup gate is passed by this experiment. No existing reader or attack source was modified.

Run: https://github.com/xrey167/FinX-BB/actions/runs/33985611376
Artifact: https://github.com/xrey167/FinX-BB/actions/runs/33985611376/artifacts/9975101455
