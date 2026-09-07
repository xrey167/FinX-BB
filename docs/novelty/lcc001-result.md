# LCC-001 — decisive boundary on post-hoc lineage certification

Date: 2026-09-05. Status: **executed restricted structural falsification; NOT a major useful technical novelty**.

Base: `research/e000051-clean-bystanders` at `947206b72a75dc3616ca44dc5434a0a293f4bca7`.
Preregistration: `b0d25b651ff02c5b847f30c30778e51febd8ce90`.
Source: `research/lcc001/assay.py`, commit `4c3275c0e7d19217a573f17eba7782f47f82d2a8`.
Local result: `research/lcc001/local-summary.json`.

## Decision and scope

After E-000086R, post-hoc neural source-lineage discovery/certification was a possible alternative to explicit generic dependency tracking. This screen excludes two restricted versions of that alternative:

1. Recovering arbitrary historical source-generation provenance from an unlabelled numerical activation without retained origin information.
2. Promoting finite intervention/local-derivative agreement to an exact independence certificate over an untested domain, for a function class admitting the constructed gates.

It does not reject white-box verification, restricted classes with a proved complete identification procedure, provenance-bearing representations, or new efficient numerical repair. It does not establish that a particular trained E81/E95 reader is broken. The indistinguishability argument is elementary and receives no mathematical-novelty credit.

## A. Equal activations do not identify historical source generations

Two distinct canonical generations A and B have the same payload. Identical downstream computation creates equal cached activations. A is revoked while B remains live. The observer can even receive the same current authority registry in both histories: without knowing which source produced the particular activation, its observations coincide while the required validity decision differs.

Any deterministic readout of the identical activation with the same downstream model, including a fixed lens, also coincides. An improved decoder cannot recover origin information the representation never preserved. Explicitly encoding or retaining origin escapes the restriction; that is provenance, not this programme's new invention.

The exact screen constructs 48 history pairs, using 16 integer payloads under three synthetic identity labels. All 48 have equal numerical states and opposite required validity. Supplying provenance resolves all 96 individual decisions.

Historical provenance is not functional influence. Same-value UPDATE and equal-valued alias RELINK can change generation validity without changing a content-only activation. The strict generation contract remains in force; value equality must not silently replace it.

## B. Finite intervention agreement does not certify the whole domain

Compare `H0(x,m)=base(x)` and `H1(x,m)=base(x)+m*B(x)*v`, with source-presence flag m, context x, and `v=(1,-2,3,-4)`. The downstream readout is identical. B is a triangular gate:

`B(x)=(relu(x-a)-2*relu(x-b)+relu(x-c))/(b-a)`, with `b=(a+c)/2`.

Its support is strictly inside a gap between the finite probe locations. At each probe it is identically zero on an open neighbourhood. The outputs and local derivatives of every finite order agree there, as do all tested present/revoked comparisons. Both graphs can compute the same intermediate gates and differ only in their final gate connection, leaving their observed finite numeric transcripts identical. Parameter/graph inspection is outside the restricted observer's information and can distinguish them.

At the untested midpoint b, source removal changes H1 by exactly v and changes H0 by zero. Thus the finite observations do not identify full-domain functional support.

Three seeds and probe counts 8,32,128 were fixed before execution. Selecting a gap after fixing the finite probe set is an intentional preregistered worst-case construction, not a natural-workload error estimate. The derivative checks execute through total order four; the neighbourhood argument supplies the arbitrary-finite-order statement.

This is conditional on the admitted function class. It does not prove that the same exact compact-support construction exists inside a particular fixed smooth-activation transformer. A direct fresh comparison can soundly settle that particular mutation. A white-box proof or complete finite identification under additional restrictions can also escape. What is excluded is an unrestricted universal certificate inferred solely from successful finite probes.

## C. A scoped white-box proof succeeds as a control

For the known triangular response, source independence on a closed interval [l,u] is equivalent to `u<=a OR l>=c`. This certificate was compared with an independently evaluated exact piecewise-linear extremum reference, including all breakpoints and singleton intervals.

All 1,890 comparisons agree. Of these intervals 949 are certified independent. Every whole-domain request is rejected. Support-boundary singleton intervals are accepted and the nonzero midpoint singleton is rejected. This is ordinary program/interval reasoning with zero novelty credit; it demonstrates the valid escape through explicit proof scope rather than more probe-based confidence.

## Executed local evidence

| Check | Result |
|---|---:|
| Standard-library unit tests | 9 passed, no skips |
| Equal-state provenance pairs requiring opposite validity | 48/48 |
| Correct decisions with explicit provenance | 96/96 |
| Exact audited output equalities | 1,008 |
| Equal shared numeric traces | 1,008 |
| Equal mixed-derivative vectors through total order four | 15,120 |
| Audited revocations with zero source effect | 504 |
| Constructed unprobed revocations with nonzero effect | 9/9 |
| Scoped certificates matching the exact range reference | 1,890/1,890 |

Arithmetic: Python Fraction, no numerical tolerances. Python 3.13.5. These are three synthetic construction seeds, **zero trained-reader seeds and zero language backbones**. The actual Anthropic J-lens was not executed.

Source SHA-256: `8f6ba1186c24a9e464725c3dd953c0f3a5edd374476ac83841b0cf7e10eb1698`.
The local source bytes were verified against repository Git blob `59286683dde19dced0bd2cc15991a36ebf4f9aed` and match.
Full local result SHA-256: `d60693ab500a71b22b84ca7d78ba5c00e766fee12e54b9ac76a3237ee082a813`.

Reproduce with `python research/lcc001/assay.py --out lcc001-result.json`. The full generated result contains every rational probe and counterexample witness.

Actions replication was submitted as run `33985611376`, workflow/source commit `27b31ee52b429339c60eb3811e7ebb30846055e2`. At this report's evidence cutoff it was queued. No completed CI or independent-laboratory replication is asserted.

## Invention-programme consequence

Keep ordinary authoritative generation provenance in both the candidate and strongest guarantee-matched baseline. Do not replace missing origin information with a semantic activation decoder. J-space/J-lens remains independent audit for measured causal effects and counterexample discovery; it must not become runtime authority or be presented as a universal completeness certificate.

An exact selective-reuse candidate must declare its proof-bearing information: a complete source/control-path account, or a white-box structural independence certificate valid on an explicit mutation/context domain. Classical dependency tracking and formal verification remain occupied baseline ingredients.

The still-open invention opportunity is a new numerical-computation or representation mechanism that makes such a sound domain substantially cheaper while retaining useful nonlinear cross-source computation and capable reading. E95's semantic-address/exact-payload mechanism is an ordinary baseline, not a novelty claim. Avoiding LCC001's countermodels does not itself promote a candidate.

No existing reader, trainer, threshold or attack implementation was changed. No E83/E84/E85/E95 result is promoted by LCC001. Every held-out real-symlink template still must be >=0.95 in each exact trained-reader attack job.

All system gates remain unmeasured here and unchanged: >=10x mutation-to-ready over the strongest guarantee-matched baseline at scale; <=5% inference overhead; matched total memory; >=95% fresh/unseen-paraphrase reading and lifecycle propagation; <=2% deleted-object leakage; >=90% UNKNOWN in declared missing-key scope; exact bypass or <=0.05 nats generic divergence; >=3 trained-reader seeds and preferably >1 backbone.

Primary context, not claimed as new:
- Causal abstraction and intervention-based evaluation: https://arxiv.org/html/2106.02997v2
- Original J-lens/global-workspace work: https://transformer-circuits.pub/2026/workspace/
- Incremental neural certification: https://arxiv.org/html/2305.19521v2

**Final classification: restricted falsification that narrows post-hoc discovery/certification, not a major technical novelty and not a rejection of the entire Symlink-Pod programme.**
