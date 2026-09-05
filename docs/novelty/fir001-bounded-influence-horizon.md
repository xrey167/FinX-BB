# FIR-001 — exact bounded influence horizon collapses to finite memory or late binding

**Architecture-boundary result, not a major invention.** Date: 2026-09-05.
Parent BHC001: `2153cab1bf64ef8bb73b263b8b624354319ff378`.
Preregistered before numerical execution at `6715b00c18814745b063aaa6ad6e4527f41a94e8`.
Executed source: `5ce3be2572177055a33b825dccfc07970b87f5a5`.
Completed Actions run: **33989196381**.

## Decision

Structurally forcing one pod's numerical influence to disappear from the complete future-relevant Markov state after a fixed K steps can make old states trivially safe after that horizon. But once two source worlds have the same complete state and receive identical future forcing, all later reads are identical. Persistent source recall then requires source-dependent information to be retained or reintroduced somewhere else.

In the tested construction, rereading the canonical pod at query time restores late recall and makes edits immediately visible without repairing old source-free states. The strongest conventional baseline receives the exact same canonical payload and performs the exact same materialization. Timings are ~1x. Therefore this route does not create a distinct repair primitive; it becomes finite-window memory plus late binding/lookup.

This is elementary deterministic-state sufficiency, not a new theorem. It is NOT a claim that all architectures must use raw source replay or that compact sufficient statistics cannot exist. Any such retained statistic is another source-dependent state channel and must be included in repair/memory accounting.

## Executed construction

Five seeded source-independent background recurrences. Background width128. A source channel has four stages of width32. A source payload enters stage0 and is shifted through zero-preserving nonlinear maps; no stage feeds backward. Without reinjection, structural support is exactly nilpotent: after four transitions all four source stages are zero, independent of payload value. Python object integers and an independent Fraction scalar recurrence give exact arithmetic.

Two distinct payload worlds are compared for80 steps, checking every complete `(background, source-channel)` write.

| Result | All five seeds |
|---|---|
| First complete-state equality | write index **4** |
| Equality preserved from4 through79 | yes |
| Source distinguishable before horizon | yes |
| Source distinguishable at step64 without reread | **no** |
| Source distinguishable at step64 with canonical reread | **yes** |
| Leaky control with source written into long-term background ever coalesces | **no** |

The leaky control is deliberate: once the source is allowed to change the long-term recurrent state, persistent source dependence returns, but the exact four-step horizon disappears. This is a boundary within this tested deterministic architecture, not a frequency statement about real LLMs.

## Strong baseline and timing correction

The first code draft accidentally timed a candidate read from a pre-materialized query state against a baseline that performed source materialization inside the timed call. That would have been an invalid benchmark. It was corrected BEFORE the executed evidence reported here. The final executed source gives the candidate and conventional baseline the same current canonical payload, the same source-channel initialization, the same one-step source materialization and the same readout.

CI median conventional/candidate query-time ratios across seeds:

`1.00068, 1.00165, 0.99449, 1.00375, 1.00408`.

No meaningful candidate advantage survives. The full 80-step replay is about 177–187x slower than the query-time reread microbenchmark, but that comparison does not establish novelty because the conventional reread baseline gets the same benefit. It is also not an end-to-end LLM mutation-to-ready measurement.

The finite-horizon state stores256 integer words (`128 + 4*32`). The canonical source adds32 words. A literal four-event finite window would use128 source words. Python object overhead and system metadata are not counted, so this is NOT the matched total-memory gate.

## Architecture consequence

A fixed exact horizon can only remove source influence from the state that is actually included in the equality contract. If long-term recall still depends on the source, then at least one of the following must be true:

1. the source dependence never left some retained state channel;
2. the source is reread/reinjected later;
3. another retained source-dependent sufficient statistic exists but was excluded from the purported complete state.

Case1 forfeits the claimed horizon unless that channel has its own exact edit contract. Case2 is late binding/retrieval. Case3 may be useful, but the extra representation is the actual remaining research problem and its storage, lifecycle and repair costs must be paid.

Therefore a nilpotent/finite-horizon source channel alone is not the sought breakthrough. A future candidate needs a source-dependent retained representation that is both revision-sufficient and materially cheaper to update than guarantee-matched replay/lookup, without collapsing into ordinary finite-window storage, symbolic/event-sourced state, or late binding.

## Prior-art boundary

Finite-impulse-response and finite-memory systems are classical. A 2025 IEEE Swiss Conference on Data Science paper explicitly frames attention as an FIR-style model and SSMs as IIR-style models. Structurally bounding temporal support is therefore not novel.

Forgetting Transformer (ICLR2025) adds a learned forget gate to attention, but does not provide our exact finite-step lifecycle contract. Recurrent Memory Transformer, ELMUR and recent bounded-memory long-horizon systems further occupy the general finite/bounded memory design space. ReWorld (arXiv2608.23565) explicitly separates short-horizon control from long-horizon landmark memory, reinforcing that bounded local state plus a separate long-term store is an established architectural pattern.

This experiment does not replicate or refute those models. Generic FIR memory, sliding windows, external stores, reread/late binding, event sourcing and replay receive zero novelty credit.

Primary sources reviewed:
- https://ieeexplore.ieee.org/document/11081496/
- https://proceedings.iclr.cc/paper_files/paper/2025/hash/add3d389197ad2267f660ad060ef61f4-Abstract-Conference.html
- https://arxiv.org/abs/2207.06881
- https://openreview.net/pdf?id=bm3rbtEMFj
- https://arxiv.org/abs/2608.23565

## Reproducibility

Run **33989196381** completed successfully. Selected suite: **26 passed in1.55s**, no skips. Artifact **9976088608** was downloaded and its ZIP SHA-256 verified against GitHub metadata:
`6c5f428f64501098d6b27ab2dd1c31d2ca2b7a90f719139589b98d7f6348d856`.

Archived hashes:
- experiment: `eef67d08f4fe3dbb413f46ef926d75fc18644e4b2262dccc666713568c5f76cd`
- tests: `fcea899af2553619ce4f3ad8a68846e0f0c754718b5adf584bb36c6fbffd59fe`
- CI JSON: `03c8d9fdbedae61b0a9cf9f56f6c2e67d8686e0324d72338887b7fde7587809e`

This is a same-source CI execution of synthetic exact operators, not independent laboratory replication, a pretrained backbone or a trained semantic reader. Main and historical scientific files are untouched.

All full-system gates remain NOT EVALUATED: >=10x complete mutation-to-ready against strongest matched baseline, <=5% inference-throughput loss, matched total memory, >=95% fresh/unseen-paraphrase/lifecycle reading, >=90% scoped UNKNOWN, <=0.05 nats generic divergence/exact bypass, independent J-space auditing, generation/publication safety, trained seeds and second-backbone qualification.
