# R306 Result — Persistent Coherence Recovery

Status: **persistent authority+lifetime recovery gate passed under the declared safety model; not DoD.**

## Scale and result

- 8,000 canonical Pods;
- 25,000 authority transitions;
- 100,000 final derived states;
- 100 crash/recovery trials from stale lifetime-index snapshots;
- pre-crash factor-vs-authority audit mismatches: **0**;
- stale-valid restored states observed before replay: **184,378** across sampled recovery probes;
- stale-valid states after authenticated authority replay: **0**;
- post-replay factor-vs-authority mismatches: **0**;
- stale replica capability rejection: **100%**;
- same-digest/new-generation old capability remains dead: **pass**.

Recovery cost on the CI host:

- median replay: **71.1 ms**;
- p99 replay: **149.2 ms**;
- median factor nodes touched during recovery: **23,907**;
- median online invalidated derived states/update: **1**;
- p99 online invalidated derived states/update: **33**.

Artifact ZIP SHA256: `324a99da3e571214023c10fecb694163e2671f09504f43ea98c84f174ce9f586`.
Report SHA256: `4cc36598080290b7d1bec28828020034097b33573c01f352f0fa21216620f567`.

## Architectural decision

Authority and derived-neural lifetime state are independently durable. A restored J/cache/lifetime snapshot is never trusted as current simply because its bytes and local validity bits are intact.

Recovery rule:

`restore lifetime snapshot @ watermark S -> recover authenticated authority ledger to tip T -> replay all authority transitions S<T -> only then enable J/cache serving`

This turns restart/replica recovery into the same generation-coherence problem as online serving. Stale derived artifacts can remain persisted, but authenticated authority replay kills their old lifetime factors before admission.

This is not a consensus protocol; CKCA still uses the safety-first linearizable-verifier model and fails closed when authoritative verification is unavailable.
