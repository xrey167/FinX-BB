# E-000093 — Passive In-Band Freshness Sidecar Reduction

Date: 2026-09-05
Status: **PREREGISTERED KILL SCREEN / scoped reduction**

## Motivation

E-000090/E-000090B tested generation information embedded in neural-derived state and E-000092 tested generation-conditioned neural addressing. Both are now closed as major-invention seams. Before spending more experiments on alternative watermarks, signatures, syndromes, residual markers, tensor provenance fields, or other passive freshness encodings, this experiment asks whether the entire passive class is guarantee-equivalent to a correctly co-located sidecar.

This is deliberately narrower than a universal impossibility theorem.

## Candidate class under test

A **passive in-band freshness scheme** is any reuse scheme satisfying all of the following:

1. a cached neural-derived artifact is a tensor/byte object `T`;
2. current lifecycle authority is `A`;
3. the reuse decision is deterministic and can be written as `R(A, T)`;
4. all lifecycle-specific information used by that decision is recoverable from the stored artifact through some deterministic statistic `D(T)`;
5. after materialization, lifecycle mutations do not alter `T` itself before the reuse decision; they alter only `A`;
6. accepting or rejecting `T` is a boundary decision rather than a lifecycle-dependent transformation of the neural computation represented by `T`.

The class includes passive generation watermarks, embedded version/freshness codes, tensor-resident provenance fields, residual markers, parity/syndrome-style freshness codes, and similar designs **when they are used only to decide whether an otherwise unchanged cached artifact may be reused**.

It does not include a mechanism in which the lifecycle transition actively changes the mathematical neural operator applied to the stored state, destroys information required for stale reconstruction, or otherwise makes the computation itself non-factorable into a reusable artifact plus an accept/reject predicate.

## Reduction

If the candidate's decision uses `T` only through a sufficient statistic `Z = D(T)`, then define a co-located sidecar at materialization time:

`S := D(T)`.

Define the baseline reuse decision:

`R_sidecar(A, S) := R_hat(A, S)`

where `R_hat` is exactly the candidate decision after replacing the internal decode step with its stored result.

For every authority state `A` and every artifact `T` in the registered domain:

`R_inband(A, T) == R_sidecar(A, D(T))`.

Therefore the passive in-band location of the sufficient statistic cannot by itself provide a stronger freshness/reuse guarantee than a correctly co-located sidecar carrying the same statistic.

If the candidate instead consumes additional tensor content beyond `D(T)`, the sidecar baseline is strengthened to a content-addressed digest or the minimal sufficient statistic actually consumed by the gate. If no smaller sufficient statistic is known, storing a digest plus candidate-specific verification data is allowed. The comparison is guarantee-matched, not artificially weak.

## Registered fault comparison

The executable assay enumerates deterministic candidate decoders/gates over finite artifact and authority domains and checks exact decision equality between the in-band scheme and its materialized sidecar shadow under:

- no mutation;
- current-generation update;
- unrelated object mutation;
- external metadata deletion;
- external metadata swap;
- tensor relocation/serialization;
- stale artifact replay;
- ABA authority return;
- rollback to an older authority state;
- same-content/different-generation authority states.

Transport faults are applied to metadata that is **not part of `T`**. A sidecar is defined as correctly co-located/atomically transported with the cached artifact; otherwise it would not be a guarantee-matched baseline.

## Kill rule

Kill passive in-band freshness as a **correctness/guarantee novelty seam** if the sidecar shadow exactly reproduces every registered reuse decision for all enumerated cases.

This does not prove that an in-band representation can never be faster, smaller, more hardware-friendly, or operationally preferable. Any such surviving claim must beat the co-located sidecar under matched correctness with measured systems evidence. Location inside the tensor receives zero novelty credit.

## Escape condition

A successor mechanism is interesting only if at least one of the following is true:

1. the lifecycle transition changes the neural computation itself, not merely an accept/reject predicate;
2. stale state loses a computational capability that cannot be restored from the cached tensor plus public/current authority;
3. the mechanism permits exact reuse/repair that a guarantee-matched sidecar cannot obtain without materially more work or memory;
4. the mechanism discovers/certifies causal source lineage that was not supplied as ordinary dependency metadata;
5. it establishes a measured systems frontier (major programme gate remains >=10x mutation-to-ready versus the strongest guarantee-matched baseline, <=5% normal inference overhead, matched memory).

## Explicit non-claims

No novelty is claimed for watermarking, provenance, version stamps, hashes, sidecars, cryptographic binding, ECC, capabilities, key rotation, cache validation, dependency tags, content addressing, activation markers, or the reduction itself.

The purpose of E-000093 is to prevent the programme from repeatedly rediscovering passive tensor metadata under new names.

## Programme consequence if the kill screen passes

Close the passive in-band freshness family as a major correctness novelty route. Continue only with mechanisms that change the **computational frontier**: active lifecycle-conditioned operators, exact affected-work compression beyond generic change propagation, or neural causal-lineage discovery/certification. Every later real-reader result still requires >=0.95 on every held-out symlink template in the exact interpreted job and all previously registered lifecycle/system gates remain unchanged.
