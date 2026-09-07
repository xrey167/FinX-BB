# E-000091 — Tensor-Bound Knowledge Freshness

Status: conditional preregistration; run only if E-000090 Phase A establishes a decodable in-band generation signal on >=2 backbones.
Date: 2026-09-05

## Question

Does an in-band `(pod_id,generation)` freshness signature provide a real systems property that a normal sidecar metadata tag cannot provide under the **same transport fault model**?

The intended fault model is narrow and explicit: neural tensors survive a transport/storage boundary but their sidecar metadata is absent, stale, or paired with the wrong tensor. Arbitrary malicious tensor rewriting is out of scope.

## Strong baselines

1. external sidecar `(pod_id,generation)` tag;
2. co-located sidecar serialized in the same artifact container as the tensor;
3. content-addressed tensor hash + generation registry;
4. global epoch invalidation;
5. full recomputation.

If a co-located sidecar or content hash solves every registered fault with lower cost, kill the neural-signature claim.

## Real stale-state attack

Use a capable real Symlink->Pod reader and at least two public backbones where feasible.

For Pod `P@g`:

1. resolve via a linguistic alias and materialize model-derived Hidden and KV state;
2. store three representations: raw tensor only, tensor+external sidecar, tensor+co-located sidecar;
3. update the canonical Pod once to `P@g+1`;
4. deliberately replay the raw stale tensors from `g` after stripping all sidecars;
5. deliberately pair stale tensor `g` with sidecar `g+1` (metadata-swap fault);
6. require the in-band detector to recover `(P,g)` from the stale neural state itself;
7. require a consumption gate to reject `g` against current authority `g+1`;
8. rebuild current `g+1` state and require the detector to recover `(P,g+1)`;
9. preserve unrelated Pod state and normal model behavior.

## Decision rule

### KILL

Kill if any of the following holds:

- E-000090 does not reproduce on >=2 backbone families;
- the in-band code cannot recover Pod identity + generation from stale materialized state;
- decoding requires per-Pod detector training;
- a co-located sidecar tag is equally robust under every registered transport fault and materially cheaper;
- the signature materially changes task behavior or locality;
- stale tensor replay is not behaviorally consequential;
- generation detection works only on freshly generated Hidden state but not on reused KV/Hidden artifacts.

### SURVIVE

Survive only if:

- stale raw neural state remains self-identifying after side metadata loss;
- swapped external metadata is detected as inconsistent with the tensor-carried identity;
- a single backbone-specific detector works across held-out Pods/generations;
- current-generation behavior remains correct;
- the property reproduces on >=2 backbones and >=3 seeds;
- there exists a realistic deployment boundary where raw tensor transfer/storage can outlive or become detached from its side metadata.

## Novelty guard

Even a pass is not automatically a major invention. Activation watermarking, data provenance, embedded checksums, information-flow labels and cryptographic tagging are established. The only potentially distinct mechanism is a **knowledge-lifecycle freshness code that is causally carried by reusable neural state and directly gates that state's eligibility for reuse**.

Major-break status additionally requires real LINK->Pod capability, lifecycle propagation, stale-state attacks, low overhead, and a systems benefit large enough to justify the in-band mechanism over normal metadata engineering.
