# E-000092 — Generation-Keyed Neural Addressing

Status: preregistered falsification experiment; **not a novelty claim**.
Date: 2026-09-05

## Motivation

E-000090 falsified the idea that a lightweight generation watermark could simply survive ordinary transformer computation and later be decoded reliably from final hidden state. Across DistilGPT-2 and Pythia-70M, exact generation accuracy was 0 for every registered signal strength. We therefore stop treating generation recovery from arbitrary downstream hidden state as the mechanism.

The stronger question is functional rather than diagnostic:

> Can the current Pod generation participate in the neural addressing rule itself, so that stale memory state from an older generation becomes unaddressable after one lifecycle transition even when the stale state remains physically present and external version metadata is missing or wrong?

This is **not** ordinary cache invalidation: the stale row is deliberately retained. It is also not a watermark: no decoder is asked to infer a generation after arbitrary downstream computation.

## Candidate mechanism

For a canonical Pod `P`, each generation `g` has a generation code `c(P,g)` in a reserved addressing subspace. A memory key is

`K(P,g) = semantic_key(P) + alpha * c(P,g)`

and a query resolved against the current authority is

`Q(P,g_current) = semantic_query(P) + alpha * c(P,g_current)`.

The stale and current rows can coexist. The desired property is:

- `Q(P,g+1)` selects `K(P,g+1)`;
- `Q(P,g+1)` does not select or materially mix `K(P,g)`;
- unrelated Pods remain addressable;
- normal model output is preserved except for the intended memory payload effect.

A production design would bind both Pod identity and generation; generation-only coding is only a Phase-A existence test.

## Strong prior-art boundary

The following are established and excluded from novelty:

- content-addressed and associative memory;
- key/query attention and modern Hopfield memory;
- cache invalidation and version checks;
- cryptographic key rotation and revocable capabilities;
- activation watermarking/provenance;
- key/value rotations used for KV quantization;
- external memory access control;
- selective KV eviction/retrieval.

The candidate survives only if **knowledge-lifecycle key rotation changes neural addressability of stale state itself** and yields a systems property that cannot be reduced to a cheaper sidecar check.

## Phase A — functional stale-state kill screen

Run on at least DistilGPT-2 and Pythia-70M.

1. Derive a semantic query/key from a frozen model state for a fixed prompt.
2. Construct two rows for the same Pod: stale generation `g` with payload `v_old`, and current generation `g+1` with payload `v_new`.
3. Keep both rows physically present.
4. Run a baseline without generation coding; because semantic keys are otherwise identical, the system is ambiguous/stale-order-sensitive.
5. Add generation-keyed addressing and current authority `g+1`.
6. Require current row selection >= 0.99 and stale row selection <= 0.01 across registered prompts/seeds.
7. Inject the selected payload into a frozen backbone and require the guarded output to match a current-only gold run while stale-only and unguarded ambiguous controls materially differ.
8. Swap or delete external metadata. Generation-keyed addressing must still select the current row because the code is part of the neural key/query state.
9. Preserve unrelated-Pod selection and generic-prompt locality.

## Baselines

- external sidecar generation check that deletes/masks stale row;
- co-located sidecar stored atomically with the tensor;
- content hash of tensor + metadata;
- global epoch;
- full recomputation/current-only bank;
- no-generation-code bank with stale/current row-order swaps.

If a co-located sidecar provides the same correctness, failure coverage and practical cost, the neural addressing mechanism is not a major invention even if Phase A works.

## Validity bars

Per backbone and registered seed/prompt set:

- current-generation selection >= 0.99;
- stale-generation selection <= 0.01;
- metadata swap/delete does not change the in-band selection result;
- guarded current output matches current-only gold within maxabs <= 5e-3;
- stale/ambiguous control materially differs from current gold by maxabs > 1e-4 in at least 95% of registered cases;
- unrelated Pod selection >= 0.99;
- generic output KL relative to the same current-only memory system <= 0.05 nats;
- top-1 agreement on unrelated/generic controls >= 0.98.

## KILL conditions

Kill this seam if:

- the generation term required to suppress stale rows destroys semantic routing/locality;
- stale rows still receive material routing mass;
- success depends on deleting/masking stale rows externally;
- metadata loss causes the mechanism to fail;
- the result does not reproduce on >=2 backbone families;
- a co-located sidecar is guarantee-equivalent and cheaper;
- the mechanism is merely a standard cryptographic/access-control key wrapped around an external memory rather than a neural addressability effect.

## If Phase A survives

Phase B must use the **real LINK -> canonical Pod reader**, encode `(pod_id, generation)` rather than generation alone, and attack stale Bank/router/resolved-payload/Hidden/KV material after UPDATE, RELINK, REVOKE, SHRED, DELETE, RESTORE/ABA and concurrent in-forward transitions. J-space/J-lens remains an independent content-accessibility audit and is never used as the addressing key.

## Potential narrow claim if all later phases survive

> A canonical knowledge-object lifecycle can rotate the neural address of a Pod generation so that previously materialized stale memory state becomes functionally unreachable under current inference while current and unrelated knowledge remain addressable, without requiring physical cache deletion before correctness is restored.

This statement is a hypothesis until strong baselines, real readers, multiple seeds/backbones, locality, performance and prior-art checks survive.
