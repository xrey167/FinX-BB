# R304 Result — Governed Fact Virtualization

Status: **governed single-token fact virtualization gate passed; not DoD and not a novelty claim.**

Backbone: frozen `Qwen/Qwen2.5-0.5B`.

## Result

A shared output-row transform was calibrated on an earlier subset of one-token country-capital values and then frozen. Future held-out capital values were not used to select or fit their per-value representation.

The selected shared transform was row LayerNorm. On held-out future values:

- full-vocabulary greedy value emission: **100%**;
- int8 value-state greedy emission: **100%**;
- 5,000 post-calibration future rewrites: **100%**;
- 1,000 revoke/no-value cases routed to explicit `unknown`: **100%**;
- natural aliases collapsed by the resolver to byte-identical canonical J/value state: **100%**.

Artifact ZIP SHA256: `f82d5bf56bf04b6e276451e730ccadd6cd797cc8838719ef383490f34be99b50`.

## Architectural consequence

For a **governed relation**, CKCA should not pass the natural lexical entity identity into the factual value-emission path after Symlink resolution. The resolver supplies the canonical Pod/capability, and the verified Port supplies current mutable value state. Revocation admits no value page and routes to a declared null state rather than allowing the generative path to fall back to lexical parametric recall.

This yields a stronger separation than prompt instructions such as “prefer external memory”. It is a namespace/access-path rule:

`natural alias -> resolver -> canonical pod_id -> verified current generation -> model-native value/J state`

The alias itself does not remain an alternate factual address inside governed execution.

## Claim boundary

Entity anonymization/aliasing, vocabulary/output embeddings and shared transforms are established techniques. The result is evidence that a **parametric-bypass firewall is implementable** inside the larger CKCA lifecycle contract; none of the individual ingredients is claimed as the invention.

The test is limited to single-token values and assumes a correct external resolver. Complex reasoning over multiple governed entities requires the J-Space path rather than direct value emission.
