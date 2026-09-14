# R316 Result — Neural Lifetime Type System (NLTS)

Status: **static lifecycle-safety gate passed; not DoD and not a standalone novelty claim.**

## Scale

- 250,000 valid CKVM programs
- 250,000 adversarially mutated programs
- six injected lifecycle-violation classes

## Result

The CKVM temporal type checker classified registers as `static`, `plan`, `borrowed`, or `sealed` and enforced three architecture rules:

1. generation-bearing values may not enter the reusable B-plane cache;
2. retained J artifacts may not escape a knowledge transaction without an exact lifetime seal;
3. publication requires a Neural Commit Barrier covering the exact dynamic generation read-set.

Observed:

- valid-program false rejects: **0 / 250,000**
- lifecycle-leak mutant false accepts: **0 / 250,000**
- lifecycle-leak detection rate: **100%**
- mean Python type-check cost: **8,580.7 ns/program**

Injected failures included borrowed-to-B-cache leaks, unsealed retained J state, incomplete lifetime factors, incomplete commit read sets, publication without commit, and explicit lifetime erasure.

Report SHA256: `6102bdbe87e3184631b6dff645f5c409c9b87d3b51f79ff83f0398bfd9d7f780`.
Artifact ZIP SHA256: `a1c8c41dbe90b7e0d1e93e53a4009086d1b4ebf6ea0a65e996bf568f70e6156d`.

## Architectural decision

Promote **NLTS / temporal borrow checking** into the CKVM compiler contract.

The runtime coherence machinery should not be responsible for catching every accidental lifecycle leak after execution. The compiler must make several classes of invalid architecture literally untypeable:

`mutable generation -> reusable B cache` is illegal.

`borrowed J -> retained artifact` is illegal until the exact generation dependency set has been sealed into a lifetime factor.

This does not replace runtime commit validation or CLFD; it constrains what programs may reach them.

Type systems, taint tracking, borrow checking and information-flow control are established ideas. The evidence here supports their CKVM-specific composition, not standalone novelty.
