# E-000110 Result — Lifecycle Equivariance and Edit-Group Normalizer Reduction

Status: **DECISIVE FALSIFICATION / STANDALONE NOVELTY SEAM CLOSED**

Decision:

`KILL_LIFECYCLE_EQUIVARIANCE_AND_GROUP_NORMALIZER_AS_STANDALONE_EXACT_TRANSPORT_ADVANTAGE`

Major invention promoted: **no**.

## Registered evidence identity

- preregistration + implementation source commit: `69a137136ee00446ada296cee097f1cbdff1c7b2`
- GitHub Actions run: `34019683776`
- run conclusion: `success`
- artifact id: `9985056391`
- artifact digest: `sha256:b3c742ef041ee874e09ee3c37c5af3241296a202ed1f5498e6347b7f6051675e`
- preregistration source SHA-256: `c225d59fc9af2c4f882ad1cdce385cffcfc6d75585d8268827eac4e28cde78b6`
- experiment source SHA-256: `289a4025d71a98b5fbdac29c18f0db9cf43a114c2531c378390b6736093df890`
- focused-test source SHA-256: `cad0cab629cf42d9766b8034f6078ec52734c04f310614062311638a6d6ea8fc`
- result JSON SHA-256: `aa567e18cf0a6a3c7877146295354824ffab4b5ecba6f3e298b387675079a0b1`

## Registered exact result

The candidate received the strongest clean version of the surviving E-000109 escape.
A canonical read-site Pod edit began as a compact translation and was propagated
through exact nonlinear bijections that normalize a three-parameter edit-action
family.  The translation becomes state-dependent after nonlinear transport while
remaining in the bounded family.

The candidate therefore achieved exactly the topology we wanted: compile one shared
mutation action once, then apply it directly to many stale session states without
inverse reconstruction or per-session suffix replay.

Registered totals:

| Measure | Result |
|---|---:|
| network cells | 192 |
| edit cells | 576 |
| exact state-transport cases | **111,168** |
| candidate -> fresh mismatches | **0** |
| generic -> fresh mismatches | **0** |
| candidate/generic action mismatches | **0** |
| independent probe-compiler action mismatches | **0** |
| ABA failures | **0** |
| materially changed cases | **111,168 / 111,168** |
| edit cells where a state-dependent action appeared | **576 / 576** |
| candidate normalized work | **115,488** |
| generic normalized work | **115,488** |
| candidate/generic work ratio | **1.0** |
| suffix-recompute layer evaluations | **833,760** |
| aggregate replay/candidate normalized-work ratio | **7.219451x** |
| weakest registered replay/candidate ratio | **1.967480x** |
| strongest registered replay/candidate ratio | **15.160656x** |

The independent probe compiler did not use either analytic transport function: it
reconstructed every one-layer conjugated action from exact black-box evaluations of
`F o g o F^-1` and verified held-out probes.  It agreed with the candidate and the
work-matched generic engine in all 576 edit cells.

## Why this is a kill despite the strong transport result

The candidate itself is useful in the toy structural domain: it is exact, its edit
action becomes genuinely state-dependent after nonlinear computation, and at fleet
scale it is much cheaper than replay.

But the complete advantage is inherited by an ordinary group-action/equivariant
program engine once it receives the identical compact group representation and the
identical layer-induced automorphisms.  It performs exactly the same number of
shared transport steps and exactly the same number of per-session action
applications, yielding the same fresh state in every case.

Therefore the technical advantage belongs to **generic group equivariance /
normalizer algebra**, not to Pod lifecycle semantics.

Standalone claims of the form below receive zero major-invention credit after
E-000110:

- make Pod edits a compact transformation group;
- make every layer equivariant to that edit group;
- require every layer to normalize the edit group;
- propagate a bounded edit-group element instead of replaying the suffix;
- compile one transformed group action and apply it across a cached fleet.

A future architecture may still use equivariance as a component, but it must earn a
new neural-specific systems advantage that a guarantee-matched generic group-action
engine given the same representation cannot reproduce.

## Fresh literature / patent boundary checked on 2026-09-06

The targeted search did not surface a paper or patent that establishes the narrow
FinX target of one knowledge mutation producing an exact reusable correction
receipt across arbitrary already-cached neural sessions.  This is a search result,
not a freedom-to-operate opinion.

The broader equivariance machinery is unquestionably prior art.  Group-equivariant
neural networks have been patented since at least the 2016 priority family around
`WO2017142397A1 / NL2016285B1`.  `US20250094797A1` (published 2025-03-20) claims a
system for constructing networks equivariant to arbitrary matrix groups using group
representations and equivariant tensor fusion.  Recent research also continues to
make group structure explicit: Any-Subgroup Equivariant Networks (arXiv:2603.19486,
2026) builds one architecture configurable for multiple subgroup equivariances, and
Transformer-NFN (arXiv:2410.04209) explicitly derives transformer weight-space
symmetry groups and group actions for equivariant neural functionals.

The exact-mutation literature continues to reinforce the representation/replay
boundary rather than provide the missing reusable receipt.  `Subtract or Replay?
Exact Deletion from Language-Model Memory` (arXiv:2607.27539) reports that later
delta-rule writes make 12--49% of a record contribution suffix-dependent in native
Kimi and uses checkpoint replay to recover the record-omitted state bit for bit.
`KVEraser` (arXiv:2606.17034) obtains speed by learned local KV steering and is an
approximate behavioral repair rather than exact post-mutation state transport.
`KV-Direct` (arXiv:2603.19664) makes exact residual-stream reconstruction a strong
baseline by reporting bit-identical KV reconstruction across its tested families.
`Forgetful Attention` (arXiv:2607.12204) demonstrates a deliberately addressable
support-vector memory with decremental deletion, but its exact-update mechanism is
maintained classical solver/KKT state and therefore belongs in the generic dynamic
algebra baseline family rather than supplying a new cross-session neural transport
receipt.

## Surviving frontier

E-000110 closes the cleanest version of **lifecycle equivariance / edit-group
normalizer transport as the invention itself**.

The remaining target is stricter: a lifecycle-native architecture would have to
preserve a bounded exact mutation representation through useful nonlinear neural
computation **while deriving an advantage not obtainable by an ordinary engine that
is handed the same sufficient representation**.  Equivariance, reversibility,
provenance, low rank, sparse recomputation, symbolic algebra, tensor factorization,
piecewise-affine region caches, Woodbury/RLS, and generic dependency/incremental
computation remain baselines, not novelty.

No claim advances to the real LINK->Pod / two-backbone / three-seed / leakage /
UNKNOWN / generic-divergence / stale-state attack / lifecycle operation / J-space
and J-lens / <=5% steady-state-overhead promotion gates on E-000110.
