# Reference-carried memory — withdrawn, and what the attempt measured

Date: 2026-09-05
Status: **withdrawn as a novelty claim.** Twelfth retraction. What survives is a negative result and a
corrected record. Nothing here is a legal novelty or patentability opinion.

An earlier version of this document claimed that a knowledge-free reference, carried through a frozen
model's participating state with the value bound after the last cache-writing block, made persisted
state invariant under every lifecycle operation, and that this point was unoccupied. Both halves were
wrong. The claim is withdrawn under both of the programme's kill modes, and two of the six
claim-killing tests this document itself listed have fired on measurement.

## What was wrong, in the order it matters

**1. The invariance is not the carrier's contribution. It belongs to the write placement, and the
baseline has more of it.** Measured on the same substrate, with the same mutations, three arms
(`so/tests/test_write_layer.py`, and reproduced independently before being accepted):

| Store mutation | arm E, reference carrier | arm C, late write, no carrier | arm A, write in place |
|---|---:|---:|---:|
| payload UPDATE | **0.0** | **0.0** | 7.3e-02 |
| marker SHRED | **0.0** | **0.0** | 3.7e-03 |
| REVOKE (row made unroutable) | **7.1e-01** | **0.0** | 8.9e-02 |
| DELETE and compact | **3.7e-03** | **0.0** | 1.1e-02 |
| reorder a fixed row set | 6.7e-08 | **0.0** | — |

Arm C injects nothing before the final block, so nothing about the bank reaches a cache-writing
block and *no* store mutation moves its persisted state, while its answer still changes. The carrier
does not create the invariance; it inherits it from the placement and then **degrades** it, because
what rides is a routing mixture normalised over every routable key plus null, so changing the
namespace renormalises the mixture for prompts that never addressed the changed row.

**2. The document asserted the opposite of that, and its test could not catch it.** The line
"DELETE affects only references to that identity" was false at the level of persisted state. The test
that was supposed to cover this was named for reordering, compaction and growth and only ever applied
a permutation to a fixed row set — an instrument that could not fail, which is the programme's own
standing rule broken for the third time. The suite now measures and pins all five mutations for all
three arms, including the two rows where the carrier is **not** invariant.

**3. The capability the whole claim was conditioned on came back 0.0.** Arm E, seed 0, 3000 steps,
identical recipe: held-out candidate correctness **0.0000** on all four templates, against arm A at
0.955/0.990 and arm C at 0.621/0.645/0.664. Record kept at
`so/results/e000084_armE_s0_UNSUPERVISED_BIND_FAILED.json`.

The cause was diagnosed, not excused. The carrier is not at fault: E-000085 shows an injected
knowledge-free handle is linearly recoverable at the boundary of frozen GPT-2 at top-1 **1.0000** on
held-out prompts over 1024 identities (clean control exactly at chance), and on Pythia-70m at 0.972
once the injection amplitude reaches 4× the residual RMS. What failed was supervision: arm E has one
addressing decision the other arms do not — which row the transported handle names — and it had no
training target, while every addressing slot in arms A/C/D is supervised. The signature is in the
failed record: updating every payload moved the logits by 0.23 while shredding moved them by 9.6,
which is a boundary distribution that never concentrated. `bind_supervision` now trains it the way the
other arms' addressing is trained, and that rerun is in flight. **A pass would not revive the novelty
claim**; it would only decide whether the design reads at all.

**4. The point was occupied, by granted patent art.** The claim asserted that every neighbour either
puts knowledge into persisted state and repairs it, or keeps it out by not participating. That was
false.

## Prior-art boundary, corrected

Nearest neighbour, named: **Salesforce Inc., US12505252B2, "Generative responses with trust for large
language models", granted 2025-12-23** (verified: https://patents.google.com/patent/US12505252B2/en).
A knowledge-free surrogate is substituted for the value, the frozen model computes over the surrogate
through every layer — so it enters persisted state more thoroughly than a mid-stack injection — the
mapping is held outside the model, and the true value is bound back only after generation. That system
already has the asserted invariance; it goes unstated there because in that setting it is obvious. The
entire delta of this claim over it was carrier-as-mid-stack-vector rather than carrier-as-token, and
that delta currently measures 0.0.

Also occupying or narrowing the same point, all verified by fetch:

- SurrogateShield, arXiv:2606.29567 — content-free surrogates through the whole frozen forward pass, map held off-device, originals restored after generation.
- Hide and Seek, arXiv:2309.03057 — pseudonym substitution with post-hoc de-anonymisation.
- LLM-Redactor, arXiv:2604.12064 — typed stable placeholders, exact-match restoration on the response path.
- Prompt Cache, MLSys 2024, arXiv:2311.04934 — content-free placeholders occupying reserved positions in **cached prompt state**, value bound per request.

Owning the ingredients, all verified by fetch:

- Feng & Steinhardt, ICLR 2024, arXiv:2310.17191 — additive content-independent binding-ID vectors in a frozen LM's activations. It also reports that *random* vectors are not valid binding IDs, which is a risk to this construction rather than a collision.
- LLM Self-Recognition, arXiv:2606.06315 — a random semantically-empty vector injected into a frozen residual stream, participating and linearly recoverable downstream at >98%. **This owns E-000085's result outright.**
- Yang, Campbell et al., ICML 2025, arXiv:2502.20332 — abstract variables mid-stack, concrete value rebound at readout.
- Retrieval-conditioned rebinding, arXiv:2606.08644 — a relink applied at readout over content-free binding IDs.
- PANM, arXiv:2404.11870 — external neural memory with explicitly content-free address vectors, pointer assign and dereference.
- Pichay, arXiv:2603.09023 — prior published use of "late-binding retrieval handle" and of the consequence that a handle resolves to current content so cached context is never stale.
- KEEP, arXiv:2602.23592 — occupies "avoiding KV invalidation caused by memory updates", by separation plus recomputation.
- vLLM automatic prefix caching — production caches are keyed by chained hash over exact token ids; version-stamped keys are the field's discipline, i.e. the lineage metadata this claim positioned itself against.

Material and **unverified**: Anonos US12417317B1 (claims not rendered); Google US20240403564A1 / 12,417,356, whose snippet describes "a special token identifying the particular personal repository containing the personal data" — the most patent-shaped lead found, and it returned HTTP 503 twice; Byte-Exact KV-Cache Grafting, arXiv:2607.14431.

**Structurally unsearched: patents.** Google Patents full text was the only reachable corpus. No USPTO
Public Search, Espacenet, WIPO Patentscope or Lens.org query was run, no CPC sweep, no CN/JP/KR
literature, and the 18-month publication window is open. This no longer changes the verdict — the
claim dies on its own measurements — but the boundary must never be described as cleared.

## What survives, and it is a negative result

> On a frozen GPT-2 with an external mutable memory, keeping the memory out of the persisted attention
> state costs the reading capability. Moving only the write to after the last cache-writing block while
> keeping addressing deep at blocks (8, 10) gives held-out candidate means of 0.664 / 0.645 / 0.621
> across three seeds with K/V exposure verified at exactly 0.0, against 0.955 / 0.990 for the identical
> recipe writing in place (E-000084 arms A and C, run 33970654975). The pure late write is moreover the
> strictly stronger invariance: it leaves every persisted tensor bit-identical under payload update,
> marker shred, revoke, delete-with-compaction and reordering alike, where the reference carrier moves
> the cache by 7.1e-01 under revoke and 3.7e-03 under deletion.

That refutes, on measurement, the promotion of fixed late binding to "a mandatory strong baseline any
lifecycle mechanism must beat" in the E-000082 note: it is not a baseline a mechanism must beat, it is
a placement that already gives total invariance and cannot read.

## What is still worth running, and why it is not a claim

1. **Arm E with `bind_supervision`, three seeds.** Decides whether a mid-stack knowledge-free carrier
   can read at all. Positive would make the design viable and still leave it inside Salesforce's
   granted scope at the concept level; negative closes the design.
2. **Confirm the arm E collapse on seeds 1 and 2 and audit the eval path.** Exactly 0.0 is below
   candidate-restricted chance and warrants a sanity check independent of the supervision fix.
3. **A real patent search.** Nothing above may be called cleared until USPTO, Espacenet, Patentscope
   and a CPC sweep have been run.

## Errors in this document's earlier version, listed so they are not repeated

- claimed DELETE affects only references to the deleted identity — false, measured at 3.7e-03;
- claimed the reordering test covered compaction and growth — it applied a permutation only;
- reported an exposure of 2.31 as characterising the carrier — that figure came from a 4-step untrained
  smoke run; the trained run's exposure is 6.50;
- listed "a lifecycle status change" among the bit-identical operations — untested, and measured at
  7.1e-01 once a revoked row is made unroutable;
- asserted the design point was unoccupied without having run the search that found it occupied.
