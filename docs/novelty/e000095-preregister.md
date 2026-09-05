# E-000095 — semantic address plane + exact mutable payload plane

Status: **preregistered before execution.** This is a strong ordinary baseline, not a novelty claim.

## Motivation

E-000091 shows that the present softmax reader gives every routable bank row positive forward support, so an unrelated mutable Pod update changes routing, hidden, KV and logits even when the selected-Pod witness remains current. E-000092 shows the converse: post-hoc one-hot resolve+deref makes an unrelated payload mutation byte-identically invisible, but one completed seed loses 0.11 held-out accuracy on template 9 (0.945 -> 0.835) and fails the >=0.95 reader gate.

The question is therefore whether linguistic ambiguity can remain soft **without allowing mutable knowledge values or mutable alias->Pod bindings into that dense computation**.

## Baseline architecture

For every bank row j, define a stable semantic identity key k_j from subject/relation. At each neural read layer:

1. **Soft semantic address:** the text query attends over identity keys. The value used by this stage is a projection of the row's *own identity key*, never its object payload and never its alias target.
2. **Exact row identity:** a learned second query selects exactly one row (or NULL) in the executed forward pass using straight-through one-hot. This is an established estimator and receives zero novelty credit.
3. **Deterministic Symlink jump:** if the selected row is an alias, a control-plane map takes that exact row to its current canonical target row; a direct row maps to itself; a dangling target maps to NULL. The map is not mixed neurally.
4. **Exact mutable payload:** only the mapped target row's current payload is injected. The mutable support is therefore the selected identity row plus its mapped canonical target, not every row receiving nonzero semantic-attention mass.

This explicitly separates a dense linguistic **address plane** from the mutable **knowledge plane**. Decoupled addressing/storage, hard routing and pointer following are prior-art/ordinary ingredients and receive no novelty credit.

## Why this is stronger than E-000094

E-000094 makes the text-to-row decision itself one-hot. E-000095 leaves the difficult paraphrase/alias recognition stage soft, then trains a second exact identity decision from the resulting address representation. It tests whether exact mutable support can coexist with linguistic capability without requiring every semantic similarity weight to become a lifecycle dependency.

## Fixed validity gates

Three independent training seeds 0/1/2, 3000 steps, 100 symlink groups, BOS enabled, unchanged marker radius 0.35.

A seed is feasible only if all hold:

- candidate AND full-vocabulary held-out accuracy >= 0.95 on every template 8..11;
- exact no-memory bypass max-abs == 0;
- executed identity-selection forward support has <=1 positive real row per slot;
- for an unrelated canonical **payload UPDATE**, when selected identity and mapped target are unchanged: routing identity, hidden, all available KV tensors, full logits and stale-KV continuation are byte-identical;
- for an unrelated **alias RELINK**, when the queried alias and its target are unchanged: the same neural-state objects are byte-identical;
- relevant target UPDATE and queried-alias RELINK must change the current read and still produce the current truth (not merely reject stale state).

If the bank exporter physically removes unrelated rows on SHRED, that case is reported separately rather than hidden: exact reuse then requires a stable/tombstoned identity plane and E95 is not allowed to claim SHRED locality without it.

## Falsifiers

- Any seed below the >=0.95 per-template gate falsifies this implementation as the project reader substrate.
- Any mutable value/binding from an unselected row changing the materialized neural state falsifies the claimed exact mutable-support boundary.
- Byte equality without correct current answers is insufficient.
- A speedup against full replay alone is insufficient; eventual utility must beat strongest ordinary decoupled-address/selective-recompute baselines at matched capability.

## Claim boundary

E-000095 cannot establish novelty. Even a full pass only establishes a capable baseline substrate on which a later **lifecycle-shaped support training objective** may be compared. No CAVI breakthrough, J-space claim, SHRED guarantee, patentability claim or multi-backbone result follows from E95 alone.
