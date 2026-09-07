# PDX-002 — the audit run against a real vector index

**§13(c) of the draft, executed.** It named one experiment as the thing that would decide whether
this programme's findings matter outside itself: run the instrument against a *published* system
rather than a reconstruction. PDX-001 was restructured to make that configuration rather than a
rewrite. This is that configuration.

**System under test:** `hnswlib` **0.8.0**, the reference HNSW implementation, installed from PyPI
and used as documented. Nothing patched, monkeypatched or reimplemented. l2 space, 8 dims, 64 rows,
M=16, ef_construction=200, payload domain 256, 16 targets.

## Result

| policy | top-1 | candidates left | posterior | certified |
|---|---|---|---|---|
| live (validity floor) | **1.0000** | 1.00 | 1.000000 | no |
| `mark_deleted` + public API | **1.0000** | 1.00 | 1.000000 | no |
| `mark_deleted` + disk access only | **1.0000** | 1.00 | 1.000000 | no |
| rebuilt without the row | 0.0000 | 256.00 | 0.003906 | **yes** |

**Decision: `REAL_INDEX_TOMBSTONE_IS_NOT_AN_ERASURE`.** Floor met, control clean.

## What this says, precisely

The library behaves exactly as documented, and the API is *consistent*: after `mark_deleted`,
`get_items` raises `Label not found` and `knn_query` never returns the label — including after a
save/load round trip. A reader who expected the naive "soft delete still serves the payload" failure
does not get it.

What survives is the tombstone itself. `unmark_deleted` is a **documented public call**, and it
returns the payload exactly. No file parsing, no exploit, no reverse engineering. The exact bytes are
also still in the serialised index, which is the arrangement an adversary with disk access but no
process meets.

**The real library is a stronger result than the reconstruction, and in a different direction.**
PDX-001's `hnsw_tombstone` policy modelled the tombstone as a *partial* channel — the retained
adjacency narrowing 256 candidates to 2.92, a posterior of 0.391667. That was a faithful model of
one shape a tombstone can take. The reference implementation does not need a partial channel:
recovery is **exact**, at top-1 1.0000, through the public API. The reconstruction understated it.

## What is NOT claimed

**This is not a vulnerability report, and hnswlib is not doing anything undocumented.**
`mark_deleted` is specified as a reversible tombstone and is shipped paired with `unmark_deleted`.
That is the correct primitive for capacity reuse and index maintenance, and a library that offers it
is not thereby broken.

The finding is about **deployments that serve a deletion request with that primitive**. If a "delete
my data" request is discharged by `mark_deleted`, the payload is one documented call away for anyone
who reaches the index, and is in the file for anyone who reaches the disk.

No hosted service was touched. No vendor product was tested. Nothing was reverse-engineered.

## A defect this experiment found in itself

The first draft's embedding used `% 251`, which maps 256 payloads onto 251 vectors — five colliding
pairs. The run still reported top-1 1.0000, because the sixteen targets happened to miss every
collision. **Right by luck.** Its own injectivity test caught it; the modulus is 257 now, prime and
larger than the domain, so the k=0 component alone is injective and the attack is well-posed by
construction rather than by sampling.

That is the fourth instrument in this branch to fail its own test before reporting, which remains
the only evidence available that any of them can.

## What would have falsified it

The live arm had to recover at 1.0000, or the at-chance readings elsewhere would mean nothing. The
rebuilt arm had to sit at chance with the whole domain open, or the instrument would be reporting a
channel that is not there. Both held.

## What this still does not settle

The audit measures what an adversary can recover from the index. It does not show that any deployed
system serves deletion requests this way — that is a claim about operators, and this experiment
cannot make it. §13(c) asked for the instrument to be pointed at a real system; it has been, and the
result is about the primitive, not about anyone's use of it.
