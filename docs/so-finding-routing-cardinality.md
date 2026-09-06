# The finding: §31.41's cardinality attribution is architecture-specific, and one line decides it

*2026-09-05. A correction to a recorded attribution, with the mechanism traced to code. This is a
finding about this architecture, not a novelty claim, and it is filed as `so-finding-*` for that
reason. Every number is from `so/results/e000058_routing_cardinality_floor*.json`.*

## What was on record

§31.41 recorded that the reader's off-pod outputs carry the number of active rows, and attributed it:

> "The reader's off-pod outputs carry the number of active rows — **every row sits in every routing
> softmax and the dereference pass-through bias is `log(n_cells)`** — as a deterministic,
> sub-milli-nat shift a standardised probe reads perfectly."

`e000051_residue_reader.py` carries the same sentence in its pre-registered rule: *"a property of
dense routing and not of history"*. It was an attribution, never a measurement, and nothing in the
repository commissioned a test of it — checked by grep across `docs/` and `so/` before this ran.

## The test

E-000058 restricts the routing softmax to its top-k support and changes nothing else — not the
`allowed` masking, not the forward/reverse selection, not the value mixture, no retraining. Both read
sites are patched, because the recorded sentence names the *dereference* bias specifically. Every arm
and AUC is then computed by calling E-000051's own `run_reader_seed` under the patch, so the dense arm
is a reproduction of the record rather than a reimplementation of it.

Three guards decide whether an arm may be read at all, and all three earned their place by firing:

- **`present/auc_i >= 0.95`** — a reader that cannot read is not a reader.
- **patch-effectiveness** — a sparsified mode whose metrics reproduce dense to the last bit did not
  run. The first GPT-2 attempt returned four bit-identical modes and reported them as a result,
  because `KnowledgeAdapterLM` routes inside a closure that a class-level patch cannot reach.
- **the float floor** — E-000051's own rule, *"if perm AUC > 0.60 every AUC row is void"*, which this
  experiment initially failed to carry over. It voided a conclusion already published to the owner.

## Synthetic reader — three seeds, all four modes valid (worst perm 0.5804)

| mode | cells in mixture | `add2` (row-count channel) | `cascade` (marker channel) |
|---|---|---:|---:|
| dense | ~850 | 0.9782 | 0.9491 |
| top16 | 16 | 0.9613 | 0.9507 |
| top4 | 4 | **0.9833** | 0.9602 |
| top1 | 1 | **0.5001** | **0.9585** |

**The attribution fails here.** Shrinking the mixture by a factor of 200 does not dilute the channel —
it rises. `log(n_cells)` predicts decay with the normaliser's size and there is none, at any k, until
the mixture is removed outright.

**And the two floors dissociate.** At argmax every arm falls to chance — `add2`, `blank`, `dangle`,
`perm` all ≈0.50 — while the marker channel holds at 0.9585. The row-count channel is a **mixture**
artefact; the marker channel is a **selection** artefact. Argmax closes the first and leaves the
second; content-derived markers (E-000053) close the second and leave the first. Neither closes the
other.

*Conceded, before a reviewer says it:* the `top1` endpoint for `add2` is true **by construction** — at
k=1 the output is invariant to every non-argmax row. That endpoint is a confirmation, not a discovery.
The empirical content is the flat region, and the dissociation is the *converse* of that construction:
by the same argument any channel carried only by the mixture **must** collapse at k=1, and `cascade`
does not. That is what licenses calling it a selection channel.

## GPT-2 reader — one seed, dense VOID

| mode | `add2` | `cascade` | perm_ii | perm_iii | status |
|---|---:|---:|---:|---:|---|
| dense | 0.9304 | 0.9066 | 0.4748 | **0.6045** | **VOID** |
| top16 | 0.9338 | 0.8968 | 0.4751 | 0.5344 | valid |
| top4 | 0.9583 | 0.9285 | 0.5111 | 0.5756 | valid |
| top1 | **1.0000** | 0.9746 | 0.5553 | 0.4667 | valid |

The dense arm is void even at 100 pods, so **no flat-region claim is made for this reader and nothing
is anchored to that baseline.** What is readable is the ordering among the valid modes, and it runs
the opposite way: `add2` **rises** with sparsification and reaches 1.0000 at argmax, where the
synthetic reader fell to 0.5000.

## The reconciliation, and it is one line

`so/llm_adapter.py:281`:

```python
bias = self.deref_pass_bias[read_index] + float(np.log(n_cells))
```

The GPT-2 adapter puts `log(n_cells)` **directly** into the dereference pass bias. The synthetic model
has no such term. So at argmax the synthetic reader's output becomes invariant to non-winning rows and
the mixture channel disappears; the GPT-2 reader's null-column bias still moves with the row count, two
added rows shift whether the null column wins, and the result is a discrete output change — a
cardinality channel that is **not** a mixture channel, and that sparsification concentrates rather than
removes.

**So §31.41's attribution is wrong for the synthetic reader and right for the GPT-2 one, and the
deciding feature is that line.** The explanation is architecture-specific, not a property of routed
memory in general.

## What this is, and what it is not

It **is**: a correction to a recorded attribution, a mechanistic account of the discrepancy traced to
a named line, a measured dissociation of two nuisance channels on the synthetic reader, and three
guards the harness did not have.

It is **not** a technical novelty. It is a finding about this architecture. Nothing here would stand
alone as a contribution to the field, and it is not filed as one.

## What would settle it

1. **The causal test of the reconciliation.** Ablate the `log(n_cells)` term from the GPT-2
   dereference bias and re-run. If `top1/add2` falls from 1.0000 toward chance, the one-line
   explanation is measured rather than inferred. This is the single most valuable next run and it is
   not done here.
2. **A cardinality-matched arm.** Add two rows *and* remove two, holding `n_cells` fixed. If the GPT-2
   channel is cardinality, that arm is at chance; if it is content, it fires. Independent of (1).
3. **GPT-2 seeds 1–2** (~90 min of adapter training each), and whatever raises `perm_iii` off 0.6045
   so the dense baseline becomes readable.
4. **`top2`** on the synthetic reader, to locate whether the mixture channel is all-or-nothing between
   k=1 and k=4.
