# The claim: canonicalisation is an erasure–disclosure duality

*2026-09-04. One claim, its evidence, and the prior art it has to survive. Everything here is measured
in this repository; every citation was read at the source and the sentence relied on is quoted.*

## The claim

Let a fact be reachable through **k access paths**. There are two ways to hold it:

* **duplicated** — k independent records, one per path;
* **a canonical pod** — one record holding the object, and k−1 pointers to it. This is the symlink.

The two arrangements present an **identical interface**: every key resolves to the same object in
both, which is checked before anything else is measured. Their erasure costs are not identical. They
are **exactly inverted**:

| guarantee you want | canonical pod | duplicated |
|---|---|---|
| the fact is **unreachable** to the reader | **1** record | k records |
| the deletion leaves **no trace** in the store | k records | **1** record |

And at one record removed — the deletion the pod exists to make cheap — the pod **names the deleted
key**. An adversary who reads the store and nothing else, with no model, no query access and no
before-and-after comparison, identifies it uniquely:

| store | deleted key disclosed | uniquely identified | candidate keys left | false positives |
|---|---|---|---|---|
| **canonical pod** | **1.0000** | **1.0000** | **1 of 1,536** | 0.00 |
| duplicated | 0.0000 | 0.0000 | 1,536 of 1,536 | 0.00 |

Three seeds, 100 pods per seed, every pre-registered criterion passing (E-000035). The mechanism is
one line of the store: `MVCCStore.bank()` builds a LINK row's `link_subject`/`link_relation` from the
target cell, and keeps doing it after the target is evicted or deleted — deliberately, so the model
has to *discover* the miss rather than be handed it by the control plane (E-000015's recorded design).
Every surviving alias is therefore a signpost reading **a record stood at (s, r) and is gone.**

**So the sentence this programme wanted — *canonicalisation makes erasure a single certifiable
operation* — is half of what that design decision does. The other half is *and it turns every access
path into a deletion oracle*.**

## Why it is not owned

**Raeesi & Roed**, *Auditing Forgetting in Limited Memory Language Models* (arXiv:2607.00605), §9,
verbatim: *"A second direction is canonicalization at write time, in which aliases and paraphrastic
forms are stored as pointers into a single canonical record rather than as independent triplets."*
The next sentence is *"Both approaches are directly testable within our framework."* Proposed, called
testable, not tested — and no disclosure analysis anywhere in the paper, which reports post-deletion
correctness and no contingency-set size.

**Database resilience** — the minimum contingency set, with its PTIME/NP-complete dichotomy — computes
the cost of falsifying a query. It asks nothing about what the *surviving* database reveals about the
deletion that was performed. The metric is theirs and is used here under their name; the question is
not in it.

**Deletion inference** — Chen, Zhang, Wang, Backes, Humbert & Zhang, *When Machine Unlearning
Jeopardizes Privacy* (CCS 2021, arXiv:2005.02205), and Gao et al., *Deletion Inference, Reconstruction,
and Compliance in Machine (Un)Learning* (arXiv:2202.03460) — recovers the deleted record from **the
discrepancy between the model before and after unlearning**. Two snapshots, and a model. This channel
needs **one snapshot and no model**, and it exists *because of the normalisation choice* rather than
because of an update. Removing the record is what creates it.

## Why it matters outside this repository

The channel is not a property of `so/mvcc.py`. It is a property of **any store that keeps a dangling
reference after deleting its referent**, and that is the default almost everywhere: a soft-deleting
vector index that keeps its edges, a knowledge graph with `owl:sameAs` pointing at a removed node, a
deduplicating store whose manifest still lists the chunk hash, a foreign key with `ON DELETE
SET NULL`. Each of those turns "was there a record about this person, and was it deleted" — precisely
the question an erasure guarantee is supposed to make unanswerable — into a lookup.

## The mitigation, and its exact price

Blanking the dangling pointer's key closes the channel at **1.0000** and makes every such pointer
identical at **1.0000**. It also destroys what E-000015's alias criteria are about: with the key
blanked, an alias to a removed target is indistinguishable from an alias to key (0, 0), so
`delete_target/alias_unknown` stops being a discovery about the model and becomes a tautology about
the bank. The trade is recorded as a number rather than argued.

## What the claim does not say

It is a property of *this* store's bank; one that compacts its aliases on deletion, or never exports
the target key, has no such channel — which is the point, because that is a design choice a system can
now make knowingly. It measures what a reader of the store can see, not what a model exposes to
someone who cannot read the store. It says nothing about whether the disclosure matters in a given
threat model. And the *first* row of the inversion table — unreachable in one record — is Codd's
modification anomaly applied to a delete, and the remedy for it is 1971; only the second row and the
inversion are new.

## The supporting structure, which is also not free

The claim above is about a store. Making it a guarantee about a *system* needs the composition, which
is the other thing built here and which nobody else can assemble yet:

```
record-level certificate over R   +   R covers the fact's closure
-----------------------------------------------------------------
                    fact-level certificate
```

`so/audit.py::certify_fact` performs it and carries what a fact-level verdict must carry or it is not
a verdict: the **query workload** it was proved over (the default is one single-hop question per key;
a fact a multi-hop derivation still reaches is outside it, and E-000019 records that case at
`derivable_recovery_after_revoke_K3 = 1.0`), the **individuation** (the subject-relation-object triple,
not the object's value under another subject), and whether the payload is **unreachable or actually
erased** — under EVICT the store still holds it, on purpose, which is what makes RESTORE work.

Nobody else composes because nobody else has the left-hand side: E-000030 is the first deletion
certificate in this line of work, and it exists only because knowledge here lives in rows with a
finite payload domain that can be swept exhaustively.

Building it cost three corrections that are part of the claim's credibility rather than footnotes to
it. `certify_structural` on an empty row set returned the ladder's strongest label — "no path, for any
value over any domain" — on a bank whose rows were **live**, and E-000030's recorded
`delete/structurally_certified` came from exactly that call. `certify_encoding` on an empty row set
certified in one evaluation. And `check_absence` tested membership only, while `bank()` builds an
alias row's link key from the target cell, so a surviving row is a function of the removed one and a
membership test cannot see it. All three are closed, and the re-run records DELETE on evidence that
could have failed: `delete/control_reachable_before` 1.0000 at |grad| 3.3e+01 before the removal,
`delete/payload_absent` 1.0000 after.

**The rule, for the third time in this programme after §31.10 and §31.13: an instrument that cannot
fail is not evidence.**

## Reproduce

```bash
./run.sh disclosure     # E-000035, the claim. No model, no checkpoint, seconds.
./run.sh closure        # E-000032, the store-side closure and the composition.
./run.sh certify        # E-000030, the record-level certificate it composes with.
```
