# Certified deletion for memory-augmented models: the store-side closure it must compose with, and the disclosure channel canonicalisation opens

**Status: DRAFT. Not submittable as it stands.** Two experiments named in §11 have not been run, and
every external citation below is a *lead*, not a checked fact — see §0.

---

## 0. Two things a reader must know before the abstract

**External citations are unverified.** The literature positions in this draft descend from a 41-agent
literature workflow recorded in `docs/so-novelty-2026-09-04.md`, whose own provenance note says: its
claims about this repository were checked against `so/results/*.json` and match; its claims about the
external literature — titles, authors, venues, arXiv identifiers, dates, reported numbers — were
**not** independently verified, and several cited works postdate the model's training data. Nothing
in the environment this draft was written in has network access. **Every citation must be checked
before submission.** Where the sources say "verified", that is the workflow's word.

**Two experiments block honest submission.** They are named in §11 and neither is optional.

Every number attributed to this repository below is from a committed record and is reproducible by
the `make` target named beside it.

---

## Abstract

Machine unlearning for memory-augmented language models is evaluated adversarially: delete, run
extraction or membership attacks, report that they failed. That standard bounds the adversaries
someone thought of. We report a clean instance of it failing — a deletion primitive that passed four
attacks at chance over 750 pooled trials and surrendered the deleted object at top-1 1.0000 to a
fifth attack written afterwards, through an index term derived from the very payload it destroyed.

We then give the constructive alternative and its cost. Where knowledge is held in rows with a small
payload domain, independence of the model's computation from a deleted payload is *provable*
exhaustively: sweep every value the payload could hold and check nothing downstream moves. On a
frozen 124M-parameter reader this takes under a second and certifies three deletions that the attack
standard could only ever have called "not yet broken".

Three results follow that a certificate alone does not give. (i) A record-level certificate is not a
fact-level guarantee: it must compose with a *store-side closure*, which is the database literature's
**resilience** and which we compute with a certified lower bound. (ii) The arrangement that makes
erasure a single certifiable operation — canonicalising aliases into one record — **turns every
surviving access path into a deletion oracle**: from the store alone, with no model, an adversary
names the deleted key uniquely at 1.0000 where a duplicated store leaves the whole key space. (iii)
The channel in (i) is not an artefact of one implementation: restated over an abstract store it
reproduces in three unrelated shapes, and in the vector-index shape it leaks heavily while scoring
0.0000 on the top-1 metric the original attack reported.

We do not claim the architecture. An external addressable store read by a frozen core with per-entry
lifecycle operations is prior art. The contribution is the audit, the composition, and four negative
results with mechanism.

---

## 1. The evidence standard, and where it broke

The deletion primitive SHRED destroys a cell's marker; a learned gate closes; the payload becomes
unreadable. Against it we ran a calibrated linear probe, forced choice, logit rank and top-1, on
fresh seeds that took no part in choosing the configuration. Forced choice landed on exactly 375 of
750. The probe on 4 of 750 against a chance of 1 in 256. Every exact interval contained its chance
level — **and the probe read live cells at 0.893–0.927**, so the attacks demonstrably worked where
there was something to find. (`make keychannel`, E-000019.)

Then one question none of the four asked. `shred()` writes only the marker and leaves the row ACTIVE,
and the routing keys are computed **before** the gate and never gated:

```
k_f = k_fwd(LN(s + r))          # subject and relation
k_r = k_rev(LN(o + r))          # THE OBJECT
v_f = v_fwd(o) * g              # only the values are gated
```

Give the attacker what the rest of the battery gives them — a cell's subject and relation — let them
locate its column from the routing of the ordinary forward question, then sweep candidate objects
through a *reverse* query and take the one that steers the read onto that column. Five seeds, 500
pooled targets, no training:

| condition | object recovered top-1 | mean rank | winning margin |
|---|---|---|---|
| active (validity control) | 1.0000 | 0.0 | 0.6195 |
| **shred** | **1.0000** | **0.0** | **0.6195** |
| revoke / delete | 0.0040 | 128.02 | 0.0022 |
| chance | 0.0039 | 127.5 | — |

The shredded row is not leaky. It is *unchanged* — equal to the live cell to four decimals, margin
included — because the tensors the attack reads are the same before and after.

**This is not "soft-deleted data still on the medium."** The payload *was* gated; the value channel
*is* at chance; the recovery runs through a derived index the primitive never touched. The nearest
prior work we are aware of (Ghost Vectors — *unverified*) recovers *undeleted* embeddings from a
soft-deleting index. This is different in kind.

**The generalisable rule:** enumerate every quantity derived from a payload and gate all of them, or
take the row out of the addressable set. It applies to any codebook key, memory address, scope
classifier, instance key, or vector index with a soft delete.

## 2. The same channel in three unrelated store shapes

The objection to §1 is that it is a note about one file's `k_rev`. We answer it by restating the
attack over an abstract store — anything exposing rows, a deletion policy, and the observable
quantities derived from a row — and doing both halves exhaustively over a 256-value payload domain:
the **attack** (keep every candidate consistent with what the store exposes) and the **certificate**
(sweep the same domain and check whether any observable moves at all). The two are the same
statement, and the run asserts they agree on every policy. (`make pdxaudit`, PDX-001.)

Validity floor: the same attack reads a *live* payload at top-1 1.0000, so the at-chance readings
below are readings.

| policy | top-1 | candidates left | posterior on true payload | search-space cut | certified |
|---|---|---|---|---|---|
| gated value, ungated derived key | **1.0000** | 1.00 | 1.000000 | 256× | no |
| tombstoned index node, edges kept | 0.0000 | 2.92 | **0.391667** | **88×** | no |
| cleared value, codebook key kept | **1.0000** | 1.00 | 1.000000 | 256× | no |
| row removed from addressable set | 0.0000 | 256.00 | 0.003906 | 1× | **yes** |
| every derived quantity gated | 0.0000 | 256.00 | 0.003906 | 1× | **yes** |

**The tombstone row corrects the metric of §1.** A soft-deleted index node that keeps the adjacency
list built from its own embedding **never names the payload** — top-1 flat at 0.0000. Under §1's
headline that is a clean deletion. It is not: the retained edges narrow 256 candidates to 2.92, a
posterior of 0.3917 against a chance of 0.0039. **Any audit of this kind must report the
candidate-set posterior, or it will report a false negative on precisely the arrangement most
deployed systems use.**

The two clean policies are not "no attack recovered it": every value the payload could hold leaves
the observables bit-identical, so no attack of this shape exists.

**Scope.** These are faithful reconstructions of published *shapes*, written from their descriptions.
No published system is run here. See §11.

## 3. The constructive half: what a certificate is, and what it costs

`so/audit.py`. Three checks of increasing strength, and two guards that can fail.

**Differential.** Perturb the deleted payload, run the model twice with identical questions, compare
every tensor every submodule emits.

**Exhaustive.** The payload is an entity id, so the domain has 256 values. Sweep all of them. Not a
sample; every case.

**Interface.** Both models read the store in exactly one place. So if the *encoding* is bit-identical
across the whole payload domain, every downstream quantity is identical **for every possible query**
— multi-hop, reverse, phrasings nobody has written. The cost is one cheap encoding per payload value,
never running the core, which is why it took under a second for a 124M-parameter adapter and would
take the same for a 7B one.

The guards exist because reading source is how two defects entered the record. `check_mediation`
looks for an output that moves while the encoding holds still — which would mean the interface is not
the only path, and voids the certificate. `check_row_locality` checks whether one row's payload can
move another's encoding.

Three seeds, 3 targets, 1000 cells, payload domain 256, no training (`make certify`, E-000030):

| model | operation | certified for every query | first quantity that moves |
|---|---|---|---|
| synthetic | REVOKE | no (certified on 838 swept questions) | `encode_bank[v_f]` |
| synthetic | SHRED | no | `encode_bank[v_f]` |
| synthetic | DELETE | yes, structurally | the row is not in the bank |
| frozen GPT-2, soft gate | REVOKE | **CERTIFIED** | — |
| frozen GPT-2, soft gate | SHRED | no | residual 1.390e-02 |
| frozen GPT-2, hard gate | REVOKE | **CERTIFIED** | — |
| frozen GPT-2, hard gate | SHRED | **CERTIFIED** | — |

The soft/hard split measures what the hard gate buys, which had been assumed: a sigmoid never returns
zero, so under the soft gate a shredded cell retains 1.390e-02 of its payload in the value; the hard
gate thresholds to exactly zero.

## 4. A learned gate certifies the margin it was shown, not the predicate it implements

The store declares a deletion radius of 0.35. Sweeping shells over eleven checkpoints: accept rate
1.0000 out to 0.70, 0.2191 at 0.80, first reaching zero at 0.90. **The gate's operational radius is
0.90 against a declared 0.35, on every checkpoint.** The annulus the store's own predicate calls
deleted is accepted at **2,199,996 of 2,200,000**, and no training or evaluation distribution ever
populated it, because both samplers reject inside 0.7.

The published false-accept rate of 8.49e-04 reproduces at 8.550e-04 *on the distribution it was
measured on*, and is not the false-accept rate of the thing being claimed.

That learned detectors fail adversarially is old (Carlini & Wagner; learned index structures needing
an exact backup filter — *unverified*). Demonstrating the specification-versus-boundary gap as a
**swept geometry on a deletion mechanism, with both rates side by side**, is what we could not find.
The fix is to the data, not the architecture: show the gate the predicate's boundary.

## 5. A certificate about a record is not a guarantee about a fact

Everything above proves the computation is independent of a deleted **record**. Nobody asks that.
They ask whether the **fact** is gone, and the two come apart wherever the fact is reachable another
way. This repository holds the extreme case: `derivable_recovery_after_revoke_K3 = 1.0` in every seed
— a certified record deletion under which every derivable fact survives.

The gap closes by factoring the guarantee:

```
record-level certificate over R   +   R covers the fact's closure
─────────────────────────────────────────────────────────────────
                    fact-level certificate
```

The second premise is a property of the **store**, computed with a mechanical resolver and no model
at all. In a canonicalised pod the closure is **one record for any number of access keys**; under
duplication it is **exactly k**. Both proved rather than sampled: every live derivation is a must-hit
set, so a pairwise-disjoint subfamily bounds the optimum from below, and `optimal` is set only when
the greedy search *meets* that bound. (`make closure`.)

**What is not ours, and must be cited in the first paragraph, not related work.** The quantity is
**resilience** in the database literature — the size of a **minimum contingency set**, with its own
PTIME/NP-complete dichotomy and an LP-tight solver in print. The certified lower bound is that
literature's standard disjoint-witness packing bound. And the remedy — canonicalisation at write time
— is proposed verbatim in §9 of a 2026 audit of forgetting in limited-memory language models, which
calls it "directly testable within our framework" and does not test it (*all unverified*).

What is ours is the measurement, the composition with a record-level certificate, and the price the
reader charges for the indirection: 0.0954 for sharing and 0.0688 for link training, worst of three
seeds over twelve phrasings.

## 6. Canonicalisation makes erasure certifiable and opens a deletion oracle

Delete one of *k* duplicated copies and the store is a store with *k−1* copies: nothing says a
deletion happened, nothing says where. Delete a pod's object and every alias is still a LINK row
still carrying the removed cell's key.

No model, no checkpoint, no training: the adversary reads the bank and names every key a LINK row
points at that no row holds. Three seeds, 100 pods each (`make disclosure`, E-000035):

| store | deleted key disclosed | uniquely identified | candidate keys left | false positives |
|---|---|---|---|---|
| **canonical (pod)** | **1.0000** | **1.0000** | **1 of 1,536** | 0.00 |
| duplicated | 0.0000 | 0.0000 | 1,536 of 1,536 | 0.00 |

And the closure **inverts** with the guarantee you ask for:

| guarantee | canonical pod | duplicated |
|---|---|---|
| unreachable to the reader | **1.00** | 3.00 |
| no trace left in the bank | 3.00 | **1.00** |

A pod's aliases *are* the signposts, so leaving no trace costs the object plus all of them; a
duplicated store costs the one record you were removing anyway. Exact on every seed.

**So the claim must be stated in both directions: canonicalisation makes erasure a single certifiable
operation and turns every access path into a deletion oracle.** Both follow from one design decision,
and "was there a record about this person, and was it deleted" is exactly the question an erasure
guarantee is supposed to make unanswerable. Blanking the dangling key closes the channel at 1.0000.
The trade is a number, not an argument.

This is the result we expect to travel furthest: any store that keeps a dangling reference after
deleting its referent has it, which includes every soft-deleting vector index that keeps its edges.

## 7. What holding facts in rows buys, measured

One frozen GPT-2, 400 facts, 50 deletion targets, three ways of holding and removing them, attacked
identically. The weights arms get a LoRA of 2,359,296 parameters against the adapter's 2,370,692 — a
1.00× match — trained until they answer 95% of the targets. (`make compare`, E-000024.)

| | cells (SHRED) | LoRA, gradient ascent | LoRA, relabel to ' unknown' |
|---|---|---|---|
| answered the fact before deletion | 0.92 | 0.96 | 0.96 |
| answers it after | 0.02 | 0.00 | 0.02 |
| forced choice, chance 0.50 | **0.44** | 0.78 | **1.00** |
| relearning attack recovers held-out facts | **0.00** | 0.48 | **0.72** |
| perplexity on ordinary prose, from 42.9 | **42.9** | 6.19e+09 | 8.49e+06 |
| parameters changed | **0** | 2,359,296 | 2,359,296 |
| seconds to delete 50 facts | **0.0008** | 137 | 311 |
| **a certificate is even available** | **yes** | no | no |

The relabel arm is the strongest practical baseline and the worst of the three: it looks deleted at
0.02, and an attacker who fine-tunes on half the deleted facts recovers 76% of the half they never
supplied.

The last row is the one that matters, and not because of effort: in a LoRA there is no finite payload
domain to sweep and no interface the data passes through. **Putting facts in rows is what makes
deletion certifiable at all.**

## 8. Methods: an instrument that cannot fail is not evidence

Three instruments in this programme certified by *not testing*. Reachability on an empty row set
returned the strongest label in the ladder on a live bank. The payload sweep on an empty row set
certified in one evaluation. And a membership test could not see that an alias row's link key is
built *from the target cell*, so a surviving row is computed from the removed one. All three are
closed: reachability refuses an empty row set, absence needs a positive control showing the payload
*was* reachable, and store-absence sweeps the payload over its whole domain.

We state the rule as a methods contribution because we have now paid for it five times, and because
the evaluation literature this paper joins is built almost entirely from instruments of exactly this
shape:

> **An instrument that cannot fail is not evidence. Every screen needs a positive control, and every
> attack needs a validity floor.**

Concretely, every measurement in this paper carries one. §1's probe reads live cells at 0.893–0.927.
§2's attack recovers a live payload at 1.0000 and refuses to report if it does not. §3's certificate
has two guards that can void it. A mechanical sweep of all 100 recorded experiments for comparisons
whose two sides share a source found three, and the sweep's own first run failed its floor and had to
be fixed before its silence meant anything.

We think this is the half of the unlearning-evaluation problem that attack-based benchmarks leave
open, and it is cheap.

## 9. What this is not

The architecture is re-invention: an external addressable knowledge store read by a frozen core, with
per-entry lifecycle operations and a forgetting primitive, is prior art at larger scale, several
instances with explicit forgetting (*unverified*). The word "provably" is not earned for the
copy-bound claim: absence of a fact is not demonstrable from weights, only from the training
algorithm, and that argument belongs to prior work.

The certificate is about the **model**, not the system: after REVOKE or SHRED the store still holds
the payload, and anyone who can read the store does not need the model. It is exhaustive over the
payload domain and universal over queries only through the interface argument, which is a claim about
a specific model that the mediation guard can refute but not establish.

The fact-level guarantee is **unreachability, not erasure**; it is over a **declared query
workload** — the default is one single-hop question per key, and §5's counterexample shows a
multi-hop derivation outside it is not hypothetical; and it individuates a fact as a **triple**, so
canonicalising one subject's pod says nothing about the same value under another subject.

Everything is CPU, 124M parameters, synthetic worlds, single-token entities. **Nothing here shows
unlearning of knowledge already in pretrained weights** — that is precisely what the architecture
avoids rather than solves.

## 10. Contributions, stated narrowly

1. A payload-derived index channel that defeats a value-gating deletion primitive at top-1 1.0000
   while every value-channel attack sits at chance, with the generalisable rule and a
   store-independent instrument that reproduces it in three shapes (§1, §2).
2. The demonstration that top-1 recovery is the wrong metric for this audit, and that the
   candidate-set posterior is required — established on the vector-index shape, where top-1 reads
   0.0000 and the search space is cut 88× (§2).
3. Exhaustive record-level deletion certificates for a frozen memory-augmented reader, and the
   soft/hard gate measurement (§3).
4. The specification-versus-learned-boundary gap as a swept geometry on a deletion mechanism, with
   both false-accept rates side by side (§4).
5. The composition of a record-level certificate with a store-side closure, with a certified lower
   bound, and the price of the indirection (§5).
6. **Canonicalisation makes erasure a single certifiable operation and turns every access path into a
   deletion oracle** — measured from the bank alone, no model (§6).
7. A methods rule with five worked instances (§8).

## 11. What must be run before this is honest to send

**Neither is optional and neither is large.**

**(a) The payload-derived sweep on the symlink arms.** §1's attack has never been run on the aliasing
arms, so §6's "nothing recoverable" is value-channel-only for those arms. Named as still-unrun in the
2026-09-04 record; still unrun.

**(b) The closure reproduced in a chunked vector index.** Until §5's closure is shown in the
arrangement almost every deployed system uses, the result is about one store implementation rather
than about the pattern. The target exists (`make retrieval`, E-000033).

**(c) Strongly recommended, and the reason this paper would matter outside the project:** run §2's
instrument against a *published* system with its own reported deletion metric — a codebook editor, an
episodic memory, or a real soft-deleting vector index. PDX-001 made this configuration rather than a
rewrite: supply a store whose `observe()` reads the real index. Two outcomes, both publishable: a
recovery channel in a system people cite, or a null that localises the defect. No training;
inference hours on existing checkpoints.

**(d) Verify every citation.** See §0.

## 12. Venue

Not an architecture paper. A short measurement-and-audit paper, at the venue that takes empirical
audits of forgetting in memory-augmented models. §5 and §6 are the sections a reviewer will weigh;
§1 and §2 are what a practitioner will act on.

---

### Reproduction

Every number above: `make keychannel certify closure retrieval disclosure compare pdxaudit`. Records
in `so/results/`. Instrument audits: `make calibrate charged unread auditinstr`.
