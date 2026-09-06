# What this programme actually found

*2026-09-04. Read [what is new here and what is not](so-novelty-2026-09-04.md) first — it says which
half of this is re-invention. This document states the half that is not.*

## In one paragraph

Machine unlearning is evaluated by attack: delete, run extraction or membership attacks, report that
they failed. That standard bounds the adversaries someone thought of, and this repository contains a
clean instance of it failing — a deletion primitive that passed four attacks at chance over 750
pooled trials and gave the deleted fact up at 100% to a fifth attack written afterwards. The
alternative is to prove that the model's computation does not depend on the deleted payload. Where
knowledge lives in rows with a small payload domain, that proof is cheap and exhaustive: sweep every
value the payload could hold and check the computation does not move. Doing it found two defects in a
system built specifically to get deletion right, and then certified three deletions that the attack
standard could only ever have called "not yet broken". Pushing the same standard from a *record* to a
*fact* then found three more defects — in the instruments themselves, which were certifying by not
testing — and one result that cuts against the design: the arrangement that makes erasure a single
certifiable operation also turns every access path into a deletion oracle.

## 1. The evidence standard, and where it broke here

SHRED destroys a cell's marker, the learned gate closes, and the payload becomes unreadable. Against
it the programme ran a calibrated linear probe, forced choice, logit rank and top-1, on fresh seeds
that took no part in choosing the configuration. Forced choice landed on exactly 375 of 750. The
probe on 4 of 750 against a chance of 1 in 256. Every exact interval contained its chance level, and
the probe read live cells at 0.89–0.93, so the attacks demonstrably worked where there was something
to find.

Then E-000028 asked a question none of the four asked. `shred()` writes only the marker and leaves the
row ACTIVE, and in `encode_bank` the routing keys are computed **before** the gate and never gated:

    k_f = k_fwd(LN(s + r))          # subject and relation
    k_r = k_rev(LN(o + r))          # THE OBJECT
    v_f = v_fwd(o) * g              # only the values are gated

Give an attacker what the rest of the battery gives them — a cell's subject and relation — let them
find its column from the routing of the ordinary forward question, then sweep candidate objects
through a *reverse* query and take the one that steers the read onto that column. Five seeds, 500
pooled targets:

| condition | object recovered top-1 | mean rank | winning margin |
|---|---|---|---|
| active (validity control) | 1.0000 | 0.0 | 0.6195 |
| **shred** | **1.0000** | **0.0** | **0.6195** |
| revoke / delete | 0.0040 | 128.0 | 0.0022 |
| chance | 0.0039 | 127.5 | — |

The shredded row is not leaky. It is *unchanged* — equal to the live cell to four decimals, margin
included — because the tensors the attack reads are the same before and after.

## 2. The instrument

`so/audit.py`. Three checks of increasing strength, and two guards that can fail.

**Differential.** Perturb the deleted payload, run the model twice with identical questions, compare
every tensor every submodule emits. Nothing moves means the two runs are the same computation.

**Exhaustive.** The payload is an entity id, so the domain has 256 values. Sweep all of them. This is
not a sample; it is every case.

**Interface.** Both models read the store in exactly one place — `so/model.py:246`, and
`so/llm_adapter.py:323` for the adapter. So if the *encoding* is bit-identical across the whole
payload domain, every downstream quantity is identical **for every possible query**: multi-hop,
reverse, phrasings nobody has written. The cost is one cheap encoding per payload value, and it never
runs the core, which is why it took under a second for the 124M-parameter adapter and would take the
same for a 7B one.

The guards exist because reading source is exactly how the two defects below got into the record.
`check_mediation` looks for an output that moves while the encoding holds still — which would mean
the interface is not the only path, and voids the certificate. `check_row_locality` checks whether
one row's payload can move another row's encoding; it cannot at noise 0, which upgrades the joint
claim from sampled to proved, and it *can* at noise > 0, because `jitter()` takes its rms over all
rows including masked ones.

Both guards were needed. The first version of the adapter arm compared `values_payload` — the ungated
payload, which `encode_bank` returns as a diagnostic and `forward` never reads — and reported a
residual of 3.49 through a tensor the model does not look at. Naming the consumed interface is part of
the claim: too wide is over-strict, too narrow is unsound.

## 3. What it finds

Three seeds, 3 targets, 1000 cells, a payload domain of 256 values, no training.

| model | operation | certified for every query | certified on 838 swept questions | first quantity that moves |
|---|---|---|---|---|
| synthetic | REVOKE | no | **yes** | `encode_bank[v_f]` |
| synthetic | SHRED | no | **no** | `encode_bank[v_f]` |
| synthetic | DELETE | yes, structurally | yes, structurally | the row is not in the bank |
| frozen GPT-2, soft gate | REVOKE | **CERTIFIED** | — | — |
| frozen GPT-2, soft gate | SHRED | no | — | residual 1.390e-02 |
| frozen GPT-2, hard gate | REVOKE | **CERTIFIED** | — | — |
| frozen GPT-2, hard gate | SHRED | **CERTIFIED** | — | — |

Those three CERTIFIED rows are the first deletions in this programme that are not "no attack
recovered it". The computation is bit-identical for all 256 values the payload could take, so no
attack can recover it.

And the soft/hard split says precisely what the hard gate buys, which was assumed and never measured:
a sigmoid never returns zero, so under the soft gate a shredded cell keeps 1.390e-02
of its payload in the value; the hard gate thresholds to exactly zero and the value becomes exactly
the ' unknown' direction.

## 4. The two rules that transfer

**A gate on values is not a deletion primitive when another term in the computation is a function of
the same payload.** Not "soft-deleted data still on the medium" — the payload *was* gated, the value
channel *is* at chance, and recovery ran through a derived index the primitive never touched. The same
question can be put to GRACE's codebook keys, Larimar's memory addressing, SERAC's scope classifier
and any vector store with a soft delete. Enumerate every quantity derived from a payload and gate all
of them, or take the row out of the addressable set.

**A learned gate certifies the margin between the classes it was shown, not the predicate it was
written to implement.** The store declares a radius of 0.35. Sweeping shells over eleven checkpoints,
the gate accepts everything out to 0.70 at 1.0000, 0.2191 at 0.80, and first reaches zero at 0.90 — an
operational radius of 0.90, on every checkpoint. The annulus the store calls deleted is accepted at
2,199,996 of 2,200,000, and no training or evaluation distribution ever populated it, because both
samplers reject inside 0.7. The published false-accept rate of 8.49e-04 is reproduced at 8.550e-04 on
the distribution it was measured on, and is not the false-accept rate of the thing being claimed. The
fix is to the data, not the architecture: show the gate the predicate's boundary.

## 5. What holding facts in rows buys, measured

One frozen GPT-2, 400 facts, 50 deletion targets, three ways of holding and removing them, attacked
identically. The weights arms get a LoRA of 2,359,296 parameters against the
adapter's 2,370,692 — a 1.00x match — and are trained until they answer 95% of the targets.

| | cells (SHRED) | LoRA, gradient ascent | LoRA, relabel to ' unknown' |
|---|---|---|---|
| answered the fact before deletion | 0.92 | 0.96 | 0.96 |
| answers it after | 0.02 | 0.00 | 0.02 |
| forced choice, chance 0.50 | **0.44** | 0.78 | **1.00** |
| relearning attack recovers held-out facts it was never given | **0.00** | 0.48 | **0.72** |
| perplexity on ordinary prose, from 42.9 | **42.9** | 6.19e+09 | 8.49e+06 |
| parameters changed | **0** | 2,359,296 | 2,359,296 |
| seconds to delete 50 facts | **0.0008** | 137 | 311 |
| a certificate is even available | yes | no | no |

The relabel arm is the strongest practical baseline and it is the worst of the three: it looks deleted
at 0.02, and an attacker who fine-tunes on half the deleted facts recovers
76% of the half they never supplied.

The last row is the one that matters most. A certificate is available for the cells arm and not for
the others, and not because of effort: in a LoRA there is no finite payload domain to sweep and no
interface the data passes through. **Putting facts in rows is what makes deletion certifiable at all.**

The attack-validity floor is what makes the fourth row readable, and it did its job. The first seed-0
run had the cells arm's relearning attacker recovering only 0.28 of the facts it was *handed*, below
its own 0.50 floor, so its number on the held-out half meant nothing. The attacker now evaluates every
40 steps and runs until it clears the floor: on the re-run it reaches 0.60 supplied in 200 of 400
budgeted steps and recovers **0.0000** of the held-out facts, against 0.96/0.48 for gradient ascent
and 1.00/0.72 for relabel. One seed; the three-seed run is in flight.

## 6. A certificate about a record is not a guarantee about a fact

Everything above proves the model's computation is independent of a deleted **record**. Nobody asks
that. They ask whether the **fact** is gone, and the two come apart wherever the fact is reachable
another way — this repository holds the extreme case, `derivable_recovery_after_revoke_K3 = 1.0` in
every seed of E-000019: a certified record deletion under which every derivable fact survives.

The gap closes by factoring the guarantee in two:

```
record-level certificate over R   +   R covers the fact's closure
-----------------------------------------------------------------
                    fact-level certificate
```

The second premise is a property of the **store**, computed by `so/closure.py` with the mechanical
resolver and no model at all. So the expensive half is proved once against the architecture and the
cheap half is a graph search — and the search is what canonicalisation makes trivial. In a pod the
closure is **one record for any number of access keys**; under duplication it is **exactly k**. Both
are proved rather than sampled: every live derivation is a must-hit set, so a pairwise-disjoint
subfamily of them bounds the optimum from below, and `optimal` is set only when the greedy search
*meets* that bound.

The quantity is not new and this work does not name it: it is **resilience** in the database
literature, the size of a **minimum contingency set**, with a PTIME/NP-complete dichotomy of its own.
The remedy is not new either. The closest work — Raeesi and Roed, *Auditing Forgetting in Limited
Memory Language Models*, arXiv:2607.00605 — audits 12,228 deletions, concludes that "the unlearning
boundary is drawn primarily by the database administrator rather than by the model", and proposes in
§9: *"A second direction is canonicalization at write time, in which aliases and paraphrastic forms
are stored as pointers into a single canonical record rather than as independent triplets. Both
approaches are directly testable within our framework."* Named, called testable, not tested.

What is here is the measurement, and three things it forced:

**Three instruments certified by not testing.** `certify_structural` with an empty row list returns
"NO PATH" — the strongest label in the ladder — on a bank whose rows are live, because with no row
selected there is nothing to trace a path from; E-000030 recorded `delete/structurally_certified =
true` from it. `certify_encoding` with an empty row list certifies in one evaluation. And
`check_absence` tested membership only, while `MVCCStore.bank()` builds an alias row's link key **from
the target cell** — so a surviving row is computed from the removed one and a membership test cannot
see it. All three are closed: reachability refuses an empty row set, absence needs a positive control
showing the payload *was* reachable, and `certify_store_absence` sweeps the payload in the store over
its whole domain and compares the surviving bank.

**Three words used loosely.** "Gone" was doing the work of "unreachable" — under EVICT the store still
holds the payload on purpose, which is what makes RESTORE work, so the verdict now reads *FACT
UNREACHABLE, CERTIFIED … UNREACHABLE, NOT ERASED*. "The fact" was not individuated — a pod certifies
the (subject, relation, object) triple and says nothing about the same value under another subject.
And the anomaly was misnamed: this is Codd's **modification** anomaly applied to a delete, not his
deletion anomaly, which is the opposite failure.

**A recorded claim retracted.** E-000015 and E-000020 say *the model is never told that a value it has
read is a pointer — that it must learn.* A single threshold on the value norm separates the two kinds
at **1.0000**, 7.6 pooled standard deviations, disjoint ranges, on every seed. At initialisation the
ranges overlap, so the gap is learned, not architectural. Recognition is free; only **following** is
learned.

## 7. What canonicalisation costs the subject, not the reader

The pod story had been told with one number missing. Delete one of *k* duplicated copies and the store
is a store with *k−1* copies: nothing says a deletion happened, nothing says where. Delete a pod's
object and every alias is still a LINK row still carrying the removed cell's key — kept deliberately,
so the model discovers the miss instead of being handed it. Each surviving alias is a signpost reading
*a record stood at (s, r) and is gone.*

No model, no checkpoint, no training: the adversary reads `MVCCStore.bank()` and names every key a
LINK row points at that no row holds. Three seeds, 100 pods each.

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

So the claim has to be stated in both directions: **canonicalisation makes erasure a single certifiable
operation and turns every alias into a deletion oracle.** Both follow from the same design decision,
and "was there a record about this person, and was it deleted" is exactly the question an erasure
guarantee is supposed to make unanswerable. Blanking the dangling key closes the channel at 1.0000 —
and costs E-000015's `delete_target/alias_unknown`, which stops being a discovery about the model and
becomes a tautology about the bank. The trade is a number, not an argument.

## 8. What this is not

The architecture is re-invention — see the novelty document. The certificate is about the *model*, not
the system: after REVOKE or SHRED the store still holds the payload, and anyone who can read the store
does not need the model. It is exhaustive over the payload domain and over whatever query set is
swept, and universal over queries only through the interface argument, which is a claim about a
specific model that `check_mediation` can refute but not establish. Everything here is CPU, 124M
parameters, synthetic worlds, single-token entities. Nothing shows unlearning of knowledge already in
pretrained weights — that is precisely what the architecture avoids rather than solves.

The fact-level guarantee inherits every one of those limits and adds three of its own, each carried in
the certificate rather than in a footnote. It is **unreachability, not erasure**: under EVICT the store
holds the payload deliberately. It is over a **declared query workload** — the default is one
single-hop question per key, and a fact a multi-hop derivation still reaches is outside it, which
E-000019 shows is not hypothetical. And it individuates a fact as a **triple**: canonicalising one
subject's pod says nothing about the same value stored under another subject, and selecting by value
instead destroys a bystander, which there is now a test for. The closure metric is a store statistic:
a frozen core that knew the fact before the store existed is outside it, which is what E-000013
measures separately. The disclosure channel of §7 is a property of *this* store's bank; one that
compacts its aliases on deletion, or never exports the target key, has none.
