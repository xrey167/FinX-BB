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
standard could only ever have called "not yet broken".

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
| relearning attack recovers held-out facts it was never given | **0.04** | 0.48 | **0.76** |
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

One caveat, recorded rather than smoothed: the cells arm's relearning attacker recovered only
0.28 of the facts it was *handed*, so its 0.04
on the held-out half is a weak attacker's number. The attack-validity criterion caught that, and the
attackers now run until they clear it. One seed; the three-seed run is in flight.

## 6. What this is not

The architecture is re-invention — see the novelty document. The certificate is about the *model*, not
the system: after REVOKE or SHRED the store still holds the payload, and anyone who can read the store
does not need the model. It is exhaustive over the payload domain and over whatever query set is
swept, and universal over queries only through the interface argument, which is a claim about a
specific model that `check_mediation` can refute but not establish. Everything here is CPU, 124M
parameters, synthetic worlds, single-token entities. Nothing shows unlearning of knowledge already in
pretrained weights — that is precisely what the architecture avoids rather than solves.
