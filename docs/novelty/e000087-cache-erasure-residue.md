# E-000087 — RETRACTED. The headline was a tensor compared with itself.

Date: 2026-09-06
Status: **retracted within hours of being committed.** The fifth measurement error of this session,
and the one that most deserves to be recorded, because it was made in a document that opened by
boasting about controls that can fail.

## What was claimed, and why it was worthless

The claim: a span-replacement KV erasure primitive leaves the erased entity recoverable from the
retained suffix at 1.0000, against a chance of 0.0156, on two backbones.

The reality: **P_STUB rewrites only the chunk's own span. The retained suffix it then reads is bitwise
identical to the suffix of the arm where no erasure happened at all** — verified, max abs difference
`0.000e+00`. The attack matched a tensor against itself. 1.0000 was arithmetic. So was P_ZERO's.

The tell was in the results table and I did not see it: `P_STUB_top1`, `P_ZERO_top1` and `P_NOOP_top1`
were all exactly 1.0000. Three different treatments cannot agree to four decimals unless they are the
same object.

## The control that was missing, and the irony of which one it was

The document declared four controls, C1 to C4. Every one of them examined the **ideal** arm — is it
bit-identical across entities, is the attack on it at chance. Not one asked whether the **treatment**
arm differed from doing nothing. That is the only control that could have caught this, and it is now
`C0` in the code, where it fails on the original design and marks the affected numbers `VACUOUS`.

**C0: an erasure primitive must change what is retained, or an attack on the retained state is
comparing a tensor with itself.**

## It was knowable without running anything

KVEraser (arXiv:2606.17034) states in its own method that its steering block is length-preserving and
"leaves the suffix cache unchanged". Leyline (arXiv:2606.01065) preserves suffix `K_nope` and `V`
because "that attention is exactly what we want to keep". Any readout defined over retained suffix K/V
returns the same value in the erased and un-erased arms **by those systems' own equations**. An
adversarial review of this claim put it exactly right: pointing an instrument at a quantity the
treatment provably does not touch is not a measurement, and it was "the fourth measurement error of
this session waiting to happen in a fifth costume".

## The space is occupied anyway

**MEMENTO (arXiv:2604.09852, §6.2.2)** — nearest neighbour, five months earlier, and a *better*
experiment than the one proposed here. It plants a 5-digit passcode in a block, evicts that block from
the KV cache, and trains an MLP probe on the retained downstream KV: **26.7% per digit** against a 10%
floor on Qwen3-8B, with a causal control (a memento *preceding* the block) at exactly chance, the
effect localised to deep layers, replicated on an 810K-parameter toy transformer, and constant across
training checkpoints — "the channel is architectural, not learned."

Note **per digit**. Full five-digit recovery is ≈ 0.267⁵ ≈ 1/700. Quoting 26.7% as secret recovery
would repeat this session's error class in a sixth costume.

**Compute Globally, Materialize Locally (arXiv:2607.23693)** does the causal version: omitting a source
event from what is served still moves the answer 99:0 toward the omitted value, established against
byte-identical served histories, with a negative control that could fail and did not.

## What actually survives from this experiment

One null result, and it is real because the decode does attend to the replaced span: **decoding the
query against the erased cache, the erased entity is the top-1 answer in 0.0 of cases** on both
backbones (mean rank 2820 on GPT-2, 916 on Pythia-70m; gain over the ideal arm +0.66 and +1.66 nats).
On this harness the primitive works behaviourally.

## The one experiment in this area that could still fail

From the same review, and it is not the one I ran: KVEraser validates localized erasure with a **single
cooperative NIAH exact-match query**, which its own method passes at near-perfect scores from 1K to
32K, and a full-text census of v2 finds no adversarial readout of any kind — leak 0, attack 0, canary 0,
audit 0, membership 0, threat 0. An **adversarial decode-time** readout — query classes chosen to elicit
the erased value rather than the retained one — bears on states that *do* attend to the steering block,
so its training signal actually matters and the result can come out either way. That is the E-000028
shape correctly aimed: a primitive trained against one query class, tested by another.

It is recorded, not queued. After five measurement errors in one session the right next step is not
another experiment of mine but a slower one designed by someone who has not spent the day being wrong.
