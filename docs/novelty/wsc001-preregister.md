# WSC-001 — where an accessibility audit of an external memory must read

Date: 2026-09-06
Status: **pre-registered before the recorded run; partially blind, and the disclosure is below**
Classification: instrument calibration and an audit-siting measurement. **No novelty credit** for the
Jacobian lens, for J-lens accessibility auditing, for probing, for projection/erasure, for external
memory, for canonical pods or pointer aliases, or for any of these in loose combination.

## Where this comes from

E-000063 — the composed workspace-pod deletion certificate, designed on 2026-09-05 — produced its
first result ever on 2026-09-06, after `so/jlens.py` was repaired to run on a frozen core (ledger
§31.56; its CI run 33955376015 had died in the vector-Jacobian product on both seeds after 31 minutes
of training each). On the BOS-trained symlink adapter, seed 0, 16 pods:

| row | value | bar | |
|---|---|---|---|
| `active_alias_correct` | 0.8795 | ≥ 0.80 | PASS — the memory is read |
| `shred_alias_unknown` | 1.0000 | ≥ 0.90 | PASS — one SHRED closes every alias at the output |
| `shred_alias_true_object` | 0.0000 | ≤ 0.05 | PASS |
| `bystander_top1_agree` / `kl_max` | 1.0000 / 0.0000 | ≥ 0.98 / ≤ 0.05 | PASS — locality is exact |
| `finalprobe(ACTIVE − NEVER)` | **0.9018** | ≥ 0.30 | PASS — the memory is in the final state |
| `jprobe(ACTIVE − NEVER)` | **−0.0134** | ≥ 0.30 | **FAIL — the workspace readout cannot tell live memory from none** |
| `jprobe(SHRED − NEVER)` | −0.0179 | ≤ 0.05 | passes |
| `finalprobe(SHRED − NEVER)` | 0.0536 | ≤ 0.05 | **FAIL — a residual trace survives SHRED in the final state** |

The certificate's headline row passes and its validity row fails. It passes *because* the instrument
never saw the thing it certifies gone. Ten instruments in this ledger have certified by not testing;
this one certifies by not seeing, and E-000063's own pre-registered validity bar caught it.

WSC-001 asks the two questions that leaves, and nothing wider.

## The two questions, and the three answers that are distinguishable

**A — the readout side.** Four explanations are distinguishable, and the first was found by
measurement after this file's first draft was written (disclosed below):

    SITING     the audit reads a state the memory has not reached yet
    DEPTH      the write reaches that state but is not linearly decodable there
    DIRECTION  k coordinates suffice there, but not those k
    DIMENSION  k coordinates are too few, whatever they are

Separating SITING from the rest needs a **mediator**, and the mediator is what the first draft lacked:
at every candidate site, how far does the state move when the pod is shredded, and when it was never
written? A readout that sees nothing at a site where the state does not move is not a blind readout,
it is a correctly sited one looking at the wrong block. The mediator is recorded beside every probe
number, for the states AND for the injected write at each read layer.

Sites: the outputs of blocks 8, 9, 10 and 11, and the final hidden state. Five feature families over
the *same* states, probes trained on ACTIVE only and applied unchanged to SHRED and NEVER (E-000063's
transfer probe, five-fold by template):

| family | what it is | why it is here |
|---|---|---|
| `jspace` | cosine with the J-lens atoms of the pods' objects | the audit itself |
| `random` | cosine with an equally sized random orthonormal set | dimension-matched null |
| `pca` | top-k principal components of the ACTIVE states | the best k-dimensional linear readout there is |
| `unembed` | cosine with the objects' unembedding rows | the vocabulary-basis analogue |
| `raw` | the residual itself, 768 dimensions | the capacity ceiling |

E-000063 reads one site, the output of block 8, chosen in its source as "the first adapter read
site".

**B — the causal side.** A probe says what is readable, not what the model uses. The write is a tensor
this harness holds, so it can be re-injected with its entire first-order channel removed. Projecting
it out of the span of the J-lens atoms of **all 257 scored tokens** leaves a write whose first-order
effect on every scored logit is exactly zero — measured retention 1e-6 at full rank (211/217 at the
r99 truncation the first draft of this design used, which is why the full span is used instead).

Arms, per read site, all evaluation-only on the recorded checkpoints:

| arm | write |
|---|---|
| `full` | the trained write |
| `dropW` | the trained write with the 257-atom span projected out (first-order effect exactly zero) |
| `dropW_n` | the same direction, renormalised to the trained write's norm (magnitude held fixed) |
| `dropT_n` | the same, for a J-lens span built identically over 257 tokens that are **not** scored — the structure-matched null |
| `atom` | the ground-truth object's output-embedding row at the trained write's norm — **an ORACLE arm**: it is handed the answer and does not route, so its accuracy is 1.0 by construction and is never a capability comparison. It exists only to give a second write whose blind mass can be measured. |
| `atom_dropW` | the oracle write with the same span projected out |
| `perm` | another pod's object at the same magnitude — the content floor ("something was retrieved, of the wrong content") |
| `none` | no memory |

## By construction versus measured

| | |
|---|---|
| **By construction** | `keep(W) ≈ full` and `drop(W)` losing the first-order term. The atoms of the scored tokens are *defined* as the directions whose inner product with the write gives that term; ledger §31.39 already calls this a tautology. |
| **By construction** | `atom` = 1.000. It is an oracle write. |
| **By construction** | at read layer 10 the lens source is one block from the output and the atoms **are** the unembedding rows (cos 1.000, §31.56); the second site's "workspace" is vocabulary space. |
| **By construction** | an isotropic random subspace cannot be a null for "workspaceness": at matched rank the atom span is the unique maximal first-order-retaining subspace (measured keep-retention 1.000 against 0.583 random). It is reported as a geometry row only. |
| **Measured** | whether the ANSWER survives `dropW` — that is second-order and no algebra fixes it. |
| **Measured** | `dropT_n`, the token-matched null: it can fire and void the whole of part B. |
| **Measured** | every part-A family's ACTIVE − NEVER at both sites, and the ordering between families. |
| **Measured** | the write's norm-share in the atom span against the r/d floor a random vector would have. |

## Pre-registered bars, worst seed of three, every branch written out

Validity (VOID the corresponding part if any fails):
- `V1` `full/alias ≥ 0.80` and `full/direct ≥ 0.90` at the anchor template 3.
- `V2` `perm/alias ≤ 0.05` — a write of the wrong content must not answer.
- `V3` `none/alias ≤ 0.05`.
- `V4` part A: `final/raw active − never ≥ 0.30` — some readout, somewhere, must see the memory, or
  the states were not captured.
- `V5` geometry: `dropW` first-order retention `≤ 1e-4` at both sites, or the drop arm is not
  first-order null and part B is uninterpretable.

Part B, read only with V1–V3 and V5 passing:
- `B1` **first-order sufficiency**: `dropW_n/alias ≤ 0.20` at every template AND
  `dropT_n/alias ≥ full/alias − 0.20` at every template → *the scored-token atom span is the causal
  channel of the write, and the token-matched null does not reproduce it*.
- `B2` **blind mass**: `dropW_n/alias ≥ 0.50` at ≥ 3 of the twelve templates, with `dropT_n` not
  showing the same → *the answer survives with the first-order channel removed*, the outcome this
  design was originally built to look for.
- `B3` if `dropT_n/alias < full/alias − 0.20` the null has fired: part B is recorded VOID-NULL and
  reports only that removing any rank-257 J-lens span of this construction costs the answer.
- `B4` neither B1 nor B2 (mixed across templates) → recorded as mixed, per template, with no sentence.
- `B5` **norm versus effect**, reported with B1 or B2 either way: the write's mean squared-norm share
  in the atom span against the random floor r/d = 0.3346. This row is descriptive and carries no bar.

Part A, read only with V4 passing:
- `A0` **SITING**: every family at `site8` has ACTIVE − NEVER ≤ 0.05 **and** the injected write at the
  first read layer does not move when the pod is shredded (`write8_shred_moves` ≤ 1e-3) → E-000063's
  capture block is upstream of the write that carries the pod, so no readout there can attribute the
  memory and its certificate audits a state the pod never reached. The mediator is what licenses this
  rather than A1.
- `A1` **DEPTH**: `site8` families ≤ 0.05 while the write there DOES move → the write reaches that
  state and is not linearly attributable at any dimension up to the full residual.
- `A2` **DIRECTION**: at a site whose write moves, `pca` ≥ 0.30 while `jspace` ≤ 0.05.
- `A3` **DIMENSION**: at such a site, `raw` ≥ 0.30 while `pca` ≤ 0.05.
- `A4` none of these → recorded as mixed with the table and no sentence.
- `A8` reported at every site with no bar: the first site at which each family reaches ACTIVE − NEVER
  ≥ 0.30, which is the answer to "where can this audit be read".
- `A5` reported beside every row, no bar: `final/jspace`, so that "the audit works downstream" is a
  number and not an inference.

Leak-type rows take the **max** over seeds, capability rows the **min**. The resampling unit is the
**pod**: 200 alias reads per template per seed are 100 pods read through two aliases each in one world,
so a binomial interval at n = 200 is anti-conservative and the three worlds are the replicates. No bar
is adjusted after the run.

## Blindness, disclosed

This is not a blind pre-registration and saying so is part of it.

**Seen before the bars were written:** E-000063 seed 0 in full (the table at the top). Part B on seed 0
at templates 3 and 9 only, at 200 alias and 100 direct reads: `full` 0.940 / 0.905, `dropW` 0.050 /
0.020, `dropW_n` 0.110 / 0.070, `dropT_n` 0.875 / 0.770, `atom` 1.000, `atom_dropW` 0.015 / 0.010,
`perm` 0.000, `none` 0.005 / 0.000, norm-share 0.285 / 0.284. Part A on seed 0 at 6 pods and one
template, whose numbers are too small to quote and are not used.

**What that means for the bars.** B1's shape was chosen knowing seed 0 at two templates points that
way, and B2 is the outcome this experiment was designed to find and did not. B1 is therefore a
CONFIRMATION on ten further templates and two further seeds, not a discovery, and it is labelled so
wherever it is reported. B3, the null, has not fired at either seen template and remains the row that
can void the part. Part A's bars were set on the E-000063 table alone.

**Seen after the bars above were written, and disclosed here rather than folded in.** A plumbing
validation of the address-mode split (seed 0, 10 pods, one template, 1 thread) produced the part-A
table before the recorded run. Its structure: at the write site every family in BOTH address modes
reads ACTIVE - NEVER within +/-0.021 of zero, the raw 768-dimensional state included (alias +0.0071,
direct +0.0143); at the final state every family separates strongly (alias +0.800 to +0.893, direct
+0.843 to +0.957). Two things in it were not anticipated by any bar above and are therefore added as
ROWS WITH BARS SET KNOWING THEM, labelled so wherever they are reported:

- `A6` **indirection**: `direct/final/jspace - alias/final/jspace`. Seen at +0.064 on the plumbing
  cell. Bar: an audit calibrated on the canonical key transfers to the pointer path if this is
  <= 0.10 on the worst seed, and does not if it is >= 0.20. Between the two, no sentence.
- `A7` **what a deletion leaves in the model's state, by address mode**:
  `final/raw/(SHRED - NEVER)` for each mode. Seen at +0.457 direct against -0.021 alias on the
  plumbing cell -- a shredded pod's identity decodable from the final state when the pod is addressed
  by its own key, and not when it is reached through a pointer. Bar: the asymmetry is recorded only if
  direct >= 0.20 AND direct - alias >= 0.20 on the worst seed, with `shred_alias_true_object <= 0.05`
  from E-000063 as the validity floor that the OUTPUT is closed in both modes. Otherwise it is
  reported as a table and no sentence is drawn. This row is a DISCLOSED POST-HOC ADDITION: it was not
  in the design, it was found by a plumbing run, and it is the row a reader should discount most.

**The mediator measurement, made after the bars above and before the recorded run.** On seed 0 over
eight pods at template 3, comparing a live pod against a shredded one and against a never-written one,
max-abs at the last token: block 8 **0.0000** / 0.0038, block 9 **0.0000** / 0.0079, block 10 **68.84**
/ 50.57, block 11 **288.87** / 263.64; and the injected write itself, first read site (block 8)
**0.0000**, second read site (block 10) **68.84**. The first read site injects nothing pod-specific
for these prompts. This is why `A0` exists and why the first draft's DIMENSION/DIRECTION/DEPTH triple
was insufficient: it assumed the audit's site was a write site. The `A0` branch is therefore expected,
and it is reported as a CORRECTED INSTRUMENT SITING rather than as a discovery about linear
decodability.

**What is still blind:** seeds 1 and 2 entirely (their checkpoints did not exist when this was
written), ten of the twelve templates, the direct-read rows, part A at full size, and every
per-template ordering.

## Prior art, and the exact boundary

Owned, and not claimed here in any part: the Jacobian lens and the workspace framing (Gurnee et al.,
transformer-circuits.pub, 6 July 2026), including its own concept-swap and clamp experiments;
J-lens accessibility auditing of unlearning (J-Access, Song et al., arXiv:2608.11408) and its finding
that optimising such an audit games it; the theory of what a J-lens readout means and its
short-horizon limits (arXiv:2608.25347); LRP-corrected lenses (R-lens); matched random-atom-span
ablation controls (arXiv:2609.01924); activation-patching unlearning depth (arXiv:2605.24614);
information-decomposition audits (arXiv:2601.15111); activation-perturbation elicitation
(arXiv:2505.23270); probing and transfer probes generally; projection and concept erasure (LEACE);
writing unembedding rows as memory (Memory Injections; linear relational embeddings); external and
editable memory (SERAC, GRACE, WISE, DKME, Larimar, KBLaM, MUNKEY, LMLM, SILO); canonical records and
pointer aliases (Codd; Raeesi & Roed arXiv:2607.00605 §9; lemmalog; Letta blocks); versioned memory
objects (PAMSPEC; Kumiho arXiv:2603.17244); KV-cache lifecycle (IBM US20260119893A1; ReCache; Leyline;
KVEraser; FinCacheServe); varying database state at inference to audit forgetting (Raeesi & Roed's
FULL / DEL-ON / DEL-OFF).

The boundary this experiment sits inside: **where an accessibility audit of an EXTERNAL memory must be
sited.** J-Access reads a mid-to-late band of a model whose knowledge is parametric, and reports
item-level AUROC at chance — it has no counterfactual in which the item was never known. A canonical
pod store supplies exactly that counterfactual per item over identical frozen weights and an identical
prompt, and it supplies the write as a tensor that can be decomposed. What is measured here is the
siting question those two facts make answerable, and nothing more.

## What is NOT claimed

- Not that the J-lens, J-space or any audit is wrong. What is measured is where it can be read.
- Not a deletion guarantee, a certificate, or an unlearning result. `shred_alias_unknown` is 1.0000 by
  the gate's construction (§31.53 item 3) and carries nothing.
- Not a capability comparison between the trained write and the `atom` write: `atom` is an oracle.
- Not a statement about parametric knowledge, about models above 124M, about multi-token entities,
  about free text, or about any backbone but GPT-2 small.
- Not novelty for any mechanism, basis, probe, projection or store operation used here.
- Not a claim that norm-share should have predicted effect-share; that it does not is a methodological
  note, and it is the reason B5 carries no bar.

## Kill conditions

1. `dropT_n` fires (B3): the result is about removing a rank-257 lens span of this construction, not
   about the scored tokens, and part B is void.
2. Any validity row fails on any seed.
3. Part A comes out A4 (mixed): no siting sentence is licensed.
4. B1 fails to reproduce on seeds 1 and 2 or on the ten unseen templates: the seen cells were the
   result and the part is recorded as not reproducing.
5. A published work already reports where an accessibility audit of an external memory must be sited,
   with a never-memory control.

## Reproduction

```bash
SO_BOS=1 SO_CKPT_SUFFIX=_bos python -m so.experiments.e000052_symlink_bos_train --seeds 0 1 2   # ~48 min/seed
SO_BOS=1 python -m so.experiments.e000063_workspace_pod_certificate --seed 0 \
    --checkpoint so/results/checkpoints/e000020_gpt2_bos_seed0.pt --results-dir so/results/e000063
SO_BOS=1 python -m so.experiments.wsc001_workspace_share --seeds 0 1 2 --threads 4
```
