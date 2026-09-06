# The claim: a workspace accessibility audit certifies nothing until it is sited past the write

*2026-09-06. Pre-registered at `docs/novelty/wsc001-preregister.md` before the recorded run, with the
disclosure of what had already been seen written into that file. This is a measurement and an
instrument-siting result. It is not a mechanism, not a deletion guarantee, and not a claim of legal
novelty. Two of the four things this document set out to say are **withdrawn** below, by their own
pre-registered bars.*

Records: `so/results/e000063/e000063_workspace_pod_certificate-seed{0,1,2}.json`,
`so/results/wsc001_workspace_share.{json,md}`. Three seeds, worst seed reported throughout, 16 pods
per seed, frozen GPT-2 small, BOS-trained symlink adapter.

## The sentence

> On a frozen GPT-2 reading an external canonical-pod store, a Jacobian-lens accessibility audit
> placed at the memory's first read site certifies nothing, because the pod's content is not there.
> Measured: when a pod is shredded, the injected write at the first read site moves by **0.032** of
> the second's (worst seed 0.049), and at that site **no** readout of any kind separates live memory
> from never-written memory — the audit's own atoms, a dimension-matched random projection, the top-k
> principal components of the live states, the unembedding rows, and the full 768-dimensional residual
> all lie between **−0.022 and +0.000**. One block later every one of them separates them at **0.66 to
> 0.92**. The consequence is not academic: E-000063's composed store-and-audit certificate returns its
> headline verdict, *no workspace trace after deletion*, on all three seeds, while its own
> pre-registered validity row — can this instrument see a live pod at all? — fails on all three.

## What that looks like in the records

### E-000063, the composed certificate, three seeds

The experiment was designed on 2026-09-05 and had never produced a number: its CI run (33955376015)
died in the vector-Jacobian product on both seeds after 31 minutes of training each, because
`so/jlens.py` differentiated a forward whose parameters the adapter had frozen (ledger §31.56).

| row | seed 0 | seed 1 | seed 2 | bar | |
|---|---|---|---|---|---|
| `active_alias_correct` | 0.8795 | 0.9732 | 0.9821 | ≥ 0.80 | the memory is read |
| `shred_alias_unknown` | 1.0000 | 0.9955 | 1.0000 | ≥ 0.90 | one SHRED closes every alias |
| `shred_alias_true_object` | 0.0000 | 0.0000 | 0.0000 | ≤ 0.05 | |
| `bystander_top1_agree` | 1.0000 | 1.0000 | 1.0000 | ≥ 0.98 | locality is exact |
| `finalprobe(ACTIVE − NEVER)` | 0.9018 | 0.8750 | 0.8616 | ≥ 0.30 | the memory is in the final state |
| **`jprobe(ACTIVE − NEVER)`** | **−0.0134** | **−0.0045** | **−0.0134** | ≥ 0.30 | **FAIL on every seed** |
| `jprobe(SHRED − NEVER)` | −0.0179 | −0.0045 | −0.0134 | ≤ 0.05 | the headline row, passes |

The headline row passes underneath a failed validity row, on every seed, including two where the
memory is read at 0.97 and 0.98. So this is not "the memory was weakly read".

### WSC-001 part A: where the audit can be sited

ACTIVE − NEVER, worst seed, probes trained on ACTIVE states only and applied unchanged:

| family | direct site8 | site9 | site10 | final | alias site8 | site9 | site10 | final |
|---|---|---|---|---|---|---|---|---|
| `jspace` (the audit) | +0.000 | +0.000 | **+0.893** | +0.866 | −0.013 | −0.022 | **+0.915** | +0.915 |
| `random` (dimension-matched) | +0.000 | +0.000 | +0.741 | +0.812 | +0.000 | −0.004 | +0.871 | +0.835 |
| `pca` (best k-dim readout) | +0.000 | +0.000 | +0.661 | +0.857 | −0.009 | −0.018 | +0.817 | +0.893 |
| `unembed` | +0.000 | −0.009 | +0.893 | +0.866 | +0.000 | −0.018 | +0.915 | +0.915 |
| `raw` (768 dimensions) | +0.000 | +0.000 | +0.777 | +0.750 | −0.018 | −0.009 | +0.862 | +0.862 |

The mediator is what separates *blind* from *correctly sited and looking at the wrong block*. Mean
max-abs movement of the state when the pod is shredded: site8 **2.81**, site9 **4.19**, site10
**83.55**, final **27.06**; and of the injected write itself, first read site **2.81** against second
**89.70**, a ratio of **0.032**. Nothing is decodable at site8 because almost nothing is there.

**This is A0 of the pre-registered rule**, and it is the surviving sentence of this document.

### A6, the pointer: the audit is closed under indirection

`direct − alias` on the audit's own family at the final state: mean **−0.010**, range −0.063 to
+0.054. An audit calibrated on a pod addressed by its own canonical key transfers to the same pod
reached through a LINK alias the model must dereference; if anything it reads the alias path very
slightly better. This is the endorsement branch, and it is reported as such: the pointer does not
break the audit.

## Withdrawn

**Part B is VOID.** The pre-registration's kill condition 1 fired: the token-matched null, a J-lens
span built identically over 257 tokens that are *not* scored, costs the answer more than the bar
allowed — `dropT_n` deficit **−0.260** on the worst seed against a −0.20 bar (worst cell template 9,
−0.230). So the registered sentence, *the scored-token atom span is the causal channel of the write*,
is **not licensed**, and it is withdrawn rather than softened. The numbers are kept as a table with no
sentence attached: removing the scored-token span takes aliased reading from ~0.95 to **≤ 0.16** at
every template and seed, removing the token-matched span costs between 0.02 and 0.26 (worst seed, worst template), the oracle write
behaves the same way (`atom_dropW` ≤ 0.02), the content floor is 0.0000 and the no-memory floor
0.0150. The asymmetry is large; the null is not clean at every template; the bar said that voids the
part, and it does.

**A validity row also failed, and the failure is partly mine.** The pre-registration writes V1 as
`full/alias ≥ 0.80 and full/direct ≥ 0.90` **at the anchor template 3**; the code computes the minimum
over all twelve templates. Read as written, V1 passes (t3: alias 0.940, direct 1.000). Read as coded,
it fails (alias 0.690 at template 1, direct 0.835). The discrepancy is a defect in this experiment's
implementation of its own pre-registration, it is recorded rather than reconciled after the fact, and
either reading leaves part B void.

**A7 is withdrawn.** The disclosed post-hoc row — a shredded pod staying decodable from the final
state under direct addressing but not through a pointer, seen at +0.457 against −0.021 in a
ten-pod single-template plumbing run — does **not** replicate. At 16 pods over three seeds the
asymmetry is **−0.016** (direct −0.036, alias −0.019). The pre-registration called this "the row a
reader should discount most". It was right.

## What is genuinely new here, stated narrowly

Not the lens (Gurnee et al.), not lens-based accessibility auditing (J-Access, Song et al.,
arXiv:2608.11408), not probing, projection, external memory, canonical pods or pointer aliases — the
pre-registration carries the full boundary list. What this adds:

1. **A siting rule with its mediator.** For an external memory, an accessibility audit must be read at
   or after the site that carries the content, and "the audit sees nothing here" is only evidence when
   the state at that site actually moves. Every number above is paired with how far it moves.
2. **A demonstrated vacuity mode.** A composed store-and-audit certificate can return its deletion
   verdict while its instrument has never seen the object live, and the thing that catches it is a
   validity row, not the verdict. Reproducible in 36 seconds on a trained checkpoint.
3. **The counterfactual that makes it measurable.** J-Access reports item-level AUROC at chance
   because parametric unlearning has no per-item "never knew it" control. A pod store has one, over
   identical frozen weights and an identical prompt, and that is the whole reason these numbers exist.

## What is not claimed

- Not that the Jacobian lens or any audit is wrong. What is measured is where it can be read.
- Not a deletion guarantee. `shred_alias_unknown` is forced by the gate's construction (§31.53 item 3).
- Not a capability comparison against the `atom` arm, which is an oracle handed the answer.
- Not a mechanism, a novel basis, a novel probe, or a novel store operation.
- Not a statement about parametric knowledge, models above 124M, multi-token entities, free text, or
  any backbone but GPT-2 small.
- Nothing about the causal channel of the write: part B is void.

## How it could be wrong

- Sixteen pods per seed, three seeds, one synthetic world per seed, single-token entities. The
  resampling unit is the pod and the three worlds are the replicates.
- The audit's site8 numbers are near-exactly zero partly because the probe is saturated by prompt
  identity: NEVER already reads the alias name. The mediator is what rules out the alternative
  explanation, and the mediator is a norm, not a decodability measure.
- The first read site is small, not silent (ratio 0.032, not 0). The claim is relative and says so.
- `random` at site10 reaching 0.74–0.87 says the readable signal there is broad, so "the audit works
  downstream" is not evidence that the audit's particular basis is the right one.

## Reproduce

```bash
SO_BOS=1 SO_CKPT_SUFFIX=_bos python -m so.experiments.e000052_symlink_bos_train --seeds 0 1 2
for s in 0 1 2; do SO_BOS=1 python -m so.experiments.e000063_workspace_pod_certificate --seed $s \
    --checkpoint so/results/checkpoints/e000020_gpt2_bos_seed$s.pt --results-dir so/results/e000063; done
SO_BOS=1 python -m so.experiments.wsc001_workspace_share --seeds 0 1 2 --threads 4
```
