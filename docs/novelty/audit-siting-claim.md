# The claim: an accessibility audit of an external memory is causally right and evidentially blind at the write site

*2026-09-06. Status: research-level measurement claim, pre-registered at
`docs/novelty/wsc001-preregister.md` before the recorded run, with the disclosure of what was seen
first written into that file. Not a mechanism, not a deletion guarantee, and not a claim of legal
novelty.*

**PLACEHOLDER — this document is written against the pre-registration and is completed only when the
three-seed records exist. Every number below marked `[…]` is filled from
`so/results/wsc001_workspace_share.json` and `so/results/e000063/`. If a bar fails, the sentence it
supports is withdrawn here rather than softened.**

## The sentence

> On a frozen GPT-2 small reading an external canonical-pod store through pointer rows, a
> Jacobian-lens accessibility audit sited at the memory's own write site is **causally correct and
> evidentially blind**. Causally correct: the span of the lens atoms of the scored tokens is exactly
> the channel the write acts through — injecting the write with that span projected out takes reading
> from `[…]` to `[…]`, while projecting out a span built identically over the same number of tokens
> that are *not* scored costs `[…]`. Evidentially blind: at that same site no linear readout, from the
> audit's own atoms up to the full 768-dimensional residual, attributes the pod's content beyond what
> the prompt already reveals (`[…]` against a never-written control), so the audit's post-deletion
> certificate passes while its own validity row fails. The audit becomes evidential only downstream,
> where it separates live from never-written memory at `[…]` and where a trace survives SHRED at
> `[…]`.

And a corollary that is cheap to state and easy to get wrong:

> **Norm-share is not effect-share.** Only `[…]` of the write's squared norm lies in the span that
> carries all of its behaviour — less than the `0.3346` an isotropic random vector would put there.

## What this rests on, and what it does not

**It rests on an experiment that had never produced a number.** E-000063, the composed workspace-pod
deletion certificate, was designed on 2026-09-05 and its CI run died in the vector-Jacobian product on
both seeds after 31 minutes of training each, because `so/jlens.py` differentiated a forward whose
parameters the adapter had frozen (ledger §31.56). With that repaired it runs in 36 seconds on a
trained checkpoint, and its first record is the reason this claim exists.

**It does not rest on any mechanism being new.** The lens is Gurnee et al.'s; using it as an
accessibility audit is J-Access's; probes, projections, erasure, canonical records, pointer aliases,
versioned pods and KV lifecycle are all owned and are listed in the pre-registration's boundary
paragraph. What the pod store supplies is the one thing a parametric setting cannot: **a per-item
counterfactual in which the fact was never known**, over identical frozen weights and an identical
prompt — the control whose absence J-Access names when it reports item-level AUROC at chance.

## The measurements

### E-000063, the composed certificate — the first record

`[…]` seeds, `[…]` pods, BOS-trained symlink adapter.

| row | value | bar | |
|---|---|---|---|
| `active_alias_correct` | `[…]` | ≥ 0.80 | |
| `shred_alias_unknown` / `shred_alias_true_object` | `[…]` | ≥ 0.90 / ≤ 0.05 | |
| `bystander_top1_agree` / `bystander_kl_max` | `[…]` | ≥ 0.98 / ≤ 0.05 | |
| `finalprobe(ACTIVE − NEVER)` | `[…]` | ≥ 0.30 | |
| **`jprobe(ACTIVE − NEVER)`** | `[…]` | ≥ 0.30 | **the validity row** |
| `jprobe(SHRED − NEVER)` | `[…]` | ≤ 0.05 | the headline row |
| `finalprobe(SHRED − NEVER)` | `[…]` | ≤ 0.05 | |

### WSC-001 part B — the causal side

`[…]`

### WSC-001 part A — the readout side, by address mode

`[…]`

## Why each half could have come out otherwise

`[…]`

## What is not claimed

- Not that the Jacobian lens, J-space, or any audit is wrong. What is measured is **where** it can be
  read and what its verdict is worth there.
- Not a deletion guarantee. `shred_alias_unknown` is forced by the gate's construction (§31.53 item 3)
  and carries nothing on its own.
- Not a capability comparison against the `atom` arm, which is an oracle handed the answer.
- Not a mechanism, not a novel basis, not a novel probe, not a novel store operation.
- Not a statement about parametric knowledge, about models above 124M, about multi-token entities, or
  about any backbone other than GPT-2 small.

## How it could be wrong

`[…]`

## Reproduce

```bash
SO_BOS=1 SO_CKPT_SUFFIX=_bos python -m so.experiments.e000052_symlink_bos_train --seeds 0 1 2
SO_BOS=1 python -m so.experiments.e000063_workspace_pod_certificate --seed 0 \
    --checkpoint so/results/checkpoints/e000020_gpt2_bos_seed0.pt --results-dir so/results/e000063
SO_BOS=1 python -m so.experiments.wsc001_workspace_share --seeds 0 1 2 --threads 4
```
