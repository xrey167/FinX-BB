# LOD-001 — what is the smallest residue this audit could have seen?

Date: 2026-09-06
Status: **pre-registered before any number from this experiment exists.** The code
(`so/experiments/lod001_detection_limit.py`) was written first, this file second, and neither has been
run at the time of writing. The disclosure of what was already known is below and it is large.
Classification: instrument calibration. **No novelty credit** for the Jacobian lens, for lens-based
accessibility auditing, for probing, for projection, for dose-response calibration, for limits of
detection as an idea, for external memory, for canonical pods, for pointer aliases, or for any of
these in loose combination.

## Where this comes from

WSC-001 (`docs/novelty/audit-siting-claim.md`, 2026-09-06) recorded that at E-000063's capture site no
readout of any kind separates a live pod from one that was never written (−0.022 to +0.000 across five
feature families and both address modes), while one block later every family separates them at +0.66
to +0.92. It separated *the audit is blind here* from *nothing is here* with a **mediator that is a
norm** — how far the state moves when the pod is shredded — and its own "how it could be wrong"
section names that as the weakness:

> The mediator is what rules out the alternative explanation, and the mediator is a norm, not a
> decodability measure.

This experiment is the decodability measure the norm stands in for.

An audit that reports *no residual trace after deletion* is asserting something about every state
between a live memory and none: that had a residue been present, it would have been seen. Outside this
field that assertion has a name — a **detection limit** — and no accessibility audit of a language
model states one, for a reason that is structural rather than negligent: a residue is not a thing a
parametric memory can be given in a known amount. J-Access (Song et al., arXiv:2608.11408) reports
item-level AUROC at chance and says why — parametric unlearning has no per-item counterfactual in
which the item was never known. Here the memory is external, the counterfactual is a deleted row, and
the write is a tensor this harness holds. So it can be given in a known amount, and this experiment
gives it in ten.

## The two ladders, and why there are two

| | axis | what a rung is | what it idealises away |
|---|---|---|---|
| **write-side** | `alpha` ∈ {0, 0.031, 0.062, 0.125, 0.188, 0.25, 0.375, 0.5, 0.75, 1.0} | the adapter's write at `alpha` of its trained size (`so/llm_adapter.py`, `set_inject_scale`) | how a partial residue would arise in the store; it is a magnitude knob and nothing more |
| **store-side** | marker chord `c` ∈ {0, 0.5, 0.7, 0.75, 0.8, 0.825, 0.85, 0.9, 1.0} | the pod's active version signed at chord `c` from the store's centre, which moves the neural gate `g` in `encode_bank` and attenuates the payload **toward ' unknown'** | nothing about the lifecycle: it is a SHRED applied by halves. `c = 0` is the live pod; a full SHRED is an unsigned marker outside the valid radius |

The second ladder exists because a reviewer is entitled to say that scaling a write is an arbitrary
intervention rather than a deletion residue. The chord ladder is the store's own lifecycle dial:
ledger §31.48 R3 measured this gate's acceptance on three checkpoints at 1.0000 for every chord to
0.7, 0.0055 / 0.0935 / 0.0430 at 0.8, and 0.0000 at 0.9 and beyond, so the rungs bracket a transition
that was measured before this experiment was designed. **If the two ladders put the limit at the same
place, the write-side ladder is not an artefact of its own arbitrariness. If they do not, the chord
ladder is the one any sentence is read from**, because it is the operation applied by halves and the
other is a knob.

## What is measured

WSC-001's part A exactly, with the binary ACTIVE / SHRED / NEVER replaced by the ladders. Sites: the
outputs of blocks 8, 9, 10, and the final state. Five feature families over the same states — `jspace`
(the audit's own atoms), `random` (dimension-matched), `pca` (top-k of the live states), `unembed`,
`raw` (768 dimensions). Both address modes: the pod's canonical key, and a LINK alias the model must
dereference. Probes are trained on the **live arm only** and their identical weights score every rung
and the never-written control, five-fold by template — E-000063's design, and its stated reason: the
alias text in the prompt must not be readable as recovered memory.

**The detection limit** for a family at a site is the smallest rung whose separation from NEVER
reaches `delta = 0.30`. That bar is not new: it is the bar E-000063's own validity row uses to decide
that its instrument can see a live pod. Stating the limit against any other number would make it
incomparable to the certificate it is about. `detected = false` — no rung clears the bar — is the
outcome that matters, and it means the limit is **above the top of the ladder**: the audit could not
have seen the memory with all of it present.

**The behavioural limit**, in the same units: the smallest rung at which the model's answer returns
the pod's true object, `delta` above the never-written rate. The field's standing justification for
internal-state audits is that output forgetting is insufficient. These two numbers, in one unit, are
that justification as a quantity.

## Blindness, disclosed — and it is not small

**Seen before this file was written.** All of WSC-001's recorded part-A table and all of E-000063's
three-seed table, both in this repository. That means **the `alpha = 1` rung of every row below is
already known**: at site 8 the five families read −0.022 to +0.000, at site 10 and the final state
+0.66 to +0.92, and `active_alias_correct` is 0.8795 / 0.9732 / 0.9821. L1's `alpha = 1` endpoint is
therefore a **REPRODUCTION, not a discovery**, and it is labelled so wherever it is reported. The
recorded checkpoints do not survive (`so/results/checkpoints/` is ignored), so even that endpoint is a
fresh training of the named recipe and may reproduce the record's shape and not its level, exactly as
ledger §31.48 found.

**Not seen, and not predictable from what was seen:** every intermediate rung of either ladder; the
whole chord ladder; the behavioural limit at any rung below 1.0; the comparison between the audit's
limit and the output's; whether the curves are monotone; the `alpha = 0` floor with the row still
present; and every per-seed and per-template ordering. A readout that is non-monotone in the dose
would separate at an intermediate rung while failing at the top, and nothing in the record rules that
out — which is the sense in which L1 can fail.

## By construction — declared, never scored

- `alpha = 1` is the trained forward, bit-identically (`so/tests/test_inject_scale.py`).
- `alpha = 0` injects exactly zero and is therefore the same state as `set_inject_projection(None,
  "zero")`, the no-memory floor the records already use (same test).
- The first read site's injected vector is exactly `alpha` times its full-size value, and the second
  site's is **not**, because the block-8 write is in the residual the block-10 read builds its routing
  query from (ledger §31.56). The ladder is per-site for the reason the projection arms are.
- At the final state the lens degenerates to the unembedding basis (§31.56 measures cos 1.000 at layer
  11), so `jspace` and `unembed` are the same object there and their agreement is not evidence.
- `pca` is fitted on the live arm and applied unchanged to every rung. Refitting it per rung would make
  each rung a different readout and the ladder would not be one curve.
- The achieved chord of a rung is not the requested one: a valid marker already sits a short way off
  the centre (`MARKER_SCALE = 0.05`), so the achieved chord and the gate it produces are both recorded,
  and the chord rows are read in the gate.

## Validity rows — worst seed of three; a failure VOIDS the part named

- `V1` `alias/answer/a1 ≥ 0.80`. The memory is read at the top of the ladder. Reproduces E-000063's
  `active_alias_correct`. Failure voids everything.
- `V2` `alias/final/raw/a1_minus_never ≥ 0.30`. Some readout, somewhere, sees the live memory, or the
  states were not captured. WSC-001's V4. Failure voids everything.
- `V3` **the zero-dose anchor**: `alias/answer/a0 − alias/answer/never ≤ 0.05` **and**
  `|alias/final/raw/a0_minus_never| ≤ 0.10`. A row that routes but writes nothing must not be
  decodable, or `alpha = 0` is not a zero-dose anchor and the write-side limits are measured against
  the wrong floor. **This can fire**, and if it does the write-side ladder is read against NEVER only
  and the finding is reported as a tombstone footprint. Failure voids the write-side part.
- `V4` **the store-side ladder is a dose axis**: `gate_by_chord[c=0] ≥ 0.90` and
  `gate_by_chord[c=1.0] ≤ 0.10`. Otherwise the chord rungs are not doses and the store-side part is
  VOID.

## The rows, and the branches, written out before the run

- `L1` **the vacuity limit.** At `site8`, `LOD_alpha.detected == false` for **all five families in both
  address modes on all three seeds** → *the audit sited at the certificate's own block has a detection
  limit above full retention: it could not have seen the pod with all of it present, so its PASS after
  a deletion carries no information about the deletion.* The `alpha = 1` endpoint of this row is
  disclosed above as a reproduction; what is new is that **no intermediate rung does better**.
- `L2` **the resolving limit.** At `site10` and at `final`, `jspace/LOD_alpha.detected == true` on the
  worst seed, and the rung is reported as the number. If it does not detect at any rung the ladder
  resolves nothing on this substrate and `L3` is VOID.
- `L3` **audit against output, in one unit**, worst seed, `alias` mode, read only with `L2` holding:
  - `L3a` `audit_LOD ≤ output_LOD` at both `site10` and `final`, by at least one rung → *the internal
    audit resolves a residue the output does not, and this is by how much.*
  - `L3b` `audit_LOD ≥ output_LOD` at both sites, by at least one rung → *on this substrate the
    internal audit is dominated by asking the model, and the standing justification for auditing
    internals instead of outputs does not hold here.*
  - `L3c` within one rung at either site → no sentence; the table.
- `L4` **the two ladders against each other.** The gate value at the chord limit, against the gate the
  write-side limit corresponds to. Agreement within 0.10 → the write-side ladder is not an artefact of
  its own arbitrariness. Disagreement → the chord ladder carries every sentence and the write-side
  ladder is reported as a magnitude knob only.
- `L5` **the tombstone's own footprint**, reported at every site and family with a bar of 0.05:
  `a0_minus_never`. A live-but-empty row that is decodable is a finding about what a deletion leaves
  addressable, and it is `V3`'s failure branch seen from the other side.
- `L6` reported with no bar: the full curve for every family, site, mode, ladder and seed; the
  mediator rows WSC-001 records (`shred_moves`, `never_moves`, per-site write norms); the achieved
  chord and gate per rung.

## Kill conditions

1. **Any validity row fails** → the part it names is VOID and no sentence is read from it.
2. **`K1`, against `L1`:** any family detects at any rung ≤ 1.0 at `site8`, on any seed, in either
   address mode → the vacuity sentence is **not licensed**; the observed limit is reported instead and
   `L1` is withdrawn rather than softened.
3. **`K2`, against everything:** if `random` and `jspace` place the limit at the same rung at every
   site, the audit's particular basis carries nothing here and **no sentence may name the J-lens** —
   every sentence becomes about *a readout at this site*. WSC-001 already measured `random` reaching
   0.741 to 0.871 at site10, so this condition is expected to fire in part and is written as a
   constraint on wording, not as a void.
4. **`K3`, reproducibility:** any limit that differs by more than two rungs across the three seeds is
   reported as unstable and carries no sentence.
5. **`K4`, the ladder is not a ladder:** if the behavioural curve is non-monotone in `alpha` by more
   than 0.10 anywhere, the write-side rung ordering is not a dose ordering and `L3` is read from the
   chord ladder alone.

No bar in this file is adjusted after the run. A bar that turns out to have been mis-set is reported
as mis-set, in place, which is what ledger §31.54 did.

## Prior art, and the exact boundary

Owned, and claimed in no part of this: the Jacobian lens and the workspace framing (Gurnee et al.,
2026); lens-based accessibility auditing of unlearning and the finding that optimising such an audit
games it (J-Access, Song et al., arXiv:2608.11408); probing, transfer probes and control tasks;
projection and concept erasure; activation steering and its coefficient sweeps; graded and
dose-response interventions in interpretability generally; causal scrubbing and graded ablation;
limits of detection, calibration curves and power analysis as statistical ideas, which are a century
old; per-example membership-inference nulls and the "out model" construction (LiRA and the DP-auditing
literature); varying database state at inference to audit forgetting (Raeesi & Roed's FULL / DEL-ON /
DEL-OFF); external and editable memory (SERAC, GRACE, WISE, DKME, Larimar, KBLaM, LMLM); canonical
records and pointer aliases; versioned memory objects; KV-cache lifecycle (IBM US20260119893A1;
ReCache).

The boundary this experiment sits inside is one question: **what is the smallest residue an
accessibility audit of an external memory could have seen, and is that number a property of the
deletion or of where the audit was sited?** Every ingredient above is old. The number is not one any
of them reports, and the reason is the one J-Access states about itself — the dose does not exist on a
parametric substrate.

## What is NOT claimed

- Not that the J-lens, J-space, or any audit is wrong. What is measured is what it could have seen.
- Not a deletion guarantee, a certificate, or an unlearning result.
- Not a mechanism, a novel basis, a novel probe, or a novel store operation.
- Not that a scaled write is what an incomplete deletion produces; that is what the chord ladder is
  for, and if the two disagree the write-side ladder is demoted in place.
- Not a statement about parametric knowledge, about models above 124M, about multi-token entities,
  about free text, or about any backbone but GPT-2 small.
- Not a claim of legal novelty or of being first. That question is settled by a professional search,
  not by this file.

## Reproduce

```bash
SO_BOS=1 SO_CKPT_SUFFIX=_bos python -m so.experiments.e000052_symlink_bos_train --seeds 0 1 2
SO_BOS=1 python -m so.experiments.lod001_detection_limit --seeds 0 1 2 --threads 4
```
