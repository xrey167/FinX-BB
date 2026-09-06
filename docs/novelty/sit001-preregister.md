# SIT-001 pre-registration — can the audit window be reopened by moving the write?

*Written 2026-09-06, before any number was read from the `_bos_early` checkpoints and before
`so/experiments/sit001_audit_window.py` was run on anything. What had already been seen when these
bars were fixed is listed under **Disclosure** and is not small.*

## The question

`docs/novelty/audit-siting-claim.md` establishes that E-000063's accessibility audit returns its
deletion verdict from a block the pod's content never reaches, and that this is a siting error rather
than a fault of the lens, the basis, the probe or the dimension. It leaves the useful half unanswered:
**is there any siting at which that certificate would mean something, and if not, what would have to
change?**

Formalised in `so/siting.py`, a site is admissible for an accessibility audit only if both hold:

| half | what it asks | measured by |
|---|---|---|
| **ARRIVAL** | has the item's content reached this state at all? | the mediator: how far the state moves when the pod is removed, relative to the largest movement over the candidate sites |
| **DISTINCTNESS** | is this audit a different instrument from reading the output? | mean `|cos(v_u, W_U[u])|` between each token's lens vector here and its own unembedding row |

The **audit window** is the set of sites where both hold. Bars, fixed here:

* **W1 ARRIVAL** — `arrival_ratio ≥ 0.50`
* **W2 DISTINCTNESS** — `|cos| ≤ 0.90`

## Disclosure: what these bars had already seen

This matters more than usual, because on the recorded adapter both rows are already in the ledger.

1. **Arrival is measured.** WSC-001 (ledger §31.57): under SHRED the state moves 2.81 / 4.19 / 83.55 /
   27.06 at blocks 8, 9, 10 and the final state, i.e. arrival ratios 0.034 / 0.050 / 1.000 / 0.324.
2. **Distinctness is measured.** Ledger §31.56: the mean cosine between a token's J-lens vector and its
   own unembedding row on pretrained GPT-2 small is 0.310 at hidden state 1, 0.526 at 3, 0.593 at 5,
   0.698 at 7, 0.750 at 8, 0.783 at 9 and 10, and **1.000 at 11**.

So **the empty window on `read_layers=(8, 10)` is a re-statement of two measurements this repository
already holds**, assembled into one rule for the first time. It is reported as a consolidation and
never as a finding, and it is pinned as a fixture in `so/tests/test_siting.py` rather than re-measured
to look new. Its bar sensitivity is also measured rather than asserted: the window is empty for every
floor at or above 0.06 and every ceiling in [0.80, 0.99], and stops being empty only below a floor of
0.0502, which would admit a site carrying five per cent of the movement.

3. **Two training lines have been seen.** A 20-step smoke run at `read_layers=(2, 4)` (step 1: loss
   6.9152, batch accuracy 0.531) confirming the config trains at all, and seed 0 of the registered arm
   at step 100 (loss 2.3636, answer 6.9663, route 1.4049, accuracy 0.469), seen while checking the
   background process was alive. Neither is an outcome of this pre-registration; both are recorded
   because a disclosure list that omits what was seen is worthless.

Nothing else from the `_bos_early` checkpoints has been read.

## The arm

The identical symlink adapter — same trainer (`E20.train_adapter_links`), same 3000 steps, same
budget, same BOS, seeds 0, 1, 2 — with **`read_layers=(4, 6)`** in place of `(8, 10)`; checkpoints
under `SO_CKPT_SUFFIX=_bos_early`, trained by `so/experiments/sit001_early_read_train.py`, which
records nothing.

**Why (4, 6) and not (2, 4), fixed before training.** The routing query is read from the residual at
the read layer, so a read placed before the subject is resolved is this arm's most likely way to die,
and the arm is worthless if the window is bought by breaking the memory. (4, 6) leaves four blocks
after the second write and sits at cosine 0.698–0.783 over its candidate sites — inside W2 — while
keeping the query on a state deep enough to carry the subject. (2, 4) would give a less degenerate
window (down to 0.593) at a higher capability risk, and is not registered.

## Validity bars — the arm can die, and here is how

Worst seed of three, at E-000063's templates and its 16 unique-object pods:

* **VE1** `active_alias_correct ≥ 0.80` — the early-read adapter can read a pod through its aliases.
* **VE2** `shred_alias_unknown ≥ 0.90` and `shred_alias_true_object ≤ 0.05` — deletion still reaches
  the output, so there is a deletion to audit.
* **VE3** `bystander_top1_agree ≥ 0.98` — locality holds, so the audit is about this pod.

If VE1 fails, **the arm is void as a statement about auditing** and is reported only as a capability
cost of moving the write, with no window claim attached. This is the most likely failure and saying so
in advance is the point.

## Predictions, in the order they can fail

* **P0 — the siting rule replicates at a new placement.** On the early adapter the FIRST read site
  (block 4) again fails W1: `arrival_ratio(block 4) < 0.50` on all three seeds. This is not implied by
  anything measured so far — the recorded adapter's first read site could have been late-specific — and
  if arrival at block 4 is ≥ 0.50 the siting rule is placement-specific and that goes on the record
  against the claim.
* **P1 — the window opens.** At least one site is in the window on all three seeds.
* **P2 — the audit can see a live pod there.** At the registered site, `jprobe(ACTIVE − NEVER) ≥ 0.30`
  on all three seeds. **This is exactly E-000063's failed validity row**, which reads −0.0134 /
  −0.0045 / −0.0134 on the recorded adapter.
* **P3 — the verdict, now non-vacuous.** Given P2, `jprobe(SHRED − NEVER) ≤ 0.05` on all three seeds.

**Site selection rule, fixed now to prevent choosing a site by its answer:** the certificate is read at
the **lowest-numbered block in the window on all three seeds**. Every site's rows go into the record;
only that one carries P2 and P3.

## Decision branches

* **VE1 fails** → arm void for auditing; report as a capability cost of the write placement.
* **P0 fails** → record it against the siting claim: the first-read-site result does not generalise
  across placements, and `docs/novelty/audit-siting-claim.md` gains that limit.
* **P1 fails** (window empty at (4, 6) too) → the window is not reopened by this much movement; report
  the curve and state that on GPT-2 small the two halves may be irreconcilable at any placement that
  keeps capability. No claim of a general impossibility from one backbone.
* **P1 and P2 pass, P3 passes** → the strongest available outcome: **a composed store-and-audit
  certificate whose audit is sited where the content is and where the lens is not the unembedding, and
  whose deletion verdict survives.** Claim: the audit window is a design constraint on auditable
  external memory, and it is satisfiable.
* **P1 and P2 pass, P3 FAILS** → the corrected siting flips the verdict: sited admissibly, the audit
  finds a trace the vacuous siting reported absent. E-000063's headline is retracted in the ledger, and
  this is the stronger result for the claim even though it costs the certificate.
* **P1 passes, P2 fails** → the window is **necessary but not sufficient**: the audit is blind at an
  admissible site for one of WSC-001's other reasons (DEPTH, DIRECTION, DIMENSION). The siting rule
  survives as a necessary condition and the claim document says so. This outcome is judged likely
  enough that it is written here rather than discovered.

## Kill conditions

1. If the early adapter's own answer floor (a bank with no memory) exceeds 0.05 at these templates,
   the capability rows are not interpretable and the arm is void.
2. If the mediator at every candidate site is 0 for some seed, that seed's counterfactual did not fire
   and the seed is reported as such rather than averaged in.
3. If `jprobe(ACTIVE − NEVER)` at the registered site passes only on a seed where `active_alias_correct`
   is below VE1, that seed does not count toward P2.

## By construction — not findings, whatever the numbers say

* The lens cosine falling at earlier sites is a property of the architecture already measured in
  §31.56; the arm does not claim it.
* The site with the largest mediator has arrival ratio 1.000 by definition of the ratio.
* W1 and W2's bars are calibrated on the recorded adapter, as disclosed above.
* An adapter that reads earlier writes earlier: P0 is about how much of the content arrives at the
  FIRST of two read sites, which is not fixed by the placement.

## Prior-art boundary

Unchanged from `docs/novelty/wsc001-preregister.md`. Nothing here claims novelty for the Jacobian lens
(Gurnee et al.), for lens-based accessibility auditing (J-Access, Song et al., arXiv:2608.11408), for
probing, projection, external memory, canonical pods or pointer aliases. What is new is narrow: the
admissibility rule, its two halves measured together, and whatever the early-read arm returns.
