# Modular Neural OS / Mutable Knowledge Architecture

**Complete Experiment & Evidence Ledger**

**Project:** Editable / erasable in-model knowledge for neural networks  
**Research direction:** Modular Neural OS / Mutable Knowledge / Neural-MVCC / Symlink Architecture  
**Status date:** 2026-09-02  
**Document type:** Experiment ledger / evidence report  
**Status:** Active research

---

## 1. Purpose of this document

This document is the experimental counterpart to the main project architecture document.

Its purpose is to record:

* hypotheses,
* experiments,
* implementations,
* tests,
* controls,
* seeds,
* observations,
* positive results,
* negative results,
* falsification attempts,
* unresolved questions,
* evidence strength,
* and the current breakthrough status.

The central rule is:

> A promising mechanism is not a breakthrough until the relevant experiment actually demonstrates it.

The project therefore separates:

1. conceptual plausibility,
2. synthetic demonstration,
3. controlled neural-network evidence,
4. transformer evidence,
5. real-LLM evidence,
6. scalable system evidence.

---

## 2. Central research question

The project investigates whether knowledge inside a neural model can become a first-class mutable object.

Instead of treating a trained model as one inseparable parameter blob, the intended architecture should allow knowledge to have properties such as:

```text
knowledge
├── identity
├── source
├── owner
├── version
├── dependencies
├── validity
├── provenance
├── confidence
├── lifecycle
└── deletion state
```

The ultimate question is:

> Can a neural model learn knowledge in a form that remains usable like normal learned knowledge while also remaining identifiable, versionable, replaceable and selectively deletable?

---

## 3. Experimental breakthrough definition

A true breakthrough requires substantially more than deleting a database entry or disabling an adapter.

The target chain is:

```text
WRITE
  ↓
INTEGRATE
  ↓
USE
  ↓
IDENTIFY
  ↓
UPDATE / VERSION
  ↓
DELETE
  ↓
VERIFY ABSENCE
  ↓
PRESERVE UNRELATED KNOWLEDGE
  ↓
RESIST RECONSTRUCTION
```

A candidate mechanism therefore needs to demonstrate at least:

### Selectivity

Target knowledge disappears.

### Retention

Non-target knowledge remains intact.

### Generalization

Deletion affects paraphrases and alternative queries, not merely one memorized prompt.

### Causal isolation

The effect follows from manipulating the intended knowledge structure.

### Reconstruction resistance

The supposedly deleted information cannot simply be recovered through another obvious route.

### Scalability

The mechanism should have a plausible path beyond toy models.

---

## 4. Evidence scale

The project uses the following conceptual evidence hierarchy.

| Level | Meaning |
|---|---|
| E0 | Idea only |
| E1 | Analytical / conceptual support |
| E2 | Toy implementation |
| E3 | Repeated synthetic evidence |
| E4 | Controlled neural-network evidence |
| E5 | Transformer evidence |
| E6 | Real pretrained LLM evidence |
| E7 | Scalable / externally reproduced evidence |

A result at E2–E3 is valuable, but it must not be presented as E6.

---

## 5. Historical experimental branch

Before the project was narrowed to its present evidence-first core, several mechanisms were explored.

These included:

* Symlink Adapter,
* Source Marker,
* Aiko Marker,
* Crypto-Shredding,
* Immune Response,
* dependency structures,
* knowledge Pods,
* mutable knowledge cells,
* F0–F5 unlearning levels.

These ideas established much of the conceptual vocabulary of the current project.

They are not all independently validated mechanisms.

---

## 6. F0–F5 deletion model

One important early result was recognizing that “deletion” is not a binary concept.

The project therefore distinguishes several increasingly strong deletion levels.

### F0 — Access suppression

Knowledge still exists but normal access is blocked.

Example:

```text
query
  ↓
filter
  ↓
blocked answer
```

This is not neural deletion.

### F1 — Routing removal

A route to knowledge is removed.

Example:

```text
knowledge router
     ├── A
     ├── B  ← disabled
     └── C
```

The underlying information may still exist.

### F2 — Component removal

A dedicated module or adapter containing knowledge is removed.

This is stronger than F1 but still does not establish that the base model has forgotten the information.

### F3 — Functional forgetting

Behavioral tests indicate that the target information can no longer be retrieved.

However, internal traces may remain.

### F4 — Representational removal

Evidence indicates that the relevant representation itself has been removed or neutralized.

This requires stronger internal tests than behavioral evaluation alone.

### F5 — Reconstruction-resistant deletion

The strongest target.

The information:

* cannot normally be retrieved,
* cannot easily be recovered through paraphrases,
* cannot be reconstructed through known dependencies,
* does not reappear through obvious latent routes,
* while unrelated capabilities remain preserved.

F5 remains a research target rather than a completed claim.

---

## 7. Symlink hypothesis

The Symlink concept originated from a software analogy.

Instead of duplicating knowledge throughout a network:

```text
A → knowledge
B → knowledge
C → knowledge
```

the model would ideally learn something closer to:

```text
A ─┐
B ─┼──► K17
C ─┘
```

where K17 is a mutable neural knowledge object.

Deleting or replacing K17 could therefore modify multiple semantic access paths simultaneously.

### Experimental motivation

This would potentially address a major problem with ordinary model editing:

> The same fact can be represented redundantly across parameters and contexts.

A Symlink-like representation attempts to separate:

```text
semantic access
```

from:

```text
knowledge payload
```

### Current evidence

Status: promising architectural hypothesis.

What has been established so far is the usefulness and internal consistency of the abstraction.

What has not yet been established is that a large pretrained transformer naturally or controllably organizes arbitrary knowledge into clean Symlink objects.

Therefore:

| Claim | Status |
|---|---|
| Symlink concept | → supported |
| Symlink toy behavior | → investigated |
| LLM-scale Symlink | → unproven |

---

## 8. Source Marker / Aiko Marker

The Marker branch investigated whether learned information could carry a persistent provenance identity.

Conceptually:

```text
knowledge payload
      +
source identity
      +
lifecycle identity
```

rather than:

```text
anonymous gradient update
```

The Marker is intended to provide a handle for later:

* attribution,
* mutation,
* invalidation,
* deletion,
* auditing.

---

## 9. Crypto-Shredding experiment family

A further branch combined Marker identity with cryptographic destruction.

Conceptually:

```text
Knowledge K
   ↓
encoded / gated by key κ
   ↓
model can use K
DELETE:
destroy κ
   ↓
K becomes inaccessible
```

This creates an appealing analogy with cryptographic erasure.

### Important experimental conclusion

> Crypto-shredding alone does not prove neural unlearning.

Why?

Because the neural system could potentially have:

* copied the information,
* compressed it elsewhere,
* inferred it from correlated facts,
* stored alternative representations.

Therefore:

```text
key destruction
≠
proof of knowledge destruction
```

This was an important conceptual falsification.

Crypto-shredding remains potentially useful as a lifecycle/security mechanism, but cannot by itself satisfy the strongest unlearning claim.

---

## 10. Immune Response experiments

The Immune Response branch investigated whether the system could recognize invalidated knowledge and actively suppress or replace dependent representations.

Conceptually:

```text
invalidated K17
      ↓
dependency detection
      ↓
affected structures
      ↓
repair / quarantine / relearning
```

The idea remains relevant to future autonomous maintenance.

However, it was intentionally removed from the immediate core experiment because it introduces additional complexity before the fundamental mutable-knowledge mechanism has been demonstrated.

Current status:

> Architectural research branch — not core breakthrough evidence.

---

## 11. Reduction to the core experiment

A major methodological improvement was deliberately freezing much of the larger Neural OS vision.

The project moved from:

```text
Pods
+ Symlinks
+ Crypto
+ Immune System
+ Graphs
+ autonomous maintenance
+ provenance
+ distributed architecture
```

to one experimentally answerable question:

> Can we create a minimal neural system in which learned knowledge has an independently controllable lifecycle?

This produced the Mini-Transformer / Synthetic World research line.

---

## 12. Synthetic World

A synthetic environment was chosen because ground truth is completely controllable.

Example:

```text
Entity A → Property X
Entity B → Property Y
Entity C → Property Z
```

Advantages:

* exact knowledge provenance,
* exact dependency structure,
* controlled mutations,
* controlled contradictions,
* unlimited generated data,
* no contamination from pretrained knowledge,
* precise retention testing.

This is crucial for falsification.

With a pretrained LLM it can be difficult to determine whether a fact originates from:

```text
experiment training
```

or:

```text
pretraining
```

The Synthetic World removes this ambiguity.

---

## 13. Mini-Transformer experiment

The Mini-Transformer serves as the smallest meaningful neural substrate.

The goal is not to build a useful language model.

The goal is to test the architecture under controlled neural learning.

Experimental pipeline:

```text
Synthetic World
      ↓
training examples
      ↓
Mini Transformer
      ↓
Mutable Knowledge mechanism
      ↓
write / query / update / delete
      ↓
evaluation
```

---

## 14. Core experiment E-000001

The first formal experiment was defined around mutable knowledge.

### Hypothesis

A model can maintain knowledge whose lifecycle can be manipulated independently while preserving unrelated learned information.

### Required conditions

The experiment must distinguish:

```text
TARGET knowledge
CONTROL knowledge
UNRELATED knowledge
```

and measure all three before and after mutation.

### Required sequence

```text
baseline
   ↓
write
   ↓
verify learning
   ↓
update
   ↓
verify replacement
   ↓
delete
   ↓
verify forgetting
   ↓
retention test
```

This structure became the basis for subsequent test families.

---

## 15. Multi-seed testing

Single-run success was explicitly rejected as sufficient evidence.

Tests were therefore expanded across multiple random seeds.

Purpose:

* detect lucky initialization,
* detect unstable training,
* distinguish architecture effects from stochastic effects,
* estimate repeatability.

The seed work strengthened confidence in the synthetic mechanism.

However:

> Multi-seed synthetic evidence still does not equal real-LLM validation.

---

## 16. Negative-control testing

Controls were introduced to detect trivial explanations.

Examples include comparisons between:

```text
target mutation
```

and:

```text
untouched knowledge
```

as well as tests intended to establish that observed forgetting was not merely caused by general model degradation.

The desired pattern is:

```text
Target:
high → low
Control:
high → high
```

rather than:

```text
Target:
high → low
Control:
high → low
```

The latter would indicate destructive global forgetting rather than selective editing.

---

## 17. Update / replacement testing

Deletion is not the only required operation.

A mutable architecture should support:

```text
K(version 1)
     ↓
UPDATE
     ↓
K(version 2)
```

without retraining the complete system.

This motivated explicit version semantics.

---

## 18. Neural-MVCC

The project subsequently developed the concept into Neural MVCC.

MVCC is inspired by Multi-Version Concurrency Control in databases.

Instead of treating knowledge as one timeless parameter state:

```text
K
```

the architecture treats it as:

```text
K@v1
K@v2
K@v3
```

with lifecycle information.

Example:

```text
K17
├── v1 — historical
├── v2 — historical
└── v3 — active
```

A query resolves against an active knowledge state.

This enables conceptual operations analogous to:

```text
CREATE
READ
UPDATE
INVALIDATE
ROLLBACK
DELETE
```

for neural knowledge.

---

## 19. Versioned cell lifecycle tests

The strongest synthetic evidence obtained so far concerns the versioned neural-cell lifecycle / Neural-MVCC mechanism.

The tests demonstrate, at synthetic level, that the architecture can maintain distinct lifecycle states and manipulate them without simply destroying the entire learned system.

Current assessment:

| Aspect | Status |
|---|---|
| Conceptual consistency | PASS |
| Synthetic implementation | PASS |
| Repeated synthetic behavior | PASS |
| Version lifecycle | PASS |
| Selective mutation | supported |
| Real pretrained LLM | NOT YET PROVEN |

This is currently one of the strongest pieces of evidence in the project.

---

## 20. Biomarker research branch

A newer research direction asks whether mutable knowledge can leave measurable internal signatures.

These signatures are referred to as neural biomarkers.

The hypothesis is:

> If a knowledge object has a distinct lifecycle, changes to that object may produce measurable activation or representation signatures.

Potential signals include:

```text
activation patterns
representation distance
routing patterns
attention behavior
gradient response
cell utilization
dependency activation
```

The long-term goal would be:

```text
Knowledge ID
     ↓
Neural Biomarker
     ↓
observable lifecycle
```

This could provide something extremely important:

> an internal verification mechanism for knowledge deletion rather than relying solely on output behavior.

---

## 21. Why biomarkers matter

Suppose a model stops answering:

```text
“What is fact K?”
```

That alone does not establish deletion.

The model could simply have learned:

```text
do not output K
```

while still internally representing K.

A biomarker could potentially distinguish:

```text
OUTPUT SUPPRESSION
```

from:

```text
REPRESENTATIONAL CHANGE
```

This is one of the highest-value research directions currently under investigation.

---

## 22. Biomarker status

Current status:

| Aspect | Status |
|---|---|
| Concept | promising |
| Synthetic signals | research evidence exists |
| Robust causal marker | not yet established |
| Real-LLM marker | not yet established |
| Deletion certificate | not established |

Therefore no claim of a neural deletion biomarker should yet be presented as proven.

---

## 23. Dependency / reconstruction testing

A fundamental difficulty discovered throughout the project is that facts rarely exist independently.

For example:

```text
K1: Alice lives in Berlin
K2: Berlin is in Germany
K3: Alice lives in Germany
```

Deleting K3 may be meaningless if the model can reconstruct it from K1 + K2.

Therefore future deletion evaluation must distinguish:

```text
memorized knowledge
```

from:

```text
derivable knowledge
```

This substantially strengthens the definition of F5 deletion.

---

## 24. Reconstruction attacks

A deletion test must eventually include adversarial recovery attempts.

Candidate classes:

```text
direct query
paraphrase
multi-hop query
reverse query
context completion
forced-choice query
representation probe
activation probe
dependency reconstruction
```

A knowledge deletion claim becomes stronger as it survives more of these attacks.

---

## 25. Causal tests

Correlation is insufficient.

Suppose cell C17 activates whenever knowledge K17 is queried.

That gives:

```text
C17 ↔ K17
```

but does not establish:

```text
C17 → K17
```

Therefore the architecture requires interventions such as:

```text
disable C17
swap C17
restore C17
replace C17
```

and observing whether the corresponding knowledge changes predictably.

This is central to distinguishing biomarkers from merely correlated activations.

---

## 26. Ablation testing

Ablation tests are intended to determine whether architectural components are actually necessary.

Example:

```text
Full architecture
vs
without versioning
vs
without marker
vs
without routing
vs
random deletion
```

If removing a component has no measurable effect, the architecture should not claim that component is essential.

This principle remains part of the research methodology.

---

## 27. Retention testing

Every forgetting experiment must simultaneously test preservation.

A successful result therefore has two sides:

```text
FORGET TARGET
+
KEEP CONTROL
```

The desired metric structure is:

```text
target retention       ↓↓↓
control retention      ≈ constant
general capability     ≈ constant
```

Selective forgetting is more important than raw forgetting.

---

## 28. False-positive breakthrough criteria

The following outcomes explicitly do not count as breakthroughs:

### Output refusal

“I cannot answer that.”

Knowledge may still exist.

### Prompt filtering

Query never reaches the model.

### Database deletion

External memory disappears while model memory remains.

### Adapter disablement alone

Shows modular access control, not necessarily neural forgetting.

### Global degradation

Target disappears because everything became worse.

### One successful seed

Could be stochastic.

### One exact prompt forgotten

Could be surface overfitting.

### Crypto key destruction alone

Does not establish that the neural network lacks another copy.

### Correlated activation

Does not establish causal knowledge localization.

These distinctions substantially increased the rigor of the project.

---

## 29. C-series validation campaign

The project subsequently evolved into an extended validation campaign.

The C-series tests progressively attack alternative explanations and increase the strength of evidence.

Rather than interpreting every successful test as a new architecture, the series should be understood as:

```text
candidate mechanism
       ↓
attack
       ↓
survives?
       ↓
stronger attack
       ↓
survives?
       ↓
cross-model validation
```

The campaign has progressed into the C50+ range.

---

## 30. C55 → C57

The current critical validation chain is:

```text
C55
 ↓
C56
 ↓
C57
```

These tests represent the transition from the strongly controlled synthetic evidence toward the real-model / GPU evidence required for a substantially stronger claim.

### Current state

At the present project checkpoint:

| Evidence | State |
|---|---|
| Synthetic Neural-MVCC evidence | STRONG |
| C55–C57 real-model chain | OUTSTANDING |
| GPU execution | OUTSTANDING |

The inability to use the local GPUs temporarily shifted work toward everything that could still be proven without them.

This is an important boundary in the evidence.

---

## 31. Session record 2026-09-02 — recorded experiments

*Appended on 2026-09-02. Sections 1–30 above are the ledger as supplied; this section records what the code in [`so/`](../so/README.md) actually measured in one CPU-only session. All numbers come from `so/results/*.json`; the complete tables are in the [session results](so-results-2026-09-02.md), the path forward in the [roadmap](so-roadmap-2026-09-02.md).*

### 31.1 What was built

A minimal system in which the neural core is trained on worlds that are re-sampled every step, so facts cannot enter its weights; knowledge lives in addressable cells (key = subject + relation, value = object, marker, version, status) that the core reads through routing attention with a null cell for "nothing found" and a learned marker gate on the value. Lifecycle operations (WRITE / UPDATE / REVOKE / RESTORE / ROLLBACK / SHRED / RESIGN, plus SWAP / REPLACE as interventions) act on the cells only. Every experiment compares the trained core with a mechanical reference over the same store and has pre-registered pass criteria evaluated on the worst seed; sample sizes are stated in each record, and E-000001-B additionally reports exact binomial intervals for its pooled rates.

### 31.2 Evidence recorded

| Experiment | Question | Outcome | Level |
|---|---|---|---|
| E-000001-A | Are the intended semantics coherent in a mechanical reference? | 5 seeds × 1,000 cells: every family 100%, replay deviation 0 | E3 / F1 |
| E-000001-B | Does a trained core reproduce them? | 5 seeds: direct, 2-hop, provenance, reverse, update, rollback, revoke, restore, locality, alternative path 100% in every seed; 3-hop 99.8%; SHRED 97.0% worst seed; all criteria met | E4 / F3 (SHRED learned) |
| E-000002 | Does the core copy facts into its weights? | Re-sampled: layer masked → 0% answered, leak after REVOKE 0%. Fixed world with layer: also 0% / 0% (2000-step bound). No layer: 100% memorised, 100% leak | E4 (the copy-problem control) |
| E-000003 | Selectivity, retention, generalisation | REVOKE: target high → low on every paraphrase, multi-hop and reverse route. SHRED (unsupervised gate): every paraphrase and multi-hop route, reverse access 95.8% (worst seed 90.5%). Controls, unrelated, bypass paths 100%; UPDATE / ROLLBACK exact; all criteria met | E4 / F3 |
| E-000004 | Reconstruction attacks | REVOKE: everything at chance (mask). SHRED: behaviourally deleted (direct / paraphrase / multi-hop UNKNOWN 100%) but linear probe 8% and forced choice 69% (worst seed) recover a residual → **F4 criteria not met** | E4 / F3 with a trace |
| E-000005 | Causal interventions | disable / swap / restore / replace / random-other / localisation / routed-cell causality: 100% in every seed | E4 |
| E-000006 | Ablations | Marker gate necessary for SHRED (0% without); routing necessary (nothing readable); **routing loss necessary for learning** (without it the model collapses to UNKNOWN); **null cell not essential** (pre-registered expectation wrong, claim withdrawn) | E4 |
| E-000007 | Biomarker: suppression vs deletion | Suppressed model answers UNKNOWN yet keeps value contribution 8.3 and probe 86%; SHRED keeps routing mass (key unchanged) but drops the contribution to 1.3 (worst seed 1.57) and the probe to 4.4% (worst seed 8.0%); REVOKE zero by mask. The separation holds in every seed, but the two F4 criteria are evaluated on the WORST seed and both fail there, so the record is **F3**. The prose inside that record calls the shredded arm "F4, learned" eleven lines below its own F3 line; that sentence is an error in the prose, not in the criteria, and the record is left as it was produced | E4 / F3 |
| E-000009 | Verification loss on the gate | Separation improved (signed 0.89 → 0.998, unsigned mean 0.087 → 0.065) but an unsigned tail (max 0.84) lets 3–5% of shredded payloads through under hard gating → **F4 criteria still not met** | E4 |
| E-000010 | Class-balanced verification loss (weight 5) | 5 seeds: after SHRED the payload is still present and routed to, yet linear probe 0.2–0.4% (chance 0.39%), forced choice 53–54% (chance 50%), logit rank 125–127 (chance 127.5), gated value contribution 0.047 soft / 0.000 hard; SHRED 100% in the core suite, no other family degraded; **all F4 criteria met**. One seed still shows a single unsigned marker with a high soft score (max 0.995) — none of the 500 shredded targets leaked | E4 / **F4** (synthetic system) |
| E-000014 | Addressing at 10,000 cells / 2,560 entities (verified gate) | 3 seeds: direct 100% (30,000 pooled), 2-hop 99.9%, 3-hop 99.5%, provenance 99.99%, reverse 99.8%, every lifecycle operation ≥ 99.7%, locality 100%, alternative path 100%, replay 0; after SHRED with hard verification on 500 targets: probe 0.33% (chance 0.04%), true-object top-1 0.2%, mean rank 1246 of 1279.5, forced choice 0.53; the same model reads 100% at 1k / 3k / 10k cells with routing mass 0.995. Residual: the gate still passes a rare unsigned marker (0.2% of shredded targets answered) — within the binomial thresholds | E4 / **F4** (10× scale) |
| E-000016 | How far does the indirection carry, and does the model refuse what it cannot represent? | E-000015's two-slot control had failed on two-link chains. The cause was the training distribution, not the architecture: aliases there always pointed at facts, so the second dereference slot never saw a pointer. With 30% chains in training and nothing else changed, 3 seeds: two slots answer a two-link chain 100%, one slot answers 0% and returns UNKNOWN 100% — it refuses what it cannot represent instead of naming another entity, which is the falsifying arm of the design. No price elsewhere: reading, alias reading, 2-hop, provenance across the indirection and link-free worlds all 100%. E-000015's other failure is repaired too: shredding the POINTER rather than the payload reaches 97% on the worst seed against 93% before. Sharing unchanged: after ONE shred the probe recovers 1.0% through an alias and 88.7% through the copies of the duplication arm. All five pre-registered groups met; the level rule pre-registered here caps the record at F3 | E4 / F3 (all criteria met) |
| E-000012 | Is REVOKE better expressed as a status flag in the gate than as removal from routing? | Same architecture, budget and criteria as E-000011; the only change is that a revoked cell stays addressable and its status multiplies the verification gate. Answering ' unknown' after REVOKE rises 72.7% → 99.0% on the trained template (worst seed 98.0%) and 17.3% → 53.3% on the weakest held-out one, with reading 86.6% → 90.9%, 2-hop 83.3% → 91.1%, UPDATE 87.7% → 90.0%, locality 98.9% → 99.5% improving — but three families REGRESSED and the earlier wording of this row ("no family regressing") was wrong: broken paths 77.0% → 66.3%, SHRED on the weakest held-out template 75.0% → 53.3%, template 4 68.1% → 52.3%. No claim group changed state either: the same five of eight are unmet in both records. Mechanism: once a cell is masked out the routing spreads over neighbouring keys and the frozen model names another entity; a gated cell still absorbs the mass and reads as unknown. Still F1 on the pre-registered bar: held-out paraphrases stay weak and broken paths reach 66.3%; SHRED on held-out templates traded down 75.0% → 53.3% | E5 (substrate) / F1, design change validated |
| E-000013 | Can a cell override a fact the pretrained model already knows, and does the prior return after deletion? | 50 real countries whose capitals GPT-2 names correctly 96% of the time receive counterfactual capital cells among 950 prior-free fillers, 3 seeds. On the trained template the override is complete (100% among the candidates and over the full 50k vocabulary). After REVOKE and after SHRED the distribution returns to the pretrained one at 0.0004 nats and the true capital is answered again at exactly the prior's 96%. Nothing counterfactual survives: the PAIRED excess over the base model is 0.0000 for the probe, for forced choice and for the counterfactual top-1, while the same probe reads the counterfactual 87% of the time while the cell is active, which is the validity condition that makes those zeros mean anything. The copy bound is exact. What fails is addressing on unseen phrasings: prior-free reading drops to 12.6% on the weakest held-out template, the override does not transfer to them at all, and on generic sentences the distribution moves by 2.27 nats where the bar is 0.05 | E5 (substrate) / F1 |
| E-000017-A | Is the held-out failure a deletion failure or a reading failure that deletion inherits? | No training: E-000012's three checkpoints are decomposed per template into what the model answers with the cell ACTIVE against what it answers after REVOKE. Conditioned on having read the fact correctly while active, it refuses after REVOKE at 99.9% on trained templates and 96.1% on held-out ones (worst seed 94.2%), and returns the deleted object in 0.15% of those cases. Unconditionally the same templates reach only ~52% because reading itself is 69.4% there against 96.1% on trained templates. Template 4 is the clearest case: 37.3% of its answers are already another entity while the cell is ACTIVE and 35.7% are after REVOKE, the same population rather than a leak. The fired kill criterion stands, because it is written on the unconditional metric; the defect is located in the query and routing path, not in the deletion mechanism | E5 (substrate) / n.a. |
| E-000017-B | Does the held-out failure survive the template budget the roadmap prescribes for this stage? | Eight trained templates per relation instead of two, four held out, 3 seeds, everything else unchanged. No: answering ' unknown' after REVOKE and after SHRED on held-out phrasings reaches 89.8% mean / 86.5% worst seed against ~52% at the two-template budget, and conditioned on the model having read the fact while the cell was active it is 99.3% (worst seed 98.7%). The deleted object returns in exactly 0.0000 of cases, on every held-out template and in every seed. Kill criterion 5 is NOT cleared — its own bar is 95% unconditional and this run reaches 89.8% — but the failure now demonstrably scales with the template budget rather than being a property of the deletion mechanism. Two families fail and are the named open problems: reading on held-out phrasings 74.0% against a 90% bar, and injection where there is no key got WORSE rather than better (broken paths 71.8% unknown, generic text 3.27 nats against a 0.05 bar, above E-000013's 2.27). More prompt shapes in training mean more shapes that trigger a spurious injection, which matters more than the reading figure: it means the layer perturbs the frozen model on text it has no business touching | E5 (substrate) / F3 |
| E-000018 | Can the adapter be stopped from injecting into text it has no key for? | Not in this design, and the three-arm ablation says why. **WITHDRAWN in part — see the correction below.** Adding the CAPACITY to inject nothing (an absolute match score against the best real cell key) appeared to change nothing at all: 3.2681 nats against a baseline of 3.2741. That measurement is invalid: the match gate multiplied the read and the RMS-matched injection then divided by the RMS of that same gated read, so the factor cancelled exactly and the mechanism could not act. What survives is the other half: training the BEHAVIOUR on generic sentences brings the divergence to 0.6035 and both arms together to 0.6736, every arm stays a factor of twelve above the 0.05 bar, and the arms that improve it pay for it — the generic arm reads 68.9% on held-out phrasings against 74.0% and refuses at 88.0% against 89.8%, the combined arm 69.2% and 85.5%. The structural reading also survives and is what motivated the fix: answering ' unknown' when a cell is gone needs an INJECTION, changing nothing on text with no key needs NONE, and both were routed through the same null column. The capacity question is reopened and is now E-000022's to answer — **and E-000022 has now answered it: no** (§31.23) | E5 (substrate) / F1, negative with a mechanism |
| E-000022 | Does splitting the null column — a payload channel for answering ' unknown', a separate unknown direction for questions that found no cell — close the injection channel E-000018 left open? | No, on both of the claims it was built for. 3 seeds × 3,000 steps, everything else unchanged. `generic/kl_to_base` 0.5508 mean and 0.8657 worst seed against E-000018's 0.6736 and a bar of 0.05 — about a fifth of an improvement on a quantity that needs an order of magnitude — and `train/active_correct` 0.8844 against 0.90, so the fifth was not free. `no_key_no_injection` and `reading_not_traded_away` unsupported; `refusal_not_traded_away` and `deleted_object_never_returns` supported, at 0.9894 held-out refusal and 0.0000 for the deleted object. The structural reading that motivated the split stands and remains unpaid for; the split was necessary and is not sufficient | E5 (substrate) / F1, negative, hypothesis failed twice |
| E-000020 | Do several natural-language access paths share ONE knowledge object in a frozen GPT-2? | Two stores from the same world, read by the same trained adapter: 200 alias keys as LINK cells over 100 shared targets, versus the same keys as fact cells holding a copy. 3 seeds. **The sharing and deletion contrast carries.** One SHRED on the shared object leaves every alias path unreadable (99.5% unknown, worst seed 98.5%) and the object returns in exactly 0.0000 of cases; through every alias the probe recovers 0.33% and forced choice sits at 0.510 against a chance of 0.5, while the SAME probe on the duplication arm recovers 49.8% at a forced choice of 0.973. One UPDATE reaches the aliases in the symlink arm and 0% of them in the duplication arm. A dangling pointer after DELETE answers unknown 99.2%, a revoked alias 98.0%. Attack validity passes (probe calibration 48.1%, through an active alias 38.8%), so those zeros are informative. **What fails is reading**: direct facts 57.0% and aliases 50.7% against bars of 85% and 80%, which drags the update and lifecycle groups down through their readable-after rows. A five-minute diagnostic locates the cost precisely: the adapter trained WITHOUT links reads this same evaluation world at 82.7%, and 84.7% with the link and dereference machinery attached, so neither the world nor the mechanism costs anything at inference — the loss is in LEARNING the harder distribution, where a third of the routing supervision is now alias-related | E5 (substrate) / F1, sharing supported, reading not |
| E-000015 | Do several access keys sharing ONE knowledge object beat duplicating it? | Two stores with identical ground truth, read by the same trained model: 200 alias keys as LINK cells over 100 shared objects, versus the same keys as fact cells holding a copy. 3 seeds. One UPDATE on the shared object reaches every access path 100% in the symlink arm and 0% in the duplication arm. One SHRED leaves nothing readable on any path (100% unknown, 0% true object) while every copy still answers (100%); after that ONE operation the object is unrecoverable through every alias (probe 0.7%, forced choice 0.503 at a chance of 0.5) and fully recoverable through the copies (probe 87.3%, forced choice 1.000). Reading, provenance across the indirection (100%: the routing names alias AND target), revoke of one alias, relink, rollback and the dangling pointer after DELETE are all at 100%, and link-free worlds are unchanged. Disabling the dereference slot drops alias reading to 0% while fact reading stays at 100%. Withheld: shredding the ALIAS instead of the payload reaches 93% on the worst seed against a 95% bar, so F4 is not granted; and the two-slot control did not resolve two-link chains because chains never occur in training | E4 / **F3** |
| E-000011 | Frozen GPT-2 core v2: verified gate, deletion behaviour, held-out paraphrases, causal interventions | 3 seeds × 3,000 adapter steps with the class-balanced verification gate and a selecting gate (an unsigned payload READS AS ' unknown' instead of being attenuated). What carries: SHRED under hard verification 99.7% ' unknown' with the linear probe at 1.0%, forced choice 0.49 (chance 0.50) and a payload share of exactly 0.000 while the payload is physically present; every causal intervention (localisation, disable, swap, replace) above its threshold; 2-hop composition 83.3% against 41.7% for the SAME frozen model with both facts in the prompt and 0.6% with only the first — the roadmap's RAG kill criterion is passed. What does not: reading 86.6% and UPDATE 87.7% stay below the 0.95 bar imported from E-000008, REVOKE by routing mask reaches only 72.7%, and deletion does NOT generalise to held-out paraphrases (REVOKE 17.3%, UPDATE 41.3% on the weakest held-out template). The record is F1 on the pre-registered criteria; the SHRED and intervention families are supported, the deletion-behaviour family is not | E5 (substrate) / F1 |
| E-000019 | Does the verified gate hold on seeds that took no part in choosing it, and is the residual at CHANCE rather than merely below a bar? | Yes to both. E-000010's configuration unchanged on seeds 5, 6, 7, which took part in no selection; 250 targets per seed so the equivalence test is attainable and trennscharf. Pooled over 750 trials: forced choice 375/750 = exactly one half, interval [0.4636, 0.5364]; linear probe 4/750 against a chance of 1/256, interval [0.0015, 0.0136]; true object top-1 among entities 7/750, interval [0.0038, 0.0191]. Every interval contains its chance level and every upper end stays inside the pre-registered distance, so the residual is where chance puts it. Behavioural deletion 99.9%, every core family 100% on the worst seed, 3-hop 99.5%. Residual caveats kept: the hard gate still admits an unsigned marker in at least one seed (nothing recoverable follows), and the top-1 interval only just contains chance with a point estimate about 2.4x the chance rate, so a larger sample could separate them | E4 / **F4**, confirmed out of sample and at chance |
| E-000021 | How often does the verification gate accept an unsigned marker? | No training: the gate is a function of the marker alone, so all 11 recorded checkpoints of the verified-gate family were scored on fresh markers. Pooled over 2,200,000 markers per class: 1,867 false accepts, rate 8.49e-04 with a 95% interval of [8.11e-04, 8.88e-04], and 0 false rejects. Per family the rate lies between 8.0e-04 and 9.6e-04; the highest score any unsigned marker reached is 0.89, so the tail is above the 0.5 threshold but well below certainty. That is about one shredded payload read out per 1,180 — consistent with the residuals the attack batteries recorded, a real limit on the deletion claim, and now a measured quantity with an interval rather than a worst-seed maximum. Stated limit: this is the error rate on the two marker distributions the programme uses, not a security claim; an adversary searching for a passing vector is not modelled | E4 / measurement |
| E-000008 | Frozen pretrained GPT-2 core with the layer as adapter, natural-language prompts | Frozen GPT-2 small (124M parameters, never updated) with a 2.37M-parameter adapter trained on re-sampled natural-language worlds, 3 seeds × 2,000 steps. Pretrained prior 0.6%; adapter with every cell masked 0% (copy bound holds). Reading works: direct 88.9% (worst seed 88.5%), 83.7% even over the full 50k-token vocabulary, second template 99.9%, provenance 84%, 2-hop composition 75%; UPDATE / ROLLBACK / RESTORE / RESIGN 95–96%; locality 99.3%. Deletion behaviour only partly learned at this budget: ' unknown' after REVOKE 56% (worst seed 49%), after SHRED 38%, on broken paths 64% — otherwise the model names another entity. After REVOKE nothing is recoverable (probe 0%, forced choice at chance); after SHRED the unsupervised gate leaves the residual known from E-000004. Pre-registered thresholds (0.95) met for update/rollback and for the attacks after REVOKE, missed for reading (88.9%) and for deletion behaviour → recorded F1 at this budget. First engineering attempt collapsed to ' unknown'; fixed by RMS-matched injection and a routing-first curriculum | E5 (substrate) / F1 |

### 31.3 What this changes in sections 19, 22 and 30

- **Section 19 (versioned cell lifecycle):** "Synthetic implementation / repeated synthetic behaviour / version lifecycle: PASS" now has a recorded basis (E-000001-A/B, E-000003), with the copy-problem control (E-000002) that the whole deletion argument rests on.
- **Section 6 (F0–F5):** REVOKE = F1 by construction plus learned UNKNOWN behaviour; SHRED = F3 with the unsupervised gate (E-000004), F4 with the class-balanced verified gate (E-000010). F5 remains a research target: it requires attacks and reproducers from outside the system (roadmap stages 3 and 6).
- **Section 22 (biomarker status):** "Synthetic signals: research evidence exists" is now a recorded, causal result in the synthetic system (E-000005 + E-000007): the gated value contribution separates output suppression from representational removal; routing mass alone does not (a shredded cell is still routed to). With the verified gate (E-000010) the same signal reads exactly zero after SHRED while every reconstruction attack is at chance — a deletion certificate *within this system*. "Robust causal marker" remains not established beyond it.
- **Section 30 (C55–C57):** still outstanding. E-000008 is the CPU-feasible analogue on a 124M-parameter frozen core, not the GPU chain: it shows that the layer can be read by a frozen pretrained transformer from natural-language prompts and that update/rollback carry over, and it shows that the deletion behaviour (answer ' unknown' when the cell is gone) is the part that does not transfer for free — the frozen model prefers to name another entity. That is the first concrete real-model finding of the program and the first item of roadmap stage 2.

### 31.4 Negative and corrected findings (recorded, not tuned away)

1. The marker gate learned without supervision leaves a residual (≈9% of the value) that representation-level attacks exploit; SHRED is F3, not F4, until the gate verifies signatures reliably (E-000004, E-000007). A plain verification loss is not enough because unsigned markers are 5% of the cells (E-000009); a class-balanced verification loss closes the residual to chance in every seed (E-000010).
2. The null cell is not an essential component (E-000006).
3. Routing supervision is necessary for the mechanism to be learned at all at this budget (E-000006); provenance exactness is therefore a trained property.
4. The first GPT-2 adapter design did not learn (collapse to " unknown"); an untrained injection test showed the read-out path works at gain ≥ 1, and a routing-first curriculum was required (E-000008 engineering record).
5. Deletion does not generalise to unseen surface forms in the frozen GPT-2 (E-000011): with two templates trained per relation, REVOKE reaches 72.7% on the trained template and 17.3% on the weakest held-out one. The synthetic system does not show this because one canonical cell serves every paraphrase there. The ledger's "generalisation is the weakest property" is therefore not a theoretical caveat but a measured failure in the real model.
6. An absolute reconstruction threshold is only meaningful for prior-free targets. When the deleted object is a token the pretrained model already prefers, the frozen model alone wins the forced choice at 0.60 with no leakage whatsoever, so E-000013 pre-registers every attack as a PAIRED EXCESS over the base model, with the base measured on the same rows with the same distractor draws (protocol review of E-000013, before its first run).

### 31.5 Corrections found by the standing audit (2026-09-03)

An independent read-only audit of every record against every summary was run after E-000016. Its confirmed findings are applied above and listed here so the corrections are visible, not silently folded in.

1. **A pre-registered kill criterion has fired and no document said so.** Roadmap kill 5 (held-out-template deletion below 95% with train-template deletion near 100%) is satisfied by E-000012 (98% trained, 52% weakest held-out). It is now stated as fired in the roadmap, with the two qualifications that belong to it.
2. **The held-out failure is a REFUSAL failure, not recovery of the deleted object.** After REVOKE the deleted object is never the top-1 answer on any held-out template (0.0 in all three seeds of E-000011 and E-000012); the model names another entity instead. Reading on those templates is already unreliable while the cell is ACTIVE (56% correct, 7% unknown on template 4), so a summary that says "deletion does not generalise" without that qualification overstates what was measured. The next experiment must separate reading from refusal.
3. **"No family regressing" in the E-000012 row was wrong** (broken paths 77.0% → 66.3%, SHRED on the weakest held-out template 75.0% → 53.3%, template 4 68.1% → 52.3%), and no claim group changed state between E-000011 and E-000012.
4. **ANSWERED by E-000019 for the synthetic system.** F4 was a tolerance result, not a chance result. No record tests the SHRED residual against the null that it IS chance; the criteria compare it to a pre-registered threshold. In E-000014 the threshold's own justification in the source asserts a binomial tail probability about fifty times smaller than the true one, and 3 of 1,500 shredded targets still answer correctly. The claim string of that record also quotes the E-000010 thresholds (0.02 at 100 targets, rank 1100, forced choice 0.6) rather than the ones the run enforced (0.006 at 500 targets, rank 1150, 0.56).
5. **ANSWERED by E-000019.** The configuration was selected and confirmed on the same seeds. E-000009 (gate weight 1) fails 6 of 11 criteria; E-000010 changes two things at once (weight 5 and class balancing) and passes 11 of 11 on the same five seeds, and every later run uses a subset of them. A fresh-seed confirmation is owed.
6. **QUANTIFIED by E-000021: one unsigned marker in about 1,180 (8.49e-04, interval 8.11e-04 to 8.88e-04).** The gate is a learned classifier with a measured false-accept rate. The hard gate admits an unsigned marker in 1 of 5 seeds of E-000010 and in 2 of 3 seeds of E-000014, where in one seed the admitted marker is a shredded target. The exact zero of E-000010 did not survive the ten-fold scale-up.
7. **E-000007's own prose labels the shredded arm F4** eleven lines below the record's F3 line; both F4 criteria fail on the worst seed. The record is left as produced and the correction is carried here and in the results document.
8. **The roadmap misattributed the copy result** to the training regime. With the layer present, both the re-sampled and the fixed-world model answer 0% masked and leak 0%; the separating variable is the layer.
9. **E-000012's level label reads "F1 / routing removal"** although that design does not remove routing (routing mass on the revoked cell stays at 0.82). F1 is the fallback branch of a level rule that never tests whether routing was removed. The rule is wrong for status-gated designs and is to be replaced before the next GPT-2 run.

### 31.6 Second standing audit (2026-09-04): what it overturned

A read-only protocol review of the two experiments then in flight, of the four mechanisms added to the GPT-2 adapter that day, and of the summaries written the same day returned twenty-two confirmed findings. The two runs were stopped and restarted under the corrected code. The three that change what is claimed:

1. **The match gate could not act, and a published conclusion falls with it.** It multiplied the read; the RMS-matched injection one line later divided by the RMS of that same gated read. A scalar that multiplies a vector and is then divided out by that vector's own norm is not in the computation at all. E-000018's headline — "all of the improvement is the behavioural training and none is the added capacity" — was therefore measuring a mechanism that was switched off by arithmetic. It is withdrawn, not defended, and the capacity question moves to E-000022 under a restructured injection path where a closed gate demonstrably injects less.
2. **A model trained and tested on different worlds.** The link training bank marked no cell routable while the evaluation bank marked every non-deleted cell routable, so the status-gated design saw one convention in training and the other at test. Caught before any record existed.
3. **Every table printed the best seed under a "worst seed" heading for the leak metrics**, because the worst seed is the maximum for a quantity that should be small and the minimum only for one that should be large. The helper that decides this now lives beside the criteria check, and the E-000018 and E-000019 records carry the defect in their own markdown; their JSON criteria were always evaluated in the correct direction, so no criterion verdict changes.

Four more were fixed before they could reach a record: E-000020 measured "held out" on two of its four held-out templates and omitted the weakest; its F4 rule was weaker than the E-000015 rule it ports; its floor label claimed routing removal that a status-gated design does not perform; and the duplication arm's probe, which carries the headline contrast of the whole experiment, had no floor, so a merely weak probe would have printed a deletion result.

### 31.7 Symlink: which half of the concept is implemented

Section 7 and architecture-document section 10 describe the Symlink in two directions. Only one of them is realised in this session's code, and the record must say so.

- **Representation → cell (implemented, measured).** The neural representation carries no fact; it addresses a cell by routing attention, and the routing distribution *is* the provenance. This is what the copy-problem control tests: with the layer masked the model answers 0% and the leak after REVOKE is 0% (E-000002), provenance is exact (E-000001-B: 100%; E-000014 at 10,000 cells: 99.99%), and a causal intervention on the addressed cell changes the answer (E-000005).
- **Cell → cell (not implemented).** There is no link cell whose payload references another cell, so several access keys cannot share one knowledge object K42. Every key carries its own payload; a fact that is to be reachable under two keys is stored twice. The consequences that the analogy promises are therefore untested: one UPDATE on the target changing every alias at once, SHRED of the target followed by the attack battery through *every* alias, revoking one alias while target and sibling aliases stay intact, dangling links after DELETE, link chains, cycles, reference counting, and provenance across the indirection.

What is measured in place of the second half is that all *semantic* access paths that reach the same cell are removed by a single operation: after REVOKE of one cell, every paraphrase, every 2-hop route through it and reverse access answer UNKNOWN at 100% in the worst seed, while controls, unrelated cells and bypass paths stay at 100% (E-000003). That supports the deletion claim for one cell with many query forms; it does not support the shared-object claim for several keys. Closing that gap is E-000015, which is now recorded: with link cells the store reaches every access path in ONE operation and, after that one SHRED, the object is unrecoverable through every alias (probe 0.7%, forced choice at chance), while the duplication arm needs one operation per copy and leaves the object fully recoverable through the copies it did not touch (probe 87.3%, forced choice 1.000). The cell-to-cell direction of the Symlink concept is therefore no longer unimplemented; what stays open at LLM scale is that the frozen-GPT-2 chain does not yet carry link cells. E-000016 then showed that the indirection is not a one-step trick: with chains in the training distribution, two dereference slots resolve a two-link chain completely while a one-slot model refuses it rather than inventing an answer, and the number of slots is therefore the honest statement of how deep the mechanism reaches.

### 31.9 A published number that was a property of one phrasing (2026-09-04)

E-000020 recorded that a frozen GPT-2 with link cells reads a base fact at 0.5667 and reads *through*
an alias at 0.5067, and the roadmap and the summaries were written from those two numbers. Both are
taken at template 0, because `E20._answers` defaults to it and every call in that experiment's
battery used the default.

Template 0 is not representative of anything. E-000017-B had already measured the twelve templates on
the **link-free** adapter and found reading bimodal by phrasing — 0.795, 0.992, 0.792, 1.000, 1.000,
0.998, 0.785, 0.997 on the trained eight and 0.565, 0.968, 1.000, 0.427 on the held-out four — and
template 0 is one of the weak ones. E-000020's own record contained the counter-evidence it did not
act on: `alias_template1_train` 0.785–0.895 and `alias_template9_heldout` 0.870–0.920 on the very
checkpoints whose headline alias number is 0.5067.

E-000025 re-scores those checkpoints at all twelve templates, trains nothing, and separates two costs
the single-template number confounded. The world seed matches E-000020's, so the template-0 column is
that experiment's own condition: on seed 2 — the one checkpoint whose SHA-256 still matches the
E-000020 record — it returns direct 0.563 and alias 0.500 against the recorded 0.5633 and 0.50, which
is a reproduction rather than a fresh measurement.

| template | direct | alias, one shared object | alias, duplicated | alias, duplicated (link-free) |
|---|---|---|---|---|
| t0 (trained) | 0.6122 | 0.6167 | 0.6233 | 0.7900 |
| t1 (trained) | 0.9633 | 0.8367 | 0.9817 | 0.9933 |
| t2 (trained) | 0.6322 | 0.6017 | 0.6217 | 0.7650 |
| t3 (trained) | 0.9956 | 0.8933 | 0.9983 | 1.0000 |
| t4 (trained) | 0.9967 | 0.9350 | 0.9967 | 1.0000 |
| t5 (trained) | 0.9989 | 0.9250 | 0.9967 | 1.0000 |
| t6 (trained) | 0.6422 | 0.6317 | 0.6317 | 0.7633 |
| t7 (trained) | 0.9989 | 0.8917 | 0.9967 | 0.9950 |
| t8 (held out) | 0.4500 | 0.4283 | 0.4133 | 0.5200 |
| t9 (held out) | 0.9567 | 0.9033 | 0.9817 | 0.9650 |
| t10 (held out) | 0.9989 | 0.9250 | 1.0000 | 1.0000 |
| t11 (held out) | 0.3078 | 0.3433 | 0.2883 | 0.4200 |

Read at a phrasing that works, the frozen GPT-2 resolves a symlink at **0.9250 on the held-out
template 10** and 0.9350 on trained template 4, with direct reading at 0.9989. The two costs, worst
seed over three seeds and averaged over all twelve templates: **sharing costs 0.0954**
against duplicated copies read by the same adapter, and **having trained on links at all costs
0.0688** against the link-free adapter on the same store. Both
pre-registered bars pass, as do `train/alias_max` ≥ 0.75 (observed 0.8950)
and `heldout/alias_mean` ≥ 0.55 (observed 0.6013).

Three consequences the programme has to carry:

1. **The E-000020 record's reading numbers stand as produced and are not what the experiment set out
   to measure.** The record is not edited; this section is the correction, and E-000026 re-runs its
   whole lifecycle battery — update, rollback, shred, revoke, delete, and the probe and forced-choice
   attacks — at three phrasings, because every one of those claims was also a statement about
   template 0. "The alias no longer answers" is cheap when the alias answered half the time.
2. **Two thirds of the E-000020 record is no longer byte-reproducible.** A forced re-run replaced its
   seed-0 and seed-1 checkpoints after it was written; only seed 2 still matches the SHA-256 it cites.
   `guard_recorded_checkpoint` now refuses to overwrite a checkpoint whose digest a saved record
   names, in all fourteen writers.
3. **A single-template number is not a measurement of a system whose reading is bimodal.** Every
   experiment in the twelve-template line reports at least one strong and one weak phrasing from here
   on, and a headline taken at one template says which.

### 31.10 SHRED closes the value channel and leaves the key channel open (2026-09-04)

The strongest deletion claim in this programme is F4 for SHRED with the verified gate: after the
marker is destroyed the gate closes, the payload is unreadable, and every recorded attack comes back
at chance (E-000010, and on fresh seeds E-000019). Every one of those attacks reads the *value*
channel — the answer, the logits, the hidden state, the linear probe. None of them reads the keys.

`shred()` writes only `Version.marker` and leaves the row ACTIVE. In `encode_bank` the routing keys
are computed from the payload **before** the gate is applied and are never gated:

    k_f = k_fwd(LN(s + r))          # subject and relation
    k_r = k_rev(LN(o + r))          # THE OBJECT
    v_f = v_f * g                   # only the values are gated

So a shredded cell's reverse key is still a deterministic function of the object that was destroyed.
The first consequence was already in the record and went unread: E-000019 lists
`active/routing_mass_on_target` and `shred/routing_mass_on_target` as the same float in every seed,
and §31.3 noted that routing does not separate the two — without asking what else routing therefore
carries.

E-000028 asks. The attacker is given exactly what the rest of the battery gives them — a cell's
subject and relation — locates its column from the routing of the ordinary forward question, then
sweeps every candidate object through a **reverse** query and takes the one that steers the read onto
that column. Five seeds, 100 targets each, 500 targets pooled, on the recorded E-000010
checkpoints, no training:

| condition | object recovered, top-1 | top-5 | mean rank | winning margin |
|---|---|---|---|---|
| active (validity control) | 1.0000 | 1.0000 | 0.0 | 0.6195 |
| **shred** | **1.0000** | 1.0000 | 0.0 | 0.6195 |
| revoke | 0.0040 | 0.0220 | 128.0 | 0.0022 |
| delete | 0.0040 | 0.0220 | 128.0 | 0.0022 |
| chance | 0.0039 | 0.0195 | 127.5 | — |

The shredded row is not merely leaky, it is **unchanged**: its numbers equal the live cell's to four
decimals, margin included, because the keys the attack reads are the same tensors before and after.
The attacker's located column is also identical before and after SHRED in every one of the 500
targets. REVOKE and DELETE, which remove the row from routing altogether, land on chance —
0.0040 against 0.0039, mean rank 128.0
against 127.5.

What changes, precisely:

1. **The F4 claim for SHRED in the synthetic system is narrowed to the value channel.** It was never
   measured on any other, and on the key channel it fails at 1.0000 against a required 0.0039. The
   E-000010 and E-000019 records stand as produced — every number in them is about the value channel
   and every one is still correct.
2. **It is not a claim about the frozen-GPT-2 line.** That adapter's key is
   `k_proj(ln_key(subject + relation))` and contains no object at all, and its link payload is gated
   like any other value. Two tests state this as a property: with the gate forced shut, banks that
   differ only in the objects of shredded cells produce identical keys and identical routing, while
   an open gate moves the values.
3. **The architecture already contains an operation that does what SHRED claimed.** REVOKE and
   DELETE both close this channel completely, because a non-routable row has no column to steer onto.
   The crypto-shredding analogy is the source of the error: destroying the key to a ciphertext hides
   the plaintext, but here the *address* was derived from the plaintext too.
4. **A fix exists and is not yet evaluated.** `ModelConfig.gate_reverse_key` blends the reverse key
   toward a constant shared by all gate-closed rows, reusing the idiom by which `link_rev_key` makes
   alias rows non-reverse-addressable. It is off by default, because it changes what a routing target
   means for a reverse query against a shredded cell and therefore needs its own training run rather
   than a flag flipped on checkpoints trained without it. Two tests pin the defect and the repair.

The general lesson is the one worth carrying to any later design: **a gate on values is not a
deletion primitive if anything else in the computation is a function of the same payload.** Every
quantity derived from a cell has to be enumerated and gated, or the cell has to leave the addressable
set entirely.

### 31.11 The lifecycle claims were also statements about one phrasing (2026-09-04)

§31.9 corrected E-000020's *reading* numbers. Its whole battery was taken at the same template 0, so
every lifecycle and attack claim in that record — update, rollback, shred, revoke, delete, the probe,
forced choice — was a statement about a phrasing the model reads at 0.5633. E-000026 re-runs the
battery unchanged at three phrasings. The two strong ones are chosen at run time from
E-000017-B's record of the **link-free** adapter, so the choice cannot be tuned in the link arm's
favour; the rule picks trained template 3 and held-out template 10.

| measure (worst seed) | template 0 | template 3 | template 10 (held out) |
|---|---|---|---|
| base fact read directly | 0.5633 | 0.9933 | 0.9967 |
| read through an alias | 0.5000 | 0.8600 | 0.8700 |
| read through a duplicated copy | 0.5900 | 0.9950 | 1.0000 |
| one UPDATE reaches every alias | 0.5350 | 0.8650 | 0.8850 |
| the same UPDATE reaches a copy | 0.0000 | 0.0000 | 0.0000 |
| one SHRED: alias answers unknown | 0.9950 | 0.9950 | 1.0000 |
| one SHRED: alias names the object | 0.0000 | 0.0000 | 0.0000 |
| probe after that SHRED | 0.0100 | 0.0100 | 0.0100 |
| the same probe on a LIVE alias | 0.4200 | 0.7800 | 0.8000 |
| revoke one alias, sibling still reads | 0.5800 | 0.8800 | 0.8800 |
| DELETE: alias answers unknown | 0.9650 | 0.9600 | 0.9100 |

Criteria groups passed, out of six: **template 0 — 2** (attacks_through_every_alias, attack_validity); trained template 3 — 4; held-out template 10 — 4 (one_shred_deletes_every_path, attacks_through_every_alias, attack_validity, alias_lifecycle).

What the phrasing was hiding:

1. **One SHRED of the shared object deletes it from every alias, and now against an attack that
   works.** At template 0 the probe recovers only 0.42 of live aliases, so "0.01 after SHRED" was a
   weak statement. At the held-out template 10 the same probe recovers 0.80 live and 0.01 after one
   SHRED, forced choice sits at 0.43 against 0.5, and no alias names the object. `one_shred_deletes_every_path`
   and `alias_lifecycle` both fail at template 0 and both pass at either strong phrasing.
2. **The sharing contrast is not a phrasing effect.** `duplicate_update/alias_new_object` is 0.0000
   at every template: an update to one copy never reaches the others, whatever the question looks
   like. The shared-object side rises from 0.5350 to 0.8850 — a real improvement that still misses
   its pre-registered 0.90 on the worst seed, and is recorded as a miss rather than rounded up.
3. **`reading_through_an_alias` still fails at all three**, because its fourth criterion is
   `alias_heldout_min` ≥ 0.50 — the *weakest* of the four held-out templates, which is template 11,
   where even the link-free adapter reads 0.42. That is the criterion doing its job: the system has a
   phrasing it cannot read, and a strong average does not excuse it.

The methodological point outlives this experiment. A capability measured at one phrasing is a
statement about that phrasing, and a *deletion* measured at a phrasing the model barely reads is
close to vacuous — the fact was half-gone before the operation. Every claim of this kind from here on
is reported at a strong and a weak phrasing, and the attack that certifies it has to be shown working
on live cells at the same phrasing.

### 31.12 The gate's boundary is not the boundary the store declares (2026-09-04)

E-000021 measured the verification gate's false-accept rate at 8.49e-04 over 2.2 million markers and
the programme called that the bound on the deletion guarantee. The number is correct. The
distribution it was measured on is not the one the guarantee is about.

`MVCCStore.marker_valid` accepts a marker within `valid_radius = 0.35` of the centre and calls
everything beyond it deleted. But `so.data.invalid_markers` — E-000021's unsigned class — rejects
every draw within **0.7**, and `MVCCStore.new_invalid_marker`, which is what `shred()` writes,
rejects the same band (`2 * valid_radius`). So the annulus 0.35 < ‖m − κ‖ < 0.7 is populated by no
training distribution, no evaluation distribution and no store operation anywhere in the programme,
while the store's own predicate calls all of it deleted.

E-000029 measures the gate over the geometry instead of over a convenient sample. Eleven recorded
checkpoints, no training, markers placed on each shell by construction — rejection sampling cannot
reach the near shells, since a uniform unit vector in sixteen dimensions sits at ‖m − κ‖ ≈ √2.

| distance from the centre | accept rate | mean gate score |
|---|---|---|
| 0.10 | 1.0000 | 0.9985 |
| 0.20 | 1.0000 | 0.9978 |
| 0.30 | 1.0000 | 0.9962 |
| 0.40 | 1.0000 | 0.9917 |
| 0.50 | 1.0000 | 0.9777 |
| 0.60 | 1.0000 | 0.9300 |
| 0.70 | 0.9999 | 0.7725 |
| 0.80 | 0.2191 | 0.4320 |
| 0.90 | 0.0000 | 0.1299 |
| 1.00 | 0.0000 | 0.0254 |
| 1.10 | 0.0000 | 0.0044 |
| 1.20 | 0.0000 | 0.0008 |

**The gate's operational radius is 0.90 on every one of the eleven checkpoints, against a declared
0.35.** Everything out to 0.70 is accepted at 1.0000; the transition is at 0.80 (0.2191); zero is
first reached at 0.90. The annulus the store deletes is accepted at
2,199,996 of 2,200,000 — a rate of 1.0000.

| marker distribution | rate | 95% CI |
|---|---|---|
| uniform on the sphere, no rejection | 1.062e-03 | 1.035e-03 – 1.090e-03 |
| the annulus the store calls deleted | 1.000e+00 | 1.000 – 1.000 |
| rejection-sampled ≥ 0.7, i.e. E-000021's class | 8.550e-04 | 8.168e-04 – 8.945e-04 |

The third row reproduces E-000021's 8.49e-04 to within its interval, which is how one knows the
re-measurement is faithful before one looks at what it missed.

**Why the boundary sits where it does, and why that is the general lesson.** The training data has
signed markers at ‖m − κ‖ ≈ 0.194 and unsigned ones at ≥ 0.7, with nothing between. A classifier
trained on two separated clusters puts its boundary in the *margin*, not at a number nobody told it
about — so the gate learned ≈0.8 and the specification says 0.35. The recorded false-accept rate
follows from this rather than from anything mysterious about learned classifiers: `shred()` writes
markers uniformly beyond 0.7, about 5.8e-03 of which land inside 0.9, and integrating the gate's
accept curve over that tail gives the ~8.5e-04 that was measured. **A learned gate certifies the
margin between the classes it was shown, not the predicate it was meant to implement.** Making it
implement the predicate is a change to the data, not to the architecture: draw unsigned markers from
the whole region beyond the declared radius, the annulus included. That needs a training run and is
not yet evaluated.

**And the centre is not a secret.** `new_valid_marker` returns `normalise(κ + N(0, 0.05²·I))`, so
every signed marker in a bank is a noisy copy of κ. The mean of 950 of them — about what one
evaluation bank holds — estimates κ to 0.0076 on the worst
checkpoint, comfortably inside the 0.35 radius; the estimate itself passes `marker_valid` on every
checkpoint, markers minted from it pass at 0.9999 and the learned
gate at 1.0000, and a *single* signed marker copied onto a
shredded row passes as well. `make_centre` derives κ from `10_000 + seed`, and every checkpoint
serialises κ verbatim, so shipping a trained model ships the ability to mint signatures.

What this narrows: the marker gate is an **integrity check against unprivileged or accidental
modification**, not a cryptographic signature against an adversary with bank or checkpoint access.
The architecture document's crypto-shredding analogy does not survive contact with an adversary who
has either. Nothing in the recorded deletion results depends on the marker being secret — SHRED's
guarantee is that the gate closes, not that an attacker cannot open it — but the analogy was doing
rhetorical work the mechanism does not support, and the claim is withdrawn to the integrity reading.

### 31.13 Two criteria that are one measurement (2026-09-04)

E-000017-B's pre-registered groups check refusal after REVOKE and after SHRED separately, and the
ledger and roadmap both phrase the result as holding "after REVOKE and after SHRED" — which reads as
two deletion mechanisms agreeing. Per seed they are one number.

In every one of the 36 comparable cells of `e000017b_templates8.json`, `revoke/templateN_unknown`
equals `shred_hard/templateN_unknown` exactly, so `revoke_heldout_min` = `shred_heldout_min` = 0.92 /
0.91 / 0.865 and `revoke_train_min` = `shred_train_min` = 0.955 / 0.96 / 0.96. All three seeds of
`e000012_status_gated_revoke.json` show the same identity. `e000011_gpt2_v2.json`, the same
architecture *before* the status flag, separates them completely: revoke_heldout 0.510 against
shred_heldout 0.899, revoke_heldout_min 0.173 against 0.750.

The identity arrives with the status flag and is by construction: E-000012's design change makes the
flag multiply the same verification gate the marker feeds, so a revoked cell and a hard-gated
shredded cell null the injected value by the same arithmetic, and the frozen model's answer is then
decided by the same remaining computation. The two are separately executed — the experiment revokes,
measures, restores, shreds, sets `hard_gate` and measures again — and they *can* diverge, which one
divergent template pair in E-000012 shows. They simply do not.

What follows:

1. **Four of E-000017-B's ten criteria are two distinct numbers.** `refusal_generalises` rests on one
   measurement, not two, and its margin is 0.015. The `shred_train_min` ≥ 0.90 bar is vacuous beside
   the ≥ 0.95 bar on the identical quantity.
2. **No recorded verdict changes**, and the criteria are left exactly as pre-registered so that they
   stay reproducible. A warning at the definition says what they may not be read as.
3. **The adjudication of kill criterion 5 is untouched.** That criterion is a single unconditional
   held-out number against a 95% bar, and 0.898 is below 0.95 however many mechanisms produced it.
4. **The record contains no independent evidence that representational shredding refuses on held-out
   phrasings** — only that the gate closes. Getting it would mean measuring after SHRED with the
   status flag left ACTIVE, which no recorded run does.

### 31.14 A deletion certificate, and the first certified deletion in the programme (2026-09-04)

Every deletion result here, the F4 label included, is an attack that failed to recover the fact.
§31.10 is the bill for that standard: SHRED passed a calibrated probe, forced choice, logit rank and
top-1 across 750 pooled trials, all at chance, and an attack written afterwards recovered the object
at 1.0000 through a channel none of the four read. An attack bounds one adversary.

E-000030 reports the other kind of evidence. For each lifecycle operation it sweeps **every value the
deleted payload could hold** — the domain is an entity id, so 256 values, every case rather than a
sample — and asks whether the model computes anything different. Two levels:

* **interface.** Both models read the store in exactly one place: `MutableKnowledgeTransformer.forward`
  computes `enc = encode_bank(bank)` at `so/model.py:246` and thereafter touches only `enc["k_f"]`,
  `enc["v_f"]`, `enc["k_r"]`, `enc["v_r"]` and `enc["active"]`, and `KnowledgeAdapterLM.forward` takes
  only `keys`, `values` and the allowed set at `so/llm_adapter.py:323`. An invariant encoding therefore
  means an invariant computation **for every possible query** — multi-hop, reverse, phrasings nobody
  has written — not merely for a swept set, at one cheap encoding per payload value and without ever
  running the core.
* **outputs.** The returned logits over 838 questions: every candidate object as a
  reverse question, which is exhaustive over §31.10's attack surface, the targets' own forward
  questions, and 64 unrelated ones, because a target's key sits in the softmax denominator of
  questions that are not about it.

Two guards keep it honest, and both were needed. `check_mediation` looks for an output that moves
while the encoding holds still, which would mean the interface is not the only path and the
certificate is void; a test builds a runner that reads the bank behind the encoding's back to confirm
the guard can fail. And the first version of the adapter arm compared `values_payload` — the UNGATED
payload, which `encode_bank` returns as a diagnostic and `forward` never reads — and reported a
residual of 3.49 through a tensor the model does not look at. Naming the consumed interface is part of
the claim, not an implementation detail: too wide is over-strict, too narrow is unsound.

| model | operation | certified for every query | certified on the swept questions | first quantity that moves |
|---|---|---|---|---|
| synthetic (E-000010) | REVOKE | no | **yes** | `encode_bank[v_f]` |
| synthetic (E-000010) | SHRED | no | **no** | `encode_bank[v_f]` |
| synthetic (E-000010) | DELETE | yes, structurally | yes, structurally | the row is not in the bank |
| frozen GPT-2, soft gate | REVOKE | **CERTIFIED** | — | — |
| frozen GPT-2, soft gate | SHRED | no | — | `encode_bank[values]`, residual 1.390e-02 |
| frozen GPT-2, hard gate | REVOKE | **CERTIFIED** | — | — |
| frozen GPT-2, hard gate | SHRED | **CERTIFIED** | — | — |

Four things this settles.

1. **The first certified deletions in the programme.** In the frozen GPT-2, REVOKE is independent of
   the deleted payload under both gate modes, and SHRED is independent under the hard gate. Not "no
   attack recovered it" — the computation is bit-identical for all 256 values the payload could take,
   so no attack can.
2. **What the hard gate buys, exactly.** The soft gate is a sigmoid and never returns zero, so a
   shredded cell's value keeps 1.390e-02 of its payload and the certificate
   fails. The hard gate thresholds to exactly zero, the value becomes exactly the ' unknown' direction,
   and the certificate holds. No recorded experiment had made that statement; it was assumed.
3. **§31.10 restated as a proof.** SHRED in the synthetic model is certified at neither level, and the
   first quantity that moves is in the encoding. The attack was not unlucky; the dependence is there.
4. **REVOKE's two levels are not the same claim.** Its outputs are certified — nothing a user sees can
   move — while `v_f = v_fwd(o) * g` still carries the object, because REVOKE leaves the marker valid
   and masks the row instead. An adversary reading activations sees what a user cannot. Only DELETE,
   which takes the row out of the bank, is independent with nothing left to gate.

The rule the programme should have started from: **a deletion primitive is only as complete as the set
of payload-derived quantities it removes, and the cheap way to find that set is to sweep the payload
domain and watch what moves.** `so/audit.py` is that sweep; `make certify` runs it.

### 31.15 The strongest label in the certificate ladder fired on a live row (2026-09-04)

§31.14 records DELETE as "yes, structurally" at both levels, and `so/results/e000030_deletion_certificate.json`
carries `delete/structurally_certified = true` in every seed. Reading `so/audit.py` while building the
fact-level composition showed what produced that flag:

```python
if not len(list(deleted_rows)):
    return StructuralResult(False, 0.0, 0, "no deleted row remains in the bank, ...")
```

`certify_structural` was called with an EMPTY row list, because after DELETE there is no row to
perturb. With no row selected autograd has nothing to trace a path FROM, so it answers "no path" —
the strongest label in the ladder, the one whose docstring calls it "a theorem about every value the
payload could take, over any domain". Run on a bank whose rows are **all present and live**, the same
call returns the same label:

| call | bank | answer |
|---|---|---|
| `certify_structural(m, bank, [0], ...)` | row 0 live | REACHABLE, \|grad\| 2.219e-02 |
| `certify_structural(m, bank, [], ...)` | row 0 live | **NO PATH, certified** |

The flag certified the deletion by not testing it. It is the same failure as the `no_grad` runner the
same function already refuses, one step along, and the audit's own test asserted it as the desired
behaviour.

**What actually carries the claim.** Not reachability — membership. The model reads the store in
exactly one place (`so/model.py:246`, `so/llm_adapter.py:323`, and `check_mediation` is the standing
falsification of that premise), so a payload with no row in the bank is **not an input**, and no
function of the model can depend on it: over any payload domain, finite or not, for every query. That
is stronger than the sweep and stronger than the gradient, and it costs a set difference.

`so.audit.check_absence` states it, and its positive control is mandatory rather than optional:
"the row is not there" is evidence of a deletion only if the row **was** there and **mattered**, so
the same reachability test must be run on the pre-removal bank at the rows about to go and must find
a path. Without the control the check would pass on a cell that was never in the bank, which is the
identical failure one level further along; `check_absence` refuses that too, by name.

Three changes, and one correction to the record:

* `certify_structural` now raises on an empty row set instead of certifying it, naming
  `check_absence` as the instrument for a row that is gone.
* E-000030's DELETE arm reports `delete/control_reachable_before` and `delete/payload_absent`, both
  pre-registered at 1.0, in place of `delete/structurally_certified`. The DELETE verdict is unchanged
  — the row genuinely leaves the bank, `delete/rows_removed` always equalled the number of targets —
  but it now rests on evidence that could have come out the other way.
* E-000032 (§31.16) uses the same control before every eviction it certifies.

The general rule, which is the third time this programme has paid for it after §31.10 and §31.13: **an
instrument that cannot fail is not evidence.** A certificate needs a case where it says no, and the
cheapest way to find whether it has one is to run it on the state the deletion was supposed to change.

### 31.16 A pointer is separable from an object by its norm alone (2026-09-04)

E-000015's record, E-000020's record and the code that produced both say, in these words: *the store
decides which payload a row carries; the model is never told that a value it has read is a pointer —
that it must learn.* A prior-art review of the symlink concept asked whether recognising a pointer is
learned at all or given away by the geometry. E-000034 measured it on the recorded checkpoints:

| value projection | pointer norm | object norm | gap (pooled sd) | best single threshold | linear probe | direct read | alias read |
|---|---|---|---|---|---|---|---|
| recorded (separate `v_link`) | 18.557 | 11.917 | 7.6 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |

`encode_bank` carries a fact row's value through `v_fwd(ent_emb(obj))` and an alias row's through
`v_link(ln_key(ent_emb(link_subject) + cell_rel_emb(link_relation)))` — two projections whose scales
nothing couples. **At initialisation the two ranges overlap.** The gap is therefore learned, not built
in: the architecture supplies the freedom to tag the kinds apart and training takes it, completely.
The first write-up of this finding blamed the architecture and was wrong on that point.

So the recorded sentence is literally true — the model is not told — and false as a claim about
difficulty, which is what it was doing in the record.

**The claim that survives is narrower and still real: recognising a pointer is free; only following one
is learned.** E-000016's one-slot arm refuses a two-link chain rather than inventing an answer, which a
branch on a flag would not do, and E-000015's passthrough column keeps non-pointer values readable.
What is retired is the difficulty claim, not the capability.

`ModelConfig.share_link_value` is the arm that could earn the original wording back: both payload kinds
through the same LayerNorm and the same projection. That removes the freedom, not the possibility — the
model can still learn a scale difference by growing `cell_rel_emb` — so the criterion is a measurement
and not a theorem. Pre-registered at `shared/norm_threshold_accuracy <= 0.75` with
`shared/alias_direct >= 0.80`; if resolution collapses without the cue, what gets recorded is the
design's dependence on it.

### 31.17 What the prior art owns, checked against the sources (2026-09-04)

Six literatures were reviewed for the pod claim before any of §31.15–§31.16 was written up. Two
findings changed what this work may say, and both were verified against the primary source rather than
taken from the review.

**The metric already has a name.** What `so/closure.py` computes is RESILIENCE in the database sense —
the minimum number of tuples whose removal makes a Boolean query false — and the set is a CONTINGENCY
SET. It is studied under deletion propagation and causal responsibility and carries a PTIME/NP-complete
dichotomy for self-join-free conjunctive queries. The module now says so, uses those words, and exposes
`resilience` as the name of `fact_closure`. Inventing vocabulary for an existing metric would have made
the work unreadable to the people who own it.

**The remedy is already proposed, and not built.** Raeesi and Roed, *Auditing Forgetting in Limited
Memory Language Models* (arXiv:2607.00605), audit deletion in a database-backed LM over 12,228
deletions and conclude that parametric leakage is near zero while post-deletion correctness is
reconstituted from retrieval, so "the unlearning boundary is drawn primarily by the database
administrator rather than by the model". Their §9 proposes the pod verbatim: *"A second direction is
canonicalization at write time, in which aliases and paraphrastic forms are stored as pointers into a
single canonical record rather than as independent triplets."* The next sentence is *"Both approaches
are directly testable within our framework"* — so the closest work names the remedy, says it is
testable, and does not test it. Their §8 asks for "a closure procedure". They report post-deletion
correctness and no contingency-set size.

So the pod is not this programme's idea, and the ledger should stop implying it is. What remains, stated
as narrowly as the evidence allows:

1. the closure **computed** as a store statistic, with a certified lower bound so `optimal` is verified
   rather than assumed (§31.18, E-000032);
2. its **composition** with a record-level deletion certificate into a fact-level one — which nobody
   else can do yet, because E-000030 is the first such certificate in this line of work;
3. the closure **predicting** the reader: `(closure − 1) / keys_per_group`, computed before the model
   runs, against what the model actually still answers;
4. the **price** the neural reader charges for the indirection — 0.0954 for sharing and 0.0688 for link
   training, worst of three seeds over twelve phrasings (E-000025) — which is the part that is not in
   Codd, because a join is exact and a learned dereference is not.

### 31.18 Three words the certificate was using loosely (2026-09-04)

An adversarial pass over the fact-level certificate, run against HEAD rather than against a
description of it, found that two of the four defects it went looking for had already been fixed and
three claims were still stated more loosely than the code supports. All three are now instruments
rather than sentences.

**"Gone" was doing the work of "unreachable".** Every certificate in `so/audit.py` is about
reachability: the model cannot depend on the payload, no surviving bank row is a function of it, no
query in the workload yields it. None of them says the payload is *gone*, and under EVICT it
demonstrably is not — keeping the versions is the operation's entire purpose, which is what makes
RESTORE work, and the write-ahead log keeps a copy too. `check_retention` reports where the payload
still lives, `FactCertificate` carries it, and the verdict now reads **"FACT UNREACHABLE, CERTIFIED …
— UNREACHABLE, NOT ERASED: the store still holds the payload."** That two-part statement is what EVICT
actually earns, and it is a useful thing to earn: reversible, auditable, and constant-cost. It is not
erasure, and a record that said it was would be the exact mis-description this programme exists to
avoid.

**"The fact" was not individuated.** `fact_closure` measured over a pod certifies the
(subject, relation, object) triple. It says nothing about the same VALUE stored under a different
subject — and `so/closure.py`'s own `value_keys` selects on value alone, so measuring a closure over
*that* set removes a bystander's record as well, which is over-deletion, the dual of the failure the
pod exists to prevent. There is now a test that destroys the bystander on purpose so the distinction
cannot be lost again, and `FactCertificate.individuation` prints which of the two a verdict means.

**"Codd's deletion anomaly" was the wrong anomaly.** The failure the pod prevents is redundancy plus a
delete that reaches one place of many — Codd's **modification** anomaly applied to a delete. His
*deletion* anomaly is the opposite failure: losing information nobody asked to lose. The distinction
matters here because this work commits both errors in different places, and the vocabulary has to keep
them apart. Corrected in `so/closure.py`, both experiments and the tests.

One case the same pass identified as untested is now tested: a **shadowed duplicate**, a second FACT
cell holding the same key. `_key_index` is first-holder-wins, so the shadow answers nothing until the
first record goes and then starts answering. Its two derivations are never live at once, so no
disjoint pair exists, the certified lower bound is 1 while the true closure is 2, and `optimal` is
correctly False. That is the case where the greedy search does non-trivial work and where saying
"proved optimal" without checking would have been wrong.

### 31.19 What canonicalisation costs the subject, not the reader (2026-09-04, E-000035)

E-000032 measures what a pod buys: the fact closure falls from k to one, so a record-level certificate
composes into a fact-level one. E-000025 prices what it costs the READER: 0.0954 for sharing, 0.0688
for training the dereference. Neither asks what it costs the SUBJECT of the deleted fact, and there is
a cost, because the two arrangements do not leave the same trace behind.

Delete one of k duplicated copies and the store is a store with k−1 copies: nothing in it says a
deletion happened, and nothing says where. Delete a pod's object and every one of its k−1 aliases is
still a LINK row still carrying the removed cell's key in `bank["link_subject"]` and
`bank["link_relation"]` — kept deliberately, because E-000015's design makes the model discover the
miss rather than being handed it by the control plane. Each surviving alias is a signpost reading *a
record stood at (s, r) and is gone.*

No model, no checkpoint, no training. The adversary reads `MVCCStore.bank()` and names every key a
LINK row points at that no row holds. Three seeds, 100 pods each:

| store | deleted key disclosed | uniquely identified | candidate keys left | false positives | dangling before any deletion |
|---|---|---|---|---|---|
| canonical | **1.0000** | **1.0000** | **1.0** of 1,536 | 0.00 | 0.0 |
| duplicated | 0.0000 | 0.0000 | 1,536 of 1,536 | 0.00 | 0.0 |

Every pre-registered criterion passes, including the mitigation's: blanking a dangling pointer's key
closes the channel at 1.0000 and makes every such pointer identical at 1.0000. And that is the trade,
recorded as a number rather than argued — with the key blanked, an alias to a removed target is
indistinguishable from an alias to key (0, 0), so E-000015's `delete_target/alias_unknown` stops being
a discovery about the model and becomes a tautology about the bank.

**And the closure inverts with the guarantee.** E-000032 measures how many records must go before no
query yields the object. This measures how many before the bank shows no evidence a deletion happened
there. The same two stores swap places, exactly:

| guarantee | canonical pod | duplicated |
|---|---|---|
| unreachable to the reader (E-000032) | **1.00** | 3.00 |
| no trace left in the bank (E-000035) | 3.00 | **1.00** |

A pod's aliases are the signposts, so leaving no trace costs the object plus all of them; a duplicated
store costs the one record you were removing anyway. Both numbers are exact on every seed. Quoting
only the first row is quoting the half that flatters the design, and this programme has done that
until now.

**The claim this changes.** "Canonicalisation makes erasure a single certifiable operation" is now
paired with "and it turns every alias into a deletion oracle". Both are properties of the same design
decision. An erasure guarantee that does not mention the second is describing half its own system —
"was there a record about this person, and was it deleted" is exactly the question the guarantee is
supposed to make unanswerable, and here it is legible to anyone who can read the bank, without
touching the model.

Two limits, stated because the number invites over-reading. It is a property of THIS store's bank: one
that compacts its aliases on deletion, or never exports the target key, has no such channel. And it
measures what a reader of the bank can see, not what the model exposes to someone who cannot.

### 31.20 Exactly zero is not absent: the certificate compared numbers where it claimed to compare bits (2026-09-04)

`Certificate`'s docstring says the model "returns bit-identical tensors". `_compare` implemented
`(a - b).abs().max() > atol`. IEEE-754 has a signed zero, `-0.0 == 0.0` is true, and `x * 0.0`
preserves the sign of `x` — so a multiplicative gate at exactly zero maps the payload onto a vector of
**signed zeros whose sign pattern is still a function of the payload**, and the comparator could not
see it. Measured on the synthetic model with `hard_gate=True` and the gate forced shut:

```
gate on shredded rows                                        exactly 0.0
v_f numerically equal for every payload value                True
distinct SIGN-BIT patterns over the 32-value payload domain  32   <- a bijection
_compare(v_f at obj=3, v_f at obj=11)                        []   <- no violation
```

Every payload value gives its own bit pattern, and the certificate reported no difference. That is the
sound-unsafe direction, and it opens **only when the gate is exactly shut** — which is the only
condition under which a certificate is ever issued.

**The fix.** At `atol == 0` the comparison is now bitwise, via an integer view of the tensor, in
`_compare` and in both of `audit_independence`'s comparison sites; a violation found this way is
reported as `<name><bits>` with the fraction of elements whose bits move, because its numerical
magnitude is exactly zero and reporting that would hide the violation inside its own measurement. At
`atol > 0` the numerical comparison stands: a caller who accepts a tolerance has given up bit-identity
and asked a different question. The hard-gated SHRED that used to certify now fails at
`encode_bank[v_f<bits>]`, and EVICT still certifies, because a row that is not in the bank has no
tensor to carry a sign.

**Which recorded numbers move: none, and the reason matters.** E-000030's synthetic arms set
`model.cfg.hard_gate = False` (`e000030…py:110`), where the residue is numerically real and was
already recorded as a failure. The frozen-GPT-2 arm computes `values = payload·g + unk·(1−g)`
(`so/llm_adapter.py:240`, the `fallback="unknown"` branch the recorded E-000012 adapter uses), and
adding a nonzero constant destroys the sign of a zero — at the soft gate the recorded run already
reports a residual, and the hard-gate arm is being re-run under the fixed comparator rather than
assumed. What changes is not a number but **what the word "certified" is permitted to mean** from here.

Nor is this a practical attack: an adversary who can call `encode_bank` already holds `bank["obj"]`,
and the sign of a zero does not survive the next matrix multiply. It is an instrument defect, which is
the more expensive kind — this is the **fourth** after §31.10, §31.13 and §31.15, and they share one
shape: an instrument that agrees with its specification on the distribution it was checked against and
diverges on the one the system actually occupies.

**Provenance.** Found by an adversarial design agent run over the certificate, which measured it on
the recorded checkpoints before asserting it; its specification is kept verbatim at
`docs/e000036-spec.md` (renumbered from its own E-000035, which the disclosure experiment had taken).
The claim was re-derived here before the fix was written. The same document's arm K — whether
`gate_reverse_key` closes E-000028's channel to chance at n=500, where a pilot at n=20 gives 0.0500
with a 95% interval of [0.0013, 0.2487] — remains unrun and is the next thing worth running.

### 31.21 The paraphrase gap is a token-position artefact (2026-09-04)

Kill criterion 5 fired because deletion did not reach phrasings nobody trained on. §31.11 recorded the
lifecycle claims as statements about one phrasing. E-000025 recorded alias resolution at 0.9250 on one
held-out template and 0.3078 on the worst. Two days of this programme have treated that spread as
evidence about reading, about deletion, and most recently about carrier multiplicity.

It is none of those. It is **whether the subject name is the first token of the prompt.**

Split the twelve surface forms of `TEMPLATES12` by the GPT-2 token index of the subject — a property
of the tokenizer, computed from the tokenizer, never from any recorded accuracy. It gives
`{0, 2, 6, 8, 11}` subject-initial and the rest medial, identically for all four relations. E-000025's
**already-published** per-template table separates on that split perfectly:

| position | templates | recorded direct read |
|---|---|---|
| subject **initial** | 0, 2, 6, 8, 11 | 0.3078 – 0.6422 |
| medial | 1, 3, 4, 5, 7, 9, 10 | 0.9567 – 0.9989 |

No overlap. The worst medial template beats the best initial one by 0.31.

And the fix costs nothing. Prepending `"It is known that "` to the prompt — **no weight changed, no
retraining, the recorded E-000017-B seed-0 checkpoint as it stands** — on 64 targets:

| | before | with the prefix |
|---|---|---|
| subject-initial (5 templates) | 0.6813 | **0.9844** |
| medial (7 templates) | 0.9933 | 0.9464 |

with the two worst held-out templates going **t8 0.5625 → 1.0000** and **t11 0.3750 → 0.9844**. The
prefix is not free everywhere: the medial mean falls slightly, almost all of it one template (t9,
0.9688 → 0.7031), so this is a targeted fix for a positional failure and not a general improvement.

**What this costs.** Kill criterion 5 was measuring prompt formatting. Its held-out set {8, 9, 10, 11}
contains two subject-initial templates, so the held-out mean it fired on was dragged down by an
artefact of where the name sits. The criterion stays fired as recorded — records are not edited — but
it is not evidence about the deletion mechanism, and §31.11's framing of the lifecycle claims as
phrasing-dependent needs reading with this in front of it.

**What this does not cost.** E-000025's `cost_of_sharing` (0.0954) and `cost_of_link_training`
(0.0688) are differences between arms *at the same template*, so the positional artefact cancels in
them. Those numbers stand.

**What it falsifies of my own.** The carrier-multiplicity explanation for the paraphrase gap — that a
fact has one carrier per phrasing, so deleting the record misses the phrasings nobody trained on — is
wrong. The gap is addressing, and the cause is positional. E-000038's *privacy and collateral* half is
untouched, because it rests on the measured GPT-2 collateral of 1.0000 → 0.0000 rather than on this;
its *tying* motivation does not survive and the record says so.

The finding is E-000039's, from an adversarial design agent, and the numbers above are my own
re-measurement written without its code. The honest recommendation is its own: **normalise the prompt,
and do not train.**

### 31.22 Traceless erasure is invariant under canonicalisation (2026-09-04, E-000041)

§31.19 recorded that a pod and a duplicated store invert: the pod makes a fact unreachable in one
record instead of k and pays by naming the deleted key to anyone who reads the store. That inversion
has a reason, and the reason is a law rather than an observation.

Two costs, for the same goal, for a fact reachable through **k access paths**:

* **U** — the minimum records to remove so that no access path yields the object.
* **T** — the minimum to remove so that no path yields it **and no surviving row points at anything
  that is gone**.

Measured exhaustively over the whole spectrum between the two arms §31.19 compared — k from 2 to 8,
every number of links from 0 to k−1, three seeds, **105 of 105 cells**, mechanical resolver, no model:

| k | U, by links (0 = all copies … k−1 = full pod) | T, same order |
|---|---|---|
| 2 | 2 1 | 2 2 |
| 4 | 4 3 2 1 | 4 4 4 4 |
| 6 | 6 5 4 3 2 1 | 6 6 6 6 6 6 |
| 8 | 8 7 6 5 4 3 2 1 | 8 8 8 8 8 8 8 8 |

**U = 1 + (paths still stored as copies). T = k, in every cell.** Reading a row left to right is
walking from a duplicated store to a canonical pod: U falls from k to one, and T does not move.

**What it says.** Canonicalisation is not a reduction in the cost of erasure — it is a move along a
trade-off. A pod's k−1 aliases *are* its references, so cleaning them costs k−1 on top of the object;
a duplicated store has no references to clean and its k copies already cost k. The same total, reached
from opposite ends. Every claim of the form *normalise and erasure becomes one operation* is a claim
about U and is true; the same design leaves T where it was.

**Why it is the interesting half.** Codd's modification anomaly is U. Database resilience — the
minimum contingency set — is U. Raeesi and Roed's §9 proposal to store aliases "as pointers into a
single canonical record", offered with "Both approaches are directly testable within our framework",
is a proposal about U, untested. T is what a data subject is promised when they are told a record is
gone, and none of them states it.

**The pre-registered criteria that could have failed** and did not: `T_equals_k >= 1.0` and
`U_matches_prediction >= 1.0` are checked in every cell of the grid, so a single deviation anywhere
falsifies the law; `all_valid >= 1.0` is the control that a traceless search which emptied the store
would fail; and `U_min <= 1.0` with `U_max >= 4.0` require the contrast to be real rather than a flat
line described as a trade-off.

**Scope, stated because the number invites over-reading.** T is invariant *for this store's
semantics*, where `MVCCStore.bank()` exports a link's target key and keeps doing so after the target
is gone (§31.19). A store that compacts its aliases on deletion, or never exports a target key, has a
different T, and the law would have to be restated for it. That is why it is recorded as a measurement
over a spectrum and not as an identity — and it is also the actionable part, because it names a design
choice a system can now make knowingly.

### 31.23 The store side of the claim completes; the injection channel does not close (2026-09-04)

Two experiments that had been running since the session opened finished, one confirming and one
refusing to.

**E-000032, three seeds, twenty-six pre-registered criteria, all PASS.** This is the experiment
§31.22's law was extracted from. Its most-quoted number needs a correction that is recorded here
rather than left standing: `(closure − 1) / keys_per_group`, computed from the store **before the
model is run at all**, matches what the reader reads back after only the object record is
removed, with error 0.0000 in all three arms. **[Corrected in §31.33: the reader in E-000032 is the
E-000015 `MutableKnowledgeTransformer`, not the frozen GPT-2 as this section first said; and the
agreement is star arithmetic — on a chain the formula is wrong by a grid step against the store's
own resolver, with no model in the loop.]**

**That is not a forecast of a neural behaviour, and calling it one over-reads it.** Remove the object
record and the keys that still answer are exactly the ones whose own record survives — which is the
store's arithmetic, not a discovery about the model. Given a faithful reader the number cannot come
out otherwise. What the 0.0000 *does* establish is READER FIDELITY. **[The defence that followed here
— that a frozen GPT-2 with a pretrained prior over capitals could have gone on answering from the
prior and did not — is withdrawn in §31.33: the model that ran is trained from scratch on worlds
resampled every step and has no prior to answer from; on the actual GPT-2 adapter the same read is a
recorded FAIL (E-000020, direct 0.5700 / alias 0.5067 / dup 0.5483) and the test was not run there.]**
The value of the result is therefore COMPOSITIONAL rather than predictive — because fidelity is
established once as a property of the method, a fact-level guarantee reduces to a store-side search
plus a store-side sweep.

| store | fact closure | predicted still readable | measured still readable | prediction error |
|---|---|---|---|---|
| canonical | 1.00 | 0.0000 | 0.0000 | 0.0000 |
| mixed | 2.00 | 0.3333 | 0.3333 | 0.0000 |
| duplicated | 3.00 | 0.6667 | 0.6667 | 0.0000 |

The **per-KEY** closure is 1.00 in all three arms while the **per-FACT** closure separates them 1 / 2
/ 3. That pair is the whole argument for why a record-level certificate cannot be trusted to make a
fact-level statement: at the record level the three stores are indistinguishable. `proved optimal` is
1.00 everywhere — the greedy search met a certified lower bound from pairwise-disjoint derivations,
so the numbers are minima and not merely the best found.

And the cost, as this section first stated it: **1.45–1.80 s per certified fact deletion and zero
model evaluations inside it.** **[Corrected in §31.33: the zero was a literal in the code, not a
count; the certification window holds one standalone `encode_bank`, and the reachability control
runs the model once per fact inside the loop and was timed apart. The counted table is in the
re-run's report.]** E-000024
is the comparison — 129 s by gradient ascent, 335 s by relabelling, 2,359,296 parameters changed,
perplexity on ordinary prose from 42.9 to 6.19e+09, and no certificate available at all, because
there is no finite payload domain to sweep and no interface the data passes through.

**E-000022 is recorded as the negative it is.** E-000018's capacity question — can the adapter be
stopped from injecting into text it has no key for — was handed to E-000022 under a restructured
injection path where a closed gate demonstrably injects less. It does inject less, and it is not
enough:

| claim group | supported |
|---|---|
| no_key_no_injection | **no** |
| reading_not_traded_away | **no** |
| refusal_not_traded_away | yes |
| deleted_object_never_returns | yes |

`generic/kl_to_base` is 0.5508 mean and 0.8657 at the worst seed, against E-000018's 0.6736 and a
pre-registered bar of 0.05. Splitting the null column into a payload channel and an unknown channel
buys about a fifth on a quantity that needs an order of magnitude. `train/active_correct` also misses
at 0.8844 against 0.90, so the fifth was not free either. What survives is what survived in E-000018:
refusal at 0.9894 held out and the deleted object at 0.0000 — the two claims that were never the
problem. The structural reading that motivated the split is unchanged and remains unpaid for:
answering `unknown` when a cell is gone needs an injection, changing nothing on text with no key needs
none, and routing both through one column was the wrong shape. Splitting it was necessary and is not
sufficient, and no third experiment is queued on the strength of a hypothesis that has now failed
twice.

### 31.24 The gap is addressing, the law does not cross into a representation, and a theorem I mis-stated (2026-09-04)

Three things landed together, and only the first is a clean win.

**E-000039-A decides, now that it measures the quantity its own rule is written on.** The record had
been saved with `--no-oracle`, which skips the `cell_mask` arm the residual gap is computed from, so
`heldout/routing_share` was NaN and the decision rule stood on nothing. Re-run with the oracle:

| measure | mean over seeds | worst seed |
|---|---|---|
| heldout/gap | 1.1100 | 1.0400 |
| heldout/residual_gap | 0.1267 | 0.1100 |
| **heldout/routing_share** | **0.8861** | **0.8818** |

The rule, fixed before either arm was trained, was `routing_share >= 0.7 -> train the address arm
alone`. It fires, at the worst seed as well as the mean. **Forcing the address closes 88.6% of the
held-out reading gap**, so the paraphrase failure is addressing and not transport, and §31.21's
positional diagnosis has the number it was missing.

**E-000040 refuses to carry §31.22's law across the boundary, and says why.** The deletion itself
works: 3.33 directions take the answer from 0.9444 to 0.0000 while bystanders hold at 0.8611 from
1.0000. And the carrier is *mostly private* — only **0.0784** of a fact's basis is shared with another
fact. But privacy is not sufficient:

| measure | mean | worst seed |
|---|---|---|
| share of a fact's basis shared with other facts | 0.0784 | 0.0809 |
| facts silenceable using only their OWN directions | 0.5098 | 0.4706 |
| traceless gap (a_hide − a_answer) | +0.1236 | +0.0375 |
| facts where that gap is positive | 0.3611 | 0.2500 |
| `hole_detectable` (pre-registered ≥ 0.75) | **0.6667 — FAIL** | |

Half the facts cannot be silenced with nothing but their own directions even though almost none of
those directions is shared with any *single* other fact; the rest of the model is using them anyway.
And the detector fails its own validity bar, with the traceless gap positive for barely a third of
facts. **`T > U` is a statement about a store**, where an alias is a row that can be named. On this
evidence it does not transfer to a representation, and the earlier draft that said it did was reading
a mean over a minority.

The report also had to be corrected against itself: it printed "most of what carries a fact is not
that fact's" directly beneath its own measurement of 0.0784, which says the opposite.

**And a theorem I mis-stated, which costs the largest version of the E-000042 claim.** I asserted that
LEACE gives sufficiency and not a lower bound. **Theorem 4.1 is an iff** — "r(X) linearly guards Z if
and only if the columns of the cross-covariance matrix Σ_XZ are contained in the null space of P" —
whose *only if* half binds every affine erasure map to `dim ker(P) ≥ rank(Σ_XZ)`. LEACE attains that
rank, so the interval is *closed*. Theorems 4.2 and 4.3 are the sufficiency results; 4.1 is the
necessity one, and the paper deploys it only to certify that other methods suffice, which is why it
reads as a sufficiency paper. A certified lower bound on erasure cost in a representation exists and
predates this programme.

More of the same: Adolfi, Vilas and Wareham (ICLR 2025, arXiv:2410.08025) already define
**k-robustness** — "no set of k or fewer components erases the behaviour" — and their Problem 8 is
literally a hitting set over neurons, NP-hard and in Σ₂ᵖ, tractable in the size of the candidate set.
Bassan and Katz (arXiv:2210.13915) already run the **disjoint-packing dual** with sound certificates
and report the upper/lower ratio as a certified approximation factor, over input features via a
complete verifier. What is left for `so/support.py` is narrower and is written down as such in
`docs/so-claim-certified-representation-closure.md`.

**Two instrument failures found the same way, both about ablating the readout instead of the fact.**
E-000042's first run used rows of `W_U` — the logit lens, the J = I case — and went VOID: eight
directions removed, not one fact silenced. The apparent fix was worse: removing the eight logit-lens
rows *of the candidate capitals* took the answer to 0.0000, but those rows are the readout of the
candidate set, so the restricted argmax becomes noise. The workspace paper guards against exactly this
— its global ablation "does not ablate any tokens that appear in the top-10 tokens of a clean forward
pass" — and the guard is now carried over. Separately, a parallel measurement on this repository's own
checkpoint found a closure of 1.00 at collateral 0.0044 whose ablated states still yielded the object
to a **freshly fitted linear probe at 0.9300 held out**. So `probe_after` is a pre-registered criterion
in E-000042 and not a diagnostic: a closure a probe walks through removed a readout path, not a fact.

### 31.25 Superposition buys representation capacity and not deletion capacity — and GPT-2 is nowhere near the bound (2026-09-04, E-000043)

**The argument.** A clean deletion of fact *i* is an orthogonal projection removing a minimal subspace
`A_i` such that no access path of fact *i* still yields the object and every access path of every other
fact still does. Minimality makes every direction of `A_i` load-bearing for one of fact *i*'s paths;
zero collateral requires the projection to fix every other fact's readout subspace `V_j`. So `A_i ⊥ V_j`
for all *j ≠ i*, mutually orthogonal subspaces satisfy `Σ dim A_i ≤ d`, and **n ≤ d/s**. Representation
capacity is not linear in *d* — Johnson–Lindenstrauss gives exponentially many almost-orthogonal
directions and superposition is the observation that models use them.

**The caveat, stated because it changes the reading and strengthens it.** That bound is for *exactly*
zero damage. If collateral is judged by whether another fact's argmax flips, `A_i` need only be
orthogonal to `V_j` within each fact's margin, and almost-orthogonal sets are exponentially large
again. So the sharp bound is a limiting case and not a practical limit — which means an observed
failure to delete cleanly **cannot** be a capacity effect at this scale, and the allocation reading is
the only one left.

**The measurement**, frozen GPT-2 small, *d* = 768, 17 capital facts × 8 phrasings, all five
pre-registered criteria PASS:

| measure | value |
|---|---|
| facts a subspace of their own basis silences | 12 of 17 (0.7059), to 0.0312 |
| directions demanded in total | **58 of 768** |
| `pressure` (demand / d), bound 159 facts | **0.0755** |
| headroom unused | **0.9245** |
| `orthogonality` (σ_min of the stacked bases) | **0.2393** |
| overlap `A_i` vs `A_j` | 0.5566 mean, 0.8559 max |
| overlap `A_i` vs `V_j` — what the theorem needs | 0.5842 mean, 0.8575 max |
| bystander facts under the same ablation | 0.3897 from 1.0000 |

**Two corrections the run itself forced, before the verdict is quotable.** First, overlap under this
construction has a **non-zero null**: seventeen *random* states through the identical code path
overlap at 0.1448. Quoting 0.5566 against a baseline of zero overstates the effect, and the number
that means something is the **excess, +0.4118**. Second, `σ_min` was demoted from primary — a basis
built as "this fact minus the mean of the others" is a *centred* vector, n centred vectors span n−1
dimensions, and σ_min is therefore 0.0000 on real facts and on noise alike. A number that cannot
distinguish data from noise is not a summary.

**And the sharing is in the ADDRESSING, not the content**, which is the result:

| subspace | overlap | matched null | excess |
|---|---|---|---|
| content direction only (row 0) | 0.2232 | 0.0638 | **+0.1594** |
| addressing rows only (phrasing spread) | 0.5954 | 0.1930 | **+0.4024** |
| **addressing minus content** | | | **+0.2430** |

A fact's own content direction is nearly private; what is shared is the machinery that says which
phrasing asked for it. **What a store keeps in separate records — the object, and the keys that reach
it — a representation keeps in one subspace, so a deletion aimed at the content pays its collateral to
the addressing.** This closes a loop with §31.24: reading a fact through a new phrasing fails at
addressing (`routing_share` 0.8861), and deleting one damages bystanders through addressing. One
structure, measured from both ends.

**RETRACTED — the null was wrong.** The verdict above compared overlap against a *random-state* null.
Random states carry no template structure, and every fact here is asked with the same eight templates,
so the basis rows beyond the first are phrasing directions the DESIGN shares out to every fact. Against
a **design-matched permutation null** — within each template, shuffle which fact's state sits where,
preserving both marginals and destroying only the fact × template interaction — every sign reverses:

| subspace | real | random null | design-matched null |
|---|---|---|---|
| deletion subspaces (k ≈ 6) | 0.5898 | 0.1725 → **+0.4173** | 0.7895 → **−0.1997** |
| content only | 0.2232 | 0.0638 → +0.1594 | 0.1871 → **+0.0361** |
| addressing only | 0.5954 | 0.1930 → +0.4024 | 0.7762 → **−0.1809** |

Permuting the interaction *raises* the overlap, so the model's structure makes its subspaces MORE
distinct than chance. `excess_overlap` and `address_over_content` are kept as pre-registered and now
**FAIL** at −0.1306 and −0.2170. Both "allocation, not capacity" and "the sharing is in the addressing"
are withdrawn.

**What survives, and it is the stronger statement.** The collateral is real (0.3897 from 1.0000), the
capacity bound is nowhere near binding (pressure 0.0755, 92% headroom), and the model is not
allocating badly. So the damage is neither a capacity limit nor a training defect. It comes from a
fact's deletion subspace necessarily CONTAINING addressing directions, and addressing being shared
because facts are asked in the same ways — sharing that lives in the task, not in the model, and that
no allocation objective can remove. In a store the address and the object are separate records; in a
representation they cannot be pulled apart by allocation. **E-000044 is the test: if this reading is
right, the pod objective should not substantially reduce collateral.**

The withdrawn verdict, kept for the record: the model had 92% of its dimension budget free and gave
twelve facts deletion subspaces overlapping 0.41 more than a random-state null. Allocation is a training objective; capacity would
have been a law of dimension. That is the difference between a limit and a defect, and it is why the
pod objective — access paths of one fact sharing a core, cores of different facts disjoint — is worth
training rather than merely wishing for.

**The instrument found its own bug, because the two numbers it was built to report disagreed.** The
first run measured `rank(union) / Σ dim A_i` and called it efficiency. That is **linear independence**;
the theorem needs **orthogonality**. It reported **1.0000** — twelve subspaces totalling 58 directions
with rank exactly 58, a direct sum, "perfectly allocated" — in the very run whose pairwise principal
cosines were 0.5566 and 0.8559. Quoting it alone would have concluded the opposite of the truth.
`σ_min` of the stacked bases is now primary, the rank is kept beside it as the contrast, and a test
reproduces the failure with two subspaces at cosine 0.9 that rank scores at 1.00. A second fault in
the same run was also real: `answer_after` was averaging over facts the deletion never silenced, and is
now computed over the silenced ones with `silenced_rate` as its own criterion.

**E-000042 completes as a clean negative.** A J-lens ablation that respects the workspace paper's own
guard — never ablating a token in the clean top-10 output — silences **0 of 6** facts across all 256
subsets at pool 8, and 0 of 1 so far across all 1024 at pool 10. The earlier eight-direction ablation
that *did* take the answer to 0.0000 removed the unembedding rows of the candidate answers, which
blinds the readout rather than deleting anything.

### 31.26 The capacity theorem is false in the regime I measured, and two of its numbers could not fail (2026-09-04)

An adversarial review of `so/capacity.py` and E-000043 returned three findings, each demonstrated by
running code rather than by argument. All three stand and the claim is withdrawn.

**1. The theorem is FALSE under the notion of collateral the experiment actually uses.** The step
"zero collateral requires the projection to fix every other fact's readout subspace, so `A_i ⊥ A_j`"
silently reads *zero collateral* as *every other fact's ACTIVATION is fixed pointwise*. E-000043 does
not measure that. It measures whether the bystander's restricted argmax still yields its object — a
behavioural condition, which is not closed under intersection and yields no packing bound. The
reviewer built the counterexample: **d = 12, 66 facts** indexed by coordinate pairs, each with an
exactly minimal **1-dimensional** deletion subspace; **66/66 unreachable after their own deletion,
4290/4290 bystander checks exactly unchanged**, largest principal cosine 0.5000, and
**Σ dim A_i = 66 ≫ d = 12**. With k-wise readouts the same construction gives C(d,k), exponential.
So *"clean-deletion capacity is linear in d while representation capacity is exponential"* is false
where it was applied. My own write-up stated this as a caveat and then quoted the headline anyway; as
the reviewer puts it, the caveat and the headline cannot both be load-bearing.

**2. `pressure` and `headroom` are algebraic identities — the criterion could not fail.** `pressure =
Σ dim A_i / d` and `bound = d / (Σ dim A_i / n)` give `pressure ≡ n_measured / bound` identically:
12 / 158.89655 = 58 / 768 = 0.07552083333, matching to fifteen digits. And `fact_basis` returns
exactly 8 rows, so `dim A_i ≤ 8` and `n_held ≤ 17`, hence **`pressure ≤ 0.1771 < 0.50` for any model
and any data**. The pre-registered criterion "pressure ≤ 0.50, and it can fail" cannot. Worse, running
the identical pipeline on **pure Gaussian noise** returns pressure 0.0755 and rank-efficiency 58/58 —
bit-for-bit the GPT-2 values. *"The model had 92% of its dimension budget free"* restates *"twelve
facts were tested."* This is the ninth instrument in this ledger that certified by not testing, and it
is the number I quoted hardest.

**3. Greedy over an ordered basis is a PREFIX, not a minimal subspace.** The search accumulates rows
0, 1, 2, … and never tests whether an earlier one can be dropped, while the ordering feeds in the
highest-variance, most-shared template directions first. Backward elimination under the run's own
criterion cuts **58 directions to 23** — China 8→1, Cuba 6→1, France 7→3 — so **60% of the counted
directions are passengers**, and the "shared fact direction" placed first is load-bearing for only
4 of 12 facts. On the truly minimal subspaces the headline overlap falls **0.5566 → 0.2756**, the max
**0.8559 → 0.6743**, s falls **4.83 → 1.92**, and the bound moves 159 → 401.

**And the hypotheses were satisfied by nothing measured.** A clean deletion requires zero collateral;
collateral was 0.3897 from 1.0000, with **11 of 12 facts failing it**. `V_j` is also not the readout
subspace: for 5 of 17 facts, removing *all* of `V_j` leaves the answer at 0.975.

**What survives the review.** Against a dimension- and anisotropy-matched null the minimal subspaces
still show a **~2× excess** (0.2756 against 0.1334 ± 0.007) — real, and an order smaller than the 0.86
cosine the verdict quoted. And one premise held that I had not tested: **the deletion generalises to
phrasings it was never fitted to**, held-out survival 0.1333 from 0.9167, 11 of 12 facts under the bar.

### 31.27 E-000045: U measured on readers, and the three faults its controls caught (2026-09-04)

The idea behind it survives the above, because it does not use the capacity argument. E-000041's law
has k on both sides because a store's **alias row does two jobs** — a way *in* to the object, and a
record that survives carrying the object's key. That coincidence is what a symlink is. A representation
does not fuse the roles, so U is governed by fan-in and T by fan-out. E-000045 uses the workspace
paper's own **swap** on J-lens vectors rather than an ablation.

Three faults, each caught by a control that could fail and did. (1) The hook applied the swap at every
layer; the swap is an involution, so it swapped and unswapped alternately — and it read coordinates
with `lstsq`, whose default driver assumes full rank, while the identity arm builds a rank-one `V`.
(2) Predicates where two entities *share* a value scored as "followed" with no intervention at all;
the identity control read 0.2000 where an exactly-zero patch must read 0. Off-diagonal only, as the
paper does; both controls then read exactly **0.0000**. (3) **U was hardcoded to 1.0** — six of ten
entities had broadcast exactly 0.000, meaning nothing followed, and were being scored as T > U when U
had never been achieved.

**Outcome, with the failure kept: `u_rate` 0.3250 against a pre-registered 0.50 — FAIL.** GPT-2 small
is too weak for this instrument at the strength registered in advance. On the subset where U *was*
achieved: broadcast 0.3438, residue 0.8375, T/U 2.75, controls 0.0000 — consistent with the prediction
and **not a test of it**, because that subset is selected by whether the intervention worked. The
pre-registered correlation of T/U with broadcast is likewise not evidence: `T = 1 + residue × fan-out`,
so its sign was fixed by arithmetic before any state was read.

### 31.28 Broadcast counts references DESTROYED: a sign error, retracted (2026-09-04)

E-000045 identified the workspace paper's **broadcast** count with a reference count and predicted
that the U/T gap widens with it. An adversarial review killed it on the reading of the table, and the
error is mine and elementary.

**A cell "reaches rank 1" when the reader FOLLOWED the swap** — that is, when it did *not* retain the
old referent. So the paper's per-category numbers count references successfully **destroyed**, and
surviving references are the complement:

| category | cells redirected | **surviving references to the old referent** |
|---|---|---|
| countries | 42/48 | **6** |
| number relations | 0/48 | **48** |

The prediction was *"T is expensive when broadcast is high, because every reader is a place a trace can
survive."* Countries have the highest broadcast and the **fewest** surviving traces. The prediction is
inverted by a factor of eight against the numbers cited to motivate it. The paper says as much about
the complement itself: among failures *"the model's top-1 output is typically still the correct answer
for the original argument"* — a failure **is** a surviving reference.

**Two further breaks, each sufficient alone.** `T` counts removals of rows, and a downstream head is
not a row — it is weights shared across every fact, so there is no fact-indexed reader to remove and
`T = k` has no denotation in a representation. And the channel that *generates* `T = k` is
address-bearing: E-000035's disclosure is that a surviving alias **literally stores the deleted key**,
and blanking that key closes it at 1.0000. A head holds no key; it responds to a pattern.

**And the disconfirming case argues the other way.** The referent is *not* gone there — the concept
still appears in all four lens readouts — so it is *deleted-but-open*, the POSIX guarantee **against**
dangling references, not an instance of one.

A model-free brute force in the review settles the general form: **the U/T gap is exactly
`k + m_addr`, the number of KEY-BEARING references, and is invariant to fan-out.** Where no reference
stores an address the gap is **0 at every broadcast count**.

### 31.29 The currency of tracelessness (2026-09-04, E-000046)

E-000041's law carried one caveat, written when it was claimed: that T = k held *for a store which
exports a link's target key and goes on exporting it after the target is gone*. This tests it, over
the same grid, mechanically, three semantics — and its result is the same statement the review reached
by brute force, arrived at independently.

| semantics | T | T = k | T = U | exported view clean | **raw store still discloses** |
|---|---|---|---|---|---|
| exporting | 6.00 | 1.0000 | 0.2000 | 1.0000 | 0.0000 |
| compacting | 6.00 | 1.0000 | 0.2000 | 1.0000 | 0.0000 |
| opaque | 3.50 | 0.2000 | 1.0000 | 1.0000 | **0.8000** |

All five pre-registered criteria PASS, including the one that decides what OPAQUE means. Under OPAQUE
the exported view is clean **by construction**, so an experiment stopping at the fourth column would
have measured its own definition. The fifth column is the experiment: **the raw store still names the
removed key in 0.8000 of cells.**

**So the refined law: `T > U` is a property of KEY-BEARING references, not of pods and not of sharing.
The gap is the number of surviving rows that literally store the removed key.** Canonicalisation does
not impose that cost — it can be paid in three currencies: **deletions** (T = k), **repairs** (T = k
again, the aliases rewritten rather than removed, the store left functional), or **an interface that
declines to show the reference** (T = U). The third is not payment. It is access control, and the
0.8000 says so.

That is narrower than "canonicalisation costs T = k" and more useful, because it names the knob a
system designer actually holds.

### 31.30 BLANK: the law made into an operation (2026-09-04)

§31.29 established that `T > U` is a property of **key-bearing references** — the gap is the number of
surviving rows that literally store the removed key — and that the cost can be paid in three
currencies: deletions, repairs, or an interface that declines to show the reference. The third is not
payment. This turns the second into a primitive.

**`so.audit.certify_traceless`** is the missing rung of the ladder: unreachable, exported-clean,
**raw-clean**, and the store not emptied. `exported_clean` alone cannot be the certificate — it is
true by construction for any store that does not export a target, which would have made it the tenth
instrument here to certify by not testing. `raw_clean` is the one that can fail, and it does: it fails
for eviction.

**`MVCCStore.blank(kid)`** clears a LINK's target. The row stays live and addressable and points at
nothing, so `bank()` exports the row's own key in place of the target's and the dangling pointer is
gone. E-000035 measured that closing the channel *at the key* removes the disclosure at 1.0000; this
is that closure as an operation rather than an analysis. It refuses a FACT cell loudly — an operation
that quietly does nothing on the wrong input is how a certificate goes hollow.

E-000046 re-run with the real primitive in place of the eviction stand-in it used first:

| semantics | T | T = k | exported view clean | **raw store discloses** | rows left live |
|---|---|---|---|---|---|
| exporting (evict) | 6.00 | 1.0000 | 1.0000 | **0.8000** | 40.0 |
| compacting (blank) | 6.00 | 1.0000 | 1.0000 | **0.0000** | **42.5** |
| opaque | 3.50 | 0.2000 | 1.0000 | **0.8000** | 42.5 |

Seven pre-registered criteria, all PASS — and a correction the certificate forced. The first version
of this table read `dangling_targets(bank())` and called it the raw check, but `bank()` **is** the
exported view, so it was the same check twice under two names. `certify_traceless` walks `store.cells`
and asks whether a surviving version still holds the removed key, which is the question.

**Under the real check, only REPAIR pays.** `evict` retains the row's data on purpose — that is what
it is for — so an evicted alias goes on holding the removed key internally: clean in the view, not in
the store, which is the same shape as the opaque case. Blanking clears the key in the version itself
and is the only one of the three that reaches raw tracelessness — **and it keeps the most rows**
(42.5 against 40.0), because every access key still resolves, to UNKNOWN, instead of ceasing to be
addressable. Strictly stronger guarantee *and* strictly less destruction; the two do not trade off.
**[Withdrawn in §31.35. "Raw tracelessness" here is referential cleanliness — no surviving version
holds the removed key — and not history independence, the property §31.31 adopts as the meaning of
traceless. The rows blanking keeps exist only because the fact once did: under the definition they
are the residue, and the third run of E-000046 measures exporting (evicting every row of the pod) as
history independent at the exported level in 1.0000 of cells and compacting in 0.0000 of the cells
that have an alias. `blank` is `ON DELETE SET NULL` done by hand (SQL-92), and the three currencies
are the referential-action menu.]**

**Two semantics the tests pin down rather than assume.** Blanking an EVICTED row is *allowed*, because
`_alive` deliberately admits EVICTED so RESTORE and ROLLBACK can reach it — and it is not a no-op: it
clears the target the row would come back with, which the test checks by restoring. And a test asserted
a refusal that the store does not owe; the assertion was wrong, not the store, and it was corrected to
the documented behaviour rather than the behaviour being changed to fit it.

### 31.31 The U/T distinction is history independence, and has been since 2001 (2026-09-04)

The prior-art check I should have run before claiming came back, and it retracts the framing this
programme has defended longest.

**Naor and Teague, "Anti-persistence: history independent data structures" (STOC 2001), Definition
2.1 — the *weakest* notion in that literature:** *"A data structure implementation is history
independent if any two sequences S1 and S2 that yield the same content induce the same distribution on
the memory representation."*

The mapping is exact, not analogical. Take `S1` = create a fact reachable by k keys, then delete it
(U removals); `S2` = never create it. Both yield the same content — no path yields the object — and
under EXPORTING the memory representations differ, by the k−1 surviving link rows naming the removed
key. That is a violation of the weakest definition in the field.

> **U is the cost of making the CONTENT correct. T is the cost of making the MEMORY REPRESENTATION
> correct as well. `T − U` is the history-independence residue.**

And their framing sentence is the thing I thought I was pointing out: *"if some piece of information
cannot be retrieved via the legitimate interface of a system, then it should not be retrievable even
when there is full access to the system."* That is E-000046's exported-view/raw-store split, in 2001.

**They also isolated the exact knob.** Their §3 is *"Data Structures without Pointers: open
addressing"* and §4 is pointers, introduced with the memory-management problem that pointers force;
their abstract calls the general variable-size record scheme *"the main open problem we leave"*. A LINK
cell exporting a target key **is** a pointer, and `bank()` continuing to export it after the target is
gone is a non-history-independent reference discipline.

**An inversion worth recording, because it clarifies rather than merely corrects.** Hartline, Hong,
Mohr, Pentney and Rocke (ISAAC 2002 / Algorithmica 2005), Theorem 1: *"For a reversible data structure
to be SHI, a canonical representation for each state must be determined during the data structure's
initialization."* In that literature canonicalisation is the **cure** demanded by strong history
independence; in this ledger canonicalisation was written up as the **cause** of the T = k disclosure.
Both are right and they are about different objects: SHI canonicalises the *representation given the
content*, whereas this store canonicalises the *fact* and then adds address-bearing rows — which is
precisely the representation-level non-canonicality SHI forbids.

**And the field has priced the opaque option.** Buchbinder and Petrank (CRYPTO 2003) give an SHI/WHI
separation with an exponential gap and matching bounds; Blelloch and Golovin (FOCS 2007) give SHI
hashing at O(1) expected insert and delete, so in that setting opacity is asymptotically free;
Golovin's B-treap and the B-skip-list carry it to B-tree-shaped stores, and the line is active
(ACM TODS 2025). Ficklebase (Bajaj and Sion, ICDE 2013) owns the deletion-residue form outright: once
a tuple is expired *"any and all its side-effects are removed, thereby eliminating all its traces,
rendering it unrecoverable, and also guaranteeing that the deletion itself is undetectable."*

**So what is left, stated small.** The U/T *distinction* is not a contribution — it is weak history
independence, measured. What this repository has is an implementation and a measurement in a setting
where the concept had not been applied: **a store that a frozen language model reads**, where the
tracelessness certificate composes with a model-side deletion proof (§31.14, §31.16) that no
history-independence result addresses, because none of them has a reader whose behaviour has to be
certified too. `blank` and `certify_traceless` are that implementation. Whether `ON DELETE SET NULL`
already covers `blank` at the relational level is not yet checked and should be assumed until it is.
**[Checked in §31.35: it does, verbatim, and the composition this paragraph kept as the residue is
Garg, Goldwasser and Vasudevan (Eurocrypt 2020, Thm 3.4) — a history-independent store composed with
a learned reader whose deletion is certified — generalised by Godin and Vasudevan (2022) and folded
into one definition by Cohen, Smith, Swanberg and Vasudevan (CCS 2023). And the state
`certify_traceless` certified was not weakly history independent. Nothing in this paragraph
survives as a contribution; what survives is an implementation of their auditor form.]**

This is the seventh retraction of the session and the one that costs the most, because U/T was the
claim that had survived every previous review. It survived them because none of them was this check.

### 31.32 The pod objective moves the statistic and misses the bar, on a substrate that never had the failure (2026-09-04, E-000044)

E-000044 is the constructive half E-000043 asked for: train the pod objective (`so/pod.py` — every
access path of one fact pulled onto one carrier, carriers of different facts pushed apart, hinged at
the larger of the Welch bound and the centring floor) against an otherwise identical baseline, and
ask whether the excess overlap against the design-matched permutation null comes down, and at what
price in accuracy. Two arms, three seeds, 700 steps, `pod_weight = private_weight = 1.0`, 24 pod facts
per step. Both arms reach accuracy 1.0000; the price is zero.

| measure | arm A (baseline) | arm B (pod objective) | drop, mean | drop, worst seed |
|---|---|---|---|---|
| excess overlap, full | 0.2399 | 0.1998 | +0.0401 | +0.0361 |
| excess overlap, addressing | 0.2484 | 0.2172 | +0.0312 | +0.0243 |
| excess overlap, content | 0.0510 | 0.0325 | +0.0185 | — |
| bystander accuracy under deletion | 0.9983 | 1.0000 | +0.0017 | +0.0000 |
| closure size | 1.00 | 1.00 | 0.00 | 0.00 |

Per seed the full-overlap drop is +0.0371 / +0.0361 / +0.0472: the same sign every time, and one
third of the pre-registered bar of 0.10. **Both claim criteria FAIL as registered**: `excess_full_drop`
0.0361 against 0.10, `collateral_gain` 0.0000 against 0.05.

**Two things the numbers say, kept apart.** First, the objective does what it was written to do, at
zero cost — it reduces the overlap between fact subspaces, in content and in addressing, on every
seed, without touching accuracy — and it does so by an amount that the rule fixed before the run does
not reward. The rule has three branches (a drop of 0.10 at no price; a drop only at a price; no drop)
and the observation falls between the first and the third. That is a gap in the rule and it is
recorded as one; the favourable reading is not taken.

Second, and this is the finding that matters: **the collateral criterion could not have passed on
this substrate, because the failure it measures is not there.** Arm A already deletes at closure
1.00 with bystanders at 0.9983 — the synthetic model trained from scratch allocates its facts cleanly
without being asked to. The allocation failure E-000043 measured (bystanders to 0.3897, −0.13 overlap
against the permutation null with 92% of the dimension budget unused) is a property of the **frozen
GPT-2 with an adapter**, where the facts land in a representation nobody trained for them. E-000044
therefore trained the remedy on a patient without the disease. The 0.04 it moved is the only signal
the design could produce, and it is consistent with the objective working; it is not evidence that
the failure is trainable away, because the failure was absent.

**What it licenses.** Nothing beyond: the pod objective is cheap, sign-consistent and harmless on a
clean substrate. The constructive half of E-000043 remains untested, and the test it needs is now
specified by this failure: the same two arms on the E-000020 symlink checkpoints — the frozen GPT-2
adapter, where arm A exhibits the collateral loss — with the criteria unchanged. That experiment is
E-000047 and has not been run.

### 31.33 The zero-evaluation forecast: a literal, a misattributed reader, and star arithmetic (2026-09-04, E-000032 re-run)

The claim under test, stated so it could be killed: *a store-side statistic computed without the
model forecasts what the neural reader answers after a deletion, and the fact-level certificate is
discharged with zero model evaluations.* A fourteen-agent sweep — prior art first, then three
independent refuters per limb, then a landing — returned "surviving: none". Every code citation in
its verdict was checked by hand against this repository before this entry was written, and every one
holds. This is the eighth retraction of the session and it reaches the result that survived the
previous seven reviews.

**Defect 1 — the formula is not a function of the store.** `(closure − 1) / keys_per_group`
reproduces 0 / ⅓ / ⅔ on the three star arms and fails on a chain by a full grid step, with the
mechanical resolver and no model anywhere:

```
star_link    closure=1  predicted 0.0000  measured 0.0000  err 0.0000
star_mixed   closure=2  predicted 0.3333  measured 0.3333  err 0.0000
star_copy    closure=3  predicted 0.6667  measured 0.6667  err 0.0000
chain        closure=2  predicted 0.3333  measured 0.6667  err 0.3333
```

A chain is an alias that LINKs to a *copy* rather than to the object; evict the object and both the
copy and the alias still answer. The formula holds only under an invariant `load_arm` and
`load_mixed` impose by construction — every non-target closure member backs exactly one key — and
`MVCCStore.link` does not require it; chains are the subject of E-000016 and have a training-time
knob (`bank_with_links(..., p_chain)`, left at 0.0). The quantity that IS a function of the store is
the post-deletion resolver count, which `certify_fact` already computes through `store_after`; put
that in the formula's place and the "prediction" becomes, explicitly, *the adapter agrees with the
mechanical resolver* — the adapter's job description, recorded at 1.0000 in E-000015 on the same
checkpoints. §31.23 had already said the number is not a forecast; what it kept — "given a faithful
reader the number cannot come out otherwise" — is exact on stars and false off them. The test is on
record: `test_closure_minus_one_over_keys_is_star_arithmetic_and_not_a_store_law`.

**Defect 2 — the reader is not GPT-2.** `e000032_deletion_closure.py:176` loads
`train_or_load(seed, steps, n_deref=1)`, the `e000015_deref1_seed{0,1,2}.pt` checkpoints of a
`MutableKnowledgeTransformer` trained from scratch on a 256-entity world resampled at every training
step, with an explicit UNKNOWN head trained on broken queries. No GPT-2 forward pass occurs in
E-000032; `gpt` appears in that file twice, both times in prose about E-000025. §31.23's defence of
the 0.0000 — a pretrained prior over capitals *could* have gone on answering and did not — is
therefore void: the model that ran has nothing to answer from. On the frozen GPT-2 adapter the
corresponding reads are a recorded FAIL at template 0 (E-000020: direct 0.5700, alias 0.5067, dup
0.5483), so the same test there would show prediction errors several times the 0.05 bar; it was not
run there. Power, for the record: a reader guessing uniformly over 256 entities reproduces the
canonical arm's exact 0.0000 over its 225 destroyed-route keys with probability (255/256)^225 ≈ 0.41.
An exact zero is what a design with no residual variance looks like, not a calibrated forecast.

**Defect 3 — the accounting.** `model_evaluations_per_deletion = 0.0` at `e000032:298` was a
literal, not a count. `certify_encoding` over an empty row set computes its reference fingerprint
through `encode_bank` — one standalone encode per certification, which the code's own docstring
called vacuous while the table called it zero. The reachability control runs once per fact inside the
loop (3.71–4.50 s per fact in the recorded run) and was timed apart from the 1.45–1.80 s "per
certified deletion" and described as once per instrument. `check_mediation` — the falsification test
of the interface certificate's premise — was never called on the configuration in use
(`use_links=True, n_deref=1`); E-000030 ran it on the bare configuration and E-000032 inherited the
result. And `certify_fact` discharged anti-vacuity as `swept or structural_ok or absent_ok or
store_ok`, so a certificate could validate while its own `AbsenceCheck` read VOID; it did not happen
in the recorded run (`one_record_payload_absent` 1.0000 in all arms) but the instrument allowed it.

**What changed in code, all of it tested.** A `ModelCalls` wrapper over `forward` and `encode_bank`
counts what the model is asked to do inside the certification window; the per-fact cost is reported
with the control inside it; `check_mediation` on this configuration is a pre-registered control in
all three arms (a VOID voids every certificate below it); `certify_fact` treats every supplied check
as a conjunct, with two tests; the chain counterexample is a test; the report names the reader. The
re-run's counted table is in `so/results/e000032_deletion_closure.md`. Three seeds, 25 groups, all
29 registered criteria PASS including the three new mediation controls, and the counts are what the
code predicts: **0 forward passes and 1 standalone `encode_bank` per certification, 1 forward per
fact for the reachability control**, in every arm and every seed; the mediation check moved both the
encoding and the outputs on every arm, so it was a real test and not a vacuous pass. Per fact, all
in: 0.58 s canonical, 0.51 s mixed, 0.51 s duplicated on an idle machine (the recorded 1.45–1.80 s
was measured while E-000044 trained beside it). The window is not model-free; the store-side parts
of it are.

**What the sweep reports as owned by others** (recorded as reported; the citations were read by the
sweep's agents, not re-read for this entry). The closure object is query *resilience* — the minimum
contingency set — Freire, Gatterbauer, Immerman and Meliou (PVLDB 2015), upstream Buneman, Khanna and
Tan (PODS 2002); the pairwise-disjoint-witness lower bound is the standard packing bound there, and
Makhija and Gatterbauer (SIGMOD 2024) give an ILP whose LP relaxation is provably tight for
self-join-free conjunctive queries, strictly stronger than a greedy-plus-disjoint-family. The
architectural premise — "data removed from the datastore is guaranteed not to contribute to any model
predictions" — is SILO (Min et al., ICLR 2024), on right-to-erasure grounds, and "forgetting reduces to
deleting entries" is LMLM (Zhao et al., NeurIPS 2025). Hold the model fixed, vary the store, delete an
alias closure, use topology arms: Raeesi and Roed (arXiv:2607.00605, July 2026), 12,228 deletions,
Base/Alias/Collision/Noise, residual 0.7–13.6%, parametric leakage 0.11% — and their §9 names
canonicalisation at write time, aliases as pointers into one canonical record, as untested future
work. **This repository must not claim the store design.** Store-derived answerability checked
against a neural reader is GrailQAbility (Patidar et al., ACL 2023), where it fails. Zero computation
at deletion time is Sekhari et al. (NeurIPS 2021, Lemma 1) and SISA; "no statistic of a trained
network certifies deletion even in principle" is Thudi et al. (USENIX Security 2022) — the citation
that makes the case *a certificate versus none*, not *1.8 s versus 129 s*. Deletion-compliance
certificates are Garg, Goldwasser and Vasudevan (Eurocrypt 2020). The word "closure" is taken twice
in this literature (alias-closure; dependency-closure), both meaning syntactic expansion sets.

**What is left, stated small.** (1) `certify_store_absence` is a decision procedure for SILO's
asserted premise, and its value is that it returns NO on a field: `one_record_address_store_absent`
0.0000 in the canonical and mixed arms, because `bank()` builds a surviving alias row's address from
the removed cell. That is a counterexample to the blanket phrasing, on a store. (2) Per-key closure
1.00 in every arm while per-fact closure separates 1 / 2 / 3: three stores with identical interfaces
that a record-level certificate cannot tell apart. (3) The negative half (§31.24–31.27), which has
real variance. "Zero model evaluations" is not an axis this repository can claim, and the E-000032
result is an instrument control on reader fidelity — worth reporting as one, with its power.

### 31.34 The capacity slogan is assembled from print, and its s is the number of phrasings (2026-09-04)

§31.26 retracted the theorem `n ≤ d/s` on a counterexample and showed that `pressure` and
`headroom` are identities. A second sweep, fifteen agents on the slogan *superposition buys
representation capacity and not deletion capacity* ("surviving: none"), adds what §31.26 did not
have: where the parts come from, and why the measured `s` was never a property of GPT-2.

**Owned, as the sweep reports it** (citations read by the sweep's agents, not re-read here). The
linear budget is Elhage et al. (Toy Models, 2022) — per-feature dimensionalities sum to the embedding
dimension when packed efficiently — made a theorem by Scherlis et al. (arXiv:2210.01892): capacity
`C_i ≤ 1`, `Σ C_i ≤ d`, `C_i = 1` exactly when the feature is orthogonal to every other; that is the
theorem at `s = 1`, in a paper with no mention of deletion. The deletion framing is Yang et al.,
"Knowledge in Superposition" (AAAI-25, arXiv:2408.07413): without superposition the interference term
vanishes and editing is lossless, and superposition is the reason lifelong editing fails — with no
dimension count. Both halves together are Guo, "The Deterministic Horizon" (arXiv:2605.23024, §3.5,
Theorem 3.14), whose proof carries the boundary *an exact solution exists iff n − 1 ≤ d − 1* and whose
scope note reads "superposition is not merely a description of what LLMs do but a constraint on what
post-hoc editing can do" — weight-space, unrefereed, headline `K* ~ √d`, no `s`. The template
"superposition buys representation and not X" is Adler and Shavit (arXiv:2409.15318, ICLR 2026) for
computation. The term *deletion capacity* is Sekhari et al. (NeurIPS 2021). The subspace-exhaustion
argument is GPM (Saha et al., ICLR 2021) and its successors. Unclaimed: `n ≤ d/s` with a measured
`s`, the residual-stream projection setting, and the observation that the bound is slack by two orders
of magnitude while deletion fails anyway. The third is the only new thing, and the point below is
that it is not a finding either.

**Two structural points from the refuters, verified against the code where they touch it.**

*(a) Two rulers.* Representation capacity was counted at tolerance `ε > 0` — almost-orthogonal
families — and deletion capacity at tolerance `0`, by requiring the projection to fix every other
fact's subspace pointwise. Fixing `V_j` is sufficient for zero collateral; the derivation used it as
necessary, and the behavioural predicate ("every other fact still answers") is an argmax condition
with a margin. At equal tolerance the gap does not exist: at `τ = 0` representation capacity is `d`
too; at `τ > 0` the almost-orthogonal family itself gives `exp(cτ²d)` cleanly deletable facts. The
"price of superposition paid at deletion time" was introduced by the derivation. And a rank-one shear
`I + u wᵀ` silences a fact while leaving every bystander exactly fixed, with `n ≤ d` and no `s` — an
invertible map, destroying nothing, that satisfies the behavioural definition of UNREACHABLE. A
predicate an invertible map satisfies cannot ground a subspace count.

*(b) `s` is the number of phrasings.* `e000040_dangling_readers.py:186–198`: `fact_basis` returns the
mean of `res_self − res_others` plus the principal components of its centred spread — for eight
phrasings, exactly eight vectors. The search for a fact's closure never leaves that slice of `R^768`,
so `s ≤ 8` by protocol, and "`n ≤ d/s = 230`" reads "`n ≤ 768 / (how many paraphrases were
typed)`". Three further conditionings in the same run: `n_no_closure` 8.33 of 17 (no closure exists in
the fact's own basis — `s = ∞` under the protocol); closure and collateral are averaged over `spec`
only (lines 366–392), the facts that survived a 0.60 gate (lines 278 and 345) which admits as "the
fact's own" any direction whose solo removal costs bystanders up to 40%; and the readout is
`restricted()` (line 158), an argmax over the 17 candidate capitals — the coarsest predicate and the
largest margin. E-000040's numbers are statements about a search procedure over an eight-vector basis
with survivor-conditioned averaging, not about GPT-2's allocation. §31.25's "allocation, not
capacity" and the capacity reading it replaced both presuppose that the bound is the operative
constraint. The operative constraint was the search.

**The experiment that would settle the frontier's shape** is specified by the sweep's landing and
recorded here as **E-000048, not run**: three arms with one architecture (facts in weights; facts in
cells; the cell architecture with routing unsupervised, as the control that separates substrate from
architecture), nested worlds so the same sixteen probe facts appear at every `n`, four search classes
(basis prefix as E-000040 did it; minimal within the basis; free collateral-aware rank-`k` in `R^d`;
a LEACE-style oblique eraser at matched rank) with the headline taken from whichever gives the
highest clean rate, a per-bystander damage rate on a fixed window as the primary quantity, training
to criterion rather than to a step count, and `pressure`/`headroom` banned from every criterion. The
mini-transformer applies `hidden_edit` immediately before `readout`, so the whole search is
closed-form on cached states there. About seven hours on this machine.

### 31.35 The residue is the rows BLANK keeps: referential cleanliness is not history independence, and the composition is 2020 (2026-09-04, E-000046 third run)

A five-agent sweep asked for one unclaimed contribution in the symlink / J-space / pod programme,
prior art first, "none" allowed. Four angles returned none, zero candidates were proposed, and the
sweep's landing did what none of the previous eight reviews had done: it ran the definition this
ledger adopted in §31.31 against the state this ledger's own certificate certifies. I re-ran its
script and it reproduces line for line.

**The check.** S1 = write a fact, link two aliases to it, evict the fact, blank both aliases — the
"repair" currency §31.30 called the only one that pays. S2 = never wrote the fact. Naor and Teague's
Definition 2.1, weak form: same content ⇒ same memory representation.

```
certify_traceless:  TRACELESS, CERTIFIED in 3 operation(s) ... 3 of 4 rows were kept
same CONTENT (legitimate interface):  True   {(1,1): 7, (2,3): None, (4,3): None, (5,3): None}
bank rows S1 vs S2:                   3 vs 1          (2 live LINK rows that exist only because the fact did)
state_hash equal:                     False
evicted cell still in store.cells:    True   status EVICTED, tombstone_key (2, 3), payload 42 in its version
op log S1 vs S2:                      7 vs 1 entries  (the write, both links naming the target, the evict)
next kid S1 vs S2:                    5 vs 2
next marker identical:                False           (the generator's position encodes the number of prior writes)
```

Same content, different representation on every axis the definition names. **The state the
certificate called traceless is not weakly history independent, and the rows `blank` keeps are the
residue.** `raw_clean` — the check §31.30 called "the one with teeth" — is a scan for surviving
versions that still hold the removed key: referential integrity, `ON DELETE SET NULL` verified after
the fact. Necessary for weak history independence; not sufficient; and not what "traceless" was
defined to mean two sections later. §31.30's "strictly stronger guarantee *and* strictly less
destruction" is inverted: less destruction is the residue.

**The instrument, rebuilt so it can fail for the right reason.** `so.audit.check_history_independence`
builds, from the store as it stands, a fresh store with the same construction parameters holding only
what the interface can still answer — active FACT rows and LINK rows whose target is live — and
compares. EXPORTED level: `bank()` content arrays row for row (rows that exist only because something
once did are `residue_rows`); markers reported separately because the seeded generator's position
encodes history. RAW level: `store.cells`, the operation log, the next id — which an MVCC store keeps
on purpose, so `raw_hi` is false for any store that ever removed anything and the field exists so no
certificate can be read as saying otherwise. `TracelessCertificate` now says REFERENTIALLY CLEAN
where it said TRACELESS, carries the history check, and its summary names both properties in one
sentence so neither can be read as the other. The control that the comparison can pass: a store that
never removed anything is history independent at both levels. Four tests, including the S1/S2
scenario verbatim.

**E-000046, third run**, with the columns the first two did not have and predictions fixed before it:

| semantics | referentially clean (raw) | rows left live | **history independent, exported** | residue rows | history independent, raw |
|---|---|---|---|---|---|
| exporting (evict every row of the pod) | 0.2000 | 40.0 | **1.0000** | 0.0 | 0.0000 |
| compacting (blank the aliases) | 1.0000 | 42.5 | 0.2000 | 2.5 | 0.0000 |
| opaque | 0.2000 | 42.5 | 0.2000 | 2.5 | 0.0000 |

The registered rows `compacting/exported_hi ≤ 0` and `opaque/exported_hi ≤ 0` **FAIL as registered**
at 0.2000, and the 0.2 is exactly the cells with `n_links = 0`: a pod made of copies has no alias to
blank or to leave dangling, so evicting its closure leaves nothing behind under every semantics. The
criterion should have conditioned on `n_links ≥ 1`; it did not, and it is not rewritten. Over the 48
cells per semantics that have an alias — the cells the prediction was about, reported beside the
registered row and labelled post hoc — exporting is history independent in 1.0000 with 0.0 residue
rows and compacting and opaque in 0.0000 with 3.1 residue rows each. `raw_hi` is 0.0000 everywhere,
as predicted: the log alone distinguishes the stores. So the two properties come apart cleanly, and
in opposite directions: **repair buys referential cleanliness and keeps rows; deletion buys history
independence at the exported level and costs them. Neither reaches the raw level in an MVCC store,
by design.** The instrument that closed here is the tenth of this programme to certify by not
testing, and this one was certifying under a name it had defined itself.

**The prior art, as the sweep reports it** (read by its agents at source; not re-read for this
entry). `blank` is SQL-92 `ON DELETE SET NULL` performed by hand, strictly weaker than the declared
action because it can be forgotten — which is why a certificate has to look for it. The three
currencies are the referential-action menu: `CASCADE`, `SET NULL`, a view over `NO ACTION`. The
guarantee §31.30 intended is Ficklebase (Bajaj and Sion, ICDE 2013) — all side effects removed,
deletion undetectable — and the channels `raw_clean` does not walk (log, tombstones, kept versions,
allocator state) are Stahlberg, Miklau and Levine (SIGMOD 2007). Cell-NULLing as the erasure
primitive with a minimum-cost dependency search is Chakraborty et al. (VLDB 2025, arXiv:2507.00343),
NP-hard, covering inference dependencies `blank` does not.

And the residue §31.31 kept — *a tracelessness certificate composed with a model-side deletion proof
that no history-independence result addresses, because none has a reader whose behaviour must be
certified* — is false as stated. Garg, Goldwasser and Vasudevan (Eurocrypt 2020, arXiv:2002.10635)
define deletion-compliance over exactly that pair, the collector's memory state and the environment's
view of every future answer; Theorem 3.1 derives a compliant collector from a history-independent
dictionary by the two-clause argument this repository makes; §3.3, Definition 3.4 and Theorem 3.4
compose a history-independent store with a *learned model whose deletion operation is certified*.
That is a reader whose behaviour has to be certified, composed with a history-independent store, in
2020. Godin and Vasudevan (eprint 2022/033) generalise it — a collector built exclusively from
history-independent structures is weakly deletion-compliant — and propose the certificate form:
publish the collector with a simulator and let an auditor run experiments. Cohen, Smith, Swanberg and
Vasudevan (CCS 2023, arXiv:2210.07876) fold the reader in outright: a model is history independent
iff it satisfies machine unlearning. The reader-side sweep itself is bounded-exhaustive
noninterference (Goguen and Meseguer 1982; self-composition, Barthe et al. 2004), and the sweep
reports Ramesh (arXiv:2607.27539, July 2026) — exact deletion from a frozen LM's persistent memory,
zero float32 residual on logits and eighty audited intermediate arrays — against the claim document's
"E-000030 is the first such certificate in this line"; and Tavakoli and Sanderson (SIGIR 2026) for
"revocation happens at the level of records while violations surface as facts". On the J-space side:
one VJP per token is attribution patching; E-000042's 0 of 6 is the workspace paper's own reported
result plus self-repair at small scale; E-000043's same-relation collateral is what linear relation
decoding (Hernandez et al., ICLR 2024) predicts; fan-in and fan-out are separate criteria in Cohen,
Eshel, Geva and Globerson (TACL 2024).

**What is left, ninth statement, and the smallest.** The programme has (1) an executable auditor of
the Godin–Vasudevan form for one collector — a store plus a frozen reader — whose parts are each
owned and whose value is that its checks return NO: `certify_store_absence` on the address field,
`check_history_independence` on the blanked rows, `certify_traceless` on eviction; (2) ten instruments
that certified by not testing, each caught by a control that could fail and did; (3) three negatives
with controls on GPT-2 small, all of which the literature predicts; (4) the record. The symlink is a
store design pattern that Raeesi and Roed name as future work and SQL-92 implements; the J-space and
pod experiments are a documented null at 124M parameters. There is no ninth claim to put in this
paragraph, and the sweep that was asked to find one, with "none" allowed, said none.

### 31.36 The target, clause by clause: dynamic knowledge through pods that behaves like the model's own (2026-09-04)

The target, as stated at the end of the session: *knowledge added at runtime through pods, behaving
like the model's own knowledge, with its own delete / modify / version through the pod and the
marker, easy to add like a container, and made the same as internal knowledge by the symlinks.* A
five-agent prior-art sweep on exactly that bundle — runtime knowledge pods with lifecycle, overlaying
the parametric base — returned four verdicts of the same shape: **every property is owned by a
published system, the overlay semantics are 1995 union mounts, and the composition is engineering,
not novelty.** What follows is the target against this repository's own records, worst seed, every
number re-read from the result files for this entry.

| clause of the target | delivered? | the record |
|---|---|---|
| add like a container | **yes at synthetic scale; in GPT-2 at a strong template only** | E-000014: 10,000 cells, direct 1.0000, provenance 0.9998. E-000026 (GPT-2): direct 0.5633 and alias 0.5000 at template 0 — FAIL against 0.85 / 0.80 — and 0.9933 / 0.8600 at the strong template — PASS. §31.21: the gap is token position, removable by a prefix without training. |
| behaves like the model's own knowledge | **no, by the editing field's own metrics** | E-000013: override 1.0000 on the trained template, `override_heldout_min` **0.0000** on held-out phrasings; natural held-out reading 0.66 with 88.6% of the gap in addressing (E-000039-A). Injection on generic text: `generic/kl_to_base` 2.27 nats (E-000013) and 3.27, worst 3.65 (E-000017-B), against a 0.05 bar — a locality failure. ROME reports ~96% paraphrase success; RippleEdits alias 86.8–100%. |
| composes | **yes** | E-000016: two dereference slots resolve two-link chains at 1.0000, all five claim groups "yes"; GPT-2 hop2 0.9350 (E-000013). |
| own delete, modify, version | **operations yes; certified for REVOKE (both gates) and SHRED (hard gate) in GPT-2; holed elsewhere** | Thirteen operations in `so/mvcc.py`. E-000030: interface-level certificate, invariant for every query. But SHRED in the synthetic model gives the object up at **1.0000** through the ungated reverse key (E-000028), and the gate's operational radius is 0.90 against the declared 0.35 (E-000029). |
| delete reverts to the prior (the overlay) | **on the trained phrasing only, and by construction** | E-000013: `revoke/kl_to_base` 0.0004 (worst 0.0005) PASS; `revoke/heldout_kl_max` 3.70, worst **4.47**, against 0.1 FAIL; `revoke/top1_matches_base_pooled` 0.7617, worst 0.7300, against 0.95 FAIL; `fallback_after_revoke_by_construction` **no**. |
| traceless | **no — referentially clean, not history independent** | E-000046 third run: BLANK raw disclosure 0.0000, exported-level history independence **0.0000** on cells with an alias (3.1 residue rows), raw-level 0.0000 everywhere (§31.35). |
| the symlinks make external the same as internal | **half** | LINK cells share one object across alias *keys*: E-000015 update reach 1.0000 against 0.0000 in the copy arm, object recoverable by probe after one SHRED 0.7% against 87.3%; in GPT-2, update reach 0.8850 worst seed against the 0.90 bar (E-000026); rollback through an alias 0.5000 FAIL. *Phrasings* are not symlinked at all — they are learned addressing (§31.7, §31.21), and that is where the held-out failures live. |

Read across: the row has the conjunction as *operations*, and fails its own bars on the three
properties the target names as the point — behaves like own knowledge, reverts to the prior off the
trained phrasing, traceless.

**Owned, as the sweep reports it** (read at source by its agents). The overlay: union mounts
(Pendry and McKusick, 1995) — upper shadows lower with the base frozen, REVOKE is 4.4BSD's `rm -W`
that lets the lower file reappear, BLANK is a whiteout that keeps naming the key, which is the
`T > U` disclosure E-000046 measured. Runtime knowledge entering the computation over a frozen base:
SERAC (ICML 2022), GRACE (NeurIPS 2023), WISE / MEMOIR / MELO / T-Patcher, SoLA (2026, which names
"removal reverts to the base" as a primitive), Larimar (ICML 2024), KBLaM (ICLR 2025, > 10K triples,
add / remove one knowledge token), FILM and Entities-as-Experts (2020–21: one canonical object,
many mentions, stale objects replaced at inference without training). Delete reverting to the
parametric prior, *measured*: SILO (perplexity approaches the parametric-only model after removal),
LMLM with Raeesi and Roed's DEL-OFF (parametric leakage 0.11% over 12,228 alias-closure deletions).
Certified per-deletion equality with the never-ingested state: Subtract or Replay (Ramesh, July 2026,
median KL 5.4e-15 at 1B, bitwise on a 48B recurrent model) and Forgetful Attention (August 2026).
Versioned lifecycle with rollback: MemOS (MemCube, version chain), ChronoMem, OneEdit. Tracelessness
for AI memory by design: MemTrust's oblivious decay (January 2026). Dependency closure with reference
counts over agent memory: SBU. Multi-phrasing-to-one-entry: GRACE's expand rule, MELO's clusters,
MEMOIR's mask retrieval. On the "own knowledge" half the literature is ahead of this repository, not
behind it.

**The one thing the sweep could not find, at its exact size.** An *explicit pointer row* — alias as
address, rather than as copy, as learned radius, or as separate triple — inside a memory whose payload
enters the computation, run against a duplication arm of the same world under the same trained
reader, with the full battery (UPDATE, SHRED, REVOKE one alias, RELINK, dangling pointer after
DELETE) taken through every alias, and the reader's price for the indirection measured (0.0954 for
sharing, 0.0688 for link training, E-000025). Raeesi and Roed propose the row and call it directly
testable; nobody publishes the paired arm. That is a measurement contribution, not a mechanism, and
it is the same scoping §31.17 reached before any of this session's reviews. It is not an eighth or a
ninth or a tenth claim, and the four sweeps that were asked to find one, with "none" allowed, said
none.

**What the target needs next, if it is pursued as engineering.** The failures above are specific:
the prefix fix for token position (§31.21, no training); a paraphrase-generalisation measurement on
the standard benchmarks against ROME / GRACE / SoLA rather than against this repository's own bars;
E-000047 for the pod objective where the failure exists; and the key-channel sweep of E-000028 on the
symlink arms, which has still never been run. None of these is a novelty; each is a number the target
does not yet have.

### 31.37 Tying the address across phrasings does not make the invariance intrinsic (2026-09-05, E-000039-B)

E-000039 was written before this session's reviews and left half run: its decide phase (§31.24 —
88.6% of the held-out paraphrase gap is addressing, and a neutral prefix lifts held-out addressing to
0.98 and reading to 0.97 with no weight changed) and not its train phase. The train phase is the
**symlink on phrasings**: an InfoNCE tie on the routing query `q` — the only phrasing-dependent tensor
in the read path — between a subject-initial and a subject-medial rendering of the same question,
added to E-000017-B's trainer at the same budget. The control is E-000017-B itself, tie weight 0. The
prediction, fixed before the run and specific: the held-out subject-initial forms recover, because the
tie spans the axis they differ on. The bar is the prefixed ceiling reached *without* the prefix: 0.95
on held-out reading and addressing.

Three seeds, 3000 steps, 2841–3910 s each. The tie was learned — tie loss 0.82 → 0.41, 0.70 → 0.60,
0.92 → 0.41 over the run, minima 0.18–0.27 — and the trained phrasings read as before
(`train/active_correct` 0.9131 against the control's 0.9119).

| measure, worst seed | control (E-000017-B) | address tie (E-000039-B) | bar |
|---|---|---|---|
| held-out reading | 0.7288 | 0.7488 | ≥ 0.95 |
| held-out addressing, `route_hit_min` | 0.5400 (E-000039-A, template 11) | 0.5200 | ≥ 0.95 |
| addressing share of the held-out gap | 0.8818 | 0.8000 | — |
| deletion reaches the worst held-out phrasing, SHRED / REVOKE | 0.8650 / 0.8650 | 0.8650 / 0.8650 | ≥ 0.95 |
| generic-text KL to the base model | 3.6474 | 3.1971 | ≤ 3.65 |
| broken-key UNKNOWN | 0.6300 | 0.6000 | ≥ 0.63 |
| prefixed ceiling, no weight changed | 0.9700 read / 0.9800 route | — | — |

**All five claim groups unsupported.** Reading moved by +0.02, addressing by −0.02, deletion
propagation by exactly 0.0000. The tie changed the routing query's geometry on the pairs it was
trained on and left the held-out phrasings where they were: an invariance trained on eight
renderings does not reach a ninth whose difference is where the subject token sits, even when the
training pairs were built to span that axis. Two instrument notes, recorded rather than smoothed:
(i) two criteria (`query_cos_between_fact/read1`, `address_collision`) came back "−", because
`decompose()` computes them and `main()` never copied them into the record — FAIL by absence, as
registered; the record was fixed and an evaluation-only re-run from the saved checkpoints supplies
them, with the per-template held-out addressing the prediction was about, in the addendum below;
(ii) the 0.63 bar on broken-key UNKNOWN was set at the control's own value, so 0.6000 is a 0.03
regression and not a collapse.

**What it means for the target.** "Behaves like the model's own knowledge" fails on held-out
phrasings (§31.36), the failure is addressing (§31.24), and this run shows the addressing cannot be
symlinked *by training on the trained phrasings*. The pre-registered recommendation is now the
finding: normalise the prompt in the read path — the prefix, which reaches 0.98 with no training —
and carry that scope on the certificate; do not train. Prior art: paraphrase-consistency and
query-invariance training are standard (GRACE's expand rule, contrastive dense retrieval), so the
mechanism is owned; the negative is specific to a frozen LM's addressable memory whose failure is
positional, and its control is a ceiling that the same weights demonstrably reach. A clean negative
that could have come out the other way, and no claim.

**Addendum, the evaluation-only re-run (same checkpoints, nothing retrained).** The two criteria
recorded as "−" now carry values, and the per-template held-out addressing the prediction was about is
on record beside the control's:

| measure, worst seed | control (E-000039-A on E-000017-B) | address tie (E-000039-B) | bar |
|---|---|---|---|
| routing-query cosine between different facts, read layer 10 | 0.2720 | 0.1683 | ≤ 0.33 |
| address collision on held-out forms (share of targets sharing a cell) | 0.1000 | 0.1125 | ≤ 0.02 |
| held-out route_hit, template 8 (subject-initial) | 0.66 | 0.67 | — |
| held-out route_hit, template 11 (subject-initial) | 0.54 | 0.52 | — |
| held-out route_hit, template 9 (subject-medial) | 0.94 | 0.97 | — |
| held-out route_hit, template 10 (subject-medial) | 1.00 | 1.00 | — |
| prefixed ceiling, read / route_hit, no weight changed | 0.97 / 0.98 | 0.97 / 0.97 | — |

The tie did what it was asked at the level it was asked: the between-fact cosine of the routing query
fell and the collapse criterion passes. It did not move a single held-out template: the two
subject-initial forms sit at the control's numbers to within 0.03 and the two subject-medial forms
were already at the ceiling. The address-collision row FAILS on both the control and the tie — it was
registered against a bar the untouched model does not meet either, which makes it a property of the
evaluation (a hundred targets over a thousand cells at these held-out forms) and not of the tie. The
prefix still reaches 0.97 on the same weights. E-000050 (§31.38) says why: the failing templates are
exactly those that put the subject at token position 0, and GPT-2's tokenizer prepends no BOS.

### 31.39 The J-space pod is by construction in this adapter, and the one measurement that survived is predicted by §31.38 (2026-09-05)

A fifteen-agent workflow was asked for the training-free measurements at the symlink / J-space / pod
seam — whether knowledge injected through the store lands on its object's J-lens atom and has a
J-space closure of one; what fraction of the adapter lives in the model's verbalizable basis; whether
the pod's lifecycle is visible in J space; and whether the held-out addressing gap is linearly
erasable from the routing query without training. Each design stated its result sentence first; two
refuters attacked each sentence; one survived to implementation. The verdict closes the J-space angle
for this implementation, and it closes it by reading the code rather than by prior art.

**Three of four are tautologies of `so/llm_adapter.py`, not measurements.** `encode_bank` builds a
row's value from `W_U[obj]` alone (`payload_from='output'`), through `v_proj` and `o_proj` initialised
to the identity; nothing about the (subject, relation) key enters the value. So (i) every access path
of an object — target key, alias keys, every phrasing, other pods with the same object, and the
duplication arm's copies — injects one vector up to routing weight, and "one atom silences every
access path" is what the code does before any measurement; (ii) the write is *keyed by object token*
by construction, so "the J-space closure is scoped by object rather than by pod" is the definition of a
J-lens atom meeting the definition of the value; (iii) every adapter parameter is trained by
cross-entropy on the answer logit at the last token, whose gradient with respect to the injected read
is the per-prompt Jacobian applied to `W_U[u]` — the quantity whose corpus expectation *defines* the
J-lens vector — so "the write aligns with `v_u`" and "projecting `v_u` out removes the write's
first-order contribution" are consequences of the objective and the lens definition. The contrast
with E-000042's pretrained facts is void as well: the paper's guard never lets `v_u` be ablated when
the model answers `u`, so "injected facts have a closure of one where pretrained facts have none"
compares ablate-`v_u` with never-ablate-`v_u`. The J-space *fraction* of the transport is, to first
order, the identity that the restricted candidate logits respond only to the component of the read
in the span of their gradients.

**What a refuter measured on `e000020_gpt2_seed0.pt` while killing the design**, kept as a calibration
of the instrument: the trained write is far from `W_U[u]` (relative Frobenius distance of `v_proj` and
`o_proj` from the identity 2.42 / 1.71 / 1.24; cos(write at block 8, `w_u`) 0.21); the block-8 atom is
80% the logit-lens row (cos(`v_u`, `w_u`) 0.80); the write's cosine to its own object's atom is 0.24
against 0.06 to another object's — 94% of the write's energy is orthogonal to the atom; and the
block-10 atom estimated with the final post-`ln_f` residual as target is **degenerate**, cos 0.99
between any two tokens, because the VJP is dominated by the shared normalisation direction. The second
number is an instrument note on `so/jlens.py` — `target_layer=-1` on a HuggingFace GPT-2 differentiates
through `ln_f` and must not be used — and none of them is a sentence about pods.

**The survivor, E-000049 — is the held-out addressing gap a linear-erasure problem?** Implemented
(`so/experiments/e000049_template_nullspace_addressing.py`, 787 lines: an `Eraser` wrapped around
`q_ln`, orthogonal / LEACE / PCA arms, a within-fact permutation null, matched-rank random nulls, a
transport-side control that erases the same subspace after the query is taken), smoke-tested at 8 and
100 targets, decompose() reproducing the E-000039-A record to 0.0000. At smoke size every erasure,
the 7-direction random null included, *lowered* held-out addressing. The landing expected the negative
at 75%, and its completeness critic said why the run is worth less than it looks: E-000050 (§31.38) had
by then shown the failing templates are exactly those that put the subject at token position 0, so
the subject's content is *lost* at the sink position and a projection at the last token can only
remove, never add — the negative is predicted by Xiao et al. 2023 and Sun et al. 2024, and a positive
would be the function-vector literature (Todd et al. 2024; Hendel et al. 2023: format information at
the last token is additive and low-rank) plus All-but-the-Top / BERT-whitening (Mu and Viswanath 2018;
Su et al. 2021) for the PCA arms. The critic also listed what must change before the run can record
anything: the transport and null bars are one-sided, so in the negative branch no control can fail;
the LEACE-7 and orthogonal-7 arms erase the same span; the permutation null was variance-matched at
smoke size only; a NaN in any seed's recovered fraction sinks the aggregate; and the decisive
measurement — a subject-identity probe on `h_10[last]` fit on medial templates and tested on
subject-initial ones — was filed as optional when it is the decision. The file is kept as delivered,
with those defects named here rather than patched silently, and it is not run until E-000050 has
said whether there is anything left for it to explain.

**What this closes.** "Symlink – J space – pod" as a mechanism claim is closed for this repository:
in this adapter the pod's object-keying in the model's verbalizable basis is what the code writes and
the training objective aligns, and the sweep that was asked to find its unowned measurement found
three tautologies and one experiment whose answer the tokenisation finding already supplies.

### 31.40 The design sweep: what a result sentence at the seam could still be, and what it cannot (2026-09-05)

An eighteen-agent workflow was asked, for the fifteenth repetition of the request, to design the
experiment at the symlink / J-space / pod seam whose *result sentence* would be unowned — mechanisms
being conceded — with two adversarial refuters per sentence, a code-reading feasibility check per
design, a landing and a completeness critic. It ran across the subagent limit and was resumed from
cache; every agent completed. Four designs, and the verdict on each:

**The J-lens vector as the pod's carrier — refuted twice, and geometrically thin.** Replace the
value a cell injects (the trained `v_proj(W_U[u])`) by the object's J-lens vector at matched
injection RMS and measure held-out override under forced addressing. Stripped of the adapter's name
the sentence is Hernandez et al.'s linear relational embeddings, Fig. 5 (a Jacobian direction beats
the decoder row, template-stable), FishBack (the Jacobian-pullback direction on GPT-2 small with its
off-target KL at matched effect), the workspace paper's own swap across sixteen templates, and
"Memory Injections" (unembedding rows written into GPT-2 small as an external memory). Feasibility
confirmed the geometry the refuters predicted: at read layer 10 the paper's J-lens vector *is*
`W_U[u]` (cos 1.000, one block from the output), at layer 8 cos 0.858 — so "J space" is a thin variant
of the unembedding row on these read layers, and the genuinely different direction (the post-`ln_f`
atom, cos 0.06) is the degenerate one §31.39 recorded. One more confound the reviews had not seen:
E-000013's worst held-out phrasing, template 3 "{s}'s capital city is", is subject-initial — the
position-0 case E-000050 is testing — so the "transport share" this design would attribute to the
carrier sits on a prompt whose subject occupies the attention sink. Not run. With §31.39 (the
J-space pod is by construction) this closes the J-space half of the idea from both sides: as a
reading of the adapter it is a tautology of the code, and as a carrier it is owned and nearly the
unembedding row.

**The paired-arm ledger — a tautology of `bank()`, twice.** Every headline clause (the reverse key
survives SHRED and closes under EVICT/DELETE; the dangling alias discloses and rebinds; BLANK carries
neither key nor object) is what `bank()` exports by construction and E-000015/E-000026/E-000035
already measured. Feasibility found the run as specified would take ~40 h, and that the address
sweep's chance level was mis-stated (124 free keys in the GPT-2 world, not 1023). Not run.

**The residue against the reader — survived one refuter, and is narrow.** After a pod's object is
evicted, do the two live rows that `ON DELETE SET NULL` (BLANK) or `NO ACTION` (a dangling link)
leave behind change a frozen reader's answers on queries *not about the pod* — a query-only adversary
holding one system and no snapshot, against a CASCADE bank that differs in exactly those rows, a
never-wrote store, and a two-fresh-rows control? Nobody has read a SET NULL or NO ACTION pointer
residue through a reader against a never-held reference off the pod; the CASCADE-versus-never cell is
Ramesh (arXiv:2607.27539, LiRA at chance) and the adversary shape is Chen et al. (arXiv:2506.14003).
The landing ranked it first and the critic then found three defects that block it as specified: the
calibration controls (a permuted bank, a bank with two added rows) were one bank per seed, so a
standardised probe separates two point masses at AUC 1.0 and the control fails by construction; the
"dangling alias answers UNKNOWN" row is E-000026's `delete_target/alias_unknown` under a new name,
because `bank()` exports the same key for an evicted target as for a deleted one; and the smoke that
was said to exercise every call never built the never-wrote store, the controls, the feature vector,
the probe or an AUC. The landing's own expectation is a calibrated null — "the reader's off-pod
outputs carry the number of rows, not the history" — and its lifecycle finding (a BLANKed,
self-referencing row was never in the training distribution; GPT-2 answered one of two blanked
aliases with a wrong entity in the smoke) is an engineering fix, not a claim. Recorded as **E-000051,
not yet run**: per-pod controls, the identity row dropped, a real smoke, and only after E-000050 has
released the box.

**The clean-deletion frontier, trimmed — survived one refuter; owned in its headline.** "Damage
governed by n/d" is Guo's Theorem 3.14 and Larimar's Table 4 owns the memory-side load curve; the
composite measurement — per-bystander damage of a pre-readout minimal projection against the number
of facts at fixed width, against the number of access paths, and weights-versus-cells under one attack
currency with a random floor — is unmeasured, and a flat result would retire every load sentence in
this ledger. Feasibility: the trainer has no to-criterion stop, the oblique eraser and the free
rank-k search do not exist, sklearn is absent, and the honest cost is ~7 h on a free box. Recorded as
**E-000048-R, not yet run**, after E-000051.

**What this leaves of the request.** After sixteen repetitions, six sweeps and ten retractions: the
symlink is a store design pattern that SQL-92 implements and Raeesi and Roed name as future work; the
J-space reading of the pod is a tautology of this adapter and, as a carrier, the unembedding row; the
pod objective moved a statistic on the wrong substrate and the address tie moved nothing; the
tokenisation finding (E-000050) is Yang et al.'s diagnosis transported to an external memory, with its
own price. The two experiments that remain unowned are calibrated-null-shaped, and both say so before
they run.

### 31.41 The residue against the reader, synthetic half: the reader carries the row count and the marker, not the history (2026-09-05, E-000051)

§31.35 measured the residue of a pod deletion at the store level: BLANK (`ON DELETE SET NULL` by hand)
and a dangling link (`NO ACTION`) leave two live rows that a never-wrote store does not have, CASCADE
leaves none. E-000051 (§31.40's survivor, run with its critic's fixes) reads those states through the
reader and asks whether the residue changes answers on queries *not about the pod*. Query-only
adversary, one system, no snapshot: five logit features per query, a linear probe, five-fold
cross-validation over 100 pods per seed, Mann–Whitney AUC; per-pod calibration banks (two row
permutations of the live bank; the live bank plus two fresh live LINK rows); KL and top-1 agreement
beside every AUC; the E-000015 reader first, three seeds, trains nothing. The first full run exposed
one instrument leak — the two fresh rows could land on a generic-class key, giving KL 16.8 on that
class — fixed and re-run; every other number reproduced.

| arm (positive vs reference) | AUC, deleted keys | AUC, bystanders | AUC, generic | max KL, bystanders | top-1 agreement |
|---|---|---|---|---|---|
| LIVE vs NEVER (validity) | 1.000 | — | — | — | — |
| CASCADE vs NEVER (the marker channel alone) | 0.501 | **0.948** (worst 0.976) | 0.577 | 0.000 | 1.000 |
| BLANK vs CASCADE (two self-referencing rows) | 1.000 | 0.817 | 0.670 | 0.229 (one seed; 0.04 on the others) | 1.000 |
| DANGLE vs CASCADE (two tombstone-key rows) | 0.999 | 0.869 | 0.685 | 0.035 | 1.000 |
| PERM vs PERM2 (float floor) | — | 0.499 | 0.500 | 0.000 | 1.000 |
| ADD2 vs PERM (row-count floor) | — | **0.965** (worst 0.969) | 0.749 | 0.000 | 1.000 |

Validity held: the adversary sees a live row against none at 1.000, the float floor sits at chance,
the hard-gate pipeline check is bit-identical (encoding and logits 0.000), the marker channel exists
at the interface (residual 0.009–0.014), blanked aliases read UNKNOWN in 1.000 of cases, and the
store-level columns reproduce §31.35 pod for pod (CASCADE exported-HI 1.0, BLANK and DANGLE 0.0 with
two residue rows).

**Two readings fire, both pre-registered, and neither is a sentence about history.**

*Row count.* Two fresh live rows that touch no bystander are detectable on bystander queries at
0.965 and on free-key queries at 0.749, at KL 0.000 and top-1 agreement 1.000. The reader's off-pod
outputs carry the number of active rows — every row sits in every routing softmax and the
dereference pass-through bias is `log(n_cells)` — as a deterministic, sub-milli-nat shift a
standardised probe reads perfectly. Under the rule this voids the residue AUCs as history channels:
BLANK's 0.817 and DANGLE's 0.869 sit *below* the two-fresh-rows floor, so two residue rows are, if
anything, less legible than two ordinary ones.

*Marker.* CASCADE versus NEVER — identical content, identical row count, markers of every row written
after the pod drawn one generator position earlier — is separable at 0.948 on bystanders (the four
rows written after the pod's target are among them) and at chance on the deleted keys and on
free-key queries. E-000046's exported-level history independence of 1.0000 for CASCADE does not
transfer to the reader: the seeded generator's position is a history channel the soft marker gate
exposes, at KL 0.000. The fix is engineering and is named, not run — draw markers from content or
from a per-row secret, never from a position in a stream — and it is what a deployed store of this
kind would need before any history-independence claim could be made at the reader.

*What is behavioural.* One number moves an answer distribution: BLANK's residue shifts a bystander's
logits by up to 0.229 nats on one seed (0.04 on the others), with top-1 unchanged everywhere. The
locality bar of 0.05 nats fails for BLANK and holds for DANGLE. Everything else in the table is
statistical — a trained probe on standardised logits — and behaviourally invisible.

**Sentences.** The negative sentence (confined) is not licensed, because its precondition — the
row-count floor at or below 0.60 — fails. The positive sentence is not licensed, because the residue
never exceeds the floor. What is recorded is the calibrated null the landing expected: on this
reader, "detectability of a deletion residue off the pod" is a row-count and a marker number, and a
paper reporting such an AUC without the two floors would be reporting the reader's arithmetic.
Owned and cited: the adversary shape (Chen et al., arXiv:2506.14003), the cascade-versus-never cell
(Ramesh, arXiv:2607.27539 — at chance in his memory, which has no seeded markers; here it is the
marker channel), the two-clause store/reader structure (Garg, Goldwasser and Vasudevan). The GPT-2
half is queued behind E-000050 and decides whether a frozen language model's adapter, which routes
with the same dense softmax, carries the same two channels.

### 31.8 Boundary

CPU only, no GPU, no LLM above 124M parameters, synthetic worlds, single-token entities, two surface forms per relation, one session. Nothing here shows unlearning of facts already encoded in pretrained weights. Evidence levels recorded: E3–E4 for the synthetic system (F4 for SHRED with the verified gate, E-000010 — **on the value channel only**: E-000028 recovers the shredded object at 1.0000 through the ungated reverse key, where REVOKE and DELETE are at chance, so F4 for SHRED is a claim about answers, logits, hidden states and probes and not about routing); E5 as substrate for the frozen-GPT-2 experiment, with reading, composition, update and the copy bound supported and behavioural deletion not yet supported at the pre-registered thresholds.
