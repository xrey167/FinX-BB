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
| E-000018 | Can the adapter be stopped from injecting into text it has no key for? | Not in this design, and the three-arm ablation says why. **WITHDRAWN in part — see the correction below.** Adding the CAPACITY to inject nothing (an absolute match score against the best real cell key) appeared to change nothing at all: 3.2681 nats against a baseline of 3.2741. That measurement is invalid: the match gate multiplied the read and the RMS-matched injection then divided by the RMS of that same gated read, so the factor cancelled exactly and the mechanism could not act. What survives is the other half: training the BEHAVIOUR on generic sentences brings the divergence to 0.6035 and both arms together to 0.6736, every arm stays a factor of twelve above the 0.05 bar, and the arms that improve it pay for it — the generic arm reads 68.9% on held-out phrasings against 74.0% and refuses at 88.0% against 89.8%, the combined arm 69.2% and 85.5%. The structural reading also survives and is what motivated the fix: answering ' unknown' when a cell is gone needs an INJECTION, changing nothing on text with no key needs NONE, and both were routed through the same null column. The capacity question is reopened and is now E-000022's to answer | E5 (substrate) / F1, negative with a mechanism |
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

### 31.8 Boundary

CPU only, no GPU, no LLM above 124M parameters, synthetic worlds, single-token entities, two surface forms per relation, one session. Nothing here shows unlearning of facts already encoded in pretrained weights. Evidence levels recorded: E3–E4 for the synthetic system (F4 for SHRED with the verified gate, E-000010 — **on the value channel only**: E-000028 recovers the shredded object at 1.0000 through the ungated reverse key, where REVOKE and DELETE are at chance, so F4 for SHRED is a claim about answers, logits, hidden states and probes and not about routing); E5 as substrate for the frozen-GPT-2 experiment, with reading, composition, update and the copy bound supported and behavioural deletion not yet supported at the pre-registered thresholds.
