# SO — Modular Neural Operating System

**Project State, Vision & Architecture**

**Status date:** 2026-09-02  
**Project:** SO  
**Research direction:** Modular, editable, addressable and continually learnable neural knowledge  
**Current phase:** Experimental research / architecture discovery  
**Primary objective:** Demonstrate that knowledge inside a neural system can become a controllable computational object rather than an inseparable side effect of model weights.

---

## 1. Executive Summary

SO is an experimental architecture for a new class of neural systems.

The central idea is simple:

> Knowledge inside an AI system should be addressable, mutable, versionable, revocable and composable — while remaining usable directly by neural computation.

Today's large language models largely encode knowledge implicitly across distributed parameters.

This creates fundamental limitations:

* individual knowledge cannot reliably be located,
* knowledge cannot reliably be deleted,
* updates can interfere with unrelated knowledge,
* provenance is difficult to establish,
* rollback is difficult,
* continual learning causes interference,
* model behavior and stored knowledge are tightly coupled,
* proving that information has actually been removed is extremely difficult.

SO investigates whether this relationship can be changed.

Instead of treating the model as one monolithic block

```text
Input
  ↓
Transformer
  ↓
Output
```

SO explores an architecture closer to:

```text
                    ┌───────────────────────────┐
                    │        Neural Core        │
                    │                           │
Input ─────────────►│ reasoning / composition   │────────► Output
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                                  │ neural access
                                  ▼
                    ┌───────────────────────────┐
                    │ Mutable Knowledge Layer   │
                    │                           │
                    │ addressable knowledge     │
                    │ provenance                │
                    │ versions                  │
                    │ dependencies              │
                    │ revocation                │
                    │ markers                   │
                    └───────────────────────────┘
```

The long-term target is not merely a better RAG system.

The knowledge should participate in the model's internal computation.

The resulting system can be thought of as a:

> **Modular Neural Operating System**

where neural knowledge behaves increasingly like manageable state.

---

## 2. The Core Research Question

The project asks:

> Can a neural model learn and reason with knowledge that remains individually controllable after training?

More formally, we want knowledge objects

$$
K = \{k_1, k_2, \ldots, k_n\}
$$

such that the system can perform operations analogous to

```text
WRITE(k)
READ(k)
UPDATE(k)
REVOKE(k)
RESTORE(k)
TRACE(k)
COMPOSE(k1, k2, ...)
```

without retraining the entire model.

The difficult part is not implementing these operations in an external database.

That is already possible.

The difficult question is whether they can operate while the knowledge remains part of neural reasoning.

---

## 3. What SO Is Not

SO should not collapse into a conventional retrieval architecture.

It is therefore important to distinguish:

### Conventional RAG

```text
Question
   ↓
Retriever
   ↓
Database
   ↓
Text/context
   ↓
LLM
```

The knowledge remains external.

The model receives textual evidence.

### SO target architecture

```text
Question
   │
   ▼
Neural computation
   │
   ├───────────────┐
   │               │
   ▼               ▼
Core state     Mutable neural
               knowledge state
   │               │
   └───────┬───────┘
           ▼
       reasoning
           │
           ▼
         answer
```

The distinction is critical:

> SO attempts to make mutable knowledge part of the computation rather than merely additional prompt context.

---

## 4. Long-Term Vision

The ultimate system should support six major properties.

### 4.1 Addressable Knowledge

Individual knowledge should possess some form of stable identity.

```text
K_00001
K_00002
K_00003
...
```

The identity does not necessarily need to correspond to one neuron or one vector.

It can represent a distributed neural structure.

But the system needs a way to address it.

### 4.2 Mutable Knowledge

Knowledge should be updateable without retraining the entire model.

```text
K17:
Version 1
Paris → property A
UPDATE
Version 2
Paris → property B
```

The update should affect the relevant behavior while preserving unrelated knowledge.

### 4.3 Revocable Knowledge

A knowledge unit should be removable or disabled.

```text
REVOKE(K17)
```

After revocation:

```text
direct retrieval        → fail
reasoning through K17   → fail
derived answer          → fail
alternative valid path  → remain functional
unrelated knowledge     → remain functional
```

This is substantially harder than simply preventing direct retrieval.

---

## 5. Deletion vs. Suppression

One of the central distinctions in SO is:

> **SUPPRESSION ≠ DELETION**

A model may stop answering a fact while the information remains encoded internally.

Therefore a successful revocation mechanism must be tested through several attack surfaces.

```text
             ┌──────────────┐
             │ revoked fact │
             └──────┬───────┘
                    │
       ┌────────────┼─────────────┐
       ▼            ▼             ▼
    direct       indirect      composed
    query        query         reasoning
       │            │             │
       └────────────┼─────────────┘
                    ▼
             reconstruction?
```

A meaningful deletion claim requires reconstruction resistance, not merely answer suppression.

---

## 6. Versionable Knowledge

Knowledge should eventually support a history such as:

```text
K42
│
├── V1
│
├── V2
│
├── V3
│
└── V4
```

Operations:

```text
UPDATE K42 → V5
ROLLBACK K42 → V3
REVOKE K42
RESTORE K42 → V5
```

This moves neural knowledge closer to software state management.

---

## 7. Provenance

SO also targets knowledge provenance.

For an answer:

```text
A → B → C → D
```

the system should ideally be capable of identifying the contributing knowledge objects:

```text
Answer D
derived_from:
    K17
    K81
    K103
```

This enables:

* explainability,
* auditing,
* dependency tracking,
* selective deletion,
* rollback,
* debugging,
* contamination analysis.

---

## 8. Knowledge Dependency Graph

Knowledge cannot be treated as independent facts.

A fact can support other conclusions.

Example:

```text
K1: A → B
K2: B → C
K3: C → D
```

Therefore:

```text
A → B → C → D
```

If K2 is revoked, the system must not continue producing D through a stale representation of the removed relation.

This motivates an internal dependency structure.

```text
             K1
              │
              ▼
             K2
            /  \
           ▼    ▼
         K3      K4
          │       │
          └───┬───┘
              ▼
             K5
```

Revocation therefore becomes a graph problem as well as a neural one.

---

## 9. Alternative Paths

A crucial architectural requirement discovered during experimentation is preservation of legitimate alternative paths.

Suppose:

```text
A ─K1→ B ─K2→ C
A ─K3→ X ─K4→ C
```

If K2 is revoked:

```text
A → B → C
```

must stop working through that path.

But:

```text
A → X → C
```

must continue to work.

Therefore deletion cannot simply destroy the output concept C.

It must remove the contribution of the targeted knowledge relationship.

This is one of the strongest arguments for relationship-level addressability.

---

## 10. The Symlink Concept

One architectural direction explored in SO is the Symlink Adapter.

The analogy comes from filesystems.

Instead of permanently duplicating knowledge throughout the model, neural computation can learn references to mutable knowledge structures.

Conceptually:

```text
Neural representation
        │
        ▼
     SYMLINK
        │
        ▼
Knowledge Object
```

Multiple computations may reference the same object:

```text
Representation A ─┐
Representation B ─┼──► K42
Representation C ─┘
```

Updating K42 therefore does not necessarily require modifying every representation that uses it.

---

## 11. Why the Symlink Idea Matters

Standard distributed representations create something resembling:

```text
knowledge
   ↓
copy
copy
copy
copy
copy
```

SO investigates:

```text
knowledge
   ↓
canonical mutable representation
   ↑
references
```

This could theoretically reduce:

* update cost,
* deletion complexity,
* catastrophic interference,
* duplicated knowledge,
* provenance ambiguity.

The open scientific question is how far this abstraction survives once the representation becomes genuinely neural and distributed.

---

## 12. Aiko Marker / Crypto-Shredding Direction

Another architectural idea developed during the project is the Aiko Marker.

Its purpose is to strengthen revocation.

The analogy is cryptographic erasure.

Instead of attempting to rewrite every representation containing information, knowledge can depend on a small critical component.

Conceptually:

```text
Knowledge representation
        Payload
           +
        Marker M
           │
           ▼
      usable knowledge
```

After destroying or invalidating M:

```text
Payload
   +
[INVALID M]
     ↓
knowledge cannot be reconstructed through the intended path
```

The analogy resembles crypto-shredding:

```text
encrypted data + key → readable
encrypted data - key → unusable
```

For SO:

```text
distributed neural trace + required marker → usable knowledge
distributed neural trace - marker → inaccessible / non-composable
```

This remains an experimental architectural direction rather than a demonstrated solution to arbitrary neural unlearning.

---

## 13. Biomarker Direction

The project subsequently expanded this idea into a more biological analogy.

Instead of merely assigning an external ID to knowledge, a learned knowledge structure could carry an internal marker / biomarker.

Conceptually:

```text
Knowledge Unit
┌──────────────────────────┐
│ semantic representation  │
│                          │
│ marker / signature       │
│ provenance               │
│ dependency information   │
└──────────────────────────┘
```

The marker could potentially provide a mechanism for:

* identification,
* routing,
* provenance,
* selective activation,
* revocation,
* dependency propagation.

The biomarker concept should not be interpreted literally as biological machinery.

It is an architectural analogy for a learned internal signature.

---

## 14. Biological Inspiration

Several biological analogies have informed the research.

### DNA analogy

Knowledge can be imagined as compact reusable sequences rather than arbitrary duplication.

```text
sequence
   ↓
expression
   ↓
function
```

Neural equivalent:

```text
knowledge representation
        ↓
activation / composition
        ↓
behavior
```

### Gene editing analogy

Instead of retraining an organism, modify a targeted component.

```text
identify
   ↓
target
   ↓
modify
   ↓
validate
```

SO equivalent:

```text
identify knowledge
        ↓
address
        ↓
update/revoke
        ↓
behavioral validation
```

### Dependency analogy

Some biological systems require host machinery.

Similarly, SO explores whether knowledge can deliberately depend on an indispensable marker.

```text
knowledge trace
      +
critical dependency
      =
functional knowledge
```

Destroying the dependency could render the knowledge unusable even if fragments remain.

---

## 15. Current High-Level Architecture

The architecture currently converges toward approximately:

```text
┌───────────────────────────────────────────────────────────────┐
│                           SO                                  │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    Neural Core                          │  │
│  │                                                         │  │
│  │  attention                                              │  │
│  │  reasoning                                              │  │
│  │  language                                               │  │
│  │  composition                                            │  │
│  └──────────────────────┬──────────────────────────────────┘  │
│                         │                                     │
│                  knowledge interface                          │
│                         │                                     │
│  ┌──────────────────────▼──────────────────────────────────┐  │
│  │                Symlink / Routing Layer                  │  │
│  │                                                         │  │
│  │  addresses                                              │  │
│  │  routing                                                │  │
│  │  markers                                                │  │
│  │  activation                                             │  │
│  └──────────────────────┬──────────────────────────────────┘  │
│                         │                                     │
│  ┌──────────────────────▼──────────────────────────────────┐  │
│  │              Mutable Knowledge Layer                    │  │
│  │                                                         │  │
│  │  K1   K2   K3   K4   ...                               │  │
│  │                                                         │  │
│  │  value                                                  │  │
│  │  marker                                                 │  │
│  │  version                                                │  │
│  │  provenance                                             │  │
│  │  dependencies                                           │  │
│  └──────────────────────┬──────────────────────────────────┘  │
│                         │                                     │
│  ┌──────────────────────▼──────────────────────────────────┐  │
│  │                  Control Plane                          │  │
│  │                                                         │  │
│  │ WRITE / UPDATE / REVOKE / RESTORE                       │  │
│  │ VERSION / TRACE / AUDIT                                 │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

---

## 16. Logical Knowledge Object

A conceptual knowledge object can be represented as:

```text
KnowledgeObject {
    id
    representation
    marker
    version
    provenance
    dependencies
    status
}
```

Possible state machine:

```text
             WRITE
               │
               ▼
            ACTIVE
           /      \
      UPDATE      REVOKE
        │            │
        ▼            ▼
     ACTIVE       REVOKED
        ▲            │
        │            │
        └──RESTORE───┘
```

This is a conceptual model, not yet evidence that all these operations are solved in arbitrary LLMs.

---

## 17. Experimental Philosophy

The project deliberately moved away from immediately building a huge system.

Instead:

> **Evidence first. Architecture second. Scale last.**

The experimental progression is intended to be:

```text
Synthetic World
      ↓
Mini architecture
      ↓
controlled knowledge
      ↓
direct retrieval
      ↓
multi-hop reasoning
      ↓
updates
      ↓
revocation
      ↓
rollback
      ↓
provenance
      ↓
locality
      ↓
adversarial reconstruction
      ↓
natural language
      ↓
larger models
```

This is essential because otherwise a large LLM can hide architectural failures.

---

## 18. Synthetic World

The first experimental environment uses synthetic facts and relationships.

Advantages:

* exact ground truth,
* arbitrary knowledge graphs,
* controlled contradictions,
* controlled updates,
* exact provenance,
* exact dependency paths,
* deterministic evaluation.

Example world:

```text
A → B
B → C
C → D
E → F
F → G
```

Questions can then measure:

```text
Direct:
A → ?
2-hop:
A → C?
3-hop:
A → D?
Provenance:
Which edges produced D?
Revocation:
Remove B → C.
Rollback:
Restore B → C.
Locality:
Did E → F change?
```

---

## 19. Experiment E-000001-A

### Purpose

Establish a mechanical reference implementation before training a neural model.

This isolates the semantics of:

* addressing,
* composition,
* provenance,
* updates,
* rollback,
* locality,
* alternative paths.

### Test scale

```text
5 seeds
×
1,000 cells
```

### Result

The mechanical reference passed the tested suite.

Observed:

| Measure | Result |
|---|---|
| Direct | 100% |
| 2-hop | 100% |
| 3-hop | 100% |
| Provenance | 100% |
| Update/Rollback | 100% |
| Locality | 100% |
| Alternative Path | 100% |
| Replay deviation | 0 |

### Interpretation

This established that the desired knowledge semantics are internally coherent in the controlled reference system.

It did not prove that a trained neural network could reproduce them.

---

## 20. Experiment E-000001-B

The next step introduced a trained synthetic Mini-Transformer.

### Objective

Determine whether learned neural computation could interact with the controlled knowledge mechanism while preserving the required semantics.

### Seeds

**5 / 5 seeds**

successfully completed the principal test set.

### Results

The recorded core results were:

| Measure | Result |
|---|---|
| Direct | 100% |
| Noise | 100% |
| 2-hop | 100% |
| 3-hop | 100% |
| Provenance | 100% |
| Revoke | exact |
| Replay | exact |

under the principal experimental configuration.

A stronger noise condition exposed degradation.

At approximately:

```text
noise = 0.24
```

direct accuracy dropped to approximately:

```text
68.4%
```

### Interpretation

E-000001-B was an important step because the behavior was no longer purely mechanical.

A trained Mini-Transformer successfully operated over the experimental knowledge structure across multiple seeds.

However:

> This was still a synthetic experiment and therefore not proof of general LLM-scale editable knowledge.

---

## 21. What E-000001 Actually Established

The experiments provide evidence for a narrower claim:

> A trained small neural model can be constructed such that structured mutable knowledge participates in learned computation while retaining controlled update/revoke/provenance behavior in a synthetic environment.
