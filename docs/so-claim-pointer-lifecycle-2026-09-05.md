# The claim: a storage pointer keeps its semantics through a frozen model's read

**Status.** One claim, at the size the measurements support. Ten adversarial sweeps of this seam
returned "none"; this is what the eleventh has, and it is a **measurement**, not a mechanism. Every
mechanism in it is owned and is cited below. The ledger (`docs/so-experiment-ledger.md`, §31.51) holds
the tables; this document holds the sentence, its evidence, its boundary and what it does not say.

---

## The sentence

> On a frozen GPT-2 small reading an external multi-version store in which several access keys are
> **LINK rows pointing at one knowledge object** rather than copies of it, the pointer's semantics
> survive the neural read at **every one of twelve phrasings** and across the store's lifecycle: an
> aliased read returns the target's current object at ≥ 0.82, one UPDATE to the target reaches every
> alias at ≥ 0.82, one SHRED or DELETE of the target makes every alias answer UNKNOWN at ≥ 0.995 and
> return the deleted object at 0.0000, a blanked alias (`ON DELETE SET NULL`) answers with an entity
> in ≤ 0.01, and a relinked alias reads its new target at ≥ 0.82 — while the reader's price for the
> indirection, against a link-free adapter trained on the same budget and scored over the same twelve
> phrasings, is **0.088** of reading accuracy. Worst of three seeds throughout, criteria and decision
> rule fixed before the run.

Two facts make the sentence measurable rather than assumed, and both could have come out otherwise:

- **It is a property of the read, not of the store.** The store resolves an alias by construction; a
  frozen 124M-parameter language model reading a routed memory does not have to. On the same
  checkpoints read with the subject at token position 0, the identical battery collapses to 0.29–0.53
  direct and 0.30–0.50 through an alias. The pointer works **once the prompt's first position is
  occupied**, by a single space at inference with no weight changed, or by training with a BOS.
- **The reverse control fires.** The BOS-trained adapter, read without its BOS, answers a
  subject-initial alias at 0.0050–0.0100 while its subject-medial ones stand at 0.9150–0.9550.

---

## Evidence

| row | worst seed | where | the control that could have failed |
|---|---|---|---|
| aliased read, all 12 phrasings | 0.8150 – 0.9400 | E-000052 | the same read at position 0: 0.30 – 0.50 |
| one UPDATE reaches every alias | 0.8200 – 0.9550 | E-000052 | the duplication arm, 0.0000 — **store arithmetic, a baseline and not evidence** |
| one SHRED → every alias UNKNOWN | 0.9950 – 1.0000 | E-000052 | the alias must still be *addressable*: it is, and it reads UNKNOWN rather than nothing |
| one DELETE → every alias UNKNOWN | 0.9950 – 1.0000 | E-000052 | — |
| SHRED/DELETE → the deleted object | 0.0000 at every phrasing | E-000052 | the object is a random entity the frozen model has no prior on |
| BLANK → some entity (the safety row) | 0.0000 (11 of 12), 0.0100 at t9 | E-000052 | it failed at 0.175 on the sink-reading substrate (§31.45) |
| BLANK → UNKNOWN | 0.9900 – 1.0000 | E-000052 | — |
| RELINK → the new target | 0.8200 – 0.9400 | E-000052 | — |
| price of the pointer, all phrasings | **0.0879** (bar 0.10) | E-000052 | E-000025's 0.0954 did not transfer; the bar was set before the substrate existed |
| price of link training | 0.0054 (bar 0.25) | E-000052 | — |
| reverse control, alias without the BOS | 0.0050 – 0.0100 initial; 0.9150 – 0.9550 medial | E-000052 | the arm exists to make the finding fail if position 0 is not the cause |
| the same battery, **no training at all**, one space at inference | direct ≥ 0.9433, alias ≥ 0.8600, UPDATE reach ≥ 0.8500 at five phrasings | E-000050-A | bare: 0.2933 / 0.3000 / 0.2950 at the worst phrasing |
| the synthetic precedent | update reach 1.0000 vs 0.0000 duplicated; object recoverable by probe after one SHRED 0.7% vs 87.3% | E-000015 | a 2.5M model trained from scratch — the claim is that this survives to a frozen pretrained one |

### Why the update row is a property of the read, not of the store

The obvious objection to the load-bearing row is that the store resolves the alias and the model only
reads the answer. It does not. `MVCCStore.bank()` exports an alias row carrying **the target's key**,
never the target's object, and the code says so at `so/mvcc.py:522-525`:

> *A link row carries the TARGET'S KEY, not its payload and not its state: whether that key is held by
> a signed, active, existing cell is exactly what the model has to discover.* `obj` *is a constant
> placeholder for link rows (never the target's object).*

So after one UPDATE to the target, **the alias row's exported bytes do not change at all**. Only the
target row's payload does. For the alias to answer with the new object the frozen model must route to
the alias row, read out the target's key, re-query the key table through its dereference slot
(`so/llm_adapter.py:262-285`), and read the target's current value. That chain is learned, and it
fails: at 0.2950 to 0.5350 with the subject on the attention sink, and at 0.8850 against a 0.90 bar in
the earlier battery (E-000026). The duplication arm at 0.0000 is the store's arithmetic; the reach at
0.82 to 0.955 is not.

Records: `so/results/e000052_symlink_bos_battery.{json,md}`,
`so/results/e000050a_symlink_prefix.{json,md}`, `so/results/e000050a_bos_artefact.{json,md}`,
`so/results/e000015_symlink_cells.md`, `so/results/e000026_lifecycle.md`.

---

## The boundary: what is owned, and by whom

- **The mechanism is old and is not claimed.** A name that resolves to a record rather than carrying a
  copy is a symlink (Unix), a foreign key with `ON DELETE SET NULL` and `ON UPDATE CASCADE` (SQL-92),
  a versioned pointer under multi-version concurrency control, and an overlay of one namespace on
  another (union mounts, 1995). The lifecycle verbs are a database's.
- **The architecture is owned.** An external, editable memory read by a frozen model: SERAC (2022),
  GRACE (2023), WISE, MEMOIR, Larimar, KBLaM, SILO (ICLR 2024), LMLM. Removal from a non-parametric
  store reverting the model is by construction in all of them.
- **The design is somebody's stated future work.** Raeesi and Roed (arXiv:2607.00605, §9): *"a second
  direction is canonicalization at write time, in which aliases and paraphrastic forms are stored as
  pointers into a single canonical record rather than as independent triplets. Both approaches are
  directly testable within our framework."* This measurement is that direction, executed. It is the
  strongest support for the claim's novelty and simultaneously its sharpest limit: the authors could
  run it on their own released harness at a scale this one cannot.
- **The instrument correction is owned.** The first-token anomaly is attention sinks (Xiao 2023),
  massive activations (Sun 2024), the sink following position rather than token (Gu, ICLR 2025;
  Ran-Milo 2026); the subject-initial collapse of a locate-and-edit weight edit and its prefix remedy
  are Yang et al., *The Fall of ROME* (Findings of EMNLP 2024). What this programme adds there is the
  measurement on an external memory with both directions of the control (§31.44, §31.46).
- **The composition of a store-side and a reader-side guarantee** is Garg, Goldwasser and Vasudevan
  (Eurocrypt 2020), generalised by Godin and Vasudevan (2022) and Cohen et al. (CCS 2023).

**What is new is the result sentence**: no published measurement reports, for an external memory read
by a frozen language model, that a *storage pointer's* semantics — read-through, update-reach,
delete-propagation, set-null safety, relink — hold across every phrasing of a query set, with the
reader's price for the indirection measured against a link-free control.

---

## What is not claimed

1. **Not the duplication arm.** One UPDATE reaching 0.0000 of duplicated copies is store arithmetic:
   a copy is a different row and an update cannot touch it. It is a baseline for reading the alias
   number, never evidence for it.
2. **Not locality.** The adapter injects 3.4–4.2 nats of KL on generic text where no key matches,
   against a 0.05 bar it has never met, and the parallel branch's independent implementation fails the
   same bar at 3.65–5.23 (§31.49). A pointer that reads, updates and deletes correctly still speaks
   when it is not asked. This is the programme's largest open failure and it is not fixed here.
3. **Not "behaves like the model's own knowledge."** On the prior-conflict substrate a stored
   counterfactual overrides the pretrained fact on the trained phrasing at 1.0000 and on held-out
   phrasings at **0.0000**, under every prefix tried (§31.46). The position-0 correction does not
   move that row.
4. **Not deletion from weights.** Nothing here removes anything from GPT-2's parameters. Every
   guarantee is about what the store hands the model and what the model then answers.
5. **Not tracelessness.** A deleted pod's residue is measurable at the store (§31.35) and, on the real
   reader, moves a bystander's distribution by up to 4.49 nats at unchanged top-1 (§31.45) — under two
   floors the reader's own arithmetic supplies. A deleted-but-routable row contributes *less* than a
   coefficient-matched live row (§31.50).
6. **Not scale.** GPT-2 small (124M), CPU only, a synthetic world, single-token entities, twelve
   templates, three seeds. Nothing here is evidence about a production model, real subjects or free
   text.
7. **Not a claim about J space.** The J-space reading of this adapter is closed on all three of its
   readings — the write is by construction, the carrier is owned, the addressing is a tautology of the
   keying (§31.39, §31.40, §31.42).
8. **Not a security property.** Nothing here revokes anything already handed out: every certificate in
   `so/audit.py` is a statement about one export (§31.47), and a freshness check cannot live in the
   learned reader at all (§31.48).

---

## Precision the sentence needs, checked at source

**"Every one of twelve phrasings" means this, exactly.** Eight of the twelve are trained forms and four
are held out. Of the four held out, **t10 is the trained t0 under a fixed prefix**: `TEMPLATES12[0][10]`
is `"It is known that {s} lives in"` and equals `E39.PREFIX + TEMPLATES12[0][0]` character for
character, verified. So the genuinely novel forms are three, t8, t9 and t11, of which two are
subject-initial and one is subject-medial. The claim covers twelve phrasings; it does not cover twelve
*unseen* phrasings, and §31.34 already records that this programme's count of phrasings is the number
somebody typed.

**The intervals.** Each cell is 100 alias groups of two aliases, so n = 200 alias reads per template
per seed (`E20.EVAL`). At that n the worst-seed rates carry real width:

| row | worst seed | 95% Clopper–Pearson |
|---|---|---|
| alias read, worst template (t1) | 0.8150 | 0.754 to 0.866 |
| UPDATE reach and RELINK, worst (t9) | 0.8200 | 0.760 to 0.871 |
| SHRED or DELETE leaves the alias UNKNOWN, worst | 0.9950 | 0.972 to 1.000 |
| BLANK answers with some entity, worst (t9) | 0.0100 | 0.001 to 0.036 |
| alias read, best template (t4) | 0.9400 | 0.898 to 0.969 |

The lower bounds are what the sentence is entitled to: the aliased read and the update reach are
"above 0.75 with 95% confidence at the worst template of the worst seed", not "0.82". The deletion
rows and the SET NULL row survive their bars comfortably. Nothing in the claim turns on a difference
smaller than these intervals.

## How it could be wrong

- Twelve templates of two surface forms per relation is not "every phrasing" in any general sense; it
  is every phrasing in a fixed, typed set.
- The price, 0.088, is against *this* link-free control at *this* budget. A better-trained link-free
  adapter would raise it.
- The single failing validity row (a blanked alias's sibling readable at 0.79 against 0.80) is
  unexplained and is recorded rather than argued away.
- Three seeds of one architecture on one frozen model. E-000019's fresh-seed protocol is the standing
  answer to "would a fourth seed hold", and it has not been run for this battery.

---

*Ledger: §31.51 (the tables), §31.44 and §31.46 (the instrument correction), §31.45 and §31.50 (the
residue), §31.47 to §31.49 (the certificate's window, the freshness proposition, the parallel branch's
audit). Eleven retractions precede this claim and are listed in the pull request.*
