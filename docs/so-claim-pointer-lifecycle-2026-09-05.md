# The claim: a learned dereference resolves a storage pointer, at a measured price

**Status: withdrawn once and rewritten, on the day it was made.** The first version of this document
claimed seven lifecycle rows. A three-lens audit refuted five of them, found the headline price
misdescribed, and found one row excluded by this experiment's own pre-registration. Ledger §31.53 is
the retraction and lists every finding with its file and line. What follows is the sentence that
survived, and it is about a third of the size of the one it replaces. The tables are ledger §31.51,
read with the correction note at its head.

## The sentence

> On a frozen GPT-2 small reading an external store in which an access key is a **LINK row carrying
> only another row's key** rather than a copy of its object, one adapter, its routing and dereference
> slots directly supervised, reads through the pointer at all twelve phrasings of a fixed template
> set: **0.8150 to 0.9400**, worst of three seeds. Nine of those twelve are labelled reproductions in
> this experiment's own pre-registration; on the three genuinely novel phrasings the range is 0.8300
> to 0.9250. Each cell is 200 alias reads clustered on 100 target rows in a single world per seed, so
> a binomial interval on the floor is anti-conservative and three worlds are the independent
> replicates. Reading through the pointer costs **this same adapter 0.088** of accuracy against
> reading the same keys as duplicated copies, and 0.046 and 0.056 on the other two seeds, against a
> 0.10 bar that E-000025 discloses was set knowing the alias cells at templates 1, 8 and 9, the floor
> cell included. The five subject-initial phrasings collapse to 0.30 to 0.50 through an alias when
> position 0 is left to the subject, on the *recorded* checkpoints; the seven medial phrasings,
> including the floor cell, place the subject off position 0 by construction and have no such control.

Two things make it a measurement rather than a definition:

- **The store does not resolve it.** `MVCCStore.bank()` exports an alias row carrying the target's
  key and a constant placeholder in place of an object, and says so at `so/mvcc.py:522-525`: *"A link
  row carries the TARGET'S KEY, not its payload and not its state: whether that key is held by a
  signed, active, existing cell is exactly what the model has to discover."* The frozen model must
  route to the alias, read out the key, re-query the key table through its dereference slot
  (`so/llm_adapter.py:262-285`), and read the target's value.
- **It fails when the instrument is wrong, on the half where that can happen.** The read on the
  *recorded* checkpoints (`e000020_gpt2`, a different artefact from the `e000020_gpt2_bos` measured
  here), at three of its five templates, gives 0.2933 to 0.5633 direct and 0.30 to 0.50 through an
  alias with the subject at position 0. Its two subject-medial templates read 0.8700 through an alias
  bare and do not collapse. Occupying position 0 with a single space, no weight changed, restores the
  initial half.

---

## Evidence

| row | worst seed | 95% interval | status |
|---|---|---|---|
| aliased read, all twelve phrasings | 0.8150 – 0.9400 | 0.754 – 0.866 at the floor | **the claim** |
| the same read with the subject at position 0 | 0.30 – 0.50 | — | **the control that could have failed** — at the five subject-initial phrasings only; vacuous at the seven medial ones, the 0.8150 floor cell included |
| cost of sharing, same adapter, alias against duplicate | 0.0879 (0.0462, 0.0563 on the other seeds) | — | **the claim**, against a 0.10 bar E-000025 set knowing templates 1, 8 and 9 |
| one UPDATE reaches every alias | 0.8200 – 0.9550 | — | demoted: the aliased read re-run, r = 0.910 against it |
| RELINK **back to the same target** reads it again | 0.8200 – 0.9400 | — | demoted: `lifecycle_extra` relinks each blanked alias to its own original target and scores the original object; it is blank-then-restore, not a relink to a new target; r = 0.980 against the aliased read |
| SHRED or DELETE leaves every alias UNKNOWN | 0.9950 – 1.0000 | — | demoted: forced by the exporter and the gate; passes at 0.95+ where the reader reads at 0.30 |
| the deleted object returns | 0.0000 | — | demoted: cannot fail, and the pre-registration excludes such rows |
| BLANK answers with an entity | 0.0100 at t9 | 0.001 – 0.036 | reported, not claimed: the rule gates it on a neighbour row that failed at 0.79 against 0.80 |
| cost of link training, against the link-free adapter | 0.0054 | — | demoted: cannot fail, 46× headroom |

The record's own verdict on the battery is `criteria.claim_supported = False`: thirteen of fourteen
pre-registered criteria pass, and one fails.

### Where the resolution happens

`MVCCStore.bank()` exports an alias row carrying the target's key and a constant placeholder in place
of an object (`so/mvcc.py:522-525`), so the store does not hand the reader the answer. That makes the
*aliased read* a property of the read. It does not make UPDATE-reach one: the reader is stateless,
re-encoding the bank on every call, so a post-update bank is structurally identical to one that always
held the new object, and update minus alias is +0.012 with r = 0.910 over 36 cells. The earlier
argument here, that E-000026's update row "failed at 0.8850 against a 0.90 bar", was wrong: per seed
the update row is the alias row plus 0.005 to 0.035, and it failed only because its bar was 0.90 where
the alias bar was 0.80.

Records: `so/results/e000052_symlink_bos_battery.{json,md}`,
`so/results/e000050a_symlink_prefix.{json,md}`, `so/results/e000050a_bos_artefact.{json,md}`,
`so/results/e000015_symlink_cells.md`, `so/results/e000026_lifecycle.md`.

---

### What the pre-registration itself said the run would carry

Worth stating, because it is the one place where the record and the audit agree exactly. E-000052's
docstring, written before its substrate existed, declares that only three things in its table are
content and that *"everything else in the table is a reproduction of E-000026 / E-000050 and is
labelled so"*: **(P)** the reader's price for the pointer, called *"the one number whose sign nobody
here can predict"*; **(N)** the SET NULL wrong-entity rate; and **(T)** the two subject-medial held-out
rows. The audit then removed (N), because the rule gates it on a neighbour that failed, and removed
the shared-update half of (T), because it is the aliased read. What is left standing is (P) and the
aliased read itself — which is what this document now claims, and nothing else.

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

**What is new is a narrower result sentence**: no published measurement reports, for an external
memory read by a frozen language model, the resolution rate of a *storage pointer* followed inside
the computation across every phrasing of a query set, with the cost of the indirection measured
against the same reader reading copies. Raeesi and Roed already measure delete-propagation over alias
closures at far larger scale (12,228 deletions, six prompt formulations) on independent triplets, and
RippleEdits already owns "one edit must reach every alias" at the parametric tier; those conjuncts are
theirs, which is why they are demoted above. And §9's future work names a *test* this battery does not
run: re-running their audit on the modified database and checking whether the retrieval-artifact rate
falls. This executes their design, not their test.

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
- The price is a within-reader contrast, alias against duplicate on one adapter, and its reference arm
  is pinned at ceiling (dup mean 0.9888 to 0.9929), so it carries little beyond the aliased read
  itself.
- Neither the dereference slot nor the gate was ablated: both are directly supervised, and no
  `n_deref = 0` arm was run on the same link store, so crediting the slot is architectural attribution
  rather than measurement.
- Nine of the twelve phrasings are labelled reproductions in the experiment's own pre-registration, so
  "all twelve phrasings" re-imports rows that pre-registration excluded from the claim.
- The single failing validity row (a blanked alias's sibling readable at 0.79 against 0.80) is
  unexplained and is recorded rather than argued away.
- Three seeds of one architecture on one frozen model. E-000019's fresh-seed protocol is the standing
  answer to "would a fourth seed hold", and it has not been run for this battery.

---

*Ledger: §31.51 (the tables), §31.44 and §31.46 (the instrument correction), §31.45 and §31.50 (the
residue), §31.47 to §31.49 (the certificate's window, the freshness proposition, the parallel branch's
audit). Eleven retractions precede this claim and are listed in the pull request.*
