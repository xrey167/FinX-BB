# Reference-carried memory — the cache holds the address, never the value

Date: 2026-09-05
Status: **research-level technical novelty candidate; not a legal novelty or patentability opinion.**
The capability gate is running and the targeted prior-art search is incomplete. Both are named below.

## The claim, in one sentence

> In a frozen language model with an external mutable memory, the state the model persists can be made
> a function of the prompt and the alias namespace alone — carrying a knowledge-free reference that
> participates in the frozen computation, with every mutable relation (payload, alias binding, marker,
> lifecycle status) applied only after the last cache-writing block — so that UPDATE, RELINK, REVOKE
> and SHRED change the model's answer while leaving every persisted tensor bit-identical, and
> revocation of already-materialised neural state costs nothing and needs no lineage metadata.

## Why this is the seam, and not the previous three answers

The programme reached this point by elimination, and the eliminations are measured, not argued.

**Late binding was assumed free, and is not.** E-000082 established that a payload injected after the
final block leaves persisted K/V bit-identical to the no-memory forward, on two backbones, and its
note promoted fixed late binding to "a mandatory strong baseline" any lifecycle mechanism must beat.
E-000083 then trained the real symlink reader at that placement and lost the capability gate: held-out
candidate mean 0.669–0.686 against 0.984 for the identical recipe at read layers (8, 10).

**It was not the addressing depth.** E-000084 arm C kept the routing query and the dereference chain at
blocks 8 and 10 and moved only the write. Run `33970654975`, three seeds, K/V exposure verified at
exactly 0.0 on real prompts:

| Arm | Seed | Held-out candidate mean | Worst full-vocabulary template | Strict gate |
|---|---:|---:|---:|---|
| A, write in place at (8, 10) | 0 | 0.955 | 0.935 | fail |
| A, write in place at (8, 10) | 1 | 0.990 | 0.960 | **pass** |
| C, write after block 11 | 0 | 0.664 | 0.285 | fail |
| C, write after block 11 | 1 | 0.645 | 0.270 | fail |
| C, write after block 11 | 2 | 0.621 | 0.290 | fail |

So something has to ride through the frozen blocks. A memory that leaves no trace in them is not read
back. That is the trade-off the programme's open problems all sit on: E-000079 and E-000080 exist to
track and repair the contamination that participation causes, and the field's cache-lifecycle work
(Leyline, KVEraser, CacheBlend, AgentKVShift) repairs the same contamination by other means.

**The unexamined assumption is that what rides must be the knowledge.** It need not. The symlink supplies
a name that is not a value, and a name is enough to ride.

## The mechanism

`AdapterConfig.reference_carrier`, with `write_layer` set to the last block.

1. **Handles.** A handle is a deterministic function of the row's **stable knowledge identity** — the
   store's `kid`, mapped through a fixed untrained basis — and of nothing else: no payload, no link
   target, no marker, and **not the row's position**. Position keying was the first implementation and
   it was wrong: inserting, removing or compacting a row shifts every later row, so a handle already
   written into a cache would come to name a different pod and bind to its value, undetected. That is
   the ABA hazard the CAVI-N work exists to catch, reintroduced one layer down. Keyed by identity, a
   cached handle either still names its pod or names nothing that is present.
2. **The carrier participates.** At each read layer the routing distribution over cell keys selects a
   handle mixture, injected in place, RMS-matched, exactly where a payload write would go. It therefore
   takes part in the frozen computation and is written into every downstream K/V tensor.
3. **The address is knowledge-free.** The routing used is the resolve slot, whose keys are built from
   subject and relation only (`k_proj(ln_key(W_in[s] + rel_emb[r]))`). It is an alias address. The
   dereference chain still runs, in the hook, for routing supervision; its resolved value is not used.
4. **The value binds at the boundary.** After the last cache-writing block the handle is decoded back to
   a row distribution and bound to that row's **resolved** value — alias to pod resolution done
   store-side by `Bank.resolved_index`, from the store's own `trace_of_key`, with no model involved.
5. **Therefore** the persisted state depends on the prompt and on which rows exist under which
   (subject, relation) keys, and on nothing else that a lifecycle operation changes.

### Two invariances, and they are different in kind

This distinction matters more than it looks, and claiming the weaker one as the stronger would be false.

| Change | Effect on every persisted tensor | Why |
|---|---|---|
| payload UPDATE, alias RELINK, marker SHRED | **exactly 0.0** | the handles and the routing are literally unchanged tensors |
| reordering, compacting or growing the store | **7.5e-08** (float32 rounding; bounded at 1e-6 in the test) | permuting rows reduces the same softmax and mixture sums in a different order |
| DELETE of an identity | affects only references to **that** identity | other identities keep exactly the handles they had, which is what identity keying buys |

The first row is the claim. The second is a robustness property and is reported as numerical, not
bit-exact. The third answers the obvious objection that deletion changes the namespace and so must
invalidate everything: under position keying it would have; under identity keying it does not.

### By construction, and pinned by tests rather than asserted

These follow from the mechanism and are declared as pipeline rows, not claim rows
(`so/tests/test_write_layer.py`, 19 tests):

- handles are an untrained buffer, do not move when banks are read, are a function of the identity
  alone, and do not crowd (closest pair 0.0225 against a norm of 0.107 on the tested identities);
- a payload UPDATE, an alias RELINK and a SHRED of every marker each leave every persisted K/V tensor
  bit-identical while the answer at the last position changes;
- a store reordering does not rebind, asserted in the same test against the exactly-zero UPDATE row;
- a handle for a removed identity is reproduced by no surviving identity;
- the carrier participates: persisted K/V differs from the no-memory forward, where arm C's provably
  does not;
- `reference_carrier` without a `write_layer` is refused.

### Measured on the substrate, not asserted

On frozen GPT-2 through the experiment harness (`so/experiments/e000084_deep_read_late_write.py`, arm E,
16 held-out alias prompts at template 9): exposure against the no-memory forward 2.31 max-abs, so the
carrier is material; `update_payload`, `relink` and `shred_markers` each 0.0 K/V max-abs with the answer
moving. This is a short run of the harness, not the capability screen.

### Not yet known, and the claim depends on it

Whether a reader trained this way holds the unchanged gate: held-out candidate and full-vocabulary
correctness at least 0.95 on every one of the four held-out templates, on three seeds. Arm E is running
on seeds 0, 1, 2. **If it fails, this claim reduces to the trade-off measurement above and nothing more.**

## What the Symlink, the Pod and J-space each do here

**Symlink.** The claim is not available without it. A knowledge-free carrier needs a name that is not a
value, and the alias row is exactly that: the resolve slot addresses the alias, and the alias-to-pod
step is a store-side relation that never enters the frozen computation. Where CAVI-N needed the
reference-versus-referent distinction to detect stale state, here it removes the staleness: a relink
changes a store relation, and no persisted tensor was ever a function of it.

**Pod.** The unit whose value binds at the boundary and whose lifecycle operations are O(1) row writes.
One pod operation reaches every alias, as E-000061 and E-000064 already measured, and now reaches every
cached prefix as well, without touching one.

**J-space.** Audit only, never optimised, as the record requires. One consequence is worth stating:
under pure late binding every internal site is bit-identical to the never-memory control, so an internal
causal audit has nothing to separate and the audit plane is empty. The reference carrier restores that
plane — the handles do participate — while guaranteeing that what any internal audit can see is a name
and never a value. That is a prediction the audit can falsify, not a claim it supports.

## Prior-art boundary

**The targeted search for this specific design has not been run.** The workflow that was to run it
(verified web and patent search per candidate) died on a session limit before its search agents
completed. What follows is the boundary from the programme's own record; treat every line as a lead,
and the absence of a collision as unestablished rather than as clearance.

Not claimed, individually or in composition:

- canonical records, aliases, pointers, symlinks, MVCC, versions, epochs, capabilities, leases,
  snapshot isolation, cache invalidation, dependency tracking (verified in the record; PAMSPEC I-D,
  Codd, SQL-92, Redell 1974, Gray and Cheriton 1989);
- external or editable memory read by a frozen model — SERAC, GRACE, WISE, Larimar, KBLaM, MUNKEY,
  LMLM, SILO, kNN-LM, Facts/Entities-as-Experts (verified in the record);
- trainable KV-like memory with real-time insert, modify and delete — IBM `US20260119893A1`;
  learned side-channel layer intervention — ETRI `US20260105279A1` (both verified in the record);
- editable, erasable or selectively recomputed KV — Leyline `arXiv:2606.01065`, KVEraser
  `arXiv:2606.17034`, CacheBlend `arXiv:2405.16444`, AgentKVShift `arXiv:2607.21604`, ReCache
  `arXiv:2608.19662`, KV Packet `arXiv:2604.13226` (verified in the record);
- J-space and the Jacobian lens, and J-lens as an unlearning audit — Gurnee et al. 2026, J-Access
  `arXiv:2608.11408` (verified in the record);
- delexicalisation, placeholder tokens, copy and pointer networks, generative retrieval with atomic
  document identifiers (listed as owned in the record; **not re-verified this session**);
- store-side relation following as sparse-matrix products — Cohen et al., ICLR 2020, `arXiv:2002.06115`
  (**unverified this session**); knowledge fused at the head — K-ON `arXiv:2502.06257`, Memory Decoder
  `arXiv:2508.09874` (**unverified this session**).

The distinction being asserted, and the thing the search must be pointed at: every design above either
puts knowledge into the persisted state and then repairs it, or keeps knowledge out by not participating
at all. A carrier that participates while being knowledge-free — so that the cache is invariant under
every lifecycle operation with no metadata and no repair — is the point neither group occupies. A
verified collision on that sentence kills the claim.

## Relation to the programme's earlier candidates

- **CAVI-N** and **NDSR** asked how to revalidate authority for neural state derived from a memory read.
  Under a reference carrier the question does not arise for lifecycle changes: the derived state was
  never a function of the mutable fields, so there is no stale lineage to detect. What survives of them
  is the deletion case — removing a row changes the namespace — and the effect boundary of E-000082:
  once a memory-dependent token is committed, exact counterfactual revocation still needs rollback.
- **E-000079 / E-000080** built object-scoped lineage over derived state, and
  `cavi-qualified-prov.md` counterexample C showed a witness that names only the intended pod is
  unsound under a dense mixture. A cache with no knowledge in it has no dependency set to get wrong.
- **E-000082 / E-000083 / E-000084 arms A and C** are the measurements that make this the remaining
  seam rather than one option among several.

## Claim-killing tests

Withdraw the claim if any of these holds.

1. Arm E misses the unchanged gate on any of three seeds while arm A passes: participation by a
   knowledge-free carrier does not buy the capability, and only the trade-off measurement survives.
2. The lifecycle invariance is not exact on the trained reader at scale — any nonzero K/V max-abs under
   UPDATE, RELINK or SHRED.
3. A verified prior source specifies a knowledge-free carrier participating in a frozen model's
   persisted state with late value binding.
4. The invariance is real but useless: deletion of rows, which does change the namespace, dominates the
   workload, so caches must be invalidated anyway.
5. The reference carrier fails the unchanged battery the programme requires of any positive result —
   REVOKE and SHRED propagation at least 0.95, deleted-object leakage at most 0.02, missing-key UNKNOWN
   at least 0.90, active-memory generic KL at most 0.05 nats, exact BYPASS — or fails to replicate on a
   second public backbone.
6. An independent J-lens audit against never-memory controls finds a payload direction at a
   cache-writing block, which would mean the carrier is not knowledge-free after all.

## Honest assessment

The measured part is the trade-off: cache purity as specified costs the capability, on three seeds, with
exposure verified at exactly zero. That is a negative result about the programme's own mandatory
baseline and it is solid.

The mechanism part is structurally verified and empirically unproven. Its lifecycle invariance is exact
and tested, but exactness there is by construction — a handle that cannot depend on a payload cannot
propagate one — and the ledger's standing rule is that a result which follows from the code is a
pipeline row, not a claim. What would make it a claim is the one number nobody here can predict from the
code: whether a frozen GPT-2 can be trained to transport a random reference through its own attention
and have a linear boundary decoder recover it well enough to read at 0.95. Arm E is that measurement.

Two further cautions the record requires. The alternatives ranked below this one were all returned as
"nothing survives both kill modes" by their own generators — theorem-shaped organisers, lineage by
exhaustion, learned placement against repair cost, and every non-audit role for J-space — with verified
prior art in each case; that is evidence about the neighbourhood, not evidence for this candidate. And
the prior-art search for this design is outstanding, so the correct reading of this document today is a
well-posed falsifiable candidate with its experiment running, not a claim that has survived.
