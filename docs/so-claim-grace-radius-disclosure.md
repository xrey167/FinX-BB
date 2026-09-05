# The claim: GRACE's deferral radius is a deletion oracle

*2026-09-05. One claim, its evidence, the code it is read off, and the prior art it has to survive.
Every line of GRACE quoted here was fetched at source in this session; every number is from
`so/results/e000059_grace_radius_disclosure.json`, three seeds, real frozen-LM activations.*

## Where it comes from — the vision's own rule, pointed outward

This is not a new idea. It is **E-000028's rule applied to somebody else's system.**

E-000028 (§31.10) found that SHRED gates a cell's *value* while the reverse routing key
`k_r = k_rev(LN(o + r))` is computed before the gate and is a function of the object being destroyed —
so a candidate sweep recovered the shredded object at top-1 **1.0000** while REVOKE and DELETE, which
take the row out of the addressable set, sat at chance **0.0039**. The ledger extracted the rule:

> a gate on values is not a deletion primitive if anything else in the computation is a function of the
> same payload; enumerate every derived quantity, or take the row out of the addressable set.

E-000035 (§31.19) added the pod half: a canonical record's surviving aliases name the key that was
removed, from the store alone, with no model in the loop — **canonicalisation makes erasure one
certifiable operation and turns every surviving access path into a deletion oracle.**

`docs/so-novelty-2026-09-04.md` §5 named porting this out as *"the smallest concrete result that would
make this matter outside the project"*, listed GRACE as a target, and recorded that it was never run.

The symlink–pod idea is what tells you where to look: **enumerate every quantity that is a function of
the removed record, including quantities stored on *other* records, and check whether the system's own
delete touches them.** In GRACE that lands immediately on the deferral radius.

## What GRACE does, quoted from the shipped source

Fetched 2026-09-05 from `easyeditor/models/grace/GRACE.py` in `zjunlp/EasyEdit`; the same rules are in
the official `thartvigsen/GRACE`. On an edit, with `smallest_distance, nearest_key = cdist(keys, query).min(0)`:

```python
if smallest_distance > (self.init_epsilon + self.epsilons[nearest_key]):
    ... = self.add_key(query, new_value, self.edit_id)
else:
    if not self.label_match(self.edit_label, self.key_labels[nearest_key]):
        ... = self.add_key(query, new_value, self.edit_id)
        self.split_epsilons_in_half(nearest_key, smallest_distance)
    else:
        if smallest_distance > self.epsilons[nearest_key]:
            if self.config.eps_expand == "coverage":
                self.epsilons[nearest_key] = smallest_distance

def split_epsilons_in_half(self, nearest_key, smallest_distance):
    self.epsilons[nearest_key] = (smallest_distance / 2) - 1e-5
    self.epsilons[-1] = smallest_distance / 2

def delete_key(self, edit_id):
    index_to_remove = self.edit_ids.index(edit_id)
    self.keys       = torch.cat((self.keys[:i],     self.keys[i+1:]),     dim=0)
    self.values     = ...
    self.epsilons   = torch.cat((self.epsilons[:i], self.epsilons[i+1:]), dim=0)
    self.key_labels = ...; self.edit_ids = ...
```

Three facts follow off the source:

1. **Split residue.** Inserting edit `k*` beside an existing key `k_j` with a different label writes
   `eps_j = ||k_j − k*||/2 − 1e-5` **onto the surviving record j**. `delete_key(k*)` slices every tensor
   at the deleted index and recomputes nothing. After a "successful" delete the codebook still
   satisfies, exactly, `||k_j − k*|| = 2·(eps_j + 1e-5)`.
2. **An edit GRACE cannot delete at all.** When the nearest key's label *matches*, no row and no
   `edit_id` are created — only `eps_nearest` grows to the absorbed edit's exact distance. `delete_key`
   for that edit finds nothing to remove, while its distance stays recorded on a neighbour. GRACE's
   advertised revert-by-entry-removal is undefined for these edits.
3. **The value channel is clean.** Key, value, label and id are gone, the frozen base model is
   untouched, and behaviour on the deleted edit reverts. That is exactly the E-000028 shape: payload
   removed, value channel at chance, recovery through a derived quantity the delete never touched.

## The attack, and what it is given

The adversary holds **only** the post-deletion codebook `(keys, epsilons, labels)`, the public frozen
base model, and a pool of N candidate edit prompts. It does **not** hold the pre-deletion codebook, the
deleted key or value, any query access to the edited model, or any training.

```
k_c      = h^(l-1)(c)                                   # one frozen forward per candidate
score(c) = min_j | ||k_c − k_j|| − 2·(eps_j + 1e-5) |   # candidates that ARE live keys excluded
recovered = argmin_c score(c)
```

Excluding candidates that are themselves surviving keys is not a convenience: the adversary holds the
codebook, so a live key is by definition not the deleted edit. Without that exclusion an *earlier*
split's constraint — whose partner is still present — is satisfied exactly and outranks the target.
That is the attack nominating a record it can already see, and it is not a leak.

## The measurement

`distilgpt2`, layer 3, keys are last-token hidden states, 256 candidate prompts (32 subjects × 8
relations), 24 edits, three seeds. Chance top-1 = 1/256 = **0.0039**.

| arm | what it changes | top-1 | rank of the deleted edit | residual |
|---|---|---:|---:|---|
| **attack** | the codebook exactly as GRACE leaves it | **1.000** | **1, 1, 1** | **0.00e+00** (runner-up 7.1e-04 … 2.1e-03) |
| `repaired` | radii recomputed from the surviving keys | 0.000 | 167, 109, 109 of 256 | — |
| `nosplit` | delete an edit that caused no split | 0.000 | 216 | — |
| `shuffled` | radii permuted, marginal preserved | 0.000 | 63 | — |

The deleted edit is recovered **exactly** — the constraint residual is 0.0 in float32 while the nearest
decoy sits three orders of magnitude away — and the three controls place the channel precisely:

- **`repaired` is the one that matters.** Recomputing each radius from the surviving keys — the single
  line GRACE does not do — sends the deleted edit to rank 109–167 of 256, i.e. to chance. So the channel
  *is* the un-repaired radius and nothing else.
- **`nosplit`**: an edit that never caused a split leaves no constraint and leaks nothing.
- **`shuffled`**: permuting the radii across rows destroys it, so it is the *pairing* of a radius to a
  key that carries the deleted key, not the radii's distribution.

Both control failures during development were mine, not the channel's, and both are recorded in the
experiment: the attack first nominated a still-present key, and the first `repaired` arm kept the old
radius whenever it was smaller — which preserves exactly the constraint it was meant to destroy.

## Why it is not owned

- **Ghost Vectors, arXiv:2606.18497** — the nearest neighbour, and different in kind. There a
  soft-deleted HNSW vector is *still physically stored* and Vec2Text inverts it. Here the deleted vector
  is genuinely gone from the tensor; the recovery runs through **a scalar belonging to a different,
  surviving record**. Their defence (encrypting the vector, rotating epoch keys) does nothing to a
  neighbour's radius.
- **Embedding inversion** (Song & Raghunathan; Morris et al.) recovers text from a *retained* embedding.
  No embedding of the deleted edit is retained here.
- **Tombstones and un-vacuumed index entries** in database and vector-store literature are about the
  record's own remains. This is about a constraint written onto another record at insertion time.

## What is not claimed

- No novelty for GRACE, key-value editors, codebooks, deferral radii, nearest-neighbour indexes,
  tombstones, embedding inversion, or the persistence of deleted data on a medium.
- No novelty for E-000028's rule, which is this repository's own prior work and is the input here.
- **Not an end-to-end EasyEdit run.** This reproduces the quoted codebook rules faithfully in ~60 lines
  on real frozen-LM activations; EasyEdit is not installed on this box. What is measured is the
  disclosure property of those rules. Confirming it against the package itself is the obvious next step
  and is not done here.
- Nothing about how often the split branch fires in a real editing workload. It fired on 2 of 24 edits
  under this configuration; a deployment where it never fires has no channel.

## What would kill it

1. The split branch is unreachable under the configurations practitioners actually use.
2. EasyEdit's released `delete_key` path is preceded, somewhere I have not read, by a radius repair.
3. `init_epsilon` in real use is large enough that the first branch always fires and no split is written.
4. The recovery does not survive keys taken from the layer GRACE actually adapts, at realistic edit
   counts, with a candidate pool the adversary could plausibly assemble.

## Disclosure

This is a defect in an open-source research tool, not a deployed product. The correct next step is to
report it to the EasyEdit and GRACE maintainers with the one-line repair — recompute each surviving
radius from the surviving keys inside `delete_key` — rather than to publish a recovery harness. Nothing
in this repository should be released as an exploit against a third-party package.
