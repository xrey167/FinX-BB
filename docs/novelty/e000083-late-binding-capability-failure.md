# E-000083 — fixed final-block late binding does not preserve reader capability

Date: 2026-09-05
Status: **decisive negative screen; not a breakthrough or novelty claim**

## What was run

GitHub Actions run `33966506365` on branch `research/e000082-revocation-local-persistence`,
commit `1805300cc77a716bc72beb6d96eae1f2b07e3ef9`, workflow `E000083 late-bound symlink reader`.
The E-000081 recipe (seed 2, 3000 steps, `SO_BOS=1`, `status_gated`, `use_links`, `n_deref=1`,
100 groups, alt-supervision 0.5) with one architectural change: `read_layers=(11,)`, i.e. the
routing query, the dereference chain and the payload write all on GPT-2's final block, after the
last cache-writing attention computation (the E-000082 cache-pure placement).

Artifacts downloaded 2026-09-05 13:57Z; ZIP SHA-256 verified against the digests GitHub reports:

| Artifact | ZIP SHA-256 |
|---|---|
| `e000083-c0.05` (id 9970463653) | `4f5a7c888f07846f75efd8a9886e311383d15506934d627a62c862b484e87b41` |
| `e000083-c0.15` (id 9970561259) | `f160da0d9ec89df9ab35a5bd69af94c1c7f6effe2c58d4d9c9eb3d581dee6871` |
| `e000083-c0.3`  (id 9970451768) | `6569971cb9f5d484b73774fca34774dde61b25e3ca65084100b142261be78c3e` |

## Result

Held-out candidate-set correctness, seed 2, templates 8–11 (template 9 is the historically strict
real-symlink form):

| consistency | t8 | t9 | t10 | t11 | mean | min | full-vocab min | strict gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.05 | 0.765 | **0.49** | 0.905 | 0.585 | 0.686 | 0.49 | 0.445 | FAIL |
| 0.15 | 0.765 | **0.41** | 0.91 | 0.59 | 0.669 | 0.41 | 0.39 | FAIL |
| 0.30 | 0.72 | **0.475** | 0.895 | 0.61 | 0.675 | 0.475 | 0.38 | FAIL |

The same recipe at `read_layers=(8, 10)` (E-000081, run `33965513527`, seed 2, consistency 0.15,
artifact `e000081-seed2-c0.15`, SHA-256 `edbf9d36…`) reads 0.98375 mean / 0.96 min (template 9:
0.96) and passes the strict gate. At consistency 0.05 and 0.30 the (8, 10) reader misses the
per-template bar narrowly (min 0.925 and 0.94) while its mean stays at 0.965.

The final-block placement therefore costs roughly 0.3 of held-out accuracy on average and collapses
on the subject-medial forms (t9, t11). The workflow's own job status was "success" because the
screen step is `continue-on-error`; the scientific verdict is in the JSON `strict_pass=false`.

## What this changes

E-000082 established that a payload injected after the final block leaves every persisted K/V
tensor bit-identical to the no-memory forward, on two public backbones. E-000082's note then
declared fixed late binding "a mandatory strong baseline" that any lifecycle mechanism must beat.
E-000083 shows that this baseline, as specified, is not available: the reader loses the capability
gate when everything it does is moved to the final block.

E-000083 confounds two placements. The **address** (the routing query taken from the last-token
residual of the read block, and the dereference chain driven by it) and the **write** (the resolved
payload added to the residual stream) were moved together. Which one needs depth is the question
that decides the seam:

- if only the address needs depth, "read deep, write late" recovers capability with exposure
  exactly 0.0 and is the corrected cache-pure baseline (kNN-LM-shaped; engineering);
- if the write needs depth — the frozen blocks after it must process the payload — then capability
  requires in-model participation, cache purity and capability are in genuine tension, and the
  participation/exposure frontier is a live mechanism question rather than a tautology.

E-000084 (`so/experiments/e000084_deep_read_late_write.py`, `AdapterConfig.write_layer`) separates
the two: arm C keeps the query and the dereference chain at blocks 8 and 10, injects nothing there,
and writes the summed read once after block 11; arm A is the E-000081 placement on the same seeds.

## What E-000084 then measured, including against this document's own preregistration

Both preregistered outcomes above were too coarse, and the arm that discriminates says so.

| Arm | Write placement | Blocks of processing after the write | Seed | Held-out candidate mean | Worst template | Gate |
|---|---|---:|---:|---:|---:|---|
| A | in place at 8 and 10 | 3 and 1 | 0 | 0.955 | 0.935 (full-vocab) | fail |
| A | in place at 8 and 10 | 3 and 1 | 1 | 0.990 | 0.960 | **pass** |
| A | in place at 8 and 10 | 3 and 1 | 2 | 0.9838 | 0.960 | **pass** |
| C | once after block 11 | 0 | 0 / 1 / 2 | 0.664 / 0.645 / 0.621 | 0.270–0.290 | fail |
| D | once after block 10 | 1 | 0 | 0.615 | 0.270 | fail |

Arm A: runs 33970654975 and 33982958930, artifacts verified (`e000084-armA-seed2`, SHA-256
`846eff99f46ca72669f81362861903dd357690ebb8ef6f0f532c0eba3f64c0e0`). Arm D seed 0: run 33982958930,
artifact `e000084-armD-seed0`, SHA-256 `fa08feb006ae6b872f170f7aaeb7fc598c16ebeddea763fa036ae72eb490e779`.
Seeds 1 and 2 of arm D are still running; one seed is not a result, and this row is provisional.

**Restoring one block of processing recovers nothing.** Arm D sits at 0.615, inside arm C's range and
nowhere near arm A. So "the frozen blocks after the write must process the payload", the second
preregistered outcome, is not what the data says either — one block of processing is worth nothing
here, and the write in arm D still reaches every downstream K/V (exposure 6.30).

What separates A from both C and D is the other thing C changed: in arm A the block-8 write is
already in the residual when the block-10 read computes its routing query. C and D both remove that,
and a contract test pins that they address bit-identically at every routing slot. On the evidence so
far the operative factor is **the first write's feedback into the second read**, not the depth of
processing after a write — which is a different mechanism from the one this document predicted, and
it needs arm D's remaining seeds before it is more than a two-point reading.

## What is not claimed

Late binding, final-layer adapters, decoupled read/write placement, cache purity, symlink routing
and paraphrase-consistency training are not claimed as novelty. E-000083 is a negative capability
screen on one seed and three consistency weights; the 3-seed fixed-configuration rerun the
programme requires before any positive CAVI interpretation is still outstanding for every placement.
