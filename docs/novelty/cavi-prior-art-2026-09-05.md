# CAVI prior-art and falsification boundary — 2026-09-05

This note narrows the research claim. It is **not** a novelty or patentability opinion. Individual
ingredients below are treated as prior art unless a composed property survives direct baselines.

## Confirmed nearby work

| Work | What it already covers | Consequence for CAVI |
|---|---|---|
| STALE (arXiv:2605.06527, 2026-05-07) | stale-memory detection, state resolution, premise resistance, write-side state adjudication (CUPMem) | “detect stale memory” is not new |
| MemTX (arXiv:2607.23929, 2026-07-27) | snapshot-isolated belief transactions, validity/provenance, validate-and-commit, cascade repair | transactions/MVCC/validity/commit discipline are not new |
| Commit-Time Authorization (arXiv:2607.10487, 2026-07-11) | freshness/binding/eligibility revalidation at durable-effect commit boundary; CommitGuard | freshness witnesses, epochs and final-boundary authorization are not new |
| DKME (Findings ACL 2026.792) | decoupled semantic addressing + partitioned knowledge-memory storage | address/storage separation is not new |
| SERAC | external counterfactual memory + learned scope classifier | external memory + scope routing is not new |
| WISE | dual parametric memory, routing, sharding for lifelong editing | modular/sharded edit memory is not new |
| Knowledge Externalization (ICLR 2026) | removable/editable external memory tokens and reversible knowledge restoration | externalized editable knowledge objects are not new |
| J-Access (arXiv:2608.11408) | Jacobian-lens audit of residual knowledge; optimizing the audit causes evasion | J-space/J-lens stays an independent audit, never a training target |
| crypto-shredding / secure deletion | key destruction and versioned secure-deletion graphs | HMACs, keys, epochs, key erasure and “crypto shred” are not new |
| canonical pointer/reference systems | canonical objects, aliases/pointers, generations, dangling references | pointers, canonical ids, refcounts and generation counters are not new individually |

Sources used in this update:
- https://arxiv.org/abs/2605.06527
- https://arxiv.org/abs/2607.23929
- https://arxiv.org/abs/2607.10487
- https://aclanthology.org/2026.findings-acl.792/
- https://proceedings.iclr.cc/paper_files/paper/2026/hash/7e9c2053258b1bdd32ff2654802cd594-Abstract-Conference.html
- https://arxiv.org/abs/2608.11408
- https://www.syssec.ethz.ch/publications/2013-11-04-secure-data-deletion-from-persistent-media-20-500-11850-74089/

## Remaining candidate composed property

The defensible target is deliberately narrower than “versioned memory” or “symlink pods”:

> **Causally Attested Versioned Indirection (CAVI):** a neural-memory consumption protocol in which
> pointer-only aliases and canonical knowledge pods are independently versioned; cached resolution and
> serialized neural-memory material are non-authoritative; both the reference binding and referent
> incarnation/reachability are revalidated adjacent to actual neural consumption; out-of-scope requests
> take an exact no-memory BYPASS while in-scope stale/missing references take UNKNOWN; and an independent
> causal/J-lens audit checks deletion without being optimized.

This only remains interesting if the *composition* yields a property simpler baselines cannot match.

## Critical falsification matrix

### F1 — alias relink while both pods stay live

A cached alias A→P is captured. A is relinked to Q, while P and Q remain live and P's incarnation is
unchanged.

- no guard: should replay A→P
- commit/export-time authorization only: should replay A→P after a resolve→mutate→consume race
- pod-only consume-time version check: should replay A→P because P is still current
- full CAVI alias+pod consume validation: must reject the old A binding

If a simple pod-only check rejects this attack just as reliably without separately validating alias
binding, the claimed CAVI distinction is falsified.

### F2 — canonical pod update / SHRED / DELETE / ABA

Old serialized rows and old resolver witnesses must fail after incarnation change, including delete +
recreate under the same logical identity. Pod-only checks are expected to pass this control; CAVI cannot
claim these controls as unique.

### F3 — exact scope semantics

- BYPASS: no-memory path must be exactly the base path (no learned null injection)
- RESOLVE: current alias+pod remains usable
- UNKNOWN: an in-scope stale/missing reference does not silently become BYPASS

### F4 — bystander preservation

Invalidating one alias/pod must not discard the entire Bank. Fresh unrelated pods must remain consumable.
This is why E-000069's whole-Bank rejection is a control rather than the final CAVI mechanism.

### F5 — cached/serialized/racy access paths

Test old resolver output, serialized tensors, old aliases, rollback/restore, ABA, concurrent relink/update,
and mutation in the resolve→inject gap. The final validation must occur at the neural memory read boundary,
not merely when the Bank is exported.

## Promotion rule

E-000070 is only a screening experiment. A positive result is promoted to multiple seeds, another
pretrained model, joint performance bars, E-000063 composed certificate integration, and reconstruction /
J-Access-style independent audits. No individual positive row is a novelty claim.
