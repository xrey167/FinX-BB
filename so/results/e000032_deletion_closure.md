# E-000032 — the deletion closure of a store, and the certificate that composes with it

Seeds [0, 1, 2], 25 alias groups per seed, the recorded E-000015 one-slot
checkpoints, no training. The reader is the E-000015 `MutableKnowledgeTransformer` -- trained
from scratch on a 256-entity world with an explicit UNKNOWN head, worlds resampled every
training step so no evaluation fact is in its weights -- and NOT the frozen GPT-2 adapter; an
earlier version of this report said GPT-2 and was wrong (ledger §31.33). Both arms are built
from the SAME world with the same ground truth, so they present an identical interface: every
key resolves to the same object in both.

## The gap a record-level certificate cannot see

| store | closure per KEY | closure per FACT | proved optimal | one record: fact certified | one record: still readable | predicted from the closure | whole closure: fact certified |
|---|---|---|---|---|---|---|---|
| canonical | 1.00 | 1.00 | 1.00 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| mixed | 1.00 | 2.00 | 1.00 | 0.0000 | 0.3333 | 0.3333 | 1.0000 |
| duplicated | 1.00 | 3.00 | 1.00 | 0.0000 | 0.6667 | 0.6667 | 1.0000 |

`closure per KEY` is how many records must go before THAT KEY stops answering. It is one in
both arms, which is the point: at the record level the two stores are indistinguishable.
`closure per FACT` is how many must go before NO key in the group yields the object, and it
is where they separate. `proved optimal` is the fraction where the greedy search MET a
certified lower bound (every live derivation is a must-hit set, so a pairwise-disjoint
subfamily bounds the optimum from below) rather than merely being assumed exact.

`one record: fact certified` removes exactly the object -- what a record-level certificate
covers today -- and asks whether that licenses a fact-level statement. `still readable` is
the model's own answer afterwards, so the verdict is confirmed by behaviour and not only by
bookkeeping. `predicted from the closure` is `(closure - 1) / keys_per_group`, computed from
the store before the model is run at all: removing only the object leaves exactly the copies
that are separate records.

What that agreement IS, stated after a review rather than before it (ledger §31.33): on the
star topologies these arms are built from, every non-target closure member backs exactly one
key, so the formula is that invariant restated and the 0.0000 measures the reader's FIDELITY
to the store's own resolver, which E-000015 had already recorded at 1.0000 on these
checkpoints. It is not a forecast. On a chain -- an alias pointing at a copy rather than at
the object -- the formula is wrong by a full grid step against the mechanical resolver with no
model in the loop (`test_closure_minus_one_over_keys_is_star_arithmetic_and_not_a_store_law`);
the quantity that IS a function of the store is the post-deletion resolver count, and
`certify_fact` already checks that one through `store_after`.

## What a certified fact deletion costs, counted

The model calls inside the certification window are COUNTED by wrapping `forward` and
`encode_bank` -- the first version of this report wrote `0` as a literal, while
`certify_encoding`'s reference fingerprint runs `encode_bank` once even over an empty row set.
The reachability control is the positive control for each fact's payload and runs once PER
FACT, so it belongs inside the per-fact cost; the first version timed it apart and called it
once per instrument. `all in` is search + certification + control.

| store | closure search (s) | certification (s) | forwards inside | standalone encodes inside | control (s) | control forwards | per fact, all in (s) |
|---|---|---|---|---|---|---|---|
| canonical | 0.0273 | 0.4803 | 0.0 | 1.0 | 0.10 | 1.0 | 0.58 |
| mixed | 0.0331 | 0.4679 | 0.0 | 1.0 | 0.04 | 1.0 | 0.51 |
| duplicated | 0.0390 | 0.4670 | 0.0 | 1.0 | 0.05 | 1.0 | 0.51 |

The store-side parts -- the closure search and the exhaustive store counterfactual -- run
no forward pass; the certification window as a whole is not model-free, and the number in
the table is what it costs.

E-000024 is the comparison: deleting 50 facts from a LoRA took 129 s by gradient ascent and
335 s by relabelling, changed 2,359,296 parameters, moved perplexity on ordinary prose from
42.9 to 6.19e+09 and 6.39e+06, and admits no certificate at all -- there is no finite payload
domain to sweep and no interface the data passes through.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| control/interface_identical | >= 1.0 | 1.0000 | PASS |
| control/read_before_deletion | >= 0.9 | 1.0000 | PASS |
| canonical/per_key_closure_max | <= 1.0 | 1.0000 | PASS |
| duplicated/per_key_closure_max | <= 1.0 | 1.0000 | PASS |
| canonical/fact_closure_max | <= 1.0 | 1.0000 | PASS |
| duplicated/fact_closure_min | >= 3.0 | 3.0000 | PASS |
| canonical/fact_closure_optimal_rate | >= 1.0 | 1.0000 | PASS |
| duplicated/fact_closure_optimal_rate | >= 1.0 | 1.0000 | PASS |
| canonical/control_reachable_before | >= 1.0 | 1.0000 | PASS |
| canonical/mediation_consistent | >= 1.0 | 1.0000 | PASS |
| mixed/mediation_consistent | >= 1.0 | 1.0000 | PASS |
| duplicated/mediation_consistent | >= 1.0 | 1.0000 | PASS |
| canonical/one_record_retained_in_store | >= 1.0 | 1.0000 | PASS |
| canonical/one_record_payload_store_absent | >= 1.0 | 1.0000 | PASS |
| duplicated/one_record_payload_store_absent | >= 1.0 | 1.0000 | PASS |
| duplicated/control_reachable_before | >= 1.0 | 1.0000 | PASS |
| canonical/one_record_fact_certified | >= 1.0 | 1.0000 | PASS |
| duplicated/one_record_fact_certified | <= 0.0 | 0.0000 | PASS |
| canonical/one_record_still_readable | <= 0.1 | 0.0000 | PASS |
| duplicated/one_record_still_readable | >= 0.6 | 0.6667 | PASS |
| canonical/whole_closure_fact_certified | >= 1.0 | 1.0000 | PASS |
| duplicated/whole_closure_fact_certified | >= 1.0 | 1.0000 | PASS |
| duplicated/whole_closure_still_readable | <= 0.1 | 0.0000 | PASS |
| mixed/fact_closure_mean | >= 2.0 | 2.0000 | PASS |
| mixed/fact_closure_max | <= 2.0 | 2.0000 | PASS |
| mixed/one_record_fact_certified | <= 0.0 | 0.0000 | PASS |
| canonical/prediction_error | <= 0.05 | 0.0000 | PASS |
| mixed/prediction_error | <= 0.05 | 0.0000 | PASS |
| duplicated/prediction_error | <= 0.05 | 0.0000 | PASS |

## What this is and is not

The gap between the two arms is Codd's MODIFICATION anomaly applied to a delete -- his
DELETION anomaly is the opposite failure, unintended loss -- and normalization is its 1971
remedy; this experiment does not claim otherwise. What it adds is that the anomaly decides whether a
DELETION CERTIFICATE for a neural memory means anything, that the store-side half of the
guarantee (closure search and store counterfactual) is computable without the model, and that in a neural memory the normalization is
not free -- E-000025 prices it at 0.0954 for sharing and 0.0688 for link training on a frozen
GPT-2, worst of three seeds across all twelve phrasings.
