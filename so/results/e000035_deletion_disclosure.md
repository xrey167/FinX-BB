# E-000035 — a pod's aliases are signposts, and a deletion leaves them pointing at it

Seeds [0, 1, 2], 100 pods per seed. No model, no checkpoint, no training:
the adversary reads `MVCCStore.bank()` and nothing else, and names every key a LINK row points
at that no row holds.

## What a deletion leaves behind

| store | deleted key disclosed | uniquely identified | candidate keys left | false positives | dangling before any deletion |
|---|---|---|---|---|---|
| canonical | 1.0000 | 1.0000 | 1.0 | 0.00 | 0.0 |
| duplicated | 0.0000 | 0.0000 | 1536.0 | 0.00 | 0.0 |

`dangling before any deletion` is the control: E-000015 puts pointers to nothing in the
training distribution on purpose, so some dangle without any deletion having happened, and
only the NEW ones are counted as disclosure.

The asymmetry is the finding. Deleting one of k duplicated copies leaves a store with k-1
copies and no trace of the operation — the adversary is left the whole key space. Deleting a
pod's object leaves every alias still pointing at it, and `MVCCStore.bank()` keeps that key
deliberately so the model has to discover the miss rather than be handed it. Each surviving
alias is therefore a signpost reading *a record stood here and is gone*.

## The closure inverts with the guarantee

| guarantee | canonical pod | duplicated |
|---|---|---|
| unreachable to the reader (E-000032) | 1.00 | 3.00 |
| no trace left in the bank (here) | 3.00 | 1.00 |

E-000032 measures the first row: how many records must go before no query yields the object.
This experiment measures the second: how many before the bank shows no evidence a deletion
happened there. The same two stores swap places. A pod's aliases are the signposts, so they
must go too; a duplicated store costs the one record you were removing anyway. Quoting only
the first row would be quoting the half that flatters the design.

## The mitigation, and what it costs

Blanking a dangling pointer's key closes the channel (1.0000) and makes every such pointer identical (1.0000). It also removes what E-000015's
alias criteria are about: with the key blanked, an alias to a removed target is
indistinguishable from an alias to key (0, 0), so discovering the miss stops being a
discovery. The trade is recorded as a number rather than argued.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| canonical/deleted_key_disclosed | >= 0.95 | 1.0000 | PASS |
| canonical/uniquely_identified | >= 0.9 | 1.0000 | PASS |
| duplicated/deleted_key_disclosed | <= 0.05 | 0.0000 | PASS |
| duplicated/candidate_keys_mean | >= 1000.0 | 1536.0000 | PASS |
| canonical/trace_closure_mean | >= 3.0 | 3.0000 | PASS |
| duplicated/trace_closure_mean | <= 1.0 | 1.0000 | PASS |
| blanked/channel_closed | >= 1.0 | 1.0000 | PASS |
| blanked/pointers_indistinguishable | >= 1.0 | 1.0000 | PASS |

## What this does not show

It is a property of this store's bank, not of canonicalisation in general: a store that
compacts its aliases on deletion, or that never exports the target key, has no such channel.
It says nothing about whether the disclosure matters, which is a threat-model question. And
it measures the bank an adversary can read, not what the model exposes to one who cannot.
