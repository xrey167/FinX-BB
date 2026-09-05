# E-000054 — two-token subjects: identity at position 0, or the single-token collapse

Seeds [0, 1, 2], 3000 steps per surface and seed, trained without a BOS; 100 targets per
seed for the decomposition. Only the string a subject is rendered as changes; keys, objects and the trainer
are E-000050-B's. Worst seed everywhere.

| surface | reading | held-out subject-initial read / route | held-out subject-medial read / route | trained subject-initial / medial read_min | held-out reading | trained reading | mean generic KL (worst seed) |
|---|---|---|---|---|---|---|---|
| product | bare | 0.91 / 0.90 | 0.94 / 0.90 | 0.96 / 0.98 | 0.9475 | 0.9900 | 3.899 |
| product | <|endoftext|> at inference | 0.99 / 0.99 | 0.74 / 0.69 | 0.99 / 0.80 | 0.9275 | 0.9763 | 4.084 |
| product | a lone space at inference | 0.99 / 0.99 | 0.93 / 0.90 | 1.00 / 0.96 | 0.9813 | 0.9950 | 3.951 |
| second | bare | 1.00 / 0.97 | 0.97 / 0.96 | 1.00 / 0.98 | 0.9950 | 0.9975 | 3.923 |
| second | <|endoftext|> at inference | 0.99 / 0.97 | 0.90 / 0.85 | 0.99 / 0.82 | 0.9688 | 0.9819 | 4.012 |
| second | a lone space at inference | 1.00 / 0.98 | 0.97 / 0.97 | 1.00 / 0.97 | 0.9938 | 0.9956 | 3.874 |
| single (the record) | bare / BOS (E-000050 A / B) | 0.37 / 0.54 ; 0.97 / 0.98 | 0.95 / 0.94 ; 0.70 / 0.64 | 0.75 / — ; — / — | 0.7288 ; 0.9175 | 0.9119 ; 0.9719 | 3.647 ; 3.920 |

## The rule, fixed before the run

Worst seed. VOID if either V row fails: the surface is not learnable at this budget and nothing else is read. With V and M holding: H1 and H2 both PASS -> the failure is IDENTITY at position 0 -- a multi-token subject whose first token is redundant is read bare, one whose first token is needed is not, and the census's 80% condition splits by whether the first token carries identity, which no benchmark records. H1 PASS, H2 FAIL -> any subject-initial multi-token surface fails for this learned router, so the 80% condition is live for adapters of this kind (and says nothing against ROME, which reads the last subject token). H1's bare row FAILS (product reads bare) with H2 PASS -> the single-token case is special: E-000050's failure is Yang et al.'s 0.36% condition and does not extend even to a product code. H1's recovery rows FAIL (product stays low under a space and a BOS) -> the product surface fails for a reason other than position 0 and no sink sentence is read. M failing on a surface -> that surface's rows are reported and not interpreted. The trained subject-initial rows and the space reading's medial price are reported and never scored.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| product/N/train_medial/read_min | >= 0.85 | 0.9800 | PASS |
| second/N/train_medial/read_min | >= 0.85 | 0.9800 | PASS |
| product/N/heldout_initial/read_min | <= 0.5 | 0.9200 | FAIL |
| product/S/heldout_initial/read_min | >= 0.9 | 0.9900 | PASS |
| product/B/heldout_initial/read_min | >= 0.9 | 0.9900 | PASS |
| second/N/heldout_initial/read_min | >= 0.9 | 1.0000 | PASS |
| product/N/heldout_medial/read_min | >= 0.9 | 0.9400 | PASS |
| second/N/heldout_medial/read_min | >= 0.9 | 0.9700 | PASS |
