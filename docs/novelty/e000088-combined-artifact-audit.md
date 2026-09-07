# E-000088 — combined audit of the completed three-seed artifacts

Audited 2026-09-05. This is a read/verification of the previously completed run, not new training or a new invention. Run **33973442114**, source anchor **75de1de0adea29b28c3e94609de138684187c4ae**, completed successfully. Its three seed artifacts were downloaded and their ZIP SHA-256 digests checked against GitHub metadata.

| Seed | Fresh alias candidate accuracy | Fixed-template >=0.95 |
|---|---:|---|
| 0 | 0.975 | Pass |
| 1 | 0.965 | Pass |
| 2 | 0.950 | Pass |

All records use 3000 training steps and template9. The seed set is exactly {0,1,2}; each per-seed record contains one row and the structural checks pass. The wrapper's all_three_seed_gate is false in each individual artifact because each job receives only one seed, not because the combined rate fails. Combining all three records satisfies this narrowly defined candidate-space/fixed-template prerequisite. This does not rewrite E77's failed0.99/0.95/0.94 result.

The executed source was inspected: E88 calls E70.run after installing the strict marker contract. E70's fresh-read metric uses cand.argmax, not the full vocabulary logits, on an independently sampled alias world under one selected template. Therefore this result must NOT be promoted to a full-vocabulary/unseen-paraphrase lifecycle reading gate. No checkpoint rerun or independent training replication was done in this audit.

Two further inherited metric limitations remain visible in the source: bystanders_preserved checks retained row masks, not measured bystander-output invariance; exact_bypass compares two calls with bank=None, not a learned scope decision. Actual stale-alias row rejection matches an explicit row-rejection neural reference, but the stale row is rejected rather than a complete corrected descendant state being reconstructed. In seed1, the recorded cavi_pred is159 while old/new truths are4/20, demonstrating why row rejection must not be misreported as current-answer correctness or universal UNKNOWN behavior.

Artifact hashes:
- seed0, ID9972343734: ZIP1eaef4a264b6c537cd48d999dfd2a8e31c0d4f9f376c3e76f8ec2098cfc11a3b; JSON03d8e799d91c6a6a260ae43f804fa808a4dde20b7535724b0d536ec24bd596bf.
- seed1, ID9972352857: ZIP5cf816747b1e4abf141760b69b7baac3a5f217b5a95ba44b1e797ed21080b137; JSON9d4354c7b2a08387549063a54a6ecf1f357bc2a719a241466c2cdec0eac47604.
- seed2, ID9972354149: ZIP960b064c4881113f19aa5da5181592f69ff38c1d69d092f61654f6105adf6ccd; JSON2df39d316c5dd16ea5368040515a5850fca11c1c51cbd8b1e8feb56a8d7ac714.

The artifacts contain result JSON, not saved model weights or a complete source-hash manifest. The execution SHA and inspected workflow tie the results to the recorded source; this audit does not claim a fresh reproduction. Main and historical scientific files are untouched. All stronger application/novelty gates remain separate and unqualified by this record.
