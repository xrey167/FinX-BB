# R305 Result — CKCA Zero-Touch Serving Overhead

Status: **local serving-overhead mechanism gate passed; not a production GPU benchmark and not DoD.**

Backbone: frozen `Qwen/Qwen2.5-0.5B` on the GitHub Actions CPU runner.

## Result

For a non-governed request, CKCA performs a route-table miss and then executes the exact unchanged B-plane model call. No Port or J-Space state is allocated.

- direct frozen-model forward median: **6,920.12 ms**;
- CKCA no-Port wrapper forward median: **6,922.73 ms**;
- observed wrapper overhead: **0.0378%**;
- no-Port route miss/control median: **120 ns**;
- active route + authority HMAC + lifetime-factor check median: **2,514 ns**;
- no-Port full-vocabulary logit delta versus direct model: **0.0**.

Artifact ZIP SHA256: `989a935b6ff63a5274e416705963d04e36365e747388c2b2c4cdef3bd9cc9b45`.
Report SHA256: `ae5ae5a53fc012ec8a93a8d1cdf4282597245cbc7f7e2824bdd6dfdb0bce869d`.

## Architectural decision

Keep CKCA **conditional**. The ordinary language/skill path must not pay a permanent neural-memory branch cost. Governance is attached only to relations/entities routed into the knowledge plane.

The hot-path shape is therefore:

`ordinary request -> route miss -> unchanged B-plane`

or, for governed knowledge:

`route hit -> current authority capability -> O(1) lifetime factor -> Port/J execution`.

This directly addresses the earlier ~73% lifecycle-overhead problem at the architectural level: lifecycle work is not injected into every ordinary model token. The measured 0.0378% wrapper overhead is encouraging but cannot be generalized to GPU production serving; a fused GPU/runtime benchmark remains mandatory.
