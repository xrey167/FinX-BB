# Symlink–Pod / J-space: evidence repair and the next useful technology test

Date: 2026-09-05. Status: completed instrument work and a read-only reassessment; **no breakthrough or novelty claim**.

## Evidence actually checked

The three E-000076 artifacts from workflow run `33958409436`, executing commit
`f2d3bcb237b31f2c3afb323bdc8fade9c5bcc43e`, were downloaded and their ZIP SHA-256
hashes checked against GitHub's artifact metadata. The workflow uses the E-000070
module, so the JSON records identify themselves as E-000070. That provenance is
preserved rather than silently relabelled.

| Seed | Fresh alias accuracy | Required | Strong capability gate | Legacy screening flag |
|---|---:|---:|---|---|
| 0 | 0.875000 | 0.95 | FAIL | true |
| 1 | 0.958333 | 0.95 | PASS | true |
| 2 | 0.854167 | 0.95 | FAIL | true |

The worst-seed capability is **0.8541666667**, not at least 0.95. The original
module deliberately uses a 0.60 screening floor. Thus the workflow correctly
reports its own screening success, but it does not test the stronger goal stated
in the research claim. No historical criteria or results were changed here.

All three records report zero difference between CAVI and the explicit row-mask
rejection reference. That supports the recorded implementation equivalence; it
is not independent evidence that a fresh-current answer is correct. On the one
selected stale-answer case per seed, the old answer is returned by the three
simpler baselines in seeds 0 and 1, but not seed 2. No population-wide recovery
rate can be inferred from those three selected cases.

The original `exact_bypass` code invokes `model(None, ...)` twice. This is a base
path repeatability control, not a test that the learned scope decision actually
chooses bypass for unrelated inputs. Likewise, retaining two row-mask bits is not
a measurement of retained-pod output locality. Fresh-current neural correctness
after relink, independent workspace attestation, and latency are not supplied by
these artifacts. They must remain **NOT_MEASURED**, not PASS.

The new `so/strict_evidence.py` reads these legacy records without mutation,
requires the complete expected seed set, enforces the stated 0.95 capability bar,
rejects invalid/non-finite rates, and reports the absent evidence explicitly. It
never promotes this legacy record format into a full breakthrough verdict.

## A concrete repair to the J-space audit

The reviewed `so/jlens.py` computes VJPs of the model's returned hidden states.
Under the fully frozen-backbone condition used by the adapter, with ordinary
integer token inputs and no other gradient-enabling hook, those states do not
require gradients. `torch.enable_grad()` alone does not create a differentiable
source. This failure condition was reproduced on a local frozen causal model.
This is a local instrument test, not a rerun of E-000063 on pretrained GPT-2.

`so/jlens_frozen.py` provides a separate, explicitly versioned replacement:

- It introduces a detached, gradient-enabled leaf at the selected decoder-block
  input without unfreezing, updating, or accumulating gradients into model weights.
- Source and target indexing are explicit. Source 9 is after block 8, whereas
  source 8 is before block 8. Same-layer/upstream and final-normalized targets are
  rejected under this estimator's declared convention.
- The sum of causal VJPs is divided by the count of valid source/target position
  pairs. Source-token and pair counts are reported separately.
- Raw mean vectors and unit directions are distinct. A direction-only dot product
  is not advertised as a calibrated vocabulary-logit ranking.
- Temporary hooks are removed on normal completion and errors. Training mode,
  invalid masks and inference-mode execution fail explicitly.

This implements established autograd/J-lens mathematics. It is **not a new J-lens
method, a neural memory mechanism, or a deletion certificate**.

Example integration in a new, separately named evaluation run:

```python
from so.jlens_frozen import estimate_frozen_jlens

gk.model.lm.eval()
jl = estimate_frozen_jlens(
    gk.model.lm, source=JL_SOURCE, token_ids=obj_token_ids,
    input_ids=corpus["input_ids"], attention_mask=corpus["attention_mask"],
    w_out=gk.model.lm.get_output_embeddings().weight, target=-2, batch_size=4,
)
jvec = jl.directions.cpu()  # Direction-only diagnostic, not token probabilities.
```

Use a non-shared model instance with no active memory injection during lens
estimation. The caller must confirm the model's hidden-state indexing convention.
Do not optimize the memory against the held-out audit.

## Tests completed now

**36 new tests passed locally**, CPU, Python 3.13.5, PyTorch 2.10.0+cpu,
NumPy 2.3.5, pytest 9.0.2. These are the new modules' tests, not the entire
repository test suite and not three pretrained-backbone replications.

The tests include analytic causal cross-position derivatives; finite-difference
checks over three initializations of nonlinear models and three actual PyTorch
attention-block stacks; weight/gradient preservation; padding, batching and corpus
replication invariance; positional and keyword block inputs; error cleanup; and
strict evidence assessment, including the actual E-000076 accuracy pattern.

An initial numerical check failed because its perturbation mask multiplied a
Python epsilon in float32 before conversion to float64. The test perturbation was
corrected to float64; its tolerance was not relaxed. The final analytic and
nonlinear checks pass. This correction is recorded so the test history is not
presented as an uninterrupted first-attempt success.

```bash
OMP_NUM_THREADS=1 python -m pytest \
  so/tests/test_jlens_frozen.py so/tests/test_strict_evidence.py -q

# Supply the three unmodified downloaded JSON records. Exit code 2 means the
# strong capability gate is not met; the report is still written.
python -m so.strict_evidence e76-seed0.json e76-seed1.json e76-seed2.json \
  --output e76_strict_assessment.json
```

No new pretrained GPT-2 training or evaluation is claimed. The downloaded
artifacts contain metrics, not reusable model checkpoints. Network access from
the local execution container was unavailable; the GitHub connector remained
usable. No paid compute, recurring job or heavyweight new training run was launched.

## The useful technology target: correct continuation, not merely rejected handles

The strongest practical next target is **fresh-current-world equivalence after a
legitimate knowledge edit**, with selective recomputation rather than a full
restart whenever a provably unaffected prefix can be reused.

For a fixed input x, fixed model/tokenizer and an explicitly declared execution
context, require:

`Run_after_edit(x, managed_cached_state) = Run_fresh(x, current_retained_memory)`

at the declared numerical tolerance. The reference retains every unrelated pod.
It is not the empty-bank/base-model output. Updates compare against the new value;
revocations compare against a freshly recomputed current world without the revoked
generation. This tests a useful result rather than just whether a permission
check returns false.

For nonlinear downstream computation, zeroing a future injection does not in
general undo an earlier injection. If `h_next = F(h_base + injected_value)`,
continuing from `h_next` is not equivalent to recomputing `F(h_base)`. Therefore a
managed cache can be safely reused only where its complete dependency set is still
valid, or where an independent argument proves the affected contribution absent.
The engineering candidate is to resume at the last unaffected computational
boundary and recompute every affected descendant. This is a proposed integration,
**not an implemented or novel result in this change**.

The Symlink matters because both reference binding and referent generation belong
to that dependency set. A relink can invalidate a reference-derived state without
changing the old referent. The Pod provides a single lifecycle identity. J-space
provides an independent, partial view of the resulting causal influence; it does
not replace dependency tracking or authorize cache reuse.

A linear lens A observing m directions cannot prove equality of d-dimensional
states when rank(A) < d: `A(h_post-h_reference)=0` leaves the kernel of A
unobserved. Nor does a token name uniquely identify a pod or a generation. The
same token can be legitimately supplied by another retained pod or the base
model. The audit must therefore compare matched interventions with the same
prompt and retained memory; disappearing token scores alone are insufficient.

The guarantee must explicitly cover the managed runtime after the declared
revocation completion point. It cannot revoke an already exported plaintext copy
or outputs already received by another party. Protected runtime invalidation,
physical erasure, and removal of pretrained knowledge are distinct claims.

## A bounded decisive evaluation

Save one reproducible checkpoint per seed, with model/tokenizer versions, BOS
policy and SHA-256. Separate model selection from the final evaluation set. First
establish the 0.95 capability bar on the prespecified aliases and unseen phrasings;
never choose only successful individual facts after seeing the answers.

Then compare full fresh recomputation, a conventional complete dependency-aware
cache implementation, and the proposed selective continuation under identical
legitimate edits. Use the same input, random-number state where relevant, surviving
memory, precision, and generation policy. Instrument actual memory reads and
recomputed layers/tokens. Measure complete distributions and retained-pod answers,
not only masks or the top token.

The decisive table must include fresh-current answer agreement; retained-pod
locality; actual scoped-path versus base-path equivalence on negatives; independent
J-space and full-state diagnostics against the matched reference; latency,
recomputed work and cache-memory cost. All results are conjunctive across at least
three seeds. A single-template result is not all-linguistic-path coverage.

For a useful outcome, match full-recompute correctness while saving meaningful
recomputation. For a novelty claim, also beat or differ substantively from the
conventional dependency-aware cache baseline. Failure of a deliberately weaker
pod-only check does not establish that distinction. A full-cache baseline that
matches the result at the same cost ends that particular novelty claim.

## Research boundary checked at primary sources

Gurnee et al. introduce the Jacobian lens and describe its incompleteness. Song et
al.'s J-Access uses it as an independent diagnostic and reports model-level recovery
prediction, not identification of which individual facts will recover. PAMSPEC
already separates authoritative versioned memory from derived state; it is an
Internet-Draft, not an IETF standard. CacheBlend already combines KV reuse and
selective recomputation for RAG. None establishes novelty for this proposal, and
none should be omitted from the baseline discussion. The search performed here is
not exhaustive and is not a patentability assessment.

Primary references:

- https://transformer-circuits.pub/2026/workspace/index.html
- https://arxiv.org/abs/2608.11408
- https://datatracker.ietf.org/doc/html/draft-infantado-agent-memory-architecture-00
- https://arxiv.org/abs/2405.16444
- https://docs.pytorch.org/docs/stable/generated/torch.no_grad

## Artifact provenance

Workflow: https://github.com/xrey167/FinX-BB/actions/runs/33958409436

| Seed | Artifact ID | ZIP SHA-256 |
|---|---|---|
| 0 | 9967786138 | dbf157fc957f10033a816fed10b14e11c8c3ab91c7970109b5ea8d5152220368 |
| 1 | 9967788181 | eba009f227ca46176c73e3932702a74663888cba70df9d1429acf488bbdaa23f |
| 2 | 9967875472 | 0d883f1ea24375bc72e31d3659acad5987b7cd986eebeab8f3bcb1f6aa8b5456 |

Reviewed branch base: `cb0b630220fa7637e72473a6eaf8079a910ff13f`.
Historical experiment and result files are unchanged by this work.
