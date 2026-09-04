# SO — reproduction targets. Every timing below was measured on a 4-core CPU box
# with no GPU; the per-model figures come from the 'train_seconds' field of the
# recorded results, so they are what this code actually took, not an estimate.
#
#   make test | smoke | synthetic | gpt2 | demo | compare | rescore | keychannel | untied | report
#
# PY       which interpreter to use          (default: python3, or .venv/bin/python if present)
# THREADS  torch threads per experiment      (default: all cores)
# SEEDS    seeds for the heavy targets       (default: 0 1 2)

PY ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
THREADS ?= $(shell nproc)
SEEDS ?= 0 1 2
RUN = OMP_NUM_THREADS=$(THREADS) SO_THREADS=$(THREADS) $(PY) -m

.PHONY: help test smoke synthetic gpt2 demo compare rescore certify closure retrieval pointers \
        disclosure keychannel untied report clean-results env

help:
	@echo "make test        unit tests, ~3 min (the deletion certificate sweeps its whole payload domain)"
	@echo "make smoke       reduced synthetic chain from scratch, ~35 min on 4 cores, writes *-quick records"
	@echo "make synthetic   recorded synthetic chain, ~3 h on 4 cores"
	@echo "make gpt2        frozen-GPT-2 chain, ~20 h on 4 cores, downloads GPT-2 once"
	@echo "make demo        watch one fact get deleted from a frozen GPT-2, ~3 min (needs a checkpoint)"
	@echo "make compare     deletion from weights vs from cells, head to head (needs a checkpoint)"
	@echo "make rescore     read the symlink checkpoints at all twelve templates, no training"
	@echo "make certify     prove the model cannot depend on a deleted payload, over its whole domain"
	@echo "make closure     how many records must go before a FACT is gone: canonical vs duplicated"
	@echo "make retrieval   the same closure in a chunked vector index, where practitioners meet it"
	@echo "make pointers    what the store gives away: a pointer separable from an object by its norm"
	@echo "make disclosure  what a deletion leaves behind: a pod's aliases point at what was removed"
	@echo "make keychannel  the channel SHRED does not close: recover a shredded object from the keys"
	@echo "make untied      the layer on a model that does not tie its embeddings (downloads Pythia-160m)"
	@echo "make report      rebuild docs/so-results-2026-09-02.md from so/results/"
	@echo "make env         print what will be used"
	@echo ""
	@echo "variables: PY=$(PY)  THREADS=$(THREADS)  SEEDS=$(SEEDS)"

env:
	@$(PY) -c "import sys, torch, numpy; print('python', sys.version.split()[0]); print('torch', torch.__version__); print('numpy', numpy.__version__); print('threads', torch.get_num_threads())"
	@echo "free disk:"; df -h . | tail -1
	@echo "note: the full gpt2 target writes about 500 MB of checkpoints into so/results/checkpoints/"

test:
	$(PY) -m pytest so/tests -q

# ------------------------------------------------- smoke: trains everything at a reduced budget
# Its models and records live in a -quick namespace, so it never reuses or overwrites a recorded one.
smoke:
	$(RUN) so.experiments.run_all --quick

# ------------------------------------------------------- synthetic chain: hours
# measured per model: E-000001-B 2.2 min, E-000002 1.2 min, E-000014 19.7 min,
# E-000015 7.7 min. Everything here is the small transformer trained from scratch.
synthetic:
	$(RUN) so.experiments.run_all
	$(MAKE) report

# ------------------------------------------------------------ GPT-2 chain: many hours
# measured per model: E-000008 20 min, E-000011 47 min, E-000012 48 min,
# E-000013 66 min, E-000017-B 42 min, E-000020 89 min. Three seeds each.
gpt2:
	$(RUN) so.experiments.e000008_gpt2_adapter --seeds $(SEEDS)
	$(RUN) so.experiments.e000011_gpt2_v2 --seeds $(SEEDS)
	$(RUN) so.experiments.e000012_status_gated_revoke --seeds $(SEEDS)
	$(RUN) so.experiments.e000013_prior_conflict --seeds $(SEEDS)
	$(RUN) so.experiments.e000017_paraphrase_gap --phase train --seeds $(SEEDS)
	$(RUN) so.experiments.e000017_paraphrase_gap --phase diagnose --seeds $(SEEDS)
	$(RUN) so.experiments.e000020_symlink_gpt2 --seeds $(SEEDS)
	$(RUN) so.experiments.e000025_template_rescoring --seeds $(SEEDS)
	$(RUN) so.experiments.e000026_lifecycle_at_a_readable_template --seeds $(SEEDS)
	$(RUN) so.experiments.e000024_weights_vs_cells --seeds $(SEEDS)
	$(MAKE) report

# the claim as a transcript on one fact, from a checkpoint the gpt2 target produces
demo:
	$(RUN) so.demo --threads $(THREADS)

# what the architecture buys: the same 50 facts deleted three ways and attacked identically.
# Trains a LoRA for the weights arms, so budget roughly 40 min per seed on 4 cores.
compare:
	$(RUN) so.experiments.e000024_weights_vs_cells --seeds $(SEEDS)

# no training at all: re-read the recorded symlink checkpoints at every template, ~7 min per seed
rescore:
	$(RUN) so.experiments.e000025_template_rescoring --seeds $(SEEDS)

# no training: prove the computation does not depend on the deleted payload, by sweeping every value
# it could hold. ~10 min; add --with-gpt2 for the frozen-LM arm.
certify:
	$(RUN) so.experiments.e000030_deletion_certificate --seeds $(SEEDS) --with-gpt2

# no training: the store-side half of an erasure guarantee, composed with the model-side certificate.
# Needs the recorded E-000015 one-slot checkpoints. ~15 min per seed on 4 free cores.
closure:
	$(RUN) so.experiments.e000032_deletion_closure --seeds $(SEEDS)

# no training: the same measurement in a chunked vector index with a frozen GPT-2 as the embedder,
# which is the arrangement almost every deployed system uses. ~10 min per seed.
retrieval:
	$(RUN) so.experiments.e000033_retrieval_closure --seeds $(SEEDS)

# no training for the diagnostic; --phase train adds the shared-projection arm (~40 min per seed)
pointers:
	$(RUN) so.experiments.e000034_pointer_separability --phase diagnose --seeds $(SEEDS)

# no model at all: whether an adversary reading the bank can name the key that was deleted. Seconds.
disclosure:
	$(RUN) so.experiments.e000035_deletion_disclosure --seeds $(SEEDS)

# no training: the channel SHRED does not close, against the recorded E-000010 checkpoints, ~3 min per seed
keychannel:
	$(RUN) so.experiments.e000028_key_channel --seeds 0 1 2 3 4

# the layer on a model that does NOT tie its embeddings; downloads Pythia-160m once, ~40 min per seed per arm
untied:
	$(RUN) so.experiments.e000027_untied_model --arm output --seeds $(SEEDS)
	$(RUN) so.experiments.e000027_untied_model --arm input  --seeds $(SEEDS)

report:
	$(PY) -m so.report

# results are the record; this only removes the reduced smoke output
clean-results:
	rm -f so/results/*-quick.json so/results/*-quick.md so/results/*-smoke.json so/results/*-smoke.md
