# =============================================================================
# KG-ProWeight Makefile
# =============================================================================
# Convenience targets for the full experimental pipeline.
#
# Usage examples:
#   make install              # install kgproweight + dev deps
#   make prepare-corpus       # Step 0/1/2/3
#   make phase1               # generate silver trajectories
#   make phase2               # train PRM + alpha-gate
#   make phase3-sft           # supervised fine-tuning of the student
#   make phase3-ppo           # PPO + GAE + Critic (default on Pro 6000 96GB)
#   make eval-baselines       # all 6 baselines under RRF top-50
#   make eval-kgpw            # KG-ProWeight inference + alpha distribution
#   make eval-ablations       # paper §7 ablations (variants 1..5)
#   make rigor-ihr            # GPT-4o IHR judge (with kappa)
#   make rigor-data-eff       # data-efficiency curve
#   make rigor-variance       # theorem-2 empirical variance check
#   make summarize            # build all paper tables / figures
# =============================================================================

SHELL          := /bin/bash
PYTHON         ?= python
PIP            ?= pip
DATASETS       ?= hotpotqa 2wikimultihopqa musique
SEEDS          ?= 13 42 2024
SPLIT          ?= dev
GPU_ID         ?= 0
CHECKPOINT_DIR ?= checkpoints/kg_proweight_final/final

.DEFAULT_GOAL  := help

# ── Setup ────────────────────────────────────────────────────────────────────

.PHONY: install install-dev
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e ".[dev]"

# ── Data preparation ─────────────────────────────────────────────────────────

.PHONY: prepare-corpus build-dense-index build-bm25-index download-datasets prewarm dropout
prepare-corpus:
	$(PYTHON) scripts/prepare/00_convert_corpus.py

build-dense-index:
	bash scripts/prepare/01_build_dense_index.sh

build-bm25-index:
	bash scripts/prepare/02_build_bm25_index.sh

download-datasets:
	$(PYTHON) scripts/prepare/03_download_datasets.py --datasets $(DATASETS) --splits train dev

prewarm:
	$(PYTHON) scripts/prepare/04_prewarm_wikidata_cache.py --datasets $(DATASETS) d_dropout --split $(SPLIT)

dropout:
	$(PYTHON) scripts/prepare/05_build_d_dropout.py --sample_size 1000

# ── Training phases ──────────────────────────────────────────────────────────

.PHONY: phase1 phase2 phase3-sft phase3-ppo phase3-grpo
phase1:
	$(PYTHON) scripts/train/phase1_generate_silver.py --config configs/training/phase1_silver.yaml

phase2:
	$(PYTHON) scripts/train/phase2_train_prm.py --config configs/training/phase2_prm.yaml

phase3-sft:
	$(PYTHON) scripts/train/phase3_sft.py --config configs/training/phase3_sft.yaml

phase3-ppo:
	$(PYTHON) scripts/train/phase3_ppo.py --config configs/training/phase3_ppo.yaml

phase3-grpo:
	$(PYTHON) scripts/train/phase3_grpo.py --config configs/training/phase3_grpo.yaml

# ── Evaluation ───────────────────────────────────────────────────────────────

.PHONY: eval-baselines eval-kgpw eval-ablations
eval-baselines:
	$(PYTHON) scripts/eval/run_baselines.py --datasets $(DATASETS) --split $(SPLIT) --gpu_id $(GPU_ID) --seeds $(SEEDS)

eval-kgpw:
	$(PYTHON) scripts/eval/run_kg_proweight.py --datasets $(DATASETS) d_dropout \
	    --checkpoint $(CHECKPOINT_DIR) --split $(SPLIT) --gpu_id $(GPU_ID) --seeds $(SEEDS)

eval-ablations:
	$(PYTHON) scripts/eval/run_ablations.py --datasets $(DATASETS) --gpu_id $(GPU_ID) --seeds $(SEEDS)

# ── Rigor experiments ────────────────────────────────────────────────────────

.PHONY: rigor-ihr rigor-data-eff rigor-variance
rigor-ihr:
	$(PYTHON) scripts/eval/run_ihr_judge.py --datasets $(DATASETS) --split $(SPLIT)

rigor-data-eff:
	$(PYTHON) scripts/eval/run_data_efficiency.py --sizes 1000 2000 5000 10000 15000

rigor-variance:
	$(PYTHON) scripts/eval/run_variance_validation.py --max_steps 500

# ── Summary & cleanup ────────────────────────────────────────────────────────

.PHONY: summarize lint test clean
summarize:
	$(PYTHON) scripts/eval/summarize_results.py --datasets $(DATASETS) d_dropout

lint:
	ruff check kgproweight scripts tests
	black --check kgproweight scripts tests

test:
	pytest -v tests

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# ── Help ─────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Common targets:"
	@echo "  make install              install kgproweight + production deps"
	@echo "  make install-dev          install dev tooling (pytest / ruff)"
	@echo "  make prepare-corpus       Step 0: corpus format conversion"
	@echo "  make build-dense-index    Step 1: e5-base-v2 + FAISS Flat"
	@echo "  make build-bm25-index     Step 2: BM25s sparse index"
	@echo "  make download-datasets    Step 3: HotpotQA / 2WikiMultiHopQA / MuSiQue"
	@echo "  make phase1               Phase 1: Teacher distillation"
	@echo "  make phase2               Phase 2: PRM + alpha-gate"
	@echo "  make phase3-sft           Phase 3a: supervised fine-tuning"
	@echo "  make phase3-ppo           Phase 3b: PPO + GAE + Critic (default)"
	@echo "  make eval-baselines       Baseline evaluation under RRF top-50"
	@echo "  make eval-kgpw            KG-ProWeight evaluation"
	@echo "  make eval-ablations       Paper §7 ablations"
	@echo "  make rigor-ihr            IHR via GPT-4o LLM-as-Judge"
	@echo "  make rigor-data-eff       Data-efficiency curve"
	@echo "  make rigor-variance       Theorem-2 empirical validation"
	@echo "  make summarize            Build all paper tables/figures"
	@echo "  make test                 pytest suite"
	@echo "  make lint                 ruff + black --check"
