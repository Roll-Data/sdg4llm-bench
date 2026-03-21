# SDG4LLM-Bench Makefile
#
# All targets use uv for environment management.
# Set up your environment first:
#   uv venv
#   source .venv/bin/activate
#   make install-dev

PYTHON ?= python
UV     ?= uv

.DEFAULT_GOAL := help

.PHONY: help install install-dev lint lint-fix test test-quick \
        seed-corpus-build seed-corpus-push seed-corpus-test \
        eval-base benchmark diagnostic leaderboard \
        clean clean-runs

# ─── Help ─────────────────────────────────────────────────────────────────────

help: ## Show all available targets
	@echo "SDG4LLM-Bench Makefile"
	@echo ""
	@echo "Setup:"
	@echo "  make install         Install the package"
	@echo "  make install-dev     Install with all dev dependencies"
	@echo ""
	@echo "Code quality:"
	@echo "  make lint            Run ruff linter"
	@echo "  make lint-fix        Run ruff linter with auto-fix"
	@echo ""
	@echo "Testing:"
	@echo "  make test            Run all non-integration, non-GPU tests"
	@echo "  make test-quick      Same as test (alias)"
	@echo ""
	@echo "Seed corpus:"
	@echo "  make seed-corpus-build   Build seed corpus locally (./seed_corpus_local)"
	@echo "  make seed-corpus-push    Build + push to HuggingFace Hub"
	@echo "  make seed-corpus-test    Build with 100 samples per task (fast test)"
	@echo ""
	@echo "Benchmark:"
	@echo "  make eval-base           Evaluate base model (no fine-tuning) on all tasks"
	@echo "  make benchmark METHOD=<name> TEACHER=<model> SDG_DATA=<file.jsonl>"
	@echo "  make diagnostic METHOD=<name> SDG_DATA_DIR=<dir>"
	@echo ""
	@echo "Leaderboard:"
	@echo "  make leaderboard         Render LEADERBOARD.md from submissions/"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean           Remove build artifacts and cache"
	@echo "  make clean-runs      Remove ./runs/ directory"

# ─── Installation ─────────────────────────────────────────────────────────────

install: ## Install the package
	$(UV) pip install -e .

install-dev: ## Install with all dev dependencies (evalplus, pytest, ruff)
	$(UV) pip install -e ".[all]"

# ─── Code quality ─────────────────────────────────────────────────────────────

lint: ## Run ruff linter (check only)
	$(UV) run ruff check sdg4llm_bench/ tests/ scripts/
	$(UV) run ruff format --check sdg4llm_bench/ tests/ scripts/

lint-fix: ## Run ruff linter with auto-fix
	$(UV) run ruff check --fix sdg4llm_bench/ tests/ scripts/
	$(UV) run ruff format sdg4llm_bench/ tests/ scripts/

# ─── Testing ──────────────────────────────────────────────────────────────────

test: ## Run tests (skip integration and gpu)
	$(UV) run pytest tests/ -v -m "not integration and not gpu" --tb=short

test-quick: test ## Alias for test

test-integration: ## Run integration tests (requires network)
	$(UV) run pytest tests/ -v -m "integration" --tb=short

# ─── Seed corpus ──────────────────────────────────────────────────────────────

seed-corpus-build: ## Build seed corpus locally from upstream datasets
	$(PYTHON) scripts/build_seed_corpus.py --output-dir ./seed_corpus_local

seed-corpus-push: ## Build seed corpus and push to HuggingFace Hub
	$(PYTHON) scripts/build_seed_corpus.py \
		--push \
		--hf-repo sdg4llm-bench/seed-corpus-v1

seed-corpus-test: ## Build with 100 samples per task (fast sanity check)
	$(PYTHON) scripts/build_seed_corpus.py \
		--output-dir /tmp/seed_corpus_test \
		--max-samples 100

# ─── Benchmark ────────────────────────────────────────────────────────────────

eval-base: ## Evaluate the base model (no fine-tuning) on all tasks
	$(PYTHON) -m sdg4llm_bench.evaluate --base-model-only --tasks all \
		--output-dir ./runs/base_model_eval

benchmark: ## Run official benchmark: 1 SDG training run + eval + submit
	@if [ -z "$(METHOD)" ]; then echo "ERROR: METHOD is required. Usage: make benchmark METHOD=<name> TEACHER=<model> SDG_DATA=<file.jsonl>"; exit 1; fi
	@if [ -z "$(TEACHER)" ]; then echo "ERROR: TEACHER is required. Usage: make benchmark METHOD=<name> TEACHER=<model> SDG_DATA=<file.jsonl>"; exit 1; fi
	@if [ -z "$(SDG_DATA)" ]; then echo "ERROR: SDG_DATA is required. Usage: make benchmark METHOD=<name> TEACHER=<model> SDG_DATA=<file.jsonl>"; exit 1; fi
	./scripts/run_benchmark.sh "$(METHOD)" "$(TEACHER)" "$(SDG_DATA)" $(BENCHMARK_ARGS)

diagnostic: ## Run per-task diagnostic (8 training runs)
	@if [ -z "$(METHOD)" ]; then echo "ERROR: METHOD is required. Usage: make diagnostic METHOD=<name> SDG_DATA_DIR=<dir>"; exit 1; fi
	@if [ -z "$(SDG_DATA_DIR)" ]; then echo "ERROR: SDG_DATA_DIR is required."; exit 1; fi
	./scripts/run_diagnostic.sh "$(METHOD)" "$(SDG_DATA_DIR)" $(DIAGNOSTIC_ARGS)

# ─── Leaderboard ──────────────────────────────────────────────────────────────

leaderboard: ## Render LEADERBOARD.md from submissions/
	$(PYTHON) scripts/render_leaderboard.py \
		--submissions-dir submissions/ \
		--output-dir .
	@echo "Updated LEADERBOARD.md and leaderboard.json"

# ─── Config integrity ─────────────────────────────────────────────────────────

config-hash: ## Recompute and update integrity hash in configs/default.yaml
	$(PYTHON) scripts/update_config_hash.py

config-check: ## Verify integrity hash matches current config (CI mode)
	$(PYTHON) scripts/update_config_hash.py --check

# ─── Cleanup ──────────────────────────────────────────────────────────────────

clean: ## Remove build artifacts and cache
	rm -rf build/ dist/ *.egg-info/
	rm -rf .ruff_cache/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-runs: ## Remove ./runs/ directory (WARNING: deletes trained checkpoints)
	@echo "WARNING: This will delete all trained checkpoints in ./runs/"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf runs/
