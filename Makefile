# Lean Proof Benchmark
# ====================
#
# Simple benchmark for LLM agents on Lean 4 theorem proving.
#
# Commands:
#   make lean-server    - Setup Lean server with Mathlib
#   make test           - Quick test (both agents on dummy data)
#   make <dataset>-<agent>  - Run specific benchmark

.PHONY: help install lean-server test dummy-simple dummy-tool minif2f-simple minif2f-tool minif2f-tier0-simple clean

help:
	@echo "Lean Proof Benchmark"
	@echo "===================="
	@echo ""
	@echo "Setup:"
	@echo "  make install        - Install Python dependencies"
	@echo "  make lean-server    - Setup Lean server with Mathlib (~2GB, 10-20min)"
	@echo ""
	@echo "Benchmarks (naming: <dataset>-<agent>):"
	@echo "  make test           - Quick test (both agents on dummy)"
	@echo "  make dummy-simple   - Simple agent on dummy data"
	@echo "  make dummy-tool     - Tool agent on dummy data"
	@echo "  make minif2f-simple - Simple agent on minif2f-lean4 (10 samples)"
	@echo "  make minif2f-tool   - Tool agent on minif2f-lean4 (10 samples)"
	@echo "  make minif2f-tier0-simple - Simple agent, pass@10, 100 easiest tier-0 problems"

# =============================================================================
# Setup
# =============================================================================

install:
	poetry install

# Setup Lean server with Mathlib + REPL
lean-server:
	@echo "Setting up Lean server with Mathlib + REPL..."
	@echo "This downloads ~2GB and takes 10-20 minutes on first run."
	@echo ""
	./scripts/setup_mathlib.sh

# =============================================================================
# Benchmarks - naming: <dataset>-<agent>
# =============================================================================

# Quick test on dummy data
test:
	poetry run python -m math_llm dummy simple
	poetry run python -m math_llm dummy tool

# Dummy dataset
dummy-simple:
	poetry run python -m math_llm dummy simple

dummy-tool:
	poetry run python -m math_llm dummy tool

# MiniF2F dataset (10 samples)
minif2f-simple:
	poetry run python -m math_llm minif2f-lean4 simple --samples 10

minif2f-tool:
	poetry run python -m math_llm minif2f-lean4 tool --samples 10

# Simple agent, pass@10, over the 100 easiest tier-0 (MATH-sourced) problems
minif2f-tier0-simple:
	./scripts/run_tier0_simple.sh

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf outputs/
	rm -rf .cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
