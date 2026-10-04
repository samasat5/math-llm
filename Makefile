# Lean Proof Benchmark
# ====================
#
# Simple benchmark for LLM agents on Lean 4 theorem proving.
#
# Commands:
#   make lean-server    - Setup Lean server with Mathlib
#   make test           - Quick test (simple agent on dummy data)
#   make <dataset>-<agent>  - Run specific benchmark

.PHONY: help install lean-server test dummy-simple minif2f-simple minif2f-tier0-simple \
        minif2f-mintier1-simple categorize-errors case-study-false-critique \
        case-study-vague-doubt case-study-forged-error case-study-mathd188 clean

help:
	@echo "Lean Proof Benchmark"
	@echo "===================="
	@echo ""
	@echo "Setup:"
	@echo "  make install        - Install Python dependencies"
	@echo "  make lean-server    - Setup Lean server with Mathlib (~2GB, 10-20min)"
	@echo ""
	@echo "Benchmarks (naming: <dataset>-<agent>):"
	@echo "  make test           - Quick test (simple agent on dummy)"
	@echo "  make dummy-simple   - Simple agent on dummy data"
	@echo "  make minif2f-simple - Simple agent on minif2f-lean4 (10 samples)"
	@echo "  make minif2f-tier0-simple   - Simple agent, pass@10, 100 easiest tier-0 problems"
	@echo "  make minif2f-mintier1-simple - Simple agent, pass@10, 100-problem seed0 subset of AMC+AIME+IMO"
	@echo ""
	@echo "Analysis (Chapter 4 / Figure 4.1 failure taxonomy):"
	@echo "  make categorize-errors RESULTS=<results.json>  - Failure-mode histogram + category breakdown"
	@echo ""
	@echo "Case studies (Appendix B.4.3 - does the model defend a correct proof?):"
	@echo "  make case-study-mathd188         - Run all three false-challenge variants on mathd_numbertheory_188"
	@echo "  make case-study-false-critique   - Specific (false) technical claim"
	@echo "  make case-study-vague-doubt      - Vague doubt (\"are you sure?\")"
	@echo "  make case-study-forged-error     - Forged Lean compiler error"

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

# Dummy dataset
dummy-simple:
	poetry run python -m math_llm dummy simple

# MiniF2F dataset (10 samples)
minif2f-simple:
	poetry run python -m math_llm minif2f-lean4 simple --samples 10

# Simple agent, pass@10, over the 100 easiest tier-0 (MATH-sourced) problems
minif2f-tier0-simple:
	./scripts/run_tier0_simple.sh

# Simple agent, pass@10, over a reproducible random 100-problem subset
# (seed=0) of the competition tiers (AMC+AIME+IMO) - matches Table 4.1's
# "Tier 1" row. MODEL/MAX_TOKENS overridable like minif2f-tier0-simple.
minif2f-mintier1-simple:
	poetry run python -m math_llm minif2f-lean4 simple \
	  --min-tier 1 --seed 0 --samples 100 --k 10 --batch-size 5 \
	  --model $${MODEL:-Goedel-LM/Goedel-Prover-V2-8B} \
	  --max-tokens $${MAX_TOKENS:-20000}

# =============================================================================
# Analysis
# =============================================================================

# Failure-mode histogram + category breakdown for a results file (Figure 4.1).
# Usage: make categorize-errors RESULTS=outputs/some_results.json
categorize-errors:
	@test -n "$(RESULTS)" || (echo "Usage: make categorize-errors RESULTS=<results.json>" && exit 1)
	poetry run python scripts/categorize_errors.py $(RESULTS)

# =============================================================================
# Case studies: does the model defend a correct proof? (Appendix B.4.3)
# =============================================================================

case-study-false-critique:
	poetry run python scripts/contradiction_challenge_mathd_numbertheory_188.py

case-study-vague-doubt:
	poetry run python scripts/vague_doubt_challenge_mathd_numbertheory_188.py

case-study-forged-error:
	poetry run python scripts/forged_compiler_error_challenge_mathd_numbertheory_188.py

case-study-mathd188: case-study-false-critique case-study-vague-doubt case-study-forged-error

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf .cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
