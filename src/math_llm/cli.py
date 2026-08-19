"""
CLI for Lean proof benchmarking.

Usage:
    python -m math_llm <dataset> <agent> [--samples N]

Examples:
    python -m math_llm dummy simple
    python -m math_llm dummy tool
    python -m math_llm minif2f-lean4 simple --samples 10
    python -m math_llm minif2f-lean4 tool --samples 10
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from math_llm.data import load_data, list_datasets
from math_llm.lean_server import LeanServer
from math_llm.agents import SimpleAgent, ToolAgent
from math_llm.training.config import TrainingConfig

def run_benchmark(
    dataset: str,
    agent_type: str,
    n_samples: Optional[int] = None,
    model_name: str = "Goedel-LM/Goedel-Prover-V2-8B",
    output_dir: str = "outputs",
    gpu: Optional[int] = None,
    k: int = 1,
    offset: int = 0,
    tier: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
) -> dict:
    """
    Run benchmark on a dataset with specified agent.

    Args:
        dataset: Dataset name ('dummy' or 'minif2f-lean4')
        agent_type: Agent type ('simple' or 'tool')
        n_samples: Number of samples (None = all)
        model_name: Model to use
        output_dir: Directory for output files
        gpu: Force specific GPU device (None = auto)
        k: Number of samples per problem for pass@k
        offset: Number of problems to skip from the start
        tier: Difficulty tier filter, minif2f-lean4 only (0=MATH/mathd, 1=AMC, 2=AIME, 3=IMO)
        max_new_tokens: Max tokens to generate per sample (None = agent default)

    Returns:
        Benchmark results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Lean Proof Benchmark")
    print(f"{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"Agent: {agent_type}")
    print(f"Model: {model_name}")
    print(f"Samples: {n_samples or 'all'}")
    print(f"Offset: {offset}")
    print(f"Tier: {tier if tier is not None else 'all'}")
    print(f"Pass@k: {k}")
    print(f"Max new tokens: {max_new_tokens or 'agent default'}")
    print(f"{'='*60}\n")

    # Load data
    problems = load_data(dataset, n_samples, offset=offset, tier=tier)
    print(f"Loaded {len(problems)} problems\n")

    # Start Lean server
    print("Starting Lean server...")
    lean_server = LeanServer()
    lean_server.start()

    # Create agent
    agent_kwargs = dict(model_name=model_name, lean_server=lean_server, gpu=gpu, k=k)
    if max_new_tokens is not None:
        agent_kwargs["max_new_tokens"] = max_new_tokens

    if agent_type == "simple":
        agent = SimpleAgent(**agent_kwargs)
    elif agent_type == "tool":
        agent = ToolAgent(**agent_kwargs)
    # elif agent_type == "grpo":
    #     agent = Policy(
    #         model_name=TrainingConfig.model_name,
    #         lean_server=lean_server,
    #         gpu=gpu,
    #     )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. Use 'simple' or 'tool'")

    # Output path (computed upfront so we can save incrementally as we go)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_slug = model_name.replace("/", "-")  # sanitize model name for filename
    tier_suffix = f"_tier{tier}" if tier is not None else ""
    maxtok_suffix = f"_maxtok{max_new_tokens}" if max_new_tokens is not None else ""
    output_file = output_path / f"{dataset}_{agent_type}_{model_slug}_k{k}{tier_suffix}{maxtok_suffix}_results.json"

    # Run benchmark
    results = []
    start_time = time.time()

    def build_summary(status: str) -> dict:
        n_total = len(results)
        n_success = sum(1 for r in results if r["success"])
        n_complete = sum(1 for r in results if r["complete"])
        avg_time = sum(r["time"] for r in results) / n_total if n_total > 0 else 0
        return {
            "dataset": dataset,
            "agent": agent_type,
            "model": model_name,
            "k": k,
            "tier": tier,
            "max_new_tokens": max_new_tokens,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "total_problems_planned": len(problems),
            "total_problems": n_total,
            "successful": n_success,
            "complete": n_complete,
            "success_rate": n_success / n_total if n_total > 0 else 0,
            "complete_rate": n_complete / n_total if n_total > 0 else 0,
            f"pass@{k}": n_complete / n_total if n_total > 0 else 0,
            "total_time": time.time() - start_time,
            "avg_time_per_problem": avg_time,
            "results": results,
        }

    def save_summary(status: str) -> dict:
        summary = build_summary(status)
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    for i, problem in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] {problem.id}")
        print(f"  Statement: {problem.statement[:80]}...")

        problem_start = time.time()
        try:
            result = agent.solve(problem)
            problem_time = time.time() - problem_start

            status = "COMPLETE" if result.complete else ("OK" if result.success else "FAIL")
            print(f"  Result: {status} ({problem_time:.2f}s)")
            if result.proof:
                print(f"  Proof: {result.proof[:60]}...")

            results.append({
                "problem_id": problem.id,
                "success": result.success,
                "complete": result.complete,
                "proof": result.proof,
                "time": problem_time,
                "error": result.error,
                "attempts": result.attempts,
            })

        except Exception as e:
            problem_time = time.time() - problem_start
            print(f"  Error: {e}")
            results.append({
                "problem_id": problem.id,
                "success": False,
                "complete": False,
                "proof": "",
                "time": problem_time,
                "error": str(e),
                "attempts": None,
            })

        save_summary("in_progress")

    total_time = time.time() - start_time

    # Stop Lean server
    lean_server.stop()

    summary = save_summary("complete")

    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total problems: {summary['total_problems']}")
    print(f"Successful: {summary['successful']} ({summary['success_rate']*100:.1f}%)")
    print(f"Complete: {summary['complete']} ({summary['complete_rate']*100:.1f}%)")
    print(f"Pass@{k}: {summary[f'pass@{k}']*100:.1f}%")
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg time/problem: {summary['avg_time_per_problem']:.2f}s")
    print(f"{'='*60}\n")
    print(f"Results saved to: {output_file}")

    return summary


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Lean Proof Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m math_llm dummy simple           # Test simple agent on dummy data
  python -m math_llm dummy tool             # Test tool agent on dummy data
  python -m math_llm minif2f-lean4 simple --samples 10
  python -m math_llm minif2f-lean4 tool --samples 10
        """,
    )

    parser.add_argument(
        "dataset",
        choices=list_datasets(),
        help="Dataset to benchmark on",
    )
    parser.add_argument(
        "agent",
        choices=["simple", "tool", "grpo"],
        help="Agent type to use",
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=None,
        help="Number of samples to run (default: all)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Goedel-LM/Goedel-Prover-V2-8B",
        help="Model to use (default: Goedel-LM/Goedel-Prover-V2-8B)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)",
    )
    parser.add_argument(
        "--gpu", "-g",
        type=int,
        default=None,
        help="Force specific GPU device (default: auto)",
    )
    parser.add_argument(
        "--k", "-k",
        type=int,
        default=1,
        help="Number of samples per problem for pass@k (default: 1)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of problems to skip from the start (default: 0)",
    )
    parser.add_argument(
        "--tier", "-t",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help=(
            "Filter minif2f-lean4 problems by difficulty tier "
            "(0=MATH/mathd, 1=AMC, 2=AIME, 3=IMO); also sorts easiest-first "
            "within the tier. Default: no filter."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max new tokens to generate per sample (default: agent default)",
    )

    args = parser.parse_args()

    run_benchmark(
        dataset=args.dataset,
        agent_type=args.agent,
        n_samples=args.samples,
        model_name=args.model,
        output_dir=args.output,
        gpu=args.gpu,
        k=args.k,
        offset=args.offset,
        tier=args.tier,
        max_new_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
