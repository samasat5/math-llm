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
from training.congif import TrainingConfig

def run_benchmark(
    dataset: str,
    agent_type: str,
    n_samples: Optional[int] = None,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: str = "outputs",
    gpu: Optional[int] = None,
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
    print(f"{'='*60}\n")

    # Load data
    problems = load_data(dataset, n_samples)
    print(f"Loaded {len(problems)} problems\n")

    # Start Lean server
    print("Starting Lean server...")
    lean_server = LeanServer()
    lean_server.start()

    # Create agent
    if agent_type == "simple":
        agent = SimpleAgent(
            model_name=model_name,
            lean_server=lean_server,
            gpu=gpu,
        )
    elif agent_type == "tool":
        agent = ToolAgent(
            model_name=model_name,
            lean_server=lean_server,
            gpu=gpu,
        )
    # elif agent_type == "grpo":
    #     agent = Policy(
    #         model_name=TrainingConfig.model_name,
    #         lean_server=lean_server,
    #         gpu=gpu,
    #     )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. Use 'simple' or 'tool'")

    # Run benchmark
    results = []
    start_time = time.time()

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
            })

    total_time = time.time() - start_time

    # Stop Lean server
    lean_server.stop()

    # Compute stats
    n_total = len(results)
    n_success = sum(1 for r in results if r["success"])
    n_complete = sum(1 for r in results if r["complete"])
    avg_time = sum(r["time"] for r in results) / n_total if n_total > 0 else 0

    summary = {
        "dataset": dataset,
        "agent": agent_type,
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "total_problems": n_total,
        "successful": n_success,
        "complete": n_complete,
        "success_rate": n_success / n_total if n_total > 0 else 0,
        "complete_rate": n_complete / n_total if n_total > 0 else 0,
        "total_time": total_time,
        "avg_time_per_problem": avg_time,
        "results": results,
    }

    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total problems: {n_total}")
    print(f"Successful: {n_success} ({summary['success_rate']*100:.1f}%)")
    print(f"Complete: {n_complete} ({summary['complete_rate']*100:.1f}%)")
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg time/problem: {avg_time:.2f}s")
    print(f"{'='*60}\n")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # Sanitize model name for filename (replace / with -)
    model_slug = model_name.replace("/", "-")
    output_file = output_path / f"{dataset}_{agent_type}_{model_slug}_results.json"

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
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
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model to use (default: Qwen/Qwen2.5-7B-Instruct)",
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

    args = parser.parse_args()

    run_benchmark(
        dataset=args.dataset,
        agent_type=args.agent,
        n_samples=args.samples,
        model_name=args.model,
        output_dir=args.output,
        gpu=args.gpu,
    )


if __name__ == "__main__":
    main()
