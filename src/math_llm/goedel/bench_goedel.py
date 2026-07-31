"""
Benchmark Goedel-Prover-V2-32B on the full MiniF2F dataset.

Usage:
    python -m math_llm.bench_goedel [--samples N] [--output FILE] [--max-tokens N]

Saves incremental results to JSONL for crash recovery / resume.
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from math_llm.data import load_minif2f
from math_llm.lean_server import LeanServer


MODEL_ID = "Goedel-LM/Goedel-Prover-V2-32B"

PROMPT_TEMPLATE = (
    "Complete the following Lean 4 code:\n\n"
    "```lean4\n"
    "{}"
    "```\n\n"
    "Before producing the Lean 4 code to formally prove the given theorem, provide a "
    "detailed proof plan outlining the main proof steps and strategies.\n"
    "The plan should highlight key ideas, intermediate lemmas, and proof structures "
    "that will guide the construction of the final formal proof."
)


def extract_proof(response: str) -> str | None:
    """Extract proof tactics from Goedel model output.

    The model outputs a reasoning section followed by a lean4 code block
    where sorry is replaced by the actual proof. We extract the tactics
    that follow ':= by' in that block.
    """
    # Take the last lean4 code block (model may have multiple in reasoning)
    matches = list(re.finditer(r"```lean4?\s*\n(.*?)```", response, re.DOTALL))
    if not matches:
        return None

    code = matches[-1].group(1).strip()

    # Extract everything after the last ':= by'
    idx = code.rfind(":= by")
    if idx != -1:
        proof = code[idx + len(":= by"):].strip()
        return proof if proof else None

    # Term-mode proof: after ':='
    idx = code.rfind(":=")
    if idx != -1:
        proof = code[idx + 2:].strip()
        # Skip 'by' prefix if present
        if proof.startswith("by "):
            proof = proof[3:].strip()
        return proof if proof else None

    return None


def load_done_ids(output_path: Path) -> set[str]:
    done = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["problem_id"])
                except Exception:
                    pass
    return done


def run_benchmark(
    n_samples: int | None = None,
    output_path: str = "results_goedel_minif2f.jsonl",
    max_new_tokens: int = 8192,
) -> None:
    output_file = Path(output_path)

    # Load problems
    print("[bench] Loading MiniF2F problems...")
    problems = load_minif2f(n_samples)
    print(f"[bench] Loaded {len(problems)} problems")

    # Resume support
    done_ids = load_done_ids(output_file)
    if done_ids:
        print(f"[bench] Resuming — {len(done_ids)} already done, skipping them")

    remaining = [p for p in problems if p.id not in done_ids]
    if not remaining:
        print("[bench] All problems already done.")
        _print_summary(output_file)
        return

    # Load model
    print(f"[bench] Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print("[bench] Model loaded")

    # Start Lean server
    print("[bench] Starting Lean server...")
    lean_server = LeanServer()
    lean_server.start()
    print("[bench] Lean server ready\n")

    total = len(problems)

    try:
        with open(output_file, "a") as out_f:
            for problem in remaining:
                idx = problems.index(problem) + 1
                print(f"[{idx}/{total}] {problem.id}")

                # Generate
                chat = [{"role": "user", "content": PROMPT_TEMPLATE.format(problem.statement)}]
                inputs = tokenizer.apply_chat_template(
                    chat,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(model.device)

                t0 = time.time()
                with torch.no_grad():
                    output_ids = model.generate(inputs, max_new_tokens=max_new_tokens)
                gen_time = time.time() - t0

                response = tokenizer.decode(
                    output_ids[0][inputs.shape[1]:],
                    skip_special_tokens=True,
                )

                proof = extract_proof(response)
                print(f"  proof extracted: {bool(proof)}  gen: {gen_time:.1f}s")

                # Verify
                success = False
                complete = False
                errors: list[str] = []
                verify_time = 0.0

                if proof:
                    t1 = time.time()
                    try:
                        lean_result = lean_server.check_proof(problem.statement, proof)
                        verify_time = time.time() - t1
                        success = lean_result.success
                        complete = lean_result.complete
                        errors = lean_result.errors
                        status = "COMPLETE" if complete else ("OK" if success else "FAIL")
                        print(f"  lean: {status}  verify: {verify_time:.1f}s")
                        if errors:
                            print(f"  error: {errors[0][:100]}")
                    except Exception as e:
                        verify_time = time.time() - t1
                        errors = [str(e)]
                        print(f"  lean exception: {e}")
                else:
                    print("  no proof extracted")

                record = {
                    "problem_id": problem.id,
                    "success": success,
                    "complete": complete,
                    "proof": proof or "",
                    "errors": errors,
                    "gen_time": gen_time,
                    "verify_time": verify_time,
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

    finally:
        lean_server.stop()

    _print_summary(output_file)


def _print_summary(output_file: Path) -> None:
    if not output_file.exists():
        return

    results = []
    with open(output_file) as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except Exception:
                pass

    n = len(results)
    if n == 0:
        return

    n_complete = sum(1 for r in results if r["complete"])
    n_success = sum(1 for r in results if r["success"])
    n_extracted = sum(1 for r in results if r["proof"])
    avg_gen = sum(r["gen_time"] for r in results) / n

    print(f"\n{'='*60}")
    print(f"MiniF2F Results  —  {MODEL_ID}")
    print(f"{'='*60}")
    print(f"Total problems : {n}")
    print(f"Proof extracted: {n_extracted} ({n_extracted/n*100:.1f}%)")
    print(f"Lean success   : {n_success} ({n_success/n*100:.1f}%)")
    print(f"Lean complete  : {n_complete} ({n_complete/n*100:.1f}%)")
    print(f"Avg gen time   : {avg_gen:.1f}s")
    print(f"Results saved  : {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Goedel-Prover-V2-32B on MiniF2F")
    parser.add_argument("--samples", "-n", type=int, default=None, help="Limit number of problems")
    parser.add_argument("--output", "-o", default="results_goedel_minif2f.jsonl", help="Output JSONL file")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max new tokens per generation")
    args = parser.parse_args()

    run_benchmark(
        n_samples=args.samples,
        output_path=args.output,
        max_new_tokens=args.max_tokens,
    )
