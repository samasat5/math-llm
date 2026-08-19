"""
Tool agent: Iterative proof with Lean tool calls.

Uses the LLM to generate tactics step by step, calling the Lean server
to verify each step and get feedback (errors, remaining goals).
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from math_llm.data import Problem
from math_llm.lean_server import LeanServer, LeanResult
from math_llm.agents.simple import AgentResult, extract_proof


# System prompt for tool-based agent
SYSTEM_PROMPT = """You are a Lean 4 theorem prover with access to a Lean REPL tool.

You can call the lean_check tool to verify tactics. After each call, you'll see:
- SUCCESS: Tactic accepted, remaining goals shown
- ERROR: Tactic failed, error message shown
- COMPLETE: Proof finished!

Strategy:
1. Analyze the goal type
2. Try appropriate tactics based on goal structure
3. If a tactic fails, CHECK THE ERROR PRINTED and try other tactics. for example if the error is "unknown tactic", offer a tactic known in lean4 that solves the problem. 
3.1 when one tactic raises a lean error, do not use exactly the same tactic for the next step!
4. Build proof incrementally

Tactics reference:
- Finishing: rfl, norm_num, decide, ring, omega, linarith, positivity, ...
- Rewriting: rw [h], simp, ring_nf, field_simp, ...
- Structure: intro, obtain, rcases, use, constructor, left/right, ...
- Application: exact, apply, have, calc, refine, ...
- Other: ext, induction, cases, simp_all, ...  

Output format: When suggesting a tactic, output ONLY the tactic on a single line.
"""


@dataclass
class Step:
    """A single step in the proof trajectory."""
    tactic: str
    result: LeanResult
    step_num: int

    def to_dict(self) -> dict:
        return {
            "step": self.step_num,
            "tactic": self.tactic,
            "success": self.result.success,
            "complete": self.result.complete,
            "errors": self.result.errors,
            "goals": self.result.goals,
        }


@dataclass
class Trajectory:
    """Complete proof attempt trajectory."""
    problem_id: str
    steps: list[Step] = field(default_factory=list)
    success: bool = False
    complete: bool = False

    def add_step(self, tactic: str, result: LeanResult) -> None:
        step = Step(tactic=tactic, result=result, step_num=len(self.steps) + 1)
        self.steps.append(step)
        if result.complete:
            self.success = True
            self.complete = True

    @property
    def final_proof(self) -> str:
        """Get the combined proof from successful steps."""
        successful = [s.tactic for s in self.steps if s.result.success]
        return "\n".join(successful)

    def to_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "num_steps": len(self.steps),
            "success": self.success,
            "complete": self.complete,
            "steps": [s.to_dict() for s in self.steps],
            "final_proof": self.final_proof,
        }


class ToolAgent:
    """
    Tool-based iterative proof agent.

    Generates tactics step by step, using Lean server feedback
    to guide the proof search.
    """

    def __init__(
        self,
        model_name: str = "Goedel-LM/Goedel-Prover-V2-8B",
        lean_server: Optional[LeanServer] = None,
        max_steps: int = 20,
        max_new_tokens: int = 200,
        temperature: float = 0.5,
        gpu: Optional[int] = None,
        log_path: Optional[str] = None,
        k: int = 1,
    ):
        self.model_name = model_name
        self.lean_server = lean_server
        self.max_steps = max_steps
        self.k = k
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.gpu = gpu
        self.log_path = log_path
        self._model = None
        self._tokenizer = None

    def load_model(self) -> None:
        """Load the LLM model."""
        if self._model is not None:
            return

        print(f"[agent] Loading model {self.model_name}...")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            if self.gpu is not None:
                # Force specific GPU
                device_map = {"": self.gpu}
                torch_dtype = torch.bfloat16
            elif torch.cuda.is_available():
                device_map = "auto"
                torch_dtype = torch.bfloat16
            elif torch.backends.mps.is_available():
                device_map = "mps"
                torch_dtype = torch.float16
            else:
                device_map = "cpu"
                torch_dtype = torch.float32

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
            )

            print(f"[agent] Model loaded on {device_map}")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _build_prompt(
        self,
        problem: Problem,
        trajectory: Trajectory,
    ) -> str:
        """Build prompt with problem and trajectory context."""
        parts = []

        # Problem statement
        if problem.description:
            parts.append(f"Problem: {problem.description}")
        parts.append(f"Theorem:\n```lean4\n{problem.statement}\n```")

        # Trajectory history
        if trajectory.steps:
            parts.append("\n--- Proof Progress ---")
            for step in trajectory.steps[-5:]:  # Last 5 steps
                status = "✓" if step.result.success else "✗"
                parts.append(f"Step {step.step_num}: `{step.tactic}` [{status}]")
                if step.result.errors:
                    parts.append(f"  Error: {step.result.errors[0][:100]}")
                elif step.result.goals:
                    parts.append(f"  Goals: {step.result.goals[0][:100]}")

            # Current goal: use last successful step's goals, not last step
            last_successful = next(
                (s for s in reversed(trajectory.steps) if s.result.success and s.result.goals),
                None,
            )
            if last_successful:
                parts.append(f"\nCurrent goal: {last_successful.result.goals[0]}")

            # Explicit blocklist so the model can't silently repeat failed tactics
            failed = [s.tactic for s in trajectory.steps[-5:] if not s.result.success]
            if failed:
                banned = ", ".join(f"`{t}`" for t in dict.fromkeys(failed))
                parts.append(f"\nBANNED (already failed — do not repeat): {banned}")

        parts.append("\nNext tactic (one line only):")
        return "\n".join(parts)

    def _generate_tactic(self, prompt: str) -> str:
        """Generate next tactic using LLM."""
        self.load_model()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        chat_prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(chat_prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature if self.temperature > 0 else None,
            do_sample=self.temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        # print(f"\n=== full response ===\n%%%{response}%%%\n====================")
        tactic = extract_proof(response)
        # print(f"[agent] Extracted tactic: {tactic[:50]}..." if len(tactic) > 50 else f"[agent] Extracted tactic: {tactic}")
        return tactic

    def _solve_once(self, problem: Problem, attempt_num: int = 0) -> Trajectory:
        """Run a single iterative proof-search trajectory."""
        trajectory = Trajectory(problem_id=problem.id)
        accumulated_proof = []

        for step_num in range(self.max_steps):
            # Build prompt with current state
            prompt = self._build_prompt(problem, trajectory)

            # print(f"\n=== Step {step_num + 1} prompt ===\n###{prompt}###")

            # Generate next tactic (exceptions propagate to caller)
            tactic = self._generate_tactic(prompt)

            # print(f"\n=== Step {step_num + 1} tactic ===\n$$${tactic}$$$")

            if self.log_path:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps({
                        "problem_id": problem.id,
                        "attempt": attempt_num + 1,
                        "step": step_num + 1,
                        "prompt": prompt,
                        "tactic": tactic,
                    }) + "\n")

            if not tactic.strip():
                continue  # Skip empty tactics

            # Build accumulated proof
            test_proof = "\n".join(accumulated_proof + [tactic])

            # Verify with Lean (exceptions propagate to caller)
            result = self.lean_server.check_proof(problem.statement, test_proof)

            trajectory.add_step(tactic, result)

            # Update accumulated proof if step succeeded
            if result.success:
                accumulated_proof.append(tactic)

            # Check for completion
            if result.complete:
                print(f"[agent] Proof complete in {step_num + 1} steps (attempt {attempt_num + 1})!")
                break

        return trajectory

    def solve(self, problem: Problem) -> AgentResult:
        """
        Solve a problem iteratively with Lean tool feedback.

        Runs up to k independent trajectories (pass@k): stops early on the
        first complete proof, otherwise reports the aggregated outcome.

        Args:
            problem: The problem to solve

        Returns:
            AgentResult with trajectory information aggregated across attempts
        """
        if self.lean_server is None:
            return AgentResult(
                problem_id=problem.id,
                success=False,
                complete=False,
                proof="",
                error="ToolAgent requires a LeanServer",
                num_attempts=0,
                attempts=[],
            )

        attempts = []
        best_trajectory = None

        for attempt_num in range(self.k):
            try:
                trajectory = self._solve_once(problem, attempt_num)
            except Exception as e:
                attempts.append({
                    "proof": "", "success": False, "complete": False,
                    "num_steps": 0, "error": str(e),
                })
                continue

            attempts.append({
                "proof": trajectory.final_proof,
                "success": trajectory.success,
                "complete": trajectory.complete,
                "num_steps": len(trajectory.steps),
                "error": None,
            })

            if best_trajectory is None:
                best_trajectory = trajectory
            if trajectory.complete:
                best_trajectory = trajectory
                break

        any_success = any(a["success"] for a in attempts)
        any_complete = any(a["complete"] for a in attempts)

        return AgentResult(
            problem_id=problem.id,
            success=any_success,
            complete=any_complete,
            proof=best_trajectory.final_proof if best_trajectory else "",
            lean_result=(
                best_trajectory.steps[-1].result
                if best_trajectory and best_trajectory.steps else None
            ),
            num_attempts=len(attempts),
            attempts=attempts,
        )
