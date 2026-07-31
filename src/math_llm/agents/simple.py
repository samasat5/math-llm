"""
Simple agent: Direct single-shot proof generation.

Uses Qwen/Qwen2.5-7B-Instruct to generate proofs in one shot.
No iterative refinement - just prompt -> proof -> verify.
"""

import re
from dataclasses import dataclass
from typing import Optional

from math_llm.data import Problem
from math_llm.lean_server import LeanServer, LeanResult


# System prompt with Lean 4 context
SYSTEM_PROMPT = """You are a Lean 4 theorem prover. Respond in English only. Given a theorem statement, output ONLY the proof tactics, nothing else.

Rules:
- Output tactics only, no explanation, no theorem statement
- Multiple tactics go on separate lines
- Do NOT wrap in code blocks
- Do NOT write "by" at the start

Common tactics:
- rfl, norm_num, decide  → numeric/definitional goals
- ring                   → polynomial equations  
- omega, linarith        → linear arithmetic
- simp, exact, apply     → general goals

Example:
Input: theorem add_comm_example : 1 + 2 = 2 + 1 := by sorry
Output: ring

Input: theorem abs_nonneg_example (x : Real) : 0 ≤ |x| := by sorry
Output: exact abs_nonneg x

"""


@dataclass
class AgentResult:
    """Result of an agent solving a problem."""
    problem_id: str
    success: bool  # Proof compiles without errors
    complete: bool  # Proof is complete (no remaining goals)
    proof: str  # Generated proof
    lean_result: Optional[LeanResult] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "success": self.success,
            "complete": self.complete,
            "proof": self.proof,
            "lean_result": self.lean_result.to_dict() if self.lean_result else None,
            "error": self.error,
        }


def extract_proof(response: str) -> str:
    """Extract proof tactics from LLM response."""
    response = response.strip()

    # Remove code blocks if present
    if "```" in response:
        # Try to extract content from code blocks
        match = re.search(r'```(?:lean4?|proof)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if match:
            response = match.group(1).strip()

    # Remove common prefixes
    prefixes = [
        "The proof is:",
        "Proof:",
        "Here's the proof:",
        "The tactics are:",
        "by ",
        "by\n",
    ]
    for prefix in prefixes:
        if response.lower().startswith(prefix.lower()):
            response = response[len(prefix):].strip()

    # Take only first meaningful lines (avoid explanations)
    lines = response.split('\n')
    proof_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Stop at explanation lines
        if line.startswith(('#', 'Note:', 'This', 'The ', 'We ')):
            break
        proof_lines.append(line)

    return '\n'.join(proof_lines) if proof_lines else response


class SimpleAgent:
    """
    Simple single-shot proof agent.

    Generates proof in one LLM call, then verifies with Lean.
    Uses Qwen/Qwen2.5-7B-Instruct by default.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        lean_server: Optional[LeanServer] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        gpu: Optional[int] = None,
    ):
        self.model_name = model_name
        self.lean_server = lean_server
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.gpu = gpu
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

            # Determine device and dtype
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

    def generate_proof(self, problem: Problem) -> str:
        """Generate a proof for the problem using the LLM."""
        self.load_model()

        # Build prompt
        user_prompt = f"Prove: {problem.statement}"
        if problem.description:
            user_prompt = f"Problem: {problem.description}\n\n{user_prompt}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Apply chat template
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature if self.temperature > 0 else None,
            do_sample=self.temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        # Decode response
        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return extract_proof(response)

    def solve(self, problem: Problem) -> AgentResult:
        """
        Solve a problem: generate proof and verify with Lean.

        Args:
            problem: The problem to solve

        Returns:
            AgentResult with success/complete status
        """
        # Generate proof
        try:
            proof = self.generate_proof(problem)
        except Exception as e:
            print(f"  [agent] Error: {e}")
            return AgentResult(
                problem_id=problem.id,
                success=False,
                complete=False,
                proof="",
                error=f"Generation failed: {e}",
            )

        # Verify with Lean (if server available)
        if self.lean_server is not None:
            try:
                lean_result = self.lean_server.check_proof(problem.statement, proof)
                return AgentResult(
                    problem_id=problem.id,
                    success=lean_result.success,
                    complete=lean_result.complete,
                    proof=proof,
                    lean_result=lean_result,
                )
            except Exception as e:
                return AgentResult(
                    problem_id=problem.id,
                    success=False,
                    complete=False,
                    proof=proof,
                    error=f"Lean verification failed: {e}",
                )

        # No Lean server - return unverified result
        return AgentResult(
            problem_id=problem.id,
            success=True,  # Assume success without verification
            complete=False,
            proof=proof,
            error="No Lean server - proof not verified",
        )
