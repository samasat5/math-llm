"""
Simple agent: Direct single-shot proof generation.

Uses Goedel-LM/Goedel-Prover-V2-8B to generate proofs in one shot.
No iterative refinement - just prompt -> proof -> verify.
"""

import re
import textwrap
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

# Pythagoras-Prover expects a full standalone Lean 4 file (imports + theorem)
# and replies with a proof plan followed by a ```lean4 fenced completion.
# See https://huggingface.co/Pythagoras-LM/Pythagoras-Prover-4B
PYTHAGORAS_HEADER = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat
"""

PYTHAGORAS_PROMPT_TEMPLATE = """Complete the following Lean 4 code:

```lean4
{}```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof."""

PYTHAGORAS_MAX_NEW_TOKENS = 8192  # matches model card recommendation

# Model's own generation_config.json: temperature=0.6, top_p=0.95, top_k=20.
# SimpleAgent's default temperature=0.1 is near-greedy and prone to repetition
# loops on this model, so use its recommended sampling settings instead.
PYTHAGORAS_TEMPERATURE = 0.6
PYTHAGORAS_TOP_P = 0.95
PYTHAGORAS_TOP_K = 20


def is_pythagoras_model(model_name: str) -> bool:
    return "pythagoras" in model_name.lower()


def extract_pythagoras_proof(response: str) -> str:
    """Extract tactic body from a Pythagoras-style completion.

    The model replies with a proof plan followed by a ```lean4 fenced block
    containing the full theorem restated with the proof filled in. We pull
    the tactic body out - everything after the OUTER theorem's own ':='
    (the first ':=' in the block, since it always opens with the restated
    "theorem ... := ..." signature) - so nested/sequential `have ... := by`
    sub-proofs stay attached to their declarations instead of being severed
    from them.

    Pythagoras sometimes writes the outer proof in tactic mode
    ("theorem ... := by\n  <tactics>") and sometimes in term mode
    ("theorem ... :=\n  have h := by ...\n  exact h", no top-level 'by').
    Splitting on ':= by' specifically breaks the term-mode case: it lands on
    the first *nested* have's ':= by' instead of the theorem's own ':=',
    orphaning that have's declaration while its later 'exact h'/'rw [h]'
    still reference it. Splitting on the bare first ':=' and only trimming
    a leading 'by' if one is actually present handles both: 'have'/'exact'
    are valid tactics too, so a term-mode body still works once embedded
    under our own synthesized 'by'.
    """
    # Pythagoras often emits several ```lean4 fences before its real answer
    # (reference snippets, draft sketches, revisions) - take the LAST one
    # that actually contains a proof attempt, not the first fence in the
    # response (which previously grabbed unrelated quoted Mathlib source or
    # an early abandoned draft instead of the model's final proof).
    blocks = re.findall(r"```lean4?\s*\n?(.*?)```", response, re.DOTALL)
    code = next((b for b in reversed(blocks) if ":= by" in b or re.search(r":=\s*\n", b)), None)
    if code is None:
        code = blocks[-1] if blocks else response

    if ":=" in code:
        tactics = code.split(":=", 1)[1]
    else:
        return extract_proof(response)

    # Strip a leading 'by' (plus same-line trailing space, e.g. ":= by ring")
    # BEFORE dedenting, not after: the literal "by" sits at whatever column
    # followed the "theorem ... := " prefix (often just 1 space), and if
    # left in place it pollutes dedent's common-margin calculation with
    # that stray 1-space indent, under-stripping every sibling `have`/tactic
    # line that follows and producing inconsistent indentation Lean's
    # whitespace-sensitive parser rejects. Consuming only "by" plus
    # trailing spaces/tabs (not the following newline) leaves the next
    # line's own indentation untouched for dedent to measure correctly.
    tactics = re.sub(r"^\s*by\b[ \t]*", "", tactics, count=1)

    # On hard problems Pythagoras sometimes abandons markdown entirely and
    # continues straight into raw code with no ```lean4 fence at all (no
    # matches above -> code fell back to the whole `response`). In that
    # case the first ':=' found is the ORIGINAL restated theorem's own
    # placeholder ("... := sorry"), and the model's real (unfenced)
    # attempt follows right after it as free-standing `have` statements -
    # so a leading 'sorry' immediately followed by more content is that
    # leftover placeholder, not the model's answer, and must be dropped
    # (otherwise the stray 'sorry' plus whatever label text follows it,
    # e.g. a bare word like "proof", breaks Lean's parser immediately,
    # before any of the real tactics that follow are even reached).
    _leftover_sorry = re.match(
        r"^\s*sorry\b[ \t]*\n?(?:[ \t]*proof\.?[ \t]*\n)?(\s*\S.*)$",
        tactics, re.DOTALL | re.IGNORECASE,
    )
    if _leftover_sorry:
        tactics = _leftover_sorry.group(1)

    tactics = textwrap.dedent(tactics).strip()
    return tactics if tactics else extract_proof(response)


@dataclass
class AgentResult:
    """Result of an agent solving a problem."""
    problem_id: str
    success: bool  # Proof compiles without errors (in >=1 of k attempts)
    complete: bool  # Proof is complete (no remaining goals) (in >=1 of k attempts)
    proof: str  # Best generated proof (first complete one, else first attempt)
    lean_result: Optional[LeanResult] = None
    error: Optional[str] = None
    num_attempts: int = 1  # Number of samples drawn for pass@k
    attempts: Optional[list[dict]] = None  # Per-attempt outcomes

    def to_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "success": self.success,
            "complete": self.complete,
            "proof": self.proof,
            "lean_result": self.lean_result.to_dict() if self.lean_result else None,
            "error": self.error,
            "num_attempts": self.num_attempts,
            "attempts": self.attempts,
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
    Uses Goedel-LM/Goedel-Prover-V2-8B by default.
    """

    def __init__(
        self,
        model_name: str = "Goedel-LM/Goedel-Prover-V2-8B",
        lean_server: Optional[LeanServer] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        gpu: Optional[int] = None,
        k: int = 1,
    ):
        self.model_name = model_name
        self.lean_server = lean_server
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.gpu = gpu
        self.k = k
        self._model = None
        self._tokenizer = None

    def load_model(self) -> None:
        """Load the LLM model."""
        if self._model is not None:
            return

        print(f"[agent] Loading model {self.model_name}...")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            # Determine device and dtype
            quantization_config = None
            if torch.cuda.is_available():
                # Force the whole model onto a single GPU. Only 8-bit quantize
                # on cards without enough VRAM to hold an 8B model in bf16
                # (~16GB) plus headroom for KV cache/activations; quantizing
                # on larger cards would only add overhead for no benefit.
                gpu_index = self.gpu if self.gpu is not None else 0
                device_map = {"": gpu_index}
                torch_dtype = torch.bfloat16
                total_vram_gb = torch.cuda.get_device_properties(gpu_index).total_memory / 1e9
                if total_vram_gb < 20:
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
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
                quantization_config=quantization_config,
                trust_remote_code=True,
            )

            print(f"[agent] Model loaded on {device_map}")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def generate_proofs(self, problem: Problem, k: int = 1):
        """Generate up to k candidate proofs for the problem, one at a time.

        A generator so callers (solve()) can verify each sample with Lean
        as soon as it's generated, instead of waiting for all k to finish.
        """
        self.load_model()

        pythagoras = is_pythagoras_model(self.model_name)

        if pythagoras:
            docstring = f"/-- {problem.description} -/\n" if problem.description else ""
            formal_statement = (
                PYTHAGORAS_HEADER + "\n" + docstring + problem.statement.strip() + "\n"
            )
            user_prompt = PYTHAGORAS_PROMPT_TEMPLATE.format(formal_statement)
            messages = [{"role": "user", "content": user_prompt}]
            max_new_tokens = max(self.max_new_tokens, PYTHAGORAS_MAX_NEW_TOKENS)
        else:
            user_prompt = f"Prove: {problem.statement}"
            if problem.description:
                user_prompt = f"Problem: {problem.description}\n\n{user_prompt}"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            max_new_tokens = self.max_new_tokens

        # Apply chat template
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {key: v.to(self._model.device) for key, v in inputs.items()}

        do_sample = self.temperature > 0
        num_samples = k if do_sample else 1
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self._tokenizer.eos_token_id,
            num_return_sequences=1,
        )
        if do_sample:
            if pythagoras:
                # Match the model's own generation_config.json exactly, no
                # extra anti-repetition penalties (those caused multilingual
                # token substitution artifacts in testing).
                gen_kwargs["temperature"] = PYTHAGORAS_TEMPERATURE
                gen_kwargs["top_p"] = PYTHAGORAS_TOP_P
                gen_kwargs["top_k"] = PYTHAGORAS_TOP_K
            else:
                gen_kwargs["temperature"] = self.temperature

        from transformers import TextStreamer

        streamer = TextStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Generate samples one at a time (batch size 1) instead of via
        # num_return_sequences=k, since a large k (e.g. pass@32) batched
        # would multiply KV-cache memory k-fold and OOM on small GPUs.
        input_len = inputs["input_ids"].shape[1]
        extract = extract_pythagoras_proof if pythagoras else extract_proof
        for i in range(num_samples):
            if num_samples > 1:
                print(f"  [agent] Sample {i + 1}/{num_samples}...")
            output = self._model.generate(**inputs, **gen_kwargs, streamer=streamer)
            response = self._tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
            proof = extract(response)
            print(f"\n  [agent] Sample {i + 1} raw response:\n{textwrap.indent(response, '    ')}")
            print(f"  [agent] Sample {i + 1} extracted proof:\n{textwrap.indent(proof, '    ')}\n")
            yield proof

    def solve(self, problem: Problem) -> AgentResult:
        """
        Solve a problem: generate up to k candidate proofs, verifying each
        with Lean immediately after it's generated.

        Pass@k: the problem counts as solved if any of the k samples succeeds.
        Stops early (without generating remaining samples) as soon as one
        sample's proof is verified complete, since that's all pass@k needs.

        Args:
            problem: The problem to solve

        Returns:
            AgentResult with aggregated success/complete status across attempts
        """
        attempts = []
        best_proof = ""
        best_lean_result = None

        try:
            proof_stream = self.generate_proofs(problem, k=self.k)
            for proof in proof_stream:
                if not best_proof:
                    best_proof = proof

                if self.lean_server is not None:
                    try:
                        lean_result = self.lean_server.check_proof(problem.statement, proof)
                    except Exception as e:
                        attempts.append({
                            "proof": proof, "success": False, "complete": False,
                            "error": f"Lean verification failed: {e}",
                        })
                        continue

                    attempts.append({
                        "proof": proof,
                        "success": lean_result.success,
                        "complete": lean_result.complete,
                        "error": lean_result.errors if lean_result.errors else None,
                    })
                    status = "COMPLETE" if lean_result.complete else ("OK" if lean_result.success else "FAIL")
                    print(f"  [agent] Sample {len(attempts)} result: {status}")
                    if lean_result.errors:
                        print(f"  [agent] Sample {len(attempts)} lean error: {lean_result.errors[0]}")

                    if lean_result.complete:
                        best_proof = proof
                        best_lean_result = lean_result
                        print(f"  [agent] Complete proof found on sample {len(attempts)} - stopping early")
                        proof_stream.close()
                        break
                else:
                    # No Lean server - assume success without verification
                    attempts.append({
                        "proof": proof, "success": True, "complete": False,
                        "error": "No Lean server - proof not verified",
                    })
        except Exception as e:
            print(f"  [agent] Error: {e}")
            if not attempts:
                return AgentResult(
                    problem_id=problem.id,
                    success=False,
                    complete=False,
                    proof="",
                    error=f"Generation failed: {e}",
                    num_attempts=0,
                    attempts=[],
                )

        any_success = any(a["success"] for a in attempts)
        any_complete = any(a["complete"] for a in attempts)

        return AgentResult(
            problem_id=problem.id,
            success=any_success,
            complete=any_complete,
            proof=best_proof,
            lean_result=best_lean_result,
            error=None if attempts else "No proofs generated",
            num_attempts=len(attempts),
            attempts=attempts,
        )
