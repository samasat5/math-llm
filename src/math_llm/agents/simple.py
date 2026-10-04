"""
Simple agent: Direct single-shot proof generation.

Uses Goedel-LM/Goedel-Prover-V2-8B to generate proofs in one shot.
No iterative refinement - just prompt -> proof -> verify.
"""
 
import math
import textwrap

import re
import textwrap
from collections import deque
from dataclasses import dataclass
from typing import Optional

from math_llm.data import Problem
from math_llm.lean_server import LeanServer, LeanResult
from math_llm.agents.autoformalizer import Autoformalizer, is_degenerate, extract_clean_plan, has_hallucinated_lemma


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


def is_goedel_model(model_name: str) -> bool:
    return "goedel" in model_name.lower()


# Goedel-Prover-V2 uses the same "Complete the following Lean 4 code" prompt
# plus proof-plan-then-```lean4-fenced-completion format as DeepSeek-Prover-V2
# (see DeepSeekCoTHandler.prover_inference in Goedel-Prover-V2's own
# src/utils.py), and its generation_config.json matches Pythagoras's
# sampling settings exactly (temperature=0.6, top_p=0.95, top_k=20). Kept as
# its own constants rather than reusing PYTHAGORAS_* so the two models' setups
# can be tuned independently.
GOEDEL_HEADER = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat
"""

GOEDEL_PROMPT_TEMPLATE = """Complete the following Lean 4 code:

```lean4
{}```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof."""

GOEDEL_MAX_NEW_TOKENS = 8192  # matches model card recommendation

GOEDEL_TEMPERATURE = 0.6
GOEDEL_TOP_P = 0.95
GOEDEL_TOP_K = 20


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
    closed = list(re.finditer(r"```lean4?\s*\n?(.*?)```", response, re.DOTALL))
    blocks = [m.group(1) for m in closed]

    # The model's actual final attempt is sometimes an unclosed trailing
    # fence - generation got cut off by the token budget mid-block, AFTER
    # one or more earlier closed fences (e.g. an abandoned `sorry`
    # placeholder it wrote before trying again). The regex above only ever
    # matches closed pairs, so without this the most-developed attempt gets
    # silently dropped in favor of an earlier, worse, already-closed one
    # just because that one happened to close in time.
    # Require ":=" specifically (not just non-empty) - the block selected
    # here must have gotten at least as far as the theorem's own placeholder
    # before being cut off, since everything below splits on ":=" and bails
    # via extract_proof() otherwise. A fragment cut off mid-signature (e.g.
    # just "theorem foo {a b : G" with no ":=" yet) is pure noise, no more
    # useful than the closed blocks, and must not preempt them.
    tail_start = closed[-1].end() if closed else 0
    trailing_open = re.search(r"```lean4?\s*\n?", response[tail_start:])
    if trailing_open:
        trailing_code = response[tail_start + trailing_open.end():]
        if ":=" in trailing_code:
            blocks.append(trailing_code)

    code = next((b for b in reversed(blocks) if ":= by" in b or re.search(r":=\s*\n", b)), None)
    if code is None and blocks:
        code = blocks[-1]
    if code is None:
        # Either no ``` fence anywhere (model abandoned markdown entirely -
        # the sorry-placeholder detection below handles that case), or the
        # last fence never closed because generation got cut off mid-block.
        # In the latter case, findall above matched nothing at all (it only
        # pairs closed fences), so falling back to the raw `response` would
        # include all the reasoning prose before the dangling opening
        # marker too - take everything after the LAST such marker instead,
        # consistent with always preferring the model's final attempt.
        opens = list(re.finditer(r"```lean4?\s*\n?", response))
        code = response[opens[-1].end():] if opens else response

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


def extract_goedel_proof(response: str) -> str:
    """Extract tactic body from a Goedel-Prover-V2-style completion.

    A standalone copy of extract_pythagoras_proof's logic (last fenced
    ```lean4 block containing a real attempt, header stripped down to just
    the tactic body) - kept separate on purpose so Goedel's extraction can
    be adjusted without touching the Pythagoras path.
    """
    # Goedel often emits several ```lean4 fences before its real answer
    # (reference snippets, draft sketches, revisions) - take the LAST one
    # that actually contains a proof attempt, not the first fence in the
    # response.
    blocks = re.findall(r"```lean4?\s*\n?(.*?)```", response, re.DOTALL)
    code = next((b for b in reversed(blocks) if ":= by" in b or re.search(r":=\s*\n", b)), None)
    if code is None and blocks:
        code = blocks[-1]
    if code is None:
        # Either no ``` fence anywhere, or the last fence never closed
        # because generation got cut off mid-block - take everything after
        # the LAST opening marker instead of the raw response, consistent
        # with always preferring the model's final attempt.
        opens = list(re.finditer(r"```lean4?\s*\n?", response))
        code = response[opens[-1].end():] if opens else response

    if ":=" in code:
        tactics = code.split(":=", 1)[1]
    else:
        return extract_proof(response)

    # Strip a leading 'by' (plus same-line trailing space, e.g. ":= by ring")
    # BEFORE dedenting: it sits at whatever column followed the restated
    # "theorem ... := " prefix, and left in place it pollutes dedent's
    # common-margin calculation, under-stripping every sibling line.
    tactics = re.sub(r"^\s*by\b[ \t]*", "", tactics, count=1)

    # If the model abandoned markdown entirely and continued straight into
    # raw code with no ```lean4 fence, the first ':=' found is the restated
    # theorem's own placeholder ("... := sorry"), and the model's real
    # (unfenced) attempt follows right after it - so a leading 'sorry'
    # immediately followed by more content is that leftover placeholder,
    # not the model's answer, and must be dropped.
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
        else:
            # Fence never closed (generation cut off mid-block) - still
            # strip the leading marker, or it survives as a literal
            # backtick token that breaks Lean's parser outright.
            response = re.sub(r"^```(?:lean4?|proof)?\s*\n?", "", response).strip()

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


class LineRepetitionStoppingCriteria:
    """Abort generation once the tail of completed lines is >= min_repeats
    repeats of a short cycle (period 1..max_period).

    Repetition loops (restating the same `have`/`#check` line, or the same
    `sorry`-placeholder block, verbatim over and over - a known failure mode
    at low temperature, but observed at 0.6+ too) otherwise run all the way
    to max_new_tokens: several minutes burned per sample for a proof that
    was never going anywhere. Streaming the output, hashing each newly
    completed line, and comparing the last few catches this early without
    needing to decode the full sequence on every step.

    Checking only period 1 (3 IDENTICAL consecutive lines) misses the
    alternating period-2 loop actually seen in practice - e.g. Goedel
    emitting `<;>` and `omega` as separate lines and cycling between them
    (`<;>\nomega\n<;>\nomega\n...`): no 3 consecutive lines are ever equal
    there, so this generalizes to period 2 (and, for the same reason, 3-8
    too).

    Lines are hashed after replacing digit runs with `#` (see _normalize)
    before the period check, not verbatim - also seen in practice: an
    8-line `have hN : ... / nlinarith` block repeated with an incrementing
    hypothesis counter (`h136`, `h137`, `h138`, ...) is structurally the
    same cycle every time, but the literal line differs each repeat (the
    number), so unnormalized hashing never matches - despite period 8 being
    within max_period's range.

    Batch-aware: tracks per-row state independently (each row in a batched
    num_return_sequences>1 call decodes a different continuation, so a
    repetition loop in one row says nothing about the others). HF's generate()
    only fully stops once every row is finished, but it stops *updating* a
    row internally once that row's own EOS/criteria fires - returning a
    per-row bool tensor here lets each repeating row freeze in place instead
    of forcing the whole batch to wait for the single slowest non-repeating
    row to hit max_new_tokens.
    """

    _DIGITS_RE = re.compile(r"\d+")

    @classmethod
    def _normalize(cls, line: str) -> str:
        return cls._DIGITS_RE.sub("#", line)

    def __init__(self, tokenizer, prompt_len: int, batch_size: int = 1, window: int = 32,
                 check_every: int = 16, max_period: int = 8, min_repeats: int = 3):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.check_every = check_every
        self.max_period = max_period
        self.min_repeats = min_repeats
        self._last_checked_len = 0
        self._num_lines_seen = [0] * batch_size
        self._line_hashes = [deque(maxlen=window) for _ in range(batch_size)]
        self.aborted = [False] * batch_size

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        cur_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        if cur_len - self._last_checked_len < self.check_every:
            return torch.tensor(self.aborted, dtype=torch.bool, device=input_ids.device)
        self._last_checked_len = cur_len

        for row in range(batch_size):
            if self.aborted[row]:
                continue
            text = self.tokenizer.decode(input_ids[row, self.prompt_len:], skip_special_tokens=True)
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            # The last line may still be mid-generation - only hash lines
            # that are actually complete (i.e. followed by a newline already).
            complete_lines = lines if text.endswith("\n") else lines[:-1]
            for line in complete_lines[self._num_lines_seen[row]:]:
                self._line_hashes[row].append(hash(self._normalize(line)))
            self._num_lines_seen[row] = len(complete_lines)

            hashes = list(self._line_hashes[row])
            for period in range(1, self.max_period + 1):
                needed = period * self.min_repeats
                if len(hashes) < needed:
                    continue
                tail = hashes[-needed:]
                cycle = tail[-period:]
                if all(tail[i] == cycle[i % period] for i in range(needed)):
                    self.aborted[row] = True
                    print(f"  [agent] Row {row}: aborting ({self.min_repeats} repeats of a "
                          f"period-{period} line cycle - repetition loop)")
                    break

        return torch.tensor(self.aborted, dtype=torch.bool, device=input_ids.device)


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
        autoformalizer: Optional[Autoformalizer] = None,
        batch_size: int = 5,
        seed: Optional[int] = None,
    ):
        self.model_name = model_name
        self.lean_server = lean_server
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.gpu = gpu
        self.k = k
        self.autoformalizer = autoformalizer
        self.batch_size = batch_size
        self.seed = seed
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

    def generate_proofs(self, problem: Problem, k: int = 1, batch_size: Optional[int] = None,
                         prefix: Optional[str] = None):
        """Generate up to k candidate proofs for the problem, one at a time.

        A generator so callers (solve()) can verify each sample with Lean
        as soon as it's generated, instead of waiting for all k to finish.

        prefix: if given, appended verbatim after the chat template's
        generation-prompt marker, so the model continues writing from this
        already-written partial answer instead of starting fresh - used by
        the case-study prefix/hint-conditioning experiments (see
        src/math_llm/agents/case_study/).
        """
        self.load_model()

        batch_size = self.batch_size if batch_size is None else batch_size

        pythagoras = is_pythagoras_model(self.model_name)
        goedel = is_goedel_model(self.model_name)
    
        if pythagoras:
            docstring = f"/-- {problem.description} -/\n" if problem.description else ""
            formal_statement = (
                PYTHAGORAS_HEADER + "\n" + docstring + problem.statement.strip() + "\n"
            )
            user_prompt = PYTHAGORAS_PROMPT_TEMPLATE.format(formal_statement)
            messages = [{"role": "user", "content": user_prompt}]
            max_new_tokens = max(self.max_new_tokens, PYTHAGORAS_MAX_NEW_TOKENS)
        elif goedel:
            docstring = f"/-- {problem.description} -/\n" if problem.description else ""
            formal_statement = (
                GOEDEL_HEADER + "\n" + docstring + problem.statement.strip() + "\n"
            )
            user_prompt = GOEDEL_PROMPT_TEMPLATE.format(formal_statement)
            messages = [{"role": "user", "content": user_prompt}]
            max_new_tokens = max(self.max_new_tokens, GOEDEL_MAX_NEW_TOKENS)
        else:
            user_prompt = f"Prove: {problem.statement}"
            if problem.description:
                user_prompt = f"Problem: {problem.description}\n\n{user_prompt}"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            max_new_tokens = self.max_new_tokens
    
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if prefix is not None:
            prompt = prompt + prefix

        # Left-padding is required for batched generation with a decoder-only
        # model: generate() appends new tokens on the right, so all sequences
        # in the batch must have their "real" content right-aligned or the
        # model would be attending to pad tokens as if they were real context
        # partway through generation. This wasn't needed at batch size 1.
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
    
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {key: v.to(self._model.device) for key, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
    
        do_sample = self.temperature > 0
        num_samples = k if do_sample else 1
    
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        if do_sample:
            if pythagoras:
                gen_kwargs["temperature"] = PYTHAGORAS_TEMPERATURE
                gen_kwargs["top_p"] = PYTHAGORAS_TOP_P
                gen_kwargs["top_k"] = PYTHAGORAS_TOP_K
            elif goedel:
                gen_kwargs["temperature"] = GOEDEL_TEMPERATURE
                gen_kwargs["top_p"] = GOEDEL_TOP_P
                gen_kwargs["top_k"] = GOEDEL_TOP_K
            else:
                gen_kwargs["temperature"] = self.temperature
        from transformers import StoppingCriteriaList

        extract = (
            extract_pythagoras_proof if pythagoras else
            extract_goedel_proof if goedel else
            extract_proof
        )

        samples_done = 0
        sample_num = 0
        n_batches = math.ceil(num_samples / batch_size)

        for batch_idx in range(n_batches):
            this_batch = min(batch_size, num_samples - samples_done)
            print(f"  [agent] Batch {batch_idx + 1}/{n_batches} "
                f"({this_batch} samples, {samples_done}/{num_samples} done)...")

            if do_sample and self.seed is not None:
                # Seeded per BATCH, not per individual sample: a batch of
                # this_batch sequences via num_return_sequences shares one
                # random draw in a single generate() call, so the unit of
                # reproducibility shifts from "sample i" (unbatched) to
                # "batch i" here - the same seed+batch_idx always reproduces
                # that whole batch, including every row in it.
                import torch
                torch.manual_seed(self.seed + batch_idx)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.seed + batch_idx)

            # num_return_sequences=this_batch: the SAME prompt, decoded into
            # `this_batch` independent samples in one generate() call. This is
            # the actual throughput win - the prompt prefill and per-token
            # weight reads are shared/batched across samples instead of being
            # repeated `this_batch` separate times.
            repetition_stop = LineRepetitionStoppingCriteria(
                self._tokenizer, input_len, batch_size=this_batch
            )
            outputs = self._model.generate(
                **inputs,
                **gen_kwargs,
                num_return_sequences=this_batch,
                stopping_criteria=StoppingCriteriaList([repetition_stop]),
            )

            for row in range(this_batch):
                sample_num += 1
                response = self._tokenizer.decode(
                    outputs[row][input_len:], skip_special_tokens=True
                )
                proof = extract(response)
                print(f"\n  [agent] Sample {sample_num} raw response:\n"
                    f"{textwrap.indent(response, '    ')}")
                print(f"  [agent] Sample {sample_num} extracted proof:\n"
                    f"{textwrap.indent(proof, '    ')}\n")
                yield proof, response

            samples_done += this_batch

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

        def verify(proof: str, source: str) -> Optional[LeanResult]:
            """Check one candidate proof with Lean and record the attempt.
            Returns the LeanResult, or None if it couldn't be verified."""
            if self.lean_server is None:
                attempts.append({
                    "proof": proof, "success": True, "complete": False,
                    "error": "No Lean server - proof not verified", "source": source,
                })
                return None

            try:
                lean_result = self.lean_server.check_proof(problem.statement, proof)
            except Exception as e:
                attempts.append({
                    "proof": proof, "success": False, "complete": False,
                    "error": f"Lean verification failed: {e}", "source": source,
                })
                return None

            attempts.append({
                "proof": proof,
                "success": lean_result.success,
                "complete": lean_result.complete,
                "error": lean_result.errors if lean_result.errors else None,
                "source": source,
            })
            status = "COMPLETE" if lean_result.complete else ("OK" if lean_result.success else "FAIL")
            print(f"  [agent] Sample {len(attempts)} ({source}) result: {status}")
            if lean_result.errors:
                print(f"  [agent] Sample {len(attempts)} lean error: {lean_result.errors[0]}")
            return lean_result

        try:
            proof_stream = self.generate_proofs(problem, k=self.k)
            for proof, response in proof_stream:
                if not best_proof:
                    best_proof = proof

                lean_result = verify(proof, source="prover")

                if lean_result is not None and lean_result.complete:
                    best_proof, best_lean_result = proof, lean_result
                    print(f"  [agent] Complete proof found on sample {len(attempts)} - stopping early")
                    proof_stream.close()
                    break

                # The prover either looped instead of writing a real proof
                # (repeated placeholder / repeated paragraph) or hallucinated
                # a Mathlib lemma name that doesn't exist - in both cases its
                # plan may still be sound (see mathd_algebra_393), so hand it
                # to a second model whose only job is translating an
                # already-correct plan into tactics, rather than burning
                # another full k-sample attempt hoping the same prover avoids
                # the same mistake.
                needs_rescue = is_degenerate(proof, response) or has_hallucinated_lemma(
                    lean_result.errors if lean_result is not None else None
                )
                if self.autoformalizer is not None and needs_rescue:
                    plan = extract_clean_plan(response)
                    print(f"  [agent] Sample {len(attempts)} looked degenerate - handing plan to autoformalizer")
                    af_result = self.autoformalizer.formalize(problem.statement, plan, lean_server=self.lean_server)
                    af_lean_result = verify(af_result.proof, source="autoformalizer")

                    if af_lean_result is not None and af_lean_result.complete:
                        best_proof, best_lean_result = af_result.proof, af_lean_result
                        print(f"  [agent] Autoformalizer rescued sample {len(attempts)} - stopping early")
                        proof_stream.close()
                        break
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
