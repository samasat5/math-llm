"""
Autoformalizer agent: rescues degenerate prover samples.

When the prover (e.g. Pythagoras) loops instead of producing a proof - either
repeating the untouched `sorry` placeholder in its code fence, or repeating a
reasoning paragraph until it runs out of tokens - the actual mathematical plan
is often still sitting in the reasoning text before the loop started. This
agent extracts that clean plan and asks a second, separate model (e.g. a Qwen
instruct/coder model) to translate it into Lean 4 tactics, since translation
from an already-correct plan is a much easier task than proving the theorem
from scratch.
"""

import re
import textwrap
from collections import Counter
from dataclasses import dataclass
from typing import Optional

# Placeholder proofs the prover extractor falls back to when it never wrote
# real tactics - these never need translating, they need replacing entirely.
_TRIVIAL_PROOFS = {"", "sorry", "```", "by sorry"}

# Paragraphs shorter than this are headers/single lines ("### Complete Lean 4
# Proof", "But perhaps..."), too short to reliably signal "the reasoning
# started repeating itself" - only longer paragraphs are compared.
_MIN_PARAGRAPH_LEN = 40

# A paragraph (or the ```-delimited code fence) recurring at least this many
# times means generation looped rather than progressed.
_REPEAT_THRESHOLD = 5

# Mathlib-style namespaced identifiers a proof might reference, e.g.
# "Real.sqrt_eq_iff_sq_eq" or "Equiv.symm_apply_apply" - deliberately
# excludes local hypothesis/variable names (h₀, σ.1, f, ...) since those
# never start with an uppercase namespace segment.
_LEMMA_NAME_RE = re.compile(r"\b[A-Z][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)+\b")

# Lean's wording for a name that doesn't resolve in the environment - if the
# *prover* hits this, it hallucinated a lemma, which is just as unrescuable
# by resampling as a repetition loop and should trigger the same rescue path.
_UNKNOWN_IDENTIFIER_ERROR_RE = re.compile(r"unknown (?:constant|identifier|class)", re.IGNORECASE)


def has_hallucinated_lemma(errors: Optional[list]) -> bool:
    """True if a LeanResult's errors report a name that doesn't resolve -
    i.e. the model invented a Mathlib lemma that doesn't exist."""
    if not errors:
        return False
    return any(_UNKNOWN_IDENTIFIER_ERROR_RE.search(e) for e in errors)


def extract_lemma_names(proof: str) -> list[str]:
    """Pull out Mathlib-style namespaced identifiers a proof references, so
    each can be checked against Lean before the proof is trusted."""
    return sorted(set(_LEMMA_NAME_RE.findall(proof)))


def is_degenerate(proof: str, response: str) -> bool:
    """True if a prover sample looped instead of producing a real proof."""
    if proof.strip() in _TRIVIAL_PROOFS:
        return True

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", response) if len(p.strip()) >= _MIN_PARAGRAPH_LEN]
    if not paragraphs:
        return False
    _, count = Counter(paragraphs).most_common(1)[0]
    return count >= _REPEAT_THRESHOLD


def extract_clean_plan(response: str) -> str:
    """Pull the reasoning text out of a prover response, stopping at the
    first sign of a repetition loop.

    Code fences are dropped outright (on a degenerate sample they're the
    untouched placeholder / the broken attempt, not useful plan content).
    Paragraphs are then kept in order until one repeats a paragraph already
    seen - that's where the model stopped reasoning and started looping.
    """
    text = re.sub(r"```.*?```", "", response, flags=re.DOTALL)
    paragraphs = re.split(r"\n\s*\n", text)

    seen: set[str] = set()
    kept = []
    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue
        if len(stripped) >= _MIN_PARAGRAPH_LEN:
            if stripped in seen:
                break
            seen.add(stripped)
        kept.append(stripped)

    return "\n\n".join(kept).strip()


AUTOFORMALIZER_SYSTEM_PROMPT = """You are a Lean 4 autoformalization assistant.

You will be given a theorem statement and an informal proof plan written by another model. Your only job is to translate that plan into Lean 4 tactics - not to write generic boilerplate, and not to reuse any example below verbatim. The tactics you write must actually use the specific hypotheses, values, and lemmas the plan describes.

Rules:
- Output ONLY the proof tactics that go after ":= by", nothing else
- Do NOT restate the theorem statement
- Do NOT wrap the tactics in a code block
- Do NOT write "by" or ":=" at the start - start directly with the first tactic (e.g. `have`, `rw`, `norm_num`)
- Multiple tactics go on separate lines
- If the plan is incomplete, wrong, or hard to follow, still produce your best-effort Lean 4 tactic proof from the statement alone
- When a `have` states a fact about a value that is only known through a given hypothesis (e.g. `h₀ : ∀ x, f x = ...`), you MUST invoke that hypothesis explicitly - e.g. `rw [h₀]` or `norm_num [h₀]`/`simp [h₀]`. `norm_num`/`simp`/`ring` alone can only prove pure numeric/algebraic identities - they know nothing about an opaque function like `f` or `σ.1` unless the relevant hypothesis is passed into them or rewritten first. Forgetting this is the single most common mistake - check every `have` about such a value for it.

This must be Lean 4 syntax, NOT Lean 3. Lean 4 differs from Lean 3 in ways models trained on mixed corpora often get wrong - do not make these mistakes:
- NO trailing commas after tactics. Lean 3: `rw [h],` -> Lean 4: `rw [h]` (newline or `<;>` to sequence, never a comma)
- `rw`/`simp`/`norm_num` always take a bracketed list. Lean 3: `rw h` -> Lean 4: `rw [h]`
- Rewriting at a hypothesis: Lean 3: `rw h at h2` -> Lean 4: `rw [h] at h2`
- No `begin...end` blocks - Lean 4 tactic blocks are just indented lines after `by`

Worked examples (illustrate required Lean 4 style and hypothesis usage only - your
actual output must be derived from the plan you are given, not copied from these):

Theorem statement: theorem example_sq (a b : ℝ) (h : a = b + 1) : a ^ 2 = (b + 1) ^ 2 := by sorry
Informal proof plan: Substitute h into the goal so both sides read (b+1)^2, then it's syntactically identical.
Output:
rw [h]

Theorem statement: theorem example_hyp (g : ℕ → ℕ) (h : ∀ n, g n = n + 5) : g 3 = 8 := by sorry
Informal proof plan: g is only known through h, so evaluate h at 3 to get g 3 = 3 + 5, then compute.
Output:
have h1 : g 3 = 3 + 5 := by rw [h]
norm_num at h1
exact h1
"""

AUTOFORMALIZER_USER_TEMPLATE = """Theorem statement:
```lean4
{statement}
```

Informal proof plan (may be incomplete or repetitive - use what's useful):
{plan}

Output the Lean 4 tactic proof:"""


@dataclass
class AutoformalizerResult:
    proof: str
    raw_response: str


class Autoformalizer:
    """Second-model rescue agent: plan text -> Lean 4 tactics."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        max_new_tokens: int = 4096,
        gpu: Optional[int] = None,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.gpu = gpu
        self._model = None
        self._tokenizer = None

    def load_model(self) -> None:
        if self._model is not None:
            return

        print(f"[autoformalizer] Loading model {self.model_name}...")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        quantization_config = None
        if torch.cuda.is_available():
            gpu_index = self.gpu if self.gpu is not None else 0
            device_map = {"": gpu_index}
            torch_dtype = torch.bfloat16
            # Sharing a GPU with the main prover model - quantize to 8-bit so
            # both models' weights + KV cache fit rather than assuming this
            # model gets the whole card to itself.
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
        print(f"[autoformalizer] Model loaded on {device_map}")

    def _generate(self, messages: list) -> str:
        self.load_model()
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        output = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        return self._tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def formalize(self, statement: str, plan: str, lean_server=None) -> AutoformalizerResult:
        """Translate an informal plan into a Lean 4 tactic proof.

        If `lean_server` is given, every Mathlib-style lemma name the proof
        references is checked for existence before it's trusted (catches a
        hallucinated lemma the same way a human reviewer would - by looking
        it up - rather than only finding out when the full proof fails to
        compile). On a hit, one retry is made telling the model exactly
        which names don't exist.
        """
        messages = [
            {"role": "system", "content": AUTOFORMALIZER_SYSTEM_PROMPT},
            {"role": "user", "content": AUTOFORMALIZER_USER_TEMPLATE.format(statement=statement.strip(), plan=plan)},
        ]
        response = self._generate(messages)
        proof = _extract_tactics(response)

        print(f"[autoformalizer] raw response:\n{textwrap.indent(response, '    ')}")
        print(f"[autoformalizer] extracted proof:\n{textwrap.indent(proof, '    ')}\n")

        if lean_server is not None:
            bogus = [name for name in extract_lemma_names(proof) if not lean_server.identifier_exists(name)]
            if bogus:
                print(f"[autoformalizer] lemma name(s) not found in Mathlib: {bogus} - retrying without them")
                messages += [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": (
                        "These lemma name(s) do not exist in Mathlib: "
                        f"{', '.join(bogus)}. Do not use them. Either find a different, real Mathlib "
                        "lemma, or prove the relevant step with basic tactics (norm_num, simp, field_simp, "
                        "ring, linarith, omega) instead. Output the full corrected Lean 4 tactic proof."
                    )},
                ]
                response = self._generate(messages)
                proof = _extract_tactics(response)
                print(f"[autoformalizer] retry raw response:\n{textwrap.indent(response, '    ')}")
                print(f"[autoformalizer] retry extracted proof:\n{textwrap.indent(proof, '    ')}\n")

        return AutoformalizerResult(proof=proof, raw_response=response)


def _extract_tactics(response: str) -> str:
    """Pull tactics out of the autoformalizer's reply (same shape as
    SimpleAgent's non-Pythagoras prompt: plain tactics, optionally fenced)."""
    response = response.strip()
    if "```" in response:
        match = re.search(r"```(?:lean4?)?\s*\n?(.*?)\n?```", response, re.DOTALL)
        if match:
            response = match.group(1).strip()
        else:
            # Generation got cut off (max_new_tokens) before the fence ever
            # closed - there's no matching pair to extract between, but the
            # leading marker must still be stripped, or it survives as a
            # literal backtick token that breaks Lean's parser outright.
            response = re.sub(r"^```(?:lean4?)?\s*\n?", "", response).strip()
    # Model sometimes echoes the theorem's own "... := by" instead of just
    # the tactics that go after it - strip a leading ":=" and/or "by".
    response = re.sub(r"^\s*:=\s*", "", response)
    response = re.sub(r"^\s*by\b[ \t]*", "", response, count=1)
    return response.strip()
