"""3-way ablation: does prompt guidance shift the hallucinated-vs-correct
lemma preference at all, and does presenting it as a code signature (closer
to the model's own generation register) do better than prose?

Conditions (only the user-prompt instructions differ; the downstream
"reasoning so far" context is held FIXED - the same real generated prefix
from the parity_guided_v1 sweep run, truncated right after `rw [` - so each
condition is scored on the exact same decision point):

  A. baseline        - no guidance at all
  B. prose guidance   - the parity_guided_v1 paragraph (isolated from the
                        unrelated omega/integrality paragraph, to avoid
                        confounding this specific ablation)
  C. code signature   - the same information, but as literal Mathlib
                        theorem signatures in a lean4 code block instead of
                        a prose sentence

For each condition: score(prefix, "Nat.odd_iff_not_even]") [wrong],
score(prefix, "Nat.not_even_iff_odd]") [correct], and the gap in nats.
"""

import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from math_llm.agents.simple import PYTHAGORAS_HEADER, PYTHAGORAS_PROMPT_TEMPLATE as BASELINE_PROMPT_TEMPLATE
from math_llm.agents.temp_sweep_aime_1987_p5 import resolve_local_snapshot
from math_llm.data import load_data

MODEL = "Pythagoras-LM/Pythagoras-Prover-4B"
WRONG = "Nat.odd_iff_not_even]"
CORRECT = "Nat.not_even_iff_odd]"

PROSE_GUIDANCE = """

Mathlib parity API: Nat.even_or_odd n : Even n ∨ Odd n gives the case split directly — do not derive one parity from the negation of the other. To relate them use Nat.not_even_iff_odd : ¬Even n ↔ Odd n or Nat.not_odd_iff_even. There is no Nat.odd_iff_not_even. omega cannot see inside Even/Odd; rewrite with Nat.odd_iff (n % 2 = 1) or Nat.even_iff first. After applying an induction hypothesis at k + c, run push_cast before linarith."""

CODE_SIGNATURE_GUIDANCE = """

Relevant Mathlib signatures:
```lean4
theorem Nat.even_or_odd (n : ℕ) : Even n ∨ Odd n
theorem Nat.not_even_iff_odd {{n : ℕ}} : ¬Even n ↔ Odd n
theorem Nat.not_odd_iff_even {{n : ℕ}} : ¬Odd n ↔ Even n
```"""

CONDITIONS = {
    "A baseline": "",
    "B + prose guidance": PROSE_GUIDANCE,
    "C + signature in code": CODE_SIGNATURE_GUIDANCE,
}


def main():
    problems = load_data("minif2f-lean4", None, min_tier=1)
    problem = next(p for p in problems if p.id == "minif2f-lean4/amc12a_2017_p7")
    docstring = f"/-- {problem.description} -/\n" if problem.description else ""
    formal_statement = PYTHAGORAS_HEADER + "\n" + docstring + problem.statement.strip() + "\n"

    local_model_path = resolve_local_snapshot(MODEL)
    print(f"[ablation] Loading tokenizer/model from {local_model_path}...")
    tok = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    model.eval()

    # Same real generated "reasoning so far" for every condition - only the
    # prompt instructions vary. Reuse the actual parity_guided_v1 generation.
    with open("outputs/amc12a_2017_p7_temperature_sweep_Pythagoras-Prover-4B_k1.json") as f:
        data = json.load(f)
    r = next(x for x in data["results"] if x.get("variant") == "parity_guided_v1")
    resp = r["raw_response"]
    idx = resp.find("rw [Nat.")
    prefix_text = resp[:idx + len("rw [")]

    def score(prefix, cont):
        n = tok(prefix, return_tensors="pt").input_ids.shape[1]
        ids = tok(prefix + cont, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            logits = model(ids).logits[0]
        cont_logits = logits[n - 1:-1].float()
        tgt = ids[0, n:]
        lp = F.log_softmax(cont_logits, dim=-1)
        result = lp[torch.arange(tgt.shape[0]), tgt].sum().item()
        del logits, cont_logits, lp
        torch.cuda.empty_cache()
        return result

    print(f"\n{'condition':<24s}{'logP(wrong)':>14s}{'logP(correct)':>16s}{'gap (nats)':>14s}")
    for label, addition in CONDITIONS.items():
        user_prompt = (BASELINE_PROMPT_TEMPLATE + addition).format(formal_statement)
        messages = [{"role": "user", "content": user_prompt}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prefix = prompt + prefix_text

        lp_wrong = score(prefix, WRONG)
        lp_correct = score(prefix, CORRECT)
        gap = lp_wrong - lp_correct
        print(f"{label:<24s}{lp_wrong:>14.3f}{lp_correct:>16.3f}{gap:>14.3f}")


if __name__ == "__main__":
    main()
