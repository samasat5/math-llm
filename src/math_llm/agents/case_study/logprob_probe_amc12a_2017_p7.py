"""Diagnostic: what does Pythagoras actually believe comes after `rw [`?

Reconstructs the exact prompt used for the parity_guided_v1 sweep run,
truncates the model's own earlier generation right after `rw [` (just before
it committed to the hallucinated `Nat.odd_iff_not_even`), and:

1. Reads off the top-k logprobs for the single next token (equivalent to an
   OpenAI-style completions call with logprobs=20, temperature 0).
2. Scores the full log-probability of two complete candidate continuations
   - the hallucinated lemma name vs. the real one - against the same
   prefix, via teacher-forced log-softmax summation.

No vLLM/OpenAI-compatible server is running in this environment, so both
run against the local transformers model directly via a forward pass.
"""

import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from math_llm.agents.simple import PYTHAGORAS_HEADER
from math_llm.agents.case_study.temp_sweep_aime_1987_p5 import PARITY_GUIDED_PROMPT_TEMPLATE, resolve_local_snapshot
from math_llm.data import load_data

MODEL = "Pythagoras-LM/Pythagoras-Prover-4B"
TOP_K = 20
TRUNCATE_AFTER = "rw ["
CANDIDATES = ["Nat.odd_iff_not_even]", "Nat.not_even_iff_odd]"]


def main():
    problems = load_data("minif2f-lean4", None, min_tier=1)
    problem = next(p for p in problems if p.id == "minif2f-lean4/amc12a_2017_p7")

    docstring = f"/-- {problem.description} -/\n" if problem.description else ""
    formal_statement = PYTHAGORAS_HEADER + "\n" + docstring + problem.statement.strip() + "\n"
    user_prompt = PARITY_GUIDED_PROMPT_TEMPLATE.format(formal_statement)
    messages = [{"role": "user", "content": user_prompt}]

    local_model_path = resolve_local_snapshot(MODEL)
    print(f"[probe] Loading tokenizer/model from {local_model_path}...")
    tok = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    model.eval()

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    with open("outputs/amc12a_2017_p7_temperature_sweep_Pythagoras-Prover-4B_k1.json") as f:
        data = json.load(f)
    r = next(x for x in data["results"] if x.get("variant") == "parity_guided_v1")
    resp = r["raw_response"]
    idx = resp.find("rw [Nat.")
    if idx == -1:
        raise RuntimeError("'rw [Nat.' not found in stored raw_response")
    prefix_text = resp[:idx + len(TRUNCATE_AFTER)]
    PREFIX = prompt + prefix_text
    print(f"[probe] PREFIX ends with: ...{PREFIX[-120:]!r}\n")

    def score(prefix, cont):
        n = tok(prefix, return_tensors="pt").input_ids.shape[1]
        ids = tok(prefix + cont, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            logits = model(ids).logits[0]
        # Only the continuation's own positions matter - slice BEFORE
        # log_softmax so we never materialize a [seq_len, vocab] float32
        # tensor for the (long, irrelevant) prompt prefix. That full-sequence
        # softmax is what OOM'd against the concurrently-running sweep.
        cont_logits = logits[n - 1:-1].float()
        tgt = ids[0, n:]
        lp = F.log_softmax(cont_logits, dim=-1)
        result = lp[torch.arange(tgt.shape[0]), tgt].sum().item()
        del logits, cont_logits, lp
        torch.cuda.empty_cache()
        return result

    print("[probe] Full-continuation log-probabilities (higher = more likely):\n")
    for c in CANDIDATES:
        print(f"  {c:28s} {score(PREFIX, c):10.3f}")

    print(f"\n[probe] Top-{TOP_K} single-next-token logprobs (for reference):\n")
    inputs = tok(PREFIX, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    log_probs = torch.log_softmax(logits[0, -1, :].float(), dim=-1)
    top = torch.topk(log_probs, TOP_K)
    for rank, (lp, tok_id) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
        piece = tok.decode([tok_id])
        print(f"  {rank:2d}. {piece!r:20s} logprob={lp:8.4f}  token_id={tok_id}")


if __name__ == "__main__":
    main()
