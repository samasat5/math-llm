"""Distribution of logP(deprecated) - logP(current) gaps across Mathlib's
real Nat.*/Int.* renamed-API surface (extracted from @[deprecated] aliases -
see outputs/mathlib_nat_int_deprecated_aliases.json).

For each (old, new) pair, scores both full identifiers from the same
minimal, generic prefix ("rw [") - not a problem-specific proof context,
since the pairs span unrelated areas of Mathlib (bitwise, computability,
floor/ceil, factorization, ...) and a fixed neutral context is the fair way
to compare staleness preference across the whole set.

Positive gap = model prefers the deprecated/old name (stale-training-data
risk); negative gap = model already prefers the current name.
"""

import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from math_llm.agents.case_study.temp_sweep_aime_1987_p5 import resolve_local_snapshot

MODEL = "Pythagoras-LM/Pythagoras-Prover-4B"
PREFIX = "rw ["
PAIRS_FILE = "outputs/mathlib_nat_int_deprecated_aliases.json"
OUT_FILE = "outputs/rename_gap_distribution.json"


def main():
    with open(PAIRS_FILE) as f:
        pairs = json.load(f)

    local_model_path = resolve_local_snapshot(MODEL)
    print(f"[dist] Loading tokenizer/model from {local_model_path}...")
    tok = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    model.eval()

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
        return result

    results = []
    for i, p in enumerate(pairs):
        old, new = p["old"], p["new"]
        lp_old = score(PREFIX, old + "]")
        lp_new = score(PREFIX, new + "]")
        gap = lp_old - lp_new
        results.append({"old": old, "new": new, "logP_old": lp_old, "logP_new": lp_new, "gap": gap})
        print(f"[{i+1}/{len(pairs)}] {old:45s} -> {new:45s}  logP_old={lp_old:8.3f} logP_new={lp_new:8.3f} gap={gap:8.3f}")
        torch.cuda.empty_cache()

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    gaps = [r["gap"] for r in results]
    gaps_sorted = sorted(gaps)
    n = len(gaps)
    mean = sum(gaps) / n
    median = gaps_sorted[n // 2] if n % 2 else (gaps_sorted[n // 2 - 1] + gaps_sorted[n // 2]) / 2
    prefers_old = sum(1 for g in gaps if g > 0)

    print(f"\n[dist] n={n}  mean={mean:.3f}  median={median:.3f}  min={min(gaps):.3f}  max={max(gaps):.3f}")
    print(f"[dist] prefers deprecated/old name: {prefers_old}/{n} ({100*prefers_old/n:.1f}%)")
    print(f"[dist] Saved full results to {OUT_FILE}")


if __name__ == "__main__":
    main()
