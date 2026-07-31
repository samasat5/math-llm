"""
Extract (statement, proof) pairs from local Mathlib source.

Mathlib is already on disk after lean_server.start() runs lake.
Typical path: ~/.lean-bench/.lake/packages/mathlib/Mathlib/
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProofPair:
    statement: str   # includes `:= by sorry` for compatibility with check_proof
    proof: str       # tactic block (dedented)


def load_mathlib_proofs(
    mathlib_path: str | None = None,
    max_samples: int = 50_000,
) -> list[ProofPair]:
    if mathlib_path is None:
        base = Path.home() / ".lean-bench"
        candidates = [
            base / ".lake" / "packages" / "mathlib" / "Mathlib",
            base / ".lake" / "packages" / "mathlib",
            base,
        ]
        for p in candidates:
            if p.exists():
                mathlib_path = p
                break

    pairs: list[ProofPair] = []
    for lean_file in Path(mathlib_path).rglob("*.lean"):
        try:
            pairs.extend(_extract_pairs(lean_file.read_text(encoding="utf-8")))
        except Exception:
            continue
        if len(pairs) >= max_samples:
            break

    return pairs[:max_samples]


def _extract_pairs(source: str) -> list[ProofPair]:
    pairs = []
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        # Match: theorem/lemma NAME ... := by  (all on one line)
        if re.match(r'\s*(theorem|lemma)\s+\w+', line) and ':= by' in line:
            stmt = re.sub(r':=\s*by\s*$', ':= by sorry', line.rstrip())
            i += 1
            proof_lines = []
            while i < len(lines):
                pl = lines[i]
                if not pl.strip():          # blank line ends the block
                    break
                if not (pl.startswith('  ') or pl.startswith('\t')):
                    break                   # dedented = new declaration
                proof_lines.append(pl.strip())
                i += 1
            if proof_lines:
                proof = '\n'.join(proof_lines)
                if len(proof) < 800:        # skip giant proofs
                    pairs.append(ProofPair(statement=stmt.strip(), proof=proof))
        else:
            i += 1
    return pairs
