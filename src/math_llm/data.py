"""
Data loading for Lean proof benchmarks.

Supports:
- dummy: Simple test problems for verification
- minif2f-lean4: Competition math problems from HuggingFace
"""

import random
import re
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
import json
from pathlib import Path

def build_trainable_set(raw_problems, lean_pool, out_path, target=3000):
    """Keep statements that compile on the CURRENT mathlib (v4.25.2), cache to disk."""
    out = Path(out_path)
    if out.exists():
        ids = set(json.loads(out.read_text()))
        return [p for p in raw_problems if p.id in ids]

    kept = []
    for p in raw_problems:
        # check the STATEMENT compiles — append a `sorry` body so an empty proof
        # still elaborates; we're testing the signature, not proving it.
        ok = lean_pool.check_compiles(p.statement + "\n  sorry")   # adapt to your API
        if ok:
            kept.append(p)
        if len(kept) >= target:
            break

    out.write_text(json.dumps([p.id for p in kept]))
    print(f"[filter] kept {len(kept)} compilable statements (scanned {raw_problems.index(p)+1})")
    return kept

@dataclass
class NuminaProblem:
    id: str
    statement: str                  
    answer: Optional[str] = None
    informal: Optional[str] = None  

_NUMINA_CACHE = None

def load_numina(n_samples=None, test=False, test_frac=0.1, seed=0):
    global _NUMINA_CACHE
    if _NUMINA_CACHE is None:
        ds = load_dataset("AI-MO/NuminaMath-LEAN", split="train")
        ds = ds.filter(lambda r: r["formal_statement"]) 
        ds = ds.shuffle(seed=seed)
        n_test = int(len(ds) * test_frac)
        _NUMINA_CACHE = {
            "test":  ds.select(range(n_test)),
            "train": ds.select(range(n_test, len(ds))),
        }
    split = _NUMINA_CACHE["test" if test else "train"]
    if n_samples is not None:
        split = split.select(range(min(n_samples, len(split))))
    return [
        NuminaProblem(
            id=r.get("uuid") or str(i),
            statement=r["formal_statement"],
            answer=r.get("answer"),
            informal=r.get("problem"),
        )
        for i, r in enumerate(split)
    ]

@dataclass
class Problem:
    """A Lean theorem proving problem."""
    id: str
    statement: str  # Lean 4 statement with 'sorry' placeholder
    description: Optional[str] = None  # Natural language description
    proof: Optional[str] = None  # Ground truth proof (if available)
    source: str = "unknown"
    difficulty: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "description": self.description,
            "proof": self.proof,
            "source": self.source,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
        }


# =============================================================================
# DUMMY DATA - Simple problems for testing
# =============================================================================

DUMMY_PROBLEMS = [
    Problem(
        id="dummy/power_one",
        statement="theorem power_one (n : Nat) : n ^ 1 = n := by sorry",
        description="Prove that any number to the power of 1 is itself",
        proof="ring",
        source="dummy",
        difficulty="easy",
    ),
    Problem(
        id="dummy/add_one",
        statement="theorem add_one : 1 + 1 = 2 := by sorry",
        description="Prove that 1 + 1 = 2",
        proof="norm_num",
        source="dummy",
        difficulty="trivial",
    ),
    Problem(
        id="dummy/nat_pos",
        statement="theorem nat_pos : 0 < 1 := by sorry",
        description="Prove that 0 is less than 1",
        proof="norm_num",
        source="dummy",
        difficulty="trivial",
    ),
    Problem(
        id="dummy/mul_comm",
        statement="theorem mul_comm_example : 2 * 3 = 3 * 2 := by sorry",
        description="Prove multiplication is commutative for 2 and 3",
        proof="ring",
        source="dummy",
        difficulty="trivial",
    ),
    Problem(
        id="dummy/add_zero",
        statement="theorem add_zero_example (n : Nat) : n + 0 = n := by sorry",
        description="Prove that adding zero doesn't change a number",
        proof="rfl",
        source="dummy",
        difficulty="easy",
    ),
    Problem(
        id="dummy/neg_neg",
        statement="theorem neg_neg_example (x : Int) : -(-x) = x := by sorry",
        description="Prove that double negation returns the original integer",
        proof="ring",
        source="dummy",
        difficulty="easy",
    ),
    Problem(
        id="dummy/square_nonneg",
        statement="theorem square_nonneg (x : Real) : 0 ≤ x^2 := by sorry",
        description="Prove that squares are non-negative",
        proof="apply sq_nonneg",
        source="dummy",
        difficulty="easy",
    ),
    Problem(
        id="dummy/abs_nonneg",
        statement="theorem abs_nonneg_example (x : Real) : 0 ≤ |x| := by sorry",
        description="Prove that absolute value is non-negative",
        proof="exact abs_nonneg x",
        source="dummy",
        difficulty="easy",
    ),
    Problem(
        id="dummy/sum_first_n",
        statement="theorem sum_first_three : (0 : Nat) + 1 + 2 = 3 := by sorry",
        description="Prove that 0 + 1 + 2 = 3",
        proof="norm_num",
        source="dummy",
        difficulty="trivial",
    ),
    Problem(
        id="dummy/iff_intro",
        statement="theorem iff_intro : (1 = 1) ↔ (2 = 2) := by sorry",
        description="Prove a simple iff statement",
        proof="constructor <;> intro <;> rfl",
        source="dummy",
        difficulty="easy",
    ),
]


def load_dummy(n_samples: Optional[int] = None) -> list[Problem]:
    """Load dummy problems for testing."""
    problems = DUMMY_PROBLEMS.copy()
    if n_samples is not None:
        problems = problems[:n_samples]
    return problems


# =============================================================================
# MINIF2F-LEAN4 - Competition math problems
# =============================================================================

def normalize_lean4_syntax(statement: str) -> str:
    """Normalize Lean syntax for Mathlib 4 compatibility."""
    # Convert "∑ k in S" to "∑ k ∈ S" (BigOperators notation)
    statement = re.sub(r'(∑\s*\w+)\s+in\s+', r'\1 ∈ ', statement)
    statement = re.sub(r'(∏\s*\w+)\s+in\s+', r'\1 ∈ ', statement)
    return statement


# Difficulty tier from miniF2F problem-id prefix. The dataset itself carries
# no difficulty field, so this buckets by provenance: tier 0 is the
# non-competition problems lifted from the MATH dataset (mathd_*/algebra_*/
# numbertheory_*/induction_*), which are consistently easier than the
# AMC/AIME/IMO competition problems in tiers 1-3.
TIER_PREFIXES = [
    ("mathd", 0),
    ("algebra", 0),
    ("numbertheory", 0),
    ("induction", 0),
    ("amc", 1),
    ("aimeII", 2),
    ("aimeI", 2),
    ("aime", 2),
    ("imosl", 3),
    ("imo", 3),
]


def compute_tier(problem_name: str) -> int:
    """Heuristic difficulty tier (0=easiest) from a miniF2F problem-id prefix."""
    name = problem_name.lower()
    for prefix, tier in TIER_PREFIXES:
        if name.startswith(prefix):
            return tier
    return 1  # unrecognized prefix -> assume competition-level, not easiest


def load_minif2f(
    n_samples: Optional[int] = None,
    test: bool = False,
    tier: Optional[int] = None,
    min_tier: Optional[int] = None,
    easiest_first: bool = False,
    shuffle_seed: Optional[int] = None,
) -> list[Problem]:
    """
    Load MiniF2F Lean 4 problems from HuggingFace.

    Dataset: cat-searcher/minif2f-lean4
    ~488 competition math problems (IMO, AMC, AIME, etc.)

    Args:
        n_samples: Optional limit on number of problems returned.
        test: If True, take the last n_samples instead of the first.
        tier: If set, keep only problems in this exact difficulty tier
            (0=MATH/mathd, 1=AMC, 2=AIME, 3=IMO). See compute_tier().
            Mutually exclusive with min_tier.
        min_tier: If set, keep problems with tier >= this value - e.g.
            min_tier=1 keeps every competition problem (AMC+AIME+IMO),
            excluding the easier tier-0 MATH-sourced ones. Mutually
            exclusive with tier.
        easiest_first: If True, sort by (tier, informal statement length)
            ascending before slicing, so the shortest/easiest problems in
            scope come first. Ignored if shuffle_seed is set.
        shuffle_seed: If set, deterministically shuffle problems (within
            whatever tier/min_tier scope applies) before slicing, so
            n_samples picks a reproducible random subset instead of a
            prefix. Takes precedence over easiest_first.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    print("[data] Loading minif2f-lean4 from HuggingFace...")
    ds = load_dataset("cat-searcher/minif2f-lean4", trust_remote_code=True)

    # Combine all splits
    all_items = []
    if hasattr(ds, 'keys'):
        for split_name in ds.keys():
            all_items.extend(ds[split_name])
    else:
        all_items = list(ds)

    problems = []
    for item in all_items:
        statement = item.get("formal_statement", item.get("statement", ""))
        name = item.get("id", item.get("name", item.get("problem_name", "")))

        # Normalize syntax
        statement = normalize_lean4_syntax(statement)

        # Get informal statement
        informal = item.get("informal_statement", item.get("informal_stmt", ""))

        # Determine tags from name
        tags = []
        if "imo" in name.lower():
            tags.append("IMO")
        elif "amc" in name.lower():
            tags.append("AMC")
        elif "aime" in name.lower():
            tags.append("AIME")

        problem = Problem(
            id=f"minif2f-lean4/{name}",
            statement=statement,
            description=informal,
            proof=item.get("proof"),
            source="minif2f-lean4",
            difficulty="competition",
            metadata={
                "tags": tags,
                "competition": item.get("source", ""),
                "year": item.get("year"),
                "tier": compute_tier(name),
            },
        )
        problems.append(problem)

    print(f"[data] Loaded {len(problems)} problems from minif2f-lean4")

    if tier is not None:
        problems = [p for p in problems if p.metadata.get("tier") == tier]
        print(f"[data] Filtered to tier {tier}: {len(problems)} problems")
    elif min_tier is not None:
        problems = [p for p in problems if p.metadata.get("tier", 0) >= min_tier]
        print(f"[data] Filtered to tier >= {min_tier}: {len(problems)} problems")

    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(problems)
    elif easiest_first:
        problems.sort(key=lambda p: (p.metadata.get("tier", 0), len(p.description or "")))

    if n_samples is not None:
        problems = problems[-n_samples:] if test else problems[:n_samples]

    return problems


# =============================================================================
# DATA LOADING API
# =============================================================================

DATASETS = {
    "dummy": load_dummy,
    "minif2f-lean4": load_minif2f,
}


def load_data(
    dataset: str,
    n_samples: Optional[int] = None,
    offset: int = 0,
    tier: Optional[int] = None,
    min_tier: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
) -> list[Problem]:
    """
    Load a benchmark dataset.

    Args:
        dataset: Dataset name ('dummy' or 'minif2f-lean4')
        n_samples: Optional limit on number of samples (None = all)
        offset: Number of problems to skip from the start (default 0)
        tier: Exact difficulty tier filter (minif2f-lean4 only). Mutually
            exclusive with min_tier. See compute_tier().
        min_tier: Keep tier >= this value, e.g. 1 for every competition
            problem (AMC+AIME+IMO). Mutually exclusive with tier.
        shuffle_seed: If set, deterministically shuffle before slicing to
            n_samples, picking a reproducible random subset instead of a
            prefix (minif2f-lean4 only).

    Returns:
        List of Problem objects
    """
    if dataset not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset: {dataset}. Available: {available}")

    if tier is not None or min_tier is not None or shuffle_seed is not None:
        if dataset != "minif2f-lean4":
            raise ValueError(f"--tier/--min-tier/--seed is only supported for minif2f-lean4, not '{dataset}'")
        loader = lambda n: load_minif2f(n, tier=tier, min_tier=min_tier, easiest_first=True, shuffle_seed=shuffle_seed)
    else:
        loader = DATASETS[dataset]

    if offset:
        problems = loader(None)
        end = offset + n_samples if n_samples is not None else None
        return problems[offset:end]

    return loader(n_samples)


def list_datasets() -> list[str]:
    """List available datasets."""
    return list(DATASETS.keys())
