from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset

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
        ds = ds.filter(lambda r: r["formal_statement"])  # drop rows w/o a statement
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
            proof=r.get("formal_ground_truth"),
            informal=r.get("problem"),
        )
        for i, r in enumerate(split)
    ]