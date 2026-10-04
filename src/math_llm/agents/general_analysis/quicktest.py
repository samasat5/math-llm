"""quick_test_alias.py — sanity-check the alias chain on one known case
before trusting it over the whole hallucination list."""

from math_llm.lean_server import LeanServer
from math_llm.agents.general_analysis.alias_resolution_map import (
    resolve_chain,
    find_alias_hops_in_current_head,
    find_alias_hops_in_history,
)

name = "Nat.odd_iff_not_even"

# Step 1: confirm it's really gone from HEAD (sanity check against your report claim)
print("in current HEAD (deprecated-but-present)?", find_alias_hops_in_current_head(name))
print("in history (deleted alias)?", find_alias_hops_in_history(name))

lean_server = LeanServer()
lean_server.start()
try:
    result = resolve_chain(name, lean_server)
    print(result)
finally:
    lean_server.stop()