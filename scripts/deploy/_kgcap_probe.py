"""Is ppo_max_kg_triples actually enforced in the rendered prompt?

The feasibility check counted '(' inside the KG block, which overcounts when a
relation or entity label itself contains a parenthesis. Count LINES instead and
compare against the cap.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kgproweight.data.prompts import build_rl_messages

OPEN, CLOSE = "[Knowledge Graph Context]", "[End of Knowledge Graph]"


def block_of(triples, cap):
    msgs = build_rl_messages(question="q", retrieved_passages=[],
                             kg_triples=triples, top_k=15, max_kg_triples=cap)
    txt = "\n\n".join(m["content"] for m in msgs)
    return txt.split(OPEN, 1)[1].split(CLOSE, 1)[0]


# 100 triples in, cap 30. Relation label contains "(x)" on purpose so the
# paren-counting artefact is visible next to the honest line count.
tri = [(f"H{i}", "rel (x)", f"T{i}") for i in range(100)]
body = block_of(tri, 30)
lines = [ln for ln in body.split("\n") if ln.strip()]
print(f"100 triples in, cap 30 -> lines={len(lines)}  paren_count={body.count('(')}")
print(f"  first: {lines[0]!r}")
print(f"  last : {lines[-1]!r}")
print(f"  VERDICT: cap {'ENFORCED' if len(lines) <= 30 else 'NOT ENFORCED'}")

# Same, with clean labels: paren count should now equal the line count.
clean = [(f"H{i}", "rel", f"T{i}") for i in range(100)]
b2 = block_of(clean, 30)
l2 = [ln for ln in b2.split("\n") if ln.strip()]
print(f"\nclean labels, cap 30 -> lines={len(l2)}  paren_count={b2.count('(')}")
