"""Why is r_kg ~0.012 when the policy cites plausible triples?

r_kg = precision x relevance (prm_annotator.py:238). Both factors have paths to
zero, and the batch mean cannot tell them apart. Replay the actual rollout
samples through the annotator and report WHICH factor collapses.

  python scripts/deploy/_rkg_diag.py outputs/split_ppo_smoke/samples/step_00040.txt
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kgproweight.data.parsers import parse_steps
from kgproweight.kg.entity_linker import EntityLinker
from kgproweight.retrieval.bootstrap import resolve_entity_cache_path
from kgproweight.reward.prm_annotator import PRMAnnotator, triple_in_subgraph

sample_path = Path(sys.argv[1] if len(sys.argv) > 1
                   else "outputs/split_ppo_smoke/samples/step_00040.txt")
text = sample_path.read_text(encoding="utf-8", errors="replace")

# Samples are separated by "--- Sample N ---".
blocks = [b for b in text.split("--- Sample ")[1:]]
print(f"{len(blocks)} rollout samples in {sample_path.name}\n")

linker = EntityLinker(cache_path=resolve_entity_cache_path(), offline=True)
ann = PRMAnnotator(entity_linker=linker, verbose=False)

n_steps = n_cited = 0
reasons: Counter = Counter()
prec_vals: list[float] = []
rel_vals: list[float] = []

for blk in blocks:
    body = blk.split("\n", 1)[1] if "\n" in blk else ""
    steps = parse_steps(body)
    for st in steps:
        n_steps += 1
        if not st.cited_triples:
            reasons["no cited triples -> NEUTRAL 0"] += 1
            continue
        n_cited += 1

        # Precision needs the subgraph this rollout actually saw. We do not have
        # it in the sample dump, so report the citation shape and let the
        # relevance factor be measured directly -- that one needs only the text.
        reasoning_only = st.raw_text.split("Knowledge Used:", 1)[0]
        rel_n = sum(
            1 for t in st.cited_triples
            if ann._triple_relevant([t], reasoning=reasoning_only,
                                    conclusion=st.intermediate_conclusion)
        )
        rel = rel_n / len(st.cited_triples)
        rel_vals.append(rel)
        if rel == 0:
            reasons["relevance = 0 (triple absent from reasoning body)"] += 1
        elif rel < 1:
            reasons["relevance partial"] += 1
        else:
            reasons["relevance = 1"] += 1

print(f"steps total        : {n_steps}")
print(f"steps with citations: {n_cited}")
print()
for k, v in reasons.most_common():
    print(f"  {v:4d}  {k}")

if rel_vals:
    print(f"\nrelevance_ratio: mean {sum(rel_vals) / len(rel_vals):.3f}, "
          f"zero in {sum(1 for r in rel_vals if r == 0)}/{len(rel_vals)}")
    print("\nr_kg = precision x relevance. If relevance is mostly 0, r_kg is 0")
    print("no matter how accurate the citations are -- the triples are real but")
    print("their surface form never reappears in the reasoning body, which is")
    print("what _triple_relevant matches on.")
