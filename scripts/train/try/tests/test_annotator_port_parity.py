"""Differential test: ported package PRMAnnotator must label IDENTICALLY to the
validated try ImprovedPRMAnnotator across a battery of step cases.

If these diverge, the port changed labelling behavior — which would corrupt the
full-scale silver generation. No GPU/model needed.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[4]
for _d in (_ROOT, _HERE.parents[1] / "shared"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from kgproweight.data.parsers import ParsedStep
from kgproweight.reward.prm_annotator import PRMAnnotator
from prm_annotator_try import ImprovedPRMAnnotator


# (cited_triples, mentioned_entities, conclusion, raw_text) per step + a subgraph
SUBGRAPH = [
    ("Albert Einstein", "place of birth", "Ulm"),
    ("Ulm", "country", "Germany"),
    ("Christopher Nolan", "country of citizenship", "United Kingdom"),
    ("Inception", "director", "Christopher Nolan"),
    ("Germany", "capital", "Berlin"),
]

CASES = [
    # verified + relevant -> +1
    dict(cited=[("Albert Einstein", "place of birth", "Ulm")],
         concl="Einstein was born in Ulm", text="Reasoning: ... Conclusion: Einstein was born in Ulm",
         prev=[]),
    # verified but filler (triple unrelated to conclusion) -> 0
    dict(cited=[("Germany", "capital", "Berlin")],
         concl="The film won an award", text="Reasoning: ... Conclusion: The film won an award",
         prev=[]),
    # cited triple absent from usable subgraph -> 0 (not -1)
    dict(cited=[("Einstein", "spouse", "Mileva Maric")],
         concl="Einstein married Mileva", text="Conclusion: Einstein married Mileva",
         prev=[]),
    # no triples, honest abstention -> 0
    dict(cited=[], concl="The graph does not contain this information",
         text="Conclusion: The graph does not contain this information", prev=["Einstein was born in Ulm"]),
    # no triples, genuine contradiction -> -1
    dict(cited=[], concl="Einstein was never born in Ulm",
         text="Conclusion: Einstein was never born in Ulm", prev=["Einstein was born in Ulm"]),
    # no triples, plain factual claim, no contradiction -> 0
    dict(cited=[], concl="Nolan directed many films",
         text="Conclusion: Nolan directed many films", prev=[]),
    # discourse, no triples -> 0
    dict(cited=[], concl=None, text="Let us now examine the next clue.", prev=[]),
    # short-entity relevance (UK) verified+relevant -> +1
    dict(cited=[("Christopher Nolan", "country of citizenship", "United Kingdom")],
         concl="Nolan is a citizen of the United Kingdom",
         text="Conclusion: Nolan is a citizen of the United Kingdom", prev=[]),
    # "is identified" positive assertion contradicted -> should NOT be vetoed as abstention
    dict(cited=[], concl="Einstein is not a physicist",
         text="Conclusion: Einstein is not a physicist", prev=["Einstein is a physicist"]),
]

SPARSE_SUBGRAPH = [("A", "r", "B")]  # < min_subgraph_for_verify


def _mk(case):
    return ParsedStep(
        index=0,
        raw_text=case["text"],
        cited_triples=case["cited"],
        mentioned_entities=[],
        intermediate_conclusion=case["concl"],
    )


def test_label_parity():
    pkg = PRMAnnotator(verbose=False)
    try_ann = ImprovedPRMAnnotator(verbose=False)
    for i, case in enumerate(CASES):
        step = _mk(case)
        lp = pkg.label(step, SUBGRAPH, case["prev"])
        lt = try_ann.label(_mk(case), SUBGRAPH, case["prev"])
        assert lp == lt, f"case {i} MISMATCH: pkg={lp} try={lt} | {case}"
        print(f"  ok  case {i}: label={lp}  | {case['text'][:48]!r}")


def test_sparse_subgraph_parity():
    pkg = PRMAnnotator(verbose=False)
    try_ann = ImprovedPRMAnnotator(verbose=False)
    step = dict(cited=[("X", "r", "Y")], concl="X relates to Y", text="Conclusion: X relates to Y", prev=[])
    lp = pkg.label(_mk(step), SPARSE_SUBGRAPH, [])
    lt = try_ann.label(_mk(step), SPARSE_SUBGRAPH, [])
    assert lp == lt == 0, f"sparse subgraph should be NEUTRAL: pkg={lp} try={lt}"
    print(f"  ok  sparse subgraph -> {lp}")


def test_trajectory_parity():
    pkg = PRMAnnotator(verbose=False)
    try_ann = ImprovedPRMAnnotator(verbose=False)
    steps = [_mk(c) for c in CASES]
    lp = pkg.annotate_trajectory(steps, SUBGRAPH)
    lt = try_ann.annotate_trajectory([_mk(c) for c in CASES], SUBGRAPH)
    assert lp == lt, f"trajectory MISMATCH: pkg={lp} try={lt}"
    print(f"  ok  trajectory labels={lp}")


def main():
    print("test_label_parity"); test_label_parity()
    print("test_sparse_subgraph_parity"); test_sparse_subgraph_parity()
    print("test_trajectory_parity"); test_trajectory_parity()
    print("\nANNOTATOR PORT PARITY PASSED ✅")


if __name__ == "__main__":
    main()
