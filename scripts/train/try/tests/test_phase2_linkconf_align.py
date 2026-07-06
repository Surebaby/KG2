"""Finding 2 regression: Phase 2's step-level link_confidence must equal what
PPO computes at inference for the same step text.

Both paths must end at the same number:
  compute_link_confidence(ParsedStep.from_text(body).mentioned_entities, EntityLinker())

Phase 2 path : parsed_step_from_silver_dict(step_dict) → .mentioned_entities
PPO path     : parse_steps("[Step N]\n<body>")[0]      → .mentioned_entities

If these diverge, the α-gate is trained on a different feature distribution than
it sees at PPO time (the original Finding 2 miscalibration). No GPU/model needed.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[4]  # .../kgpaper
for _d in (_ROOT, _HERE.parents[1] / "phase2_prm", _HERE.parents[1] / "shared"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from kgproweight.data.parsers import parse_steps, parsed_step_from_silver_dict
from kgproweight.kg.entity_linker import EntityLinker
from kgproweight.reward.alpha_gate import compute_link_confidence
from kgproweight.retrieval.bootstrap import resolve_entity_cache_path
from entity_filter_try import clean_entities


# Realistic multi-hop step bodies (entities of varying length incl. short ones).
STEP_BODIES = [
    "Reasoning: Albert Einstein was born in Ulm.\nConclusion: Einstein was born in Ulm.",
    "Reasoning: Ulm is located in Germany.\nConclusion: Ulm is in Germany.",
    "Reasoning: The film was directed by Christopher Nolan in the UK.\n"
    "Conclusion: Nolan directed it.",
    "Reasoning: No matching entity in the graph.\nConclusion: cannot determine.",
]


def _phase2_linkconf(body: str, linker: EntityLinker) -> float:
    """Exactly what _build_samples_accepted_only does per step."""
    step_dict = {"index": 0, "text": body, "label": 1, "cited_triples": []}
    parsed = parsed_step_from_silver_dict(step_dict, fallback_index=0)
    ents = clean_entities(parsed.mentioned_entities)
    return compute_link_confidence(step_entities=ents, entity_linker=linker)


def _ppo_linkconf(body: str, linker: EntityLinker) -> float:
    """Exactly what ppo_reward_try does per step: parse_steps over the response."""
    response = f"[Step 1]\n{body}\n[Final Answer]\nx"
    steps = parse_steps(response)
    assert steps, f"PPO parse produced no steps for: {body!r}"
    ents = clean_entities(steps[0].mentioned_entities)
    return compute_link_confidence(step_entities=ents, entity_linker=linker)


def test_linkconf_alignment():
    # Same cache both ends construct (Finding 2: real fix needs the populated cache,
    # else link_confidence ≡ 0 — aligned but dead).
    linker = EntityLinker(cache_path=resolve_entity_cache_path())
    cache_n = len(list(linker.cache.items()))
    print(f"  entity cache: {cache_n} entries")
    n_nonzero = 0
    for body in STEP_BODIES:
        p2 = _phase2_linkconf(body, linker)
        ppo = _ppo_linkconf(body, linker)
        assert abs(p2 - ppo) < 1e-9, (
            f"MISMATCH (Finding 2 not aligned)\n  body={body!r}\n  phase2={p2}\n  ppo={ppo}"
        )
        assert 0.0 <= p2 <= 1.0, f"link_conf out of [0,1]: {p2}"
        if p2 > 0.0:
            n_nonzero += 1
        print(f"  ok  p2={p2:.4f} == ppo={ppo:.4f}  | {body.splitlines()[0][:50]!r}")
    print(f"  ({n_nonzero}/{len(STEP_BODIES)} steps had nonzero link_confidence)")
    # With the real cache, the feature must be LIVE (not constant 0) on entity-bearing steps.
    if cache_n > 0:
        assert n_nonzero >= 2, "link_confidence is ~all-zero even with a populated cache — feature is dead"


def test_entities_identical():
    """Stronger: the parsed+filtered entity SETS themselves match, not just the score."""
    for body in STEP_BODIES:
        step_dict = {"index": 0, "text": body, "label": 1, "cited_triples": []}
        p2_ents = clean_entities(parsed_step_from_silver_dict(step_dict).mentioned_entities)
        ppo_ents = clean_entities(parse_steps(f"[Step 1]\n{body}\n[Final Answer]\nx")[0].mentioned_entities)
        assert p2_ents == ppo_ents, f"entity mismatch\n  body={body!r}\n  p2={p2_ents}\n  ppo={ppo_ents}"
        # scaffold must be gone
        assert "Reasoning" not in p2_ents and "Conclusion" not in p2_ents, f"scaffold leaked: {p2_ents}"
        print(f"  ok  entities={p2_ents}")


def test_abstention_step_has_no_entity():
    """The pure-abstention step ('No matching entity…') must yield NO real entity
    after filtering, so its link_confidence is 0 — not 0.82 from scaffold words."""
    linker = EntityLinker(cache_path=resolve_entity_cache_path())
    body = "Reasoning: No matching entity in the graph.\nConclusion: cannot determine."
    lc = _phase2_linkconf(body, linker)
    assert lc == 0.0, f"abstention step still scores {lc} (scaffold not stripped)"
    print(f"  ok  abstention link_confidence={lc}")


def main():
    print("test_linkconf_alignment"); test_linkconf_alignment()
    print("test_entities_identical"); test_entities_identical()
    print("test_abstention_step_has_no_entity"); test_abstention_step_has_no_entity()
    print("\nFINDING 2 ALIGNMENT TESTS PASSED ✅")


if __name__ == "__main__":
    main()
