"""PRM annotator labelling logic — three-class decisions on synthetic steps."""

from __future__ import annotations

import pytest

# Importing ``kgproweight.reward.*`` transitively imports torch through the
# alpha gate. Skip the whole module rather than failing collection when
# torch is not installed (e.g. fresh CI checkout).
pytest.importorskip("torch")

from kgproweight.data.parsers import ParsedStep
from kgproweight.kg.entity_linker import EntityLinker
from kgproweight.reward.prm_annotator import NEGATIVE, NEUTRAL, POSITIVE, PRMAnnotator


def _annotator(tmp_path):
    # An empty cache directory keeps the annotator from hitting Wikidata.
    cache_path = tmp_path / "entity_cache.jsonl"
    linker = EntityLinker(cache_path=str(cache_path), use_genre=False, offline=True)
    return PRMAnnotator(entity_linker=linker, neutral_pattern_match=True)


# A subgraph must hold at least ``min_subgraph_for_verify`` (3) triples before
# the annotator will verify OR refute a citation — a sparse graph cannot
# disprove anything (paper pain-point C2). These tests originally passed
# single-triple graphs, which the sparsity gate correctly sends to NEUTRAL, so
# they were asserting behaviour the policy deliberately does not have.
_FILLER = [
    ("Barack Obama", "occupation", "politician"),
    ("Barack Obama", "country of citizenship", "United States"),
]


def test_neutral_discourse_step(tmp_path):
    ann = _annotator(tmp_path)
    step = ParsedStep.from_text(
        0,
        "First, let's break down the question into sub-claims.",
    )
    assert step.cited_triples == []
    kg = [("Barack Obama", "spouse", "Michelle Obama")]
    assert ann.label(step, kg, []) == NEUTRAL


def test_positive_when_triple_grounded(tmp_path):
    ann = _annotator(tmp_path)
    step = ParsedStep(
        index=1,
        raw_text="Reasoning: (Barack Obama, spouse, Michelle Obama).",
        cited_triples=[("Barack Obama", "spouse", "Michelle Obama")],
        mentioned_entities=["Barack Obama", "Michelle Obama"],
        intermediate_conclusion="Barack is married to Michelle",
    )
    kg = [("Barack Obama", "spouse", "Michelle Obama"), *_FILLER]
    assert ann.label(step, kg, []) == POSITIVE


def test_sparse_subgraph_cannot_verify(tmp_path):
    """C2: a subgraph too small to refute anything yields NEUTRAL, not ±1."""
    ann = _annotator(tmp_path)
    step = ParsedStep(
        index=1,
        raw_text="Reasoning: (Barack Obama, spouse, Michelle Obama).",
        cited_triples=[("Barack Obama", "spouse", "Michelle Obama")],
        mentioned_entities=["Barack Obama"],
        intermediate_conclusion="Barack is married to Michelle",
    )
    assert ann.label(step, [("Barack Obama", "spouse", "Michelle Obama")], []) == NEUTRAL


def test_partial_precision_is_fractional(tmp_path):
    """R9: r_kg is CONTINUOUS (precision x relevance), not just {-1, 0, +1}.

    Guards the ``int(label)`` truncation bug: consumers that cast this to int
    map every partial credit to 0 (NEUTRAL) and silently discard the signal.
    """
    ann = _annotator(tmp_path)
    step = ParsedStep(
        index=1,
        raw_text="Reasoning: Obama married Michelle and Hillary.",
        cited_triples=[
            ("Barack Obama", "spouse", "Michelle Obama"),   # verified
            ("Barack Obama", "spouse", "Hillary Clinton"),  # not in subgraph
        ],
        mentioned_entities=["Barack Obama"],
        intermediate_conclusion="Obama married Michelle Obama",
    )
    kg = [("Barack Obama", "spouse", "Michelle Obama"), *_FILLER]
    label = ann.label(step, kg, [])
    assert 0.0 < float(label) < 1.0
    assert int(label) == 0  # documents exactly why int() must not be used


def test_taxonomic_triple_does_not_mask_relevant_one(tmp_path):
    """A leading taxonomic citation must not hide the relevant triples after it."""
    ann = _annotator(tmp_path)
    assert ann._triple_relevant(
        [("Barack Obama", "instance of", "human"),
         ("Barack Obama", "spouse", "Michelle Obama")],
        reasoning="Barack Obama married Michelle Obama.",
        conclusion="Michelle Obama is his spouse.",
    ) is True


def test_unverifiable_citation_is_neutral_not_negative(tmp_path):
    """An absent-from-subgraph citation is NEUTRAL (C2: don't punish KG gaps).

    Wikidata is incomplete, so "not in the subgraph" does not mean "false". Only
    a contradiction with a verified prior conclusion earns -1. This test
    previously expected NEGATIVE, which is the pre-R9 policy.
    """
    ann = _annotator(tmp_path)
    step = ParsedStep(
        index=2,
        raw_text="Reasoning: (Barack Obama, spouse, Hillary Clinton).",
        cited_triples=[("Barack Obama", "spouse", "Hillary Clinton")],
        mentioned_entities=["Barack Obama", "Hillary Clinton"],
        intermediate_conclusion="Wrong claim",
    )
    kg = [("Barack Obama", "spouse", "Michelle Obama"), *_FILLER]
    assert ann.label(step, kg, []) == NEUTRAL


def test_negative_on_contradiction_with_prior_conclusion(tmp_path):
    """Contradicting a prior conclusion is the ONLY negative trigger."""
    ann = _annotator(tmp_path)
    step = ParsedStep(
        index=2,
        raw_text="Reasoning: Obama was never married to Michelle Obama.",
        cited_triples=[],
        mentioned_entities=["Barack Obama"],
        intermediate_conclusion="Barack Obama was never married to Michelle Obama",
    )
    kg = [("Barack Obama", "spouse", "Michelle Obama"), *_FILLER]
    prev = ["Barack Obama was married to Michelle Obama"]
    assert ann.label(step, kg, prev) == NEGATIVE


def test_entity_drift_alone_is_not_negative(tmp_path):
    """Entity drift was removed as a standalone -1 trigger.

    It fired whenever a step's capitalised mentions failed to fuzzy-match the
    (noisy) subgraph, mislabelling legitimate world-knowledge steps.
    """
    ann = _annotator(tmp_path)
    step = ParsedStep(
        index=3,
        raw_text="The president of Atlantis is Aquaman.",
        cited_triples=[],
        mentioned_entities=["Atlantis", "Aquaman"],
        intermediate_conclusion="Aquaman rules Atlantis",
    )
    kg = [("United States", "president", "Barack Obama"), *_FILLER]
    assert ann.label(step, kg, []) == NEUTRAL


def test_honest_abstention_is_not_a_contradiction(tmp_path):
    """"No info in the KG" reports a gap; it must not be scored -1."""
    ann = _annotator(tmp_path)
    step = ParsedStep(
        index=2,
        raw_text="Reasoning: The knowledge graph does not contain Obama's spouse.",
        cited_triples=[],
        mentioned_entities=["Barack Obama"],
        intermediate_conclusion="The graph does not contain Barack Obama's spouse",
    )
    kg = [("Barack Obama", "spouse", "Michelle Obama"), *_FILLER]
    prev = ["Barack Obama was married to Michelle Obama"]
    assert ann.label(step, kg, prev) == NEUTRAL


def test_annotate_trajectory_returns_per_step_labels(tmp_path):
    ann = _annotator(tmp_path)
    steps = [
        ParsedStep.from_text(0, "Let's start by identifying entities."),
        ParsedStep(
            index=1,
            raw_text="(Barack Obama, spouse, Michelle Obama).",
            cited_triples=[("Barack Obama", "spouse", "Michelle Obama")],
            mentioned_entities=["Barack Obama"],
        ),
        ParsedStep(
            index=2,
            raw_text="(Barack Obama, spouse, Hillary Clinton).",
            cited_triples=[("Barack Obama", "spouse", "Hillary Clinton")],
            mentioned_entities=["Barack Obama"],
        ),
    ]
    kg = [("Barack Obama", "spouse", "Michelle Obama"), *_FILLER]
    labels = ann.annotate_trajectory(steps, kg)
    # discourse -> 0; verified+relevant -> +1; unverifiable -> 0 (NOT -1, see
    # test_unverifiable_citation_is_neutral_not_negative).
    assert labels == [NEUTRAL, POSITIVE, NEUTRAL]
