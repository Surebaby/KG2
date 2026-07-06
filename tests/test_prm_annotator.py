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
    linker = EntityLinker(cache_path=str(cache_path), use_genre=False)
    return PRMAnnotator(entity_linker=linker, neutral_pattern_match=True)


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
    kg = [("Barack Obama", "spouse", "Michelle Obama")]
    assert ann.label(step, kg, []) == POSITIVE


def test_negative_when_triple_hallucinated(tmp_path):
    ann = _annotator(tmp_path)
    step = ParsedStep(
        index=2,
        raw_text="Reasoning: (Barack Obama, spouse, Hillary Clinton).",
        cited_triples=[("Barack Obama", "spouse", "Hillary Clinton")],
        mentioned_entities=["Barack Obama", "Hillary Clinton"],
        intermediate_conclusion="Wrong claim",
    )
    kg = [("Barack Obama", "spouse", "Michelle Obama")]
    assert ann.label(step, kg, []) == NEGATIVE


def test_negative_on_entity_drift(tmp_path):
    ann = _annotator(tmp_path)
    step = ParsedStep(
        index=3,
        raw_text="The president of Atlantis is Aquaman.",
        cited_triples=[],
        mentioned_entities=["Atlantis", "Aquaman"],
        intermediate_conclusion="Aquaman rules Atlantis",
    )
    kg = [("United States", "president", "Barack Obama")]
    assert ann.label(step, kg, []) == NEGATIVE


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
    kg = [("Barack Obama", "spouse", "Michelle Obama")]
    labels = ann.annotate_trajectory(steps, kg)
    assert labels == [NEUTRAL, POSITIVE, NEGATIVE]
