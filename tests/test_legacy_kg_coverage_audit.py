#!/usr/bin/env python3
"""Test suite for legacy_kg_coverage_audit.py"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.diagnose.legacy_kg_coverage_audit import (
    LayerDiagnosis,
    QuestionAudit,
    check_answer_in_triples,
    check_relation_coverage,
    classify_bottleneck,
    infer_target_relations,
)


def test_infer_target_relations_temporal():
    """Test temporal relation inference."""
    question = "When was Scott Derrickson born?"
    answer = "1966"
    relations = infer_target_relations(question, answer, "hotpotqa")

    assert "temporal" in relations
    assert "date_of_birth" in relations


def test_infer_target_relations_location():
    """Test location relation inference."""
    question = "Where is the Laleli Mosque located?"
    answer = "Istanbul"
    relations = infer_target_relations(question, answer, "hotpotqa")

    assert "location" in relations


def test_infer_target_relations_identity():
    """Test identity/occupation relation inference."""
    question = "Who directed the film Big Stone Gap?"
    answer = "Adriana Trigiani"
    relations = infer_target_relations(question, answer, "hotpotqa")

    assert "identity" in relations
    assert "director" in relations


def test_check_answer_in_triples_exact_match():
    """Test exact answer detection in triples."""
    triples = [
        ("Ed Wood", "country of citizenship", "United States"),
        ("Ed Wood", "occupation", "film director"),
        ("Scott Derrickson", "occupation", "film director"),
    ]
    answer = "United States"

    has_answer, count = check_answer_in_triples(triples, answer)

    assert has_answer is True
    assert count == 1


def test_check_answer_in_triples_partial_match():
    """Test partial answer detection (word overlap)."""
    triples = [
        ("Big Stone Gap", "director", "Adriana Trigiani"),
        ("Adriana Trigiani", "place of birth", "Greenwich Village"),
    ]
    answer = "Greenwich Village, New York"

    has_answer, count = check_answer_in_triples(triples, answer)

    assert has_answer is True  # "Greenwich Village" overlaps
    assert count >= 1


def test_check_answer_in_triples_no_match():
    """Test when answer is not in triples."""
    triples = [
        ("Ed Wood", "occupation", "film director"),
        ("Scott Derrickson", "occupation", "film director"),
    ]
    answer = "yes"

    has_answer, count = check_answer_in_triples(triples, answer)

    assert has_answer is False
    assert count == 0


def test_check_relation_coverage_match():
    """Test relation coverage when target relations present."""
    triples = [
        ("Ed Wood", "date of birth", "October 10, 1924"),
        ("Ed Wood", "country of citizenship", "United States"),
    ]
    target_relations = ["temporal", "date_of_birth"]

    has_relations, count = check_relation_coverage(triples, target_relations)

    assert has_relations is True
    assert count >= 1


def test_check_relation_coverage_no_match():
    """Test relation coverage when target relations absent."""
    triples = [
        ("Ed Wood", "occupation", "film director"),
        ("Ed Wood", "sex or gender", "male"),
    ]
    target_relations = ["temporal", "date_of_birth"]

    has_relations, count = check_relation_coverage(triples, target_relations)

    assert has_relations is False
    assert count == 0


def test_classify_bottleneck_no_mentions():
    """Test bottleneck classification when no mentions extracted."""
    diagnosis = LayerDiagnosis(
        mentions_extracted=0,
        mentions_linked=0,
    )

    bottleneck, reason, strategy = classify_bottleneck(diagnosis)

    assert bottleneck == "L1_NO_MENTIONS"
    assert strategy == "improve_mention_extraction"


def test_classify_bottleneck_linking_failure():
    """Test bottleneck classification when linking fails."""
    diagnosis = LayerDiagnosis(
        mentions_extracted=3,
        mentions_linked=0,
        qid_linking_quality=0.0,
    )

    bottleneck, reason, strategy = classify_bottleneck(diagnosis)

    assert bottleneck == "L2_LINKING_FAILURE"
    assert strategy == "fix_entity_linker"


def test_classify_bottleneck_low_linking_quality():
    """Test bottleneck classification when linking quality is poor."""
    diagnosis = LayerDiagnosis(
        mentions_extracted=5,
        mentions_linked=1,
        qid_linking_quality=0.2,  # < 0.4 threshold
        raw_total_triples=0,
    )

    bottleneck, reason, strategy = classify_bottleneck(diagnosis)

    assert bottleneck == "L2_LOW_LINKING_QUALITY"
    assert strategy == "improve_entity_disambiguation"


def test_classify_bottleneck_answer_not_in_raw():
    """Test bottleneck when answer entity missing from raw cache."""
    diagnosis = LayerDiagnosis(
        mentions_extracted=3,
        mentions_linked=3,
        qid_linking_quality=1.0,
        raw_total_triples=50,
        raw_has_answer_mentions=False,
        raw_has_target_relations=False,
    )

    bottleneck, reason, strategy = classify_bottleneck(diagnosis)

    assert bottleneck == "L3_ANSWER_NOT_IN_RAW"
    assert strategy == "passage_derived_required"


def test_classify_bottleneck_relation_missing():
    """Test bottleneck when target relations missing from raw cache."""
    diagnosis = LayerDiagnosis(
        mentions_extracted=3,
        mentions_linked=3,
        qid_linking_quality=1.0,
        raw_total_triples=50,
        raw_has_answer_mentions=True,
        raw_has_target_relations=False,
        target_relation_types=["temporal", "date_of_birth"],
    )

    bottleneck, reason, strategy = classify_bottleneck(diagnosis)

    assert bottleneck == "L3_RELATION_MISSING"
    assert strategy == "passage_derived_or_expand_cache"


def test_classify_bottleneck_filtering_loss():
    """Test bottleneck when all triples filtered out."""
    diagnosis = LayerDiagnosis(
        mentions_extracted=3,
        mentions_linked=3,
        qid_linking_quality=1.0,
        raw_total_triples=50,
        raw_has_answer_mentions=True,
        raw_has_target_relations=True,
        top12_triple_count=0,
    )

    bottleneck, reason, strategy = classify_bottleneck(diagnosis)

    assert bottleneck == "L4_COMPLETE_FILTERING_LOSS"
    assert strategy == "fix_filter_threshold"


def test_classify_bottleneck_filtering_removed_useful():
    """Test bottleneck when useful edges removed by filtering."""
    diagnosis = LayerDiagnosis(
        mentions_extracted=3,
        mentions_linked=3,
        qid_linking_quality=1.0,
        raw_total_triples=50,
        raw_has_answer_mentions=True,
        raw_has_target_relations=True,
        top12_triple_count=12,
        top12_has_useful_edges=False,
    )

    bottleneck, reason, strategy = classify_bottleneck(diagnosis)

    assert bottleneck == "L4_FILTERING_REMOVED_USEFUL"
    assert strategy == "fix_reranker"


def test_classify_bottleneck_downstream():
    """Test bottleneck when KG is available but something else fails."""
    diagnosis = LayerDiagnosis(
        mentions_extracted=3,
        mentions_linked=3,
        qid_linking_quality=1.0,
        raw_total_triples=50,
        raw_has_answer_mentions=True,
        raw_has_target_relations=True,
        top12_triple_count=12,
        top12_has_useful_edges=True,
    )

    bottleneck, reason, strategy = classify_bottleneck(diagnosis)

    assert bottleneck == "L5_DOWNSTREAM"
    assert strategy == "investigate_prompt_or_model"


def test_question_audit_dataclass():
    """Test QuestionAudit dataclass creation."""
    diagnosis = LayerDiagnosis(
        raw_total_triples=25,
        mentions_extracted=3,
        mentions_linked=2,
    )

    audit = QuestionAudit(
        dataset="hotpotqa",
        qid="dev_0",
        question="Were Scott Derrickson and Ed Wood of the same nationality?",
        answer="yes",
        legacy_kg_available=True,
        legacy_kg_triple_count=12,
        layers=diagnosis,
        bottleneck="L3_RELATION_MISSING",
        repair_potential="passage_derived_required",
        seed=46,
    )

    assert audit.dataset == "hotpotqa"
    assert audit.qid == "dev_0"
    assert audit.layers.raw_total_triples == 25
    assert audit.bottleneck == "L3_RELATION_MISSING"
    assert audit.audit_version == "legacy-kg-audit-v1"


def test_layer_diagnosis_defaults():
    """Test LayerDiagnosis default values."""
    diagnosis = LayerDiagnosis()

    assert diagnosis.raw_has_answer_mentions is False
    assert diagnosis.raw_answer_entity_count == 0
    assert diagnosis.mentions_extracted == 0
    assert diagnosis.target_relation_types == []
    assert diagnosis.bottleneck_layer == "UNKNOWN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
