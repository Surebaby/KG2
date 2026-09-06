"""HotpotQA passage-aware Proof-KG builder — interface test framework.

These tests define the expected interface of the (not yet implemented) planner-free
builder.  They are skipped until the builder lands; when it does, remove the skip
and they become the regression suite.

Expected builder entry point (module TBD, likely
``scripts/prepare/build_hotpot_passage_aware_proofkg.py``):

    def build_hotpot_passage_aware_kg(
        question: str,
        retrieved_passages: list[dict],  # [{"id": ..., "contents": ...}]
        linker: EntityLinker,
        retriever: HistoricalWikidataPropertyRetriever,
        title_resolver: WikipediaTitleResolver | None,
    ) -> tuple[list[tuple[str, str, str]], dict]:
        # returns (triples, diagnostics) — triples <= 12, precision-first,
        # gold_access never read.
"""

from __future__ import annotations

import pytest

# Remove this skip once the builder is implemented.
pytest.skip("hotpot passage-aware builder not implemented", allow_module_level=True)


def _passages():
    return [{"id": "p1", "contents": "Entity One directed Film Two in 1990."}]


# --- mention extraction -----------------------------------------------------
def test_mention_extraction_from_question_and_passages():
    from scripts.prepare.build_hotpot_passage_aware_proofkg import extract_mentions

    mentions = extract_mentions("Who directed Film Two?", _passages())
    assert "Film Two" in mentions


# --- entity linking ---------------------------------------------------------
def test_high_confidence_linking_abstains_on_ambiguous():
    from scripts.prepare.build_hotpot_passage_aware_proofkg import link_high_confidence

    class AmbiguousLinker:
        def link_single(self, mention, *, question, retrieved_titles=None, passage_text=None):
            return type("R", (), {"abstained": True, "selected_qid": None})()

    linked = link_high_confidence("Ambiguous", AmbiguousLinker(), question="q", passages=_passages())
    assert linked == []


# --- relation detection -----------------------------------------------------
def test_relation_detection_uses_frozen_dictionary():
    from scripts.prepare.build_hotpot_passage_aware_proofkg import detect_relation_pids

    pids = detect_relation_pids("Who directed Film Two?")
    assert "P57" in pids  # director


# --- KG assembly ------------------------------------------------------------
def test_kg_assembly_precision_first_max_12():
    from scripts.prepare.build_hotpot_passage_aware_proofkg import assemble_kg

    triples = assemble_kg([("Film Two", "director", "Entity One")] * 20)
    assert len(triples) <= 12
    assert len(triples) == len(set(triples))


# --- constraints ------------------------------------------------------------
def test_no_gold_read():
    # The builder signature takes no gold arguments, and its output provenance
    # must carry gold_access=False.
    from scripts.prepare.build_hotpot_passage_aware_proofkg import build_hotpot_passage_aware_kg
    import inspect

    sig = inspect.signature(build_hotpot_passage_aware_kg)
    assert "gold" not in str(sig).lower()
    assert "answer" not in str(sig).lower()


def test_deterministic():
    from scripts.prepare.build_hotpot_passage_aware_proofkg import detect_relation_pids

    assert detect_relation_pids("Who directed Film Two?") == detect_relation_pids("Who directed Film Two?")
