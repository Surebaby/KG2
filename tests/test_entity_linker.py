"""Unit tests for context-aware entity linking (R9 v7).

These exercise the pure scoring logic (``_score_candidates``) and the passage
context helpers — no network, no Wikidata search.
"""

from __future__ import annotations

from kgproweight.kg.entity_linker import (
    EntityLinker,
    LinkCandidate,
    build_passage_text,
    build_passage_titles,
    passage_title,
)


def _linker() -> EntityLinker:
    return EntityLinker(use_genre=False, offline=True)


def test_build_passage_helpers_handle_dict_and_str():
    passages = [
        {"id": "Evolution_(2001_film)", "contents": "Evolution is a 2001 American science fiction comedy film directed by Ivan Reitman."},
        {"title": "GNOME", "text": "Evolution is a personal information manager for the GNOME desktop."},
        "a bare string passage",
    ]
    assert build_passage_titles(passages) == [
        "Evolution_(2001_film)", "GNOME", "a bare string passage",
    ]
    text = build_passage_text(passages)
    assert "directed by Ivan Reitman" in text
    assert "personal information manager" in text
    assert "bare string passage" in text


def test_build_passage_helpers_handle_none():
    assert build_passage_titles(None) == []
    assert build_passage_text(None) == ""


def test_wiki18_numeric_id_uses_unquoted_contents_title():
    passage = {
        "id": "3722185",
        "contents": '"Stephen Frears"\nStephen Arthur Frears is a British director.',
    }
    assert passage_title(passage) == "Stephen Frears"
    assert build_passage_titles([passage]) == ["Stephen Frears"]
    assert build_passage_text([passage]).startswith('Stephen Frears "Stephen Frears"')


def test_passage_support_disambiguates_shared_label():
    """Same surface form ("Evolution") → film should beat software when the
    retrieved passage body is about the film."""
    film = LinkCandidate(qid="Q_FILM", label="Evolution",
                         description="2001 American science fiction comedy film")
    software = LinkCandidate(qid="Q_SOFTWARE", label="Evolution",
                             description="free and open-source personal information manager")
    passage_text = "Evolution is a 2001 American science fiction comedy film directed by Ivan Reitman."

    scored = _linker()._score_candidates(
        "Evolution",
        [software, film],
        question="Which director was involved in Evolution?",
        retrieved_titles=["Evolution"],
        passage_text=passage_text,
    )
    assert scored[0].qid == "Q_FILM"
    assert scored[0].score > scored[1].score


def test_score_candidates_without_passage_text_still_valid():
    """Backward compatibility: omitting passage context must not break scoring."""
    film = LinkCandidate(qid="Q_FILM", label="Evolution",
                         description="2001 American science fiction comedy film")
    software = LinkCandidate(qid="Q_SOFTWARE", label="Evolution",
                             description="free and open-source personal information manager")
    scored = _linker()._score_candidates(
        "Evolution",
        [software, film],
        question="Which director was involved in Evolution?",
    )
    assert all(0.0 <= c.score <= 1.0 for c in scored)


def test_offline_local_candidates_enable_scoring():
    """Offline mode now returns candidates from the desc index, so
    ``_score_candidates`` can disambiguate without any network call."""
    linker = _linker()
    linker._entity_index = {
        "evolution": [
            {"qid": "Q_SOFTWARE", "label": "Evolution",
             "description": "free and open-source personal information manager"},
            {"qid": "Q_FILM", "label": "Evolution",
             "description": "2001 American science fiction comedy film"},
        ],
    }
    result = linker.link_single(
        "Evolution",
        question="Which director was involved in Evolution?",
        passage_text="Evolution is a 2001 American science fiction comedy film directed by Ivan Reitman.",
    )
    assert not result.abstained
    assert result.selected_qid == "Q_FILM"


# ---------------------------------------------------------------------------
# 2026-08-23: multi-word scaffold phrases bypassed the filter entirely.
#
# entity_filter's module header names "Knowledge Used" and "Final Answer" as
# exactly the scaffold it exists to strip, but the test was
# ``key.lower() in _SCAFFOLD`` against a set of SINGLE tokens, so any phrase of
# two or more scaffold words passed straight through. Measured over the 33,011
# accepted silver steps: "Knowledge Used" survived on 99.9% of them and pure
# scaffold made up 17.2% of all kept mentions, each contributing a spurious
# link_confidence of 0.667 to the α-gate's f_confidence feature.
# ---------------------------------------------------------------------------


def test_multiword_scaffold_is_dropped():
    from kgproweight.data.entity_filter import clean_entities

    # The exact phrase that survived on 99.9% of silver steps.
    assert clean_entities(["Knowledge Used"]) == []
    assert clean_entities(["Final Answer"]) == []
    assert clean_entities(["Reasoning Steps"]) == []


def test_partial_scaffold_mention_is_kept():
    """Only mentions that are scaffold ALL THE WAY THROUGH may be dropped.

    A real entity that happens to contain a scaffold word must survive, or the
    filter would start deleting genuine mentions -- the opposite failure.
    """
    from kgproweight.data.entity_filter import clean_entities

    assert clean_entities(["Reasoning Museum"]) == ["Reasoning Museum"]
    assert clean_entities(["Albert Einstein"]) == ["Albert Einstein"]
    # "First" is scaffold, "Women" is not -> the mention is kept.
    assert clean_entities(["First Women"]) == ["First Women"]


def test_scaffold_fix_on_a_real_silver_step_mention_list():
    """The mention list from a real silver step, as ENTITY_RE produced it."""
    from kgproweight.data.entity_filter import clean_entities

    raw = ["Reasoning", "Arthur", "Magazine", "American", "Knowledge Used", "Conclusion"]
    assert clean_entities(raw) == ["Arthur", "Magazine", "American"]
