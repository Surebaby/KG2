from __future__ import annotations

from scripts.diagnose.audit_2wiki_n300_root_projection_gap import (
    consumer_resolution,
    question_projection_state,
)


def _entity(qid: str | None) -> dict:
    return {
        "qid": qid,
        "abstained": qid is None,
    }


def test_delta_projection_cannot_inherit_root_from_removed_resolver_stack():
    state = question_projection_state(
        old_entities={"legacy-only": _entity("Q1"), "new-root": _entity(None)},
        new_entities={"legacy-only": _entity(None), "new-root": _entity("Q2")},
        resolver_rows=[{"root_anchor_surface": "new-root", "outcome": "positive"}],
    )

    assert state == {
        "old_resolution_state": "partial",
        "resolver_delta_projection_all_roots": True,
        "consumer_runtime_all_roots": False,
    }


def test_consumer_replay_uses_final_exact_precedence_and_fails_closed():
    common = {
        "title_cache": {"alpha (film)": "Q1"},
        "clean_aliases": {"alpha (film)": {"Q2"}, "beta": {"Q3"}},
        "entity_cache": {"alpha": "Q4", "gamma": "Q5"},
    }
    assert consumer_resolution(
        surface="Alpha", completed_surface="Alpha (film)", **common
    ) == ("new_exact_title_cache", "Q1")
    assert consumer_resolution(
        surface="Beta", completed_surface="Beta", **common
    ) == ("clean_v5_exact_alias", "Q3")
    assert consumer_resolution(
        surface="Gamma", completed_surface="Gamma", **common
    ) == ("new_exact_entity_cache", "Q5")
    assert consumer_resolution(
        surface="Missing", completed_surface="Missing", **common
    ) == ("all_clean_consumer_sources_miss", None)
