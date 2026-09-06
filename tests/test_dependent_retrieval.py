"""Unit tests for fail-closed plan-once dependent retrieval helpers."""

from copy import deepcopy

import pytest

from kgproweight.retrieval.dependent import (
    DependentRetrievalError,
    dependency_refs,
    extract_deterministic_bridge_candidates,
    instantiate_dependent_queries,
    merge_passages_with_provenance,
    normalize_dependency_ref,
    render_root_query,
    replace_dependency_refs,
    validate_plan_for_dependent_retrieval,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$hop_1", "slot_1"),
        ("hop_1", "slot_1"),
        ("$step_2", "slot_2"),
        ("step_2", "slot_2"),
        ("#3", "slot_3"),
        ("Step 1", None),
        ("#0", None),
    ],
)
def test_normalize_dependency_ref(raw, expected):
    assert normalize_dependency_ref(raw) == expected


def test_dependency_replacement_supports_all_three_syntaxes_and_beam():
    rendered = replace_dependency_refs(
        "$hop_1 met #2 before step_3",
        {
            "hop_1": ["Ada Lovelace", "Grace Hopper"],
            "step_2": "Charles Babbage",
            "slot_3": "London",
        },
        max_variants=2,
    )
    assert rendered == [
        "Ada Lovelace met Charles Babbage before London",
        "Grace Hopper met Charles Babbage before London",
    ]
    assert all(not dependency_refs(query) for query in rendered)


def test_dependency_replacement_fails_closed_on_missing_or_empty_value():
    with pytest.raises(DependentRetrievalError, match="unresolved dependencies"):
        replace_dependency_refs("#1 occupation", {})
    with pytest.raises(DependentRetrievalError, match="no usable values"):
        replace_dependency_refs("#1 occupation", {"step_1": []})


def test_render_relation_root_and_dependent_queries():
    root = {
        "subject": "Novum Organum",
        "relation_label": "author",
        "output_slot": "hop_1",
        "dependencies": [],
    }
    child = {
        "subject": "$hop_1",
        "relation_label": "father",
        "output_slot": "hop_2",
        "dependencies": ["hop_1"],
    }
    assert render_root_query(root, "relation_graph") == "Novum Organum author"
    assert instantiate_dependent_queries(
        child,
        "relation_graph",
        {"hop_1": ["Francis Bacon", "Roger Bacon"]},
    ) == ["Francis Bacon father", "Roger Bacon father"]


@pytest.mark.parametrize(
    ("subject", "expected"),
    [
        ("$hop_1 >> place of birth", "Francis Bacon place of birth"),
        ("$step_2 >> country", "Rabat Ajax F.C. place of birth"),
    ],
)
def test_relation_render_generically_discards_leaked_subquery_suffix(subject, expected):
    step = {
        "subject": subject,
        "relation_label": "place of birth",
        "output_slot": "hop_3",
        "dependencies": ["hop_1" if "hop_1" in subject else "step_2"],
    }
    values = {
        "hop_1": "Francis Bacon",
        "step_2": "Rabat Ajax F.C.",
    }
    assert instantiate_dependent_queries(step, "relation_graph", values) == [expected]


def test_hotpot_leaked_suffix_supplies_relation_only_when_explicit_label_is_missing():
    leaked = {
        "subject": "$hop_1 >> place of birth",
        "pid": "P19",
        "output_slot": "hop_2",
        "dependencies": ["hop_1"],
    }
    plan = {
        "steps": [
            {
                "subject": "Root Entity",
                "relation_label": "director",
                "pid": "P57",
                "output_slot": "hop_1",
                "dependencies": [],
            },
            leaked,
        ]
    }
    assert validate_plan_for_dependent_retrieval(plan, "relation_graph") == []
    assert instantiate_dependent_queries(
        leaked, "relation_graph", {"hop_1": "DRY"}
    ) == ["DRY place of birth"]

    # A separately emitted relation label remains authoritative; the suffix is
    # treated only as leaked syntax and cannot override it.
    explicit = dict(leaked, relation_label="country")
    assert instantiate_dependent_queries(
        explicit, "relation_graph", {"hop_1": "DRY"}
    ) == ["DRY country"]


def test_render_subquery_root_canonical_and_dependent_natural_language():
    root = {
        "subquery_template": "Rabat Ajax Football Ground >> occupant",
        "output_slot": "step_1",
        "dependencies": [],
    }
    child = {
        "subquery_template": "What league does #1 belong to?",
        "output_slot": "step_2",
        "dependencies": ["step_1"],
    }
    assert render_root_query(root, "subquery_graph") == (
        "Rabat Ajax Football Ground occupant"
    )
    assert instantiate_dependent_queries(
        child, "subquery_graph", {"$hop_1": "Rabat Ajax F.C."}
    ) == ["What league does Rabat Ajax F.C. belong to?"]


def test_root_render_rejects_dependency_and_relation_pid_only():
    with pytest.raises(DependentRetrievalError, match="root step contains a dependency"):
        render_root_query(
            {
                "subject": "$hop_1",
                "relation_label": "country",
                "dependencies": ["hop_1"],
            },
            "relation_graph",
        )
    with pytest.raises(DependentRetrievalError, match="no textual relation"):
        render_root_query(
            {"subject": "Ada Lovelace", "pid": "P106", "dependencies": []},
            "relation_graph",
        )


def test_plan_validation_accepts_topological_relation_and_subquery_plans():
    relation_plan = {
        "anchors": ["Novum Organum"],
        "steps": [
            {
                "subject": "Novum Organum",
                "relation_label": "author",
                "output_slot": "hop_1",
                "dependencies": [],
            },
            {
                "subject": "$hop_1",
                "relation_label": "father",
                "output_slot": "hop_2",
                "dependencies": ["hop_1"],
            },
        ],
    }
    subquery_plan = {
        "steps": [
            {
                "subquery_template": "A stadium >> occupant",
                "output_slot": "step_1",
                "dependencies": [],
            },
            {
                "subquery_template": "#1 >> member of",
                "output_slot": "step_2",
                "dependencies": ["step_1"],
            },
        ]
    }
    assert validate_plan_for_dependent_retrieval(relation_plan, "relation_graph") == []
    assert validate_plan_for_dependent_retrieval(subquery_plan, "subquery_graph") == []


def test_plan_validation_rejects_future_dependency_mismatch_duplicate_and_gold():
    plan = {
        "gold_answers": ["hidden"],
        "steps": [
            {
                "subject": "$hop_2",
                "relation_label": "author",
                "output_slot": "hop_1",
                "dependencies": [],
            },
            {
                "subject": "Book",
                "relation_label": "author",
                "output_slot": "step_1",
                "dependencies": [],
            },
        ],
    }
    errors = validate_plan_for_dependent_retrieval(plan, "relation_graph")
    assert any(error.startswith("prohibited_field:") for error in errors)
    assert any("unresolved_dependency:slot_2" in error for error in errors)
    assert any("dependency_declaration_mismatch" in error for error in errors)
    assert any("duplicate_output_slot:slot_1" in error for error in errors)


def test_bridge_candidates_are_deterministic_ranked_and_traceable():
    passages = [
        {
            "id": "d1",
            "title": "Rabat Ajax F.C.",
            "contents": "Rabat Ajax F.C.\nRabat Ajax F.C. plays in Malta.",
        },
        {
            "id": "d2",
            "title": "Maltese Premier League",
            "contents": "Maltese Premier League\nRabat Ajax F.C. won promotion.",
        },
        {
            "id": "d3",
            "title": "Rabat Ajax Football Ground",
            "contents": "Rabat Ajax Football Ground\nThe ground is in Malta.",
        },
    ]
    first = extract_deterministic_bridge_candidates(
        "Rabat Ajax Football Ground occupant",
        passages,
        exclude_surfaces=["Malta"],
        max_candidates=2,
    )
    second = extract_deterministic_bridge_candidates(
        "Rabat Ajax Football Ground occupant",
        passages,
        exclude_surfaces=["Malta"],
        max_candidates=2,
    )
    assert first == second
    assert [row["surface"] for row in first] == [
        "Rabat Ajax F.C.",
        "Maltese Premier League",
    ]
    assert first[0]["provenance"][0] == {
        "document_key": "id:d1",
        "rank": 1,
        "location": "title",
    }
    assert all(row["normalized_surface"] != "malta" for row in first)


def _doc(doc_id: str, title: str | None = None):
    return {"id": doc_id, "contents": f"{title or doc_id}\nbody for {doc_id}"}


def test_quota_merge_preserves_prefix_deduplicates_and_aggregates_provenance():
    original = [_doc(f"o{i}") for i in range(1, 7)]
    hops = [
        {
            "hop_id": "hop_1",
            "query": "first relation",
            "passages": [_doc("o1"), _doc("h1"), _doc("h2")],
        },
        {
            "hop_id": "hop_2",
            "query": "second relation",
            "passages": [_doc("h1"), _doc("h3")],
        },
    ]
    before_original, before_hops = deepcopy(original), deepcopy(hops)
    selected, telemetry = merge_passages_with_provenance(
        original, hops, original_quota=3, per_hop_quota=2, total=7
    )
    assert [row["id"] for row in selected] == ["o1", "o2", "o3", "h1", "h2", "h3", "o4"]
    assert [event["source"] for event in selected[0]["retrieval_provenance"]] == [
        "original_question",
        "dependent_query",
    ]
    assert [event["source"] for event in selected[3]["retrieval_provenance"]] == [
        "dependent_query",
        "dependent_query",
    ]
    assert telemetry["duplicate_paths_merged"] == 2
    assert telemetry["total_selected"] == 7
    assert telemetry["selected_by_source"] == {
        "hop_1": 2,
        "hop_2": 1,
        "original_backfill": 1,
        "original_prefix": 3,
    }
    assert original == before_original
    assert hops == before_hops


def test_quota_merge_is_deterministic_and_rejects_multiquery_shape_errors():
    original = [_doc("o1")]
    hops = [{"hop_id": "hop_1", "query": "q", "passages": [_doc("h1")]}]
    assert merge_passages_with_provenance(original, hops) == (
        merge_passages_with_provenance(original, hops)
    )
    with pytest.raises(DependentRetrievalError, match="invalid hop result"):
        merge_passages_with_provenance(
            original, [{"hop_id": "hop_1", "query": ["q1", "q2"], "passages": []}]
        )


def test_merge_rejects_invalid_quotas():
    with pytest.raises(DependentRetrievalError, match="cannot exceed"):
        merge_passages_with_provenance([], [], original_quota=11, total=10)
