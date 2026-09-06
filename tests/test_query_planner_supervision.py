import json

from scripts.prepare.build_query_planner_supervision import (
    _is_conservative_alias,
    _record_exclusion_reason,
    _record_leaks_source_values,
    _resolve_prior_output,
    build_2wiki_record,
    build_musique_record,
)
from kgproweight.kg.kg_filter import _RELATION_LABEL_TO_PID


def test_2wiki_supervision_replaces_intermediate_tail_and_omits_answers():
    row = {
        "id": "q1",
        "question": "Who...",
        "golden_answers": ["SECRET_FINAL"],
        "metadata": {
            "evidences": {
                "fact": ["Film A", "SECRET_DIRECTOR"],
                "relation": ["director", "country of citizenship"],
                "entity": ["SECRET_DIRECTOR", "SECRET_FINAL"],
            }
        },
    }
    record = build_2wiki_record(row)
    serialized = json.dumps(record)
    assert record["target"]["steps"][1]["subject"] == "$hop_1"
    assert record["target"]["steps"][0]["pid"] == "P57"
    assert "SECRET_DIRECTOR" not in serialized
    assert "SECRET_FINAL" not in serialized


def test_2wiki_parenthetical_head_resolves_to_prior_tail_slot():
    row = {
        "id": "q2", "question": "Which director was born earlier?", "golden_answers": ["x"],
        "metadata": {"evidences": {
            "fact": ["Film A", "Walter Edwards (director)"],
            "relation": ["director", "date of birth"],
            "entity": ["Walter Edwards", "1880"],
        }},
    }
    record = build_2wiki_record(row)
    assert record["target"]["steps"][1]["subject"] == "$hop_1"
    assert "Walter Edwards" not in record["target"]["anchors"]


def test_2wiki_unique_short_alias_resolves_but_ambiguous_alias_abstains():
    assert _is_conservative_alias("Nagaiah", "V. Nagaiah")
    assert _is_conservative_alias("Philip III", "Philip III of Spain")
    produced = {}
    assert _resolve_prior_output("Federico", produced, [("Federico Fellini", "$hop_1")]) == "$hop_1"
    assert _resolve_prior_output(
        "Federico", produced,
        [("Federico Fellini", "$hop_1"), ("Federico García Lorca", "$hop_2")],
    ) is None
    assert _is_conservative_alias("Sherry Horman", "Sherry Hormann")
    assert _is_conservative_alias("Arun Gandhi", "Arun Manilal Gandhi")
    assert not _is_conservative_alias("Charles I", "Charles III")


def test_leakage_check_ignores_json_keys_and_cross_value_token_boundaries():
    source = {
        "golden_answers": ["Abel", "U.S."],
        "metadata": {"evidences": {"entity": []}},
    }
    record = {
        "dataset": "2wikimultihopqa",
        "question": "Question without either answer",
        "target": {
            "relation_label": "father",
            "anchors": ["Lake Urru", "Steamboat Lake"],
        },
    }
    assert not _record_leaks_source_values(record, source)
    record["target"]["anchors"].append("Abel")
    assert _record_leaks_source_values(record, source)


def test_full_2wiki_relation_vocabulary_has_verified_pid_mappings():
    assert _RELATION_LABEL_TO_PID["has part"] == "P527"
    assert _RELATION_LABEL_TO_PID["presenter"] == "P371"
    assert _RELATION_LABEL_TO_PID["place of detention"] == "P2632"


def test_musique_supervision_keeps_only_subquery_templates_and_dependencies():
    row = {
        "id": "train_1",
        "question": "When was the owner founded?",
        "golden_answers": ["SECRET_FINAL"],
        "metadata": {
            "metadata": {
                "question_decomposition": [
                    {
                        "question": "Paper >> owned by",
                        "answer": "SECRET_OWNER",
                        "support_paragraph": {"title": "SECRET_TITLE", "paragraph_text": "SECRET_TEXT"},
                    },
                    {"question": "When was #1 founded?", "answer": "SECRET_FINAL"},
                ]
            }
        },
    }
    record = build_musique_record(row)
    serialized = json.dumps(record)
    assert record["target"]["steps"][1]["dependencies"] == ["step_1"]
    for secret in ("SECRET_OWNER", "SECRET_TITLE", "SECRET_TEXT", "SECRET_FINAL"):
        assert secret not in serialized


def test_literal_currency_and_number_sign_titles_are_not_dependency_slots():
    row = {
        "id": "q-dollar", "question": "Who directed $10 Raise?", "golden_answers": ["x"],
        "metadata": {"evidences": {
            "fact": ["$10 Raise", "Director X"],
            "relation": ["director", "country of citizenship"],
            "entity": ["Director X", "Country Y"],
        }},
    }
    record = build_2wiki_record(row)
    assert record["target"]["anchors"] == ["$10 Raise"]
    assert record["target"]["steps"][0]["dependencies"] == []

    musique = build_musique_record({
        "id": "m9", "question": "Whose father performed #9 Dream?", "golden_answers": ["x"],
        "metadata": {"metadata": {"question_decomposition": [
            {"question": "#9 Dream >> performer", "answer": "A"},
            {"question": "#1 >> father", "answer": "B"},
        ]}},
    })
    assert musique["target"]["steps"][0]["dependencies"] == []
    assert musique["target"]["steps"][1]["dependencies"] == ["step_1"]


def test_corrupt_one_character_anchor_is_explicitly_excluded():
    row = {
        "id": "q-corrupt", "question": "Who directed À L'Aventure?", "golden_answers": ["x"],
        "metadata": {"evidences": {
            "fact": [" À"], "relation": ["director"], "entity": ["Director X"],
        }},
    }
    record = build_2wiki_record(row)
    assert _record_exclusion_reason(record, row) == "invalid_degenerate_anchor"
    record["target"]["anchors"] = ["P"]
    record["provenance"]["has_degenerate_source_anchor"] = False
    assert _record_exclusion_reason(record, row) is None

    row["metadata"]["evidences"]["fact"] = ["  Film A "]
    normal = build_2wiki_record(row)
    assert normal["target"]["anchors"] == ["Film A"]
    assert _record_exclusion_reason(normal, row) is None
