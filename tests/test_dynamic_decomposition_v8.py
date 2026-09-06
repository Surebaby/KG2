"""CPU-only tests for the Gold-free dynamic-decomposition v8 core."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

import kgproweight.retrieval.dynamic_decomposition_v8 as v8
import kgproweight.retrieval.dynamic_decomposition_v8_cohort as cohort
from kgproweight.retrieval.dynamic_decomposition_v8 import (
    BOUND_EXCERPT_MAX_CHARS,
    DynamicDecompositionV8Error,
    MAX_QUERY_CHARS,
    MAX_SUBANSWER_CHARS,
    NO_VERIFIED_SUBANSWER,
    NO_RELEVANT_ANSWER,
    Q2_ACTION_POLICY_VERSION,
    QueryParseError,
    SUBANSWER_CONTRACT_VERSION,
    SubanswerParseError,
    bind_subanswer_provenance,
    build_dynamic_q2_action,
    build_dynamic_q2_state,
    build_static_q2_action,
    build_static_q2_state,
    merge_fixed_budget_passages,
    parse_and_bind_subanswer,
    parse_query_response,
    parse_subanswer_response,
)


QUESTION = "Where was the director of Film Alpha born?"
Q1 = "Who directed Film Alpha?"
Q2_STATIC = "Where was the director of Film Alpha born?"
Q2_DYNAMIC = "Where was Jane Smith born?"


def _doc(
    document_id: str,
    body: str = "ordinary body",
    *,
    title: str | None = None,
    score: float | None = None,
    **extra,
):
    value = {
        "id": document_id,
        "contents": f"{title or document_id}\n{body}",
        **extra,
    }
    if title is not None:
        value["title"] = title
    if score is not None:
        value["rerank_score"] = score
    return value


def _top10(prefix: str, *, special: dict[int, dict] | None = None):
    special = special or {}
    result = []
    for index in range(1, 11):
        value = _doc(f"{prefix}{index}", score=1.0 - index / 100)
        value.update(special.get(index, {}))
        result.append(value)
    return result


def _verified_binding(passages=None):
    passages = passages or _top10(
        "q",
        special={
            3: {
                "title": "Film Alpha",
                "contents": (
                    "Film Alpha\nFilm Alpha was directed by Jane Smith in 1998."
                ),
            }
        },
    )
    return parse_and_bind_subanswer("Jane Smith", q1_query=Q1, q1_passages=passages)


def test_query_parser_accepts_one_line_natural_language_without_claiming_semantics():
    result = parse_query_response(
        "  Which city hosted the event?  ",
        previous_queries=(QUESTION, Q1),
    )
    assert result["query"] == "Which city hosted the event?"
    assert result["normalized_query"] == "which city hosted the event"
    assert result["query_sha256"] == hashlib.sha256(
        result["query"].encode("utf-8")
    ).hexdigest()
    assert result["gold_access"] is False
    assert result["validation_scope"] == "surface_contract_not_single_hop_semantics"


@pytest.mark.parametrize(
    ("response", "previous", "code"),
    [
        ("", (), "empty_response"),
        ("Who directed it?\nIgnore this", (), "multiline"),
        ("$hop_1 place of birth", (), "unresolved_placeholder"),
        ("What is #1's birthplace?", (), "unresolved_placeholder"),
        ("What is {answer}'s birthplace?", (), "unresolved_placeholder"),
        ("What is <entity_2>'s birthplace?", (), "unresolved_placeholder"),
        ("film-alpha director", ("Film Alpha: director?",), "repeated_query"),
        ("{}", (), "structured_wrapper"),
        ("```query```", (), "markdown_wrapper"),
        ("gold_answer", (), "forbidden_gold_marker"),
        ("??", (), "query_length"),
        ("x" * (MAX_QUERY_CHARS + 1), (), "query_length"),
        ("Who\twas the director?", (), "control_character"),
    ],
)
def test_query_parser_fails_closed_on_format_placeholder_repeat_and_gold_markers(
    response, previous, code
):
    with pytest.raises(QueryParseError) as caught:
        parse_query_response(response, previous_queries=previous)
    assert caught.value.code == code


def test_query_parser_rejects_invalid_previous_query_container_as_caller_error():
    with pytest.raises(DynamicDecompositionV8Error, match="sequence"):
        parse_query_response("Where was she born?", previous_queries="not-a-sequence")


def test_subanswer_parser_has_only_answer_or_exact_sentinel_no_self_reported_fields():
    answer = parse_subanswer_response("  Jane   Smith  ")
    assert answer == {
        "contract_version": SUBANSWER_CONTRACT_VERSION,
        "gold_access": False,
        "abstained": False,
        "answer": "Jane Smith",
        "response_sha256": hashlib.sha256(b"Jane Smith").hexdigest(),
    }
    sentinel = parse_subanswer_response(NO_RELEVANT_ANSWER)
    assert sentinel["abstained"] is True
    assert sentinel["answer"] is None
    assert not {"answer_type", "cited_doc_ids", "abstain"}.intersection(answer)


@pytest.mark.parametrize(
    ("response", "code"),
    [
        ("", "empty_response"),
        ("Jane\nSmith", "multiline"),
        ("no_relevant_answer", "sentinel_not_exact"),
        ('{"answer":"Jane Smith"}', "structured_wrapper"),
        ("```Jane Smith```", "markdown_wrapper"),
        ("supporting_facts", "forbidden_gold_marker"),
        ("---", "no_lexical_content"),
        ("x" * (MAX_SUBANSWER_CHARS + 1), "answer_too_long"),
    ],
)
def test_subanswer_parser_rejects_multiline_wrappers_coercion_and_gold(response, code):
    with pytest.raises(SubanswerParseError) as caught:
        parse_subanswer_response(response)
    assert caught.value.code == code


def test_binder_selects_unique_document_and_context_sentence_without_model_citation():
    passages = _top10(
        "q",
        special={
            3: {
                "title": "Jane Smith",
                "contents": (
                    "Jane Smith\nFilm Alpha was directed by Jane Smith in 1998."
                ),
                "gold_answer": "DO-NOT-CONSUME",
                "supporting_facts": ["DO-NOT-CONSUME"],
            }
        },
    )
    original = deepcopy(passages)
    result = parse_and_bind_subanswer("Jane Smith", q1_query=Q1, q1_passages=passages)

    assert result["verified"] is True
    assert result["verified_answer"] == "Jane Smith"
    assert result["supporting_doc_id"] == "q3"
    assert result["supporting_doc_rank"] == 3
    assert result["supporting_sentence"] == "Film Alpha was directed by Jane Smith in 1998."
    assert result["support_location"] == "text"
    assert result["matching_document_count"] == 1
    assert result["matching_unit_count"] == 2
    assert result["verification_scope"].endswith("not_semantic_entailment")
    assert result["gold_access"] is False
    assert "DO-NOT-CONSUME" not in json.dumps(result)
    assert passages == original


def test_binder_accepts_multiple_mentions_in_one_document_but_rejects_multiple_documents():
    one_document = _top10(
        "q",
        special={
            2: {
                "contents": "Entry\nJane Smith directed the film. Jane Smith was credited."
            }
        },
    )
    accepted = _verified_binding(one_document)
    assert accepted["verified"] is True
    assert accepted["matching_document_count"] == 1
    assert accepted["matching_unit_count"] == 2

    ambiguous = deepcopy(one_document)
    ambiguous[7]["contents"] = "Other\nA separate profile also names Jane Smith."
    rejected = _verified_binding(ambiguous)
    assert rejected["verified"] is False
    assert rejected["reason"] == "answer_surface_ambiguous_across_documents"
    assert rejected["matching_document_count"] == 2


@pytest.mark.parametrize(
    ("answer", "q1", "reason"),
    [
        (NO_RELEVANT_ANSWER, Q1, "reader_sentinel"),
        ("Film Alpha", Q1, "q1_surface_echo"),
        ("unknown", Q1, "null_like_answer"),
        ("yes", Q1, "non_extractive_boolean_answer"),
        ("Jane Smith", "Who directed a different film?", "answer_surface_not_found"),
    ],
)
def test_binder_fail_closed_reasons(answer, q1, reason):
    passages = _top10("q")
    result = parse_and_bind_subanswer(answer, q1_query=q1, q1_passages=passages)
    assert result["verified"] is False
    assert result["reason"] == reason
    assert result["verified_answer"] is None


def test_binder_is_nfkc_casefold_exact_and_obeys_word_and_numeric_boundaries():
    passages = _top10(
        "q",
        special={
            1: {
                "contents": "Profile\nBamboo Man\u0303alac fronted the group."
            },
            2: {"contents": "County\nYorkshire is a ceremonial county."},
            3: {"contents": "Statistic\nThe measured value was 38.5."},
            4: {"contents": "Year\nThe event happened in 1990."},
            5: {"contents": "Count\nThe official count was 42, according to records."},
            6: {"contents": "Ratio\nThe season record was 12/38."},
            7: {"contents": "Negative\nThe signed value was -38."},
            8: {"contents": "Positive\nThe signed value was +38."},
            9: {"contents": "Decimal\nThe proportional value was .38."},
        },
    )
    unicode_result = parse_and_bind_subanswer(
        "Bamboo Mañalac", q1_query="Who fronted the group?", q1_passages=passages
    )
    substring = parse_and_bind_subanswer(
        "York", q1_query="Which county is relevant?", q1_passages=passages
    )
    decimal = parse_and_bind_subanswer(
        "38", q1_query="What was the integer value?", q1_passages=passages
    )
    year = parse_and_bind_subanswer(
        "1990", q1_query="When did the event happen?", q1_passages=passages
    )
    count = parse_and_bind_subanswer(
        "42", q1_query="What was the official count?", q1_passages=passages
    )
    ratio_left = parse_and_bind_subanswer(
        "12", q1_query="What was the numerator?", q1_passages=passages
    )
    signed_negative = parse_and_bind_subanswer(
        "-38", q1_query="What was the negative signed value?", q1_passages=passages
    )
    signed_positive = parse_and_bind_subanswer(
        "+38", q1_query="What was the positive signed value?", q1_passages=passages
    )
    assert unicode_result["verified"] is True
    assert substring["reason"] == "answer_surface_not_found"
    assert decimal["reason"] == "answer_surface_not_found"
    assert year["verified"] is True
    assert count["verified"] is True
    assert ratio_left["reason"] == "answer_surface_not_found"
    assert signed_negative["verified"] is True
    assert signed_positive["verified"] is True


@pytest.mark.parametrize("expanding_character", ["ß", "İ"])
def test_casefold_expansion_offsets_map_back_to_excerpt_and_verified_surface(
    expanding_character,
):
    passages = _top10("q")
    passages[0]["contents"] = (
        "Expansion\n" + expanding_character * 300 + " Jane Smith " + "x" * 5000
    )
    result = parse_and_bind_subanswer(
        "Jane Smith",
        q1_query=Q1,
        q1_passages=passages,
    )
    assert result["verified"] is True
    assert "Jane Smith" in result["bound_evidence_excerpt"]
    excerpt_surface = result["bound_evidence_excerpt"][
        result["bound_excerpt_surface_start"] : result["bound_excerpt_surface_end"]
    ]
    sentence_surface = result["supporting_sentence"][
        result["surface_start"] : result["surface_end"]
    ]
    assert excerpt_surface.casefold() == "Jane Smith".casefold()
    assert sentence_surface.casefold() == "Jane Smith".casefold()


def test_binder_bounds_long_evidence_excerpt_while_retaining_answer():
    passages = _top10("q")
    passages[4]["contents"] = "Long\n" + "a" * 400 + " Jane Smith " + "b" * 400
    result = _verified_binding(passages)
    assert result["verified"] is True
    assert len(result["bound_evidence_excerpt"]) == BOUND_EXCERPT_MAX_CHARS
    assert "Jane Smith" in result["bound_evidence_excerpt"]


def test_parse_failure_is_telemetry_but_broken_top10_is_caller_error():
    passages = _top10("q")
    parse_failure = parse_and_bind_subanswer(
        "Jane\nSmith", q1_query=Q1, q1_passages=passages
    )
    assert parse_failure["verified"] is False
    assert parse_failure["reason"] == "parse_error:multiline"
    assert parse_failure["parsed"] is False

    with pytest.raises(DynamicDecompositionV8Error, match="exactly 10"):
        parse_and_bind_subanswer("Jane Smith", q1_query=Q1, q1_passages=passages[:9])
    duplicate = deepcopy(passages)
    duplicate[1]["id"] = duplicate[0]["id"]
    with pytest.raises(DynamicDecompositionV8Error, match="duplicate document"):
        parse_and_bind_subanswer("Jane Smith", q1_query=Q1, q1_passages=duplicate)


def test_binder_cannot_verify_surface_beyond_model_visible_1200_chars():
    passages = _top10("q")
    passages[2] = _doc(
        "q3",
        title="Film Alpha",
        body=("x" * 1200) + " Jane Smith appears only outside the visible slice.",
    )
    binding = parse_and_bind_subanswer(
        "Jane Smith", q1_query=Q1, q1_passages=passages
    )
    assert binding["verified"] is False
    assert binding["reason"] == "answer_surface_not_found"


def test_direct_binder_rejects_forged_parser_contract():
    with pytest.raises(DynamicDecompositionV8Error, match="contract version"):
        bind_subanswer_provenance(
            {"contract_version": "forged", "gold_access": False, "abstained": False, "answer": "x"},
            q1_query=Q1,
            q1_passages=_top10("q"),
        )


def test_q2_states_are_allowlisted_and_dynamic_state_uses_system_bound_provenance():
    binding = _verified_binding()
    binding["gold_answer"] = "DO-NOT-LEAK"
    static = build_static_q2_state(original_question=QUESTION, q1_query=Q1)
    dynamic = build_dynamic_q2_state(
        original_question=QUESTION,
        q1_query=Q1,
        binding=binding,
    )
    assert static == {
        "state_version": Q2_ACTION_POLICY_VERSION,
        "mode": "q2_no_verified_subanswer",
        "gold_access": False,
        "original_question": QUESTION,
        "q1_query": Q1,
        "verified_subanswer": NO_VERIFIED_SUBANSWER,
    }
    assert dynamic["verified_subanswer"] == "Jane Smith"
    assert dynamic["bound_evidence"]["supporting_doc_id"] == "q3"
    assert "DO-NOT-LEAK" not in json.dumps(dynamic)
    assert "supporting_sentence" not in dynamic["bound_evidence"]


@pytest.mark.parametrize(
    "missing_field",
    ["verified_answer", "supporting_doc_id", "bound_evidence_excerpt"],
)
def test_dynamic_q2_state_reports_controlled_error_for_missing_verified_provenance(
    missing_field,
):
    binding = _verified_binding()
    del binding[missing_field]
    with pytest.raises(
        DynamicDecompositionV8Error,
        match="verified binding lacks provenance fields",
    ):
        build_dynamic_q2_state(
            original_question=QUESTION,
            q1_query=Q1,
            binding=binding,
        )


def test_static_q2_action_uses_valid_proposal_or_original_question_fallback():
    valid = build_static_q2_action(
        "Where was Film Alpha's director born?",
        original_question=QUESTION,
        q1_query=Q1,
    )
    invalid = build_static_q2_action(
        Q1,
        original_question=QUESTION,
        q1_query=Q1,
    )
    assert valid["selected_query"] == "Where was Film Alpha's director born?"
    assert valid["selection_source"] == "q2_static"
    assert valid["used_fallback"] is False
    assert invalid["proposal_valid"] is False
    assert invalid["parse_error"] == "repeated_query"
    assert invalid["selected_query"] == QUESTION
    assert invalid["selection_source"] == "original_question"


def test_invalid_static_proposal_safely_falls_back_to_original_question_with_literal_hash_one():
    question_with_literal_chart_rank = (
        "Which artist debuted at #1 on the national singles chart?"
    )
    action = build_static_q2_action(
        "$hop_1 birthplace",
        original_question=question_with_literal_chart_rank,
        q1_query="Which artist made the debut?",
    )
    assert action["proposal_valid"] is False
    assert action["parse_error"] == "unresolved_placeholder"
    assert action["selected_query"] == question_with_literal_chart_rank
    assert action["selection_source"] == "original_question"
    assert action["used_fallback"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda action: action.update(proposal_valid=1), "proposal_valid must be boolean"),
        (lambda action: action.update(used_fallback=1), "used_fallback must be boolean"),
        (lambda action: action.update(used_fallback=True), "valid static action cannot use fallback"),
        (
            lambda action: action.update(proposal_query="Where was somebody else born?"),
            "proposal and selected query must match",
        ),
        (
            lambda action: action.update(controller_state_sha256="0" * 64),
            "controller state hash mismatch",
        ),
        (lambda action: action.update(response_sha256=None), "lacks response SHA"),
        (
            lambda action: action.update(
                proposal_query="$hop_1 birthplace",
                selected_query="$hop_1 birthplace",
            ),
            "selected query fails contract reparse",
        ),
        (lambda action: action.update(extra_field=True), "field set mismatch"),
    ],
)
def test_static_action_validator_rejects_forged_or_internally_inconsistent_action(
    mutation,
    message,
):
    static = build_static_q2_action(
        Q2_STATIC + " using records",
        original_question=QUESTION,
        q1_query=Q1,
    )
    mutation(static)
    with pytest.raises(DynamicDecompositionV8Error, match=message):
        v8._validate_static_action(static, QUESTION, Q1)


def test_dynamic_selector_rejects_forged_invalid_static_fallback_fields():
    static = build_static_q2_action(Q1, original_question=QUESTION, q1_query=Q1)
    static["selected_query"] = "Where was somebody else born?"
    with pytest.raises(DynamicDecompositionV8Error, match="fall back byte-for-byte"):
        v8._validate_static_action(static, QUESTION, Q1)


def test_dynamic_q2_selects_valid_action_and_invalid_output_falls_back_to_original_q():
    binding = _verified_binding()
    selected = build_dynamic_q2_action(
        Q2_DYNAMIC,
        original_question=QUESTION,
        q1_query=Q1,
        binding=binding,
    )
    invalid = build_dynamic_q2_action(
        "$hop_1 birthplace",
        original_question=QUESTION,
        q1_query=Q1,
        binding=binding,
    )
    assert selected["dynamic_eligible"] is True
    assert selected["selected_query"] == Q2_DYNAMIC
    assert selected["selection_source"] == "q2_dynamic"
    assert invalid["proposal_valid"] is False
    assert invalid["selected_query"] == QUESTION
    assert invalid["selection_source"] == "original_question"
    assert invalid["fallback_reason"] == "invalid_q2_dynamic:unresolved_placeholder"
    assert not {"controller_calls", "model_calls", "call_count"}.intersection(invalid)


def test_unverified_a1_is_rejected_by_dynamic_action_builder():
    binding = parse_and_bind_subanswer(
        NO_RELEVANT_ANSWER, q1_query=Q1, q1_passages=_top10("q")
    )
    with pytest.raises(DynamicDecompositionV8Error, match="verified Gold-free binding"):
        build_dynamic_q2_action(
            "This response is ignored\nand malformed",
            original_question=QUESTION,
            q1_query=Q1,
            binding=binding,
        )


def test_missing_dynamic_response_falls_directly_to_original_question():
    action = build_dynamic_q2_action(
        None,
        original_question=QUESTION,
        q1_query=Q1,
        binding=_verified_binding(),
    )
    assert action["parse_error"] == "missing_dynamic_response"
    assert action["selected_query"] == QUESTION
    assert action["selection_source"] == "original_question"


def test_merge_exact_allocation_order_scores_provenance_and_gold_field_projection():
    root = _top10("r")
    q1 = _top10("a")
    q2 = _top10("b")
    q1[0]["gold_answer"] = "DO-NOT-LEAK"
    original = deepcopy((root, q1, q2))

    selected, telemetry = merge_fixed_budget_passages(
        root,
        q1,
        q2,
        root_query=QUESTION,
        q1_query=Q1,
        q2_query=Q2_DYNAMIC,
    )
    assert [row["id"] for row in selected] == [
        "r1", "r2", "r3", "r4", "r5", "r6", "a1", "a2", "b1", "b2"
    ]
    assert telemetry["selected_by_slot"] == {
        "root_prefix": 6,
        "q1_novel": 2,
        "q2_novel": 2,
        "root_backfill": 0,
    }
    assert telemetry["total_selected"] == 10
    assert telemetry["output_unique_document_count"] == 10
    assert selected[6]["retrieval_provenance"] == [
        {
            "source": "q1",
            "query": Q1,
            "query_sha256": hashlib.sha256(Q1.encode()).hexdigest(),
            "rank": 1,
            "score": 0.99,
        }
    ]
    assert "gold_answer" not in selected[6]
    assert (root, q1, q2) == original


def test_merge_deduplicates_and_backfills_only_from_root_ranks_7_to_10():
    root = _top10("r")
    q1 = [deepcopy(root[0]) for _ in range(10)]
    q2 = [deepcopy(root[1]) for _ in range(10)]
    selected, telemetry = merge_fixed_budget_passages(
        root,
        q1,
        q2,
        root_query=QUESTION,
        q1_query=Q1,
        q2_query=Q2_STATIC,
    )
    assert [row["id"] for row in selected] == [f"r{i}" for i in range(1, 11)]
    assert telemetry["selected_by_slot"] == {
        "root_prefix": 6,
        "q1_novel": 0,
        "q2_novel": 0,
        "root_backfill": 4,
    }
    assert selected[0]["retrieval_provenance"][1]["source"] == "q1"
    assert selected[1]["retrieval_provenance"][1]["source"] == "q2"


def test_merge_prioritizes_verified_q1_support_document_into_two_novel_slots():
    root = _top10("r")
    q1 = _top10(
        "q",
        special={
            8: {
                "title": "Film Alpha",
                "contents": "Film Alpha\nFilm Alpha was directed by Jane Smith.",
            }
        },
    )
    q2 = _top10("b")
    binding = _verified_binding(q1)

    selected, telemetry = merge_fixed_budget_passages(
        root,
        q1,
        q2,
        root_query=QUESTION,
        q1_query=Q1,
        q2_query=Q2_DYNAMIC,
        q1_binding=binding,
    )
    assert [row["id"] for row in selected[6:10]] == ["q8", "q1", "b1", "b2"]
    assert telemetry["q1_binding_document_key"] == "id:q8"
    assert telemetry["q1_binding_prioritized_into_novel_slot"] is True
    assert telemetry["selected_by_slot"]["q1_novel"] == 2


def test_merge_binding_already_in_root_prefix_keeps_two_other_q1_novel_slots():
    root = _top10("r")
    root[2]["title"] = "Film Alpha"
    root[2]["contents"] = "Film Alpha\nFilm Alpha was directed by Jane Smith."
    q1 = [deepcopy(root[2]), *_top10("q")[:9]]
    q2 = _top10("b")
    binding = _verified_binding(q1)
    selected, telemetry = merge_fixed_budget_passages(
        root,
        q1,
        q2,
        root_query=QUESTION,
        q1_query=Q1,
        q2_query=Q2_DYNAMIC,
        q1_binding=binding,
    )
    assert [row["id"] for row in selected[6:10]] == ["q1", "q2", "b1", "b2"]
    assert telemetry["q1_binding_prioritized_into_novel_slot"] is False
    assert [event["source"] for event in selected[2]["retrieval_provenance"]] == [
        "root", "q1"
    ]


def test_merge_rejects_topk_drift_root_duplicates_and_identity_content_collision():
    root, q1, q2 = _top10("r"), _top10("q"), _top10("b")
    with pytest.raises(DynamicDecompositionV8Error, match="exactly 10"):
        merge_fixed_budget_passages(
            root[:9], q1, q2, root_query=QUESTION, q1_query=Q1, q2_query=Q2_DYNAMIC
        )

    duplicate_root = deepcopy(root)
    duplicate_root[1] = deepcopy(duplicate_root[0])
    with pytest.raises(DynamicDecompositionV8Error, match="duplicate document"):
        merge_fixed_budget_passages(
            duplicate_root,
            q1,
            q2,
            root_query=QUESTION,
            q1_query=Q1,
            q2_query=Q2_DYNAMIC,
        )

    collision_q1 = deepcopy(q1)
    collision_q1[0] = _doc("r1", body="different prompt-visible bytes")
    with pytest.raises(DynamicDecompositionV8Error, match="different prompt-visible bytes"):
        merge_fixed_budget_passages(
            root,
            collision_q1,
            q2,
            root_query=QUESTION,
            q1_query=Q1,
            q2_query=Q2_DYNAMIC,
        )


def test_merge_is_byte_deterministic_for_same_inputs():
    kwargs = dict(
        root_passages=_top10("r"),
        q1_passages=_top10("q"),
        q2_passages=_top10("b"),
        root_query=QUESTION,
        q1_query=Q1,
        q2_query=Q2_DYNAMIC,
    )
    first = merge_fixed_budget_passages(**kwargs)
    second = merge_fixed_budget_passages(**kwargs)
    assert json.dumps(first, ensure_ascii=False, sort_keys=True) == json.dumps(
        second, ensure_ascii=False, sort_keys=True
    )


def _cohort_rows():
    return [
        {
            "dataset": dataset,
            "qid": f"{dataset}-dev-{index:02d}",
            "question": f"Which target belongs to {dataset} item {index}?",
        }
        for dataset in cohort.DATASETS
        for index in range(cohort.DEVELOPMENT_PER_DATASET)
    ]


def _write_locked_cohort_fixture(monkeypatch, tmp_path, rows=None):
    rows = _cohort_rows() if rows is None else rows
    freeze_dir = tmp_path / "frozen"
    freeze_dir.mkdir(parents=True)
    development = freeze_dir / "development.identity_only.jsonl"
    prospective = freeze_dir / "prospective.identity_only.jsonl"
    with development.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    prospective.write_text("SEALED; TEST MUST NOT OPEN\n", encoding="utf-8")
    development_sha = hashlib.sha256(development.read_bytes()).hexdigest()
    prospective_sha = hashlib.sha256(prospective.read_bytes()).hexdigest()
    manifest = {
        "schema_version": cohort.FREEZE_MANIFEST_SCHEMA_VERSION,
        "experiment_id": cohort.FREEZE_EXPERIMENT_ID,
        "status": cohort.FREEZE_STATUS,
        "selection_contains_gold": False,
        "output_row_field_allowlist": list(cohort.ROW_FIELDS),
        "outputs": [
            {"path": development.name, "sha256": development_sha},
            {"path": prospective.name, "sha256": prospective_sha},
        ],
    }
    manifest_path = freeze_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    monkeypatch.setattr(cohort, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cohort, "FREEZE_DIRECTORY_RELATIVE", cohort.Path("frozen"))
    monkeypatch.setattr(cohort, "MANIFEST_RELATIVE_PATH", cohort.Path("frozen/manifest.json"))
    monkeypatch.setattr(
        cohort,
        "DEVELOPMENT_RELATIVE_PATH",
        cohort.Path("frozen/development.identity_only.jsonl"),
    )
    monkeypatch.setattr(
        cohort,
        "PROSPECTIVE_RELATIVE_PATH",
        cohort.Path("frozen/prospective.identity_only.jsonl"),
    )
    monkeypatch.setattr(cohort, "EXPECTED_MANIFEST_SHA256", manifest_sha)
    monkeypatch.setattr(cohort, "EXPECTED_DEVELOPMENT_SHA256", development_sha)
    monkeypatch.setattr(cohort, "SEALED_PROSPECTIVE_SHA256", prospective_sha)
    return manifest_path, development, prospective


def test_strict_cohort_loader_accepts_only_locked_development_identity_projection(
    monkeypatch, tmp_path
):
    _write_locked_cohort_fixture(monkeypatch, tmp_path)
    loaded = cohort.load_frozen_v8_cohort()
    assert loaded["role"] == "development"
    assert loaded["gold_access"] is False
    assert loaded["prospective_unlocked"] is False
    assert loaded["row_count"] == 90
    assert loaded["per_dataset_counts"] == {
        "hotpotqa": 30,
        "2wikimultihopqa": 30,
        "musique": 30,
    }
    assert all(set(row) == {"dataset", "qid", "question"} for row in loaded["rows"])


def test_development_loader_rejects_prospective_role_or_path_before_cohort_read(
    monkeypatch, tmp_path
):
    _, development, prospective = _write_locked_cohort_fixture(monkeypatch, tmp_path)
    real_hash = cohort._sha256_file
    hashed_paths = []

    def guarded_hash(path):
        path = cohort.Path(path)
        if path.resolve() == prospective.resolve():
            raise AssertionError("sealed prospective cohort was read")
        hashed_paths.append(path.resolve())
        return real_hash(path)

    monkeypatch.setattr(cohort, "_sha256_file", guarded_hash)
    with pytest.raises(cohort.SealedProspectiveCohortError, match="sealed"):
        cohort.load_frozen_v8_cohort(role="prospective")
    with pytest.raises(cohort.SealedProspectiveCohortError, match="sealed"):
        cohort.load_frozen_v8_cohort(role="development", cohort_path=prospective)
    assert prospective.resolve() not in hashed_paths
    assert development.resolve() not in hashed_paths


def test_strict_cohort_loader_rejects_manifest_or_cohort_hash_drift(monkeypatch, tmp_path):
    manifest, development, _ = _write_locked_cohort_fixture(monkeypatch, tmp_path)
    manifest.write_text(manifest.read_text() + " ", encoding="utf-8")
    with pytest.raises(cohort.FrozenCohortError, match="manifest file SHA mismatch"):
        cohort.load_frozen_v8_cohort()

    _write_locked_cohort_fixture(monkeypatch, tmp_path / "second")
    development = tmp_path / "second" / "frozen" / "development.identity_only.jsonl"
    development.write_text(development.read_text() + "\n", encoding="utf-8")
    with pytest.raises(cohort.FrozenCohortError, match="cohort file SHA mismatch"):
        cohort.load_frozen_v8_cohort()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[0].update({"answer": "forbidden"}), "field allowlist"),
        (lambda rows: rows[1].update({"qid": rows[0]["qid"]}), "duplicate frozen cohort identity"),
        (lambda rows: rows[0].update({"dataset": "unknown"}), "unsupported dataset"),
        (lambda rows: rows[0].update({"question": " padded question? "}), "unpadded string"),
    ],
)
def test_strict_cohort_loader_validates_fields_and_identity(
    monkeypatch, tmp_path, mutate, message
):
    rows = _cohort_rows()
    mutate(rows)
    _write_locked_cohort_fixture(monkeypatch, tmp_path, rows=rows)
    with pytest.raises(cohort.FrozenCohortError, match=message):
        cohort.load_frozen_v8_cohort()
