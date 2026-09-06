"""CPU-only tests for the isolated v7 evidence-constrained subanswer helpers."""

from __future__ import annotations

import hashlib
import json

import pytest

from kgproweight.retrieval.subanswer_v7 import (
    SubanswerParseError,
    SubanswerV7Error,
    build_subanswer_reader_messages,
    parse_and_verify_subanswer,
    parse_subanswer_response,
    verify_subanswer,
)


QUESTION = "How many games are in a season for the league with the most wins?"
RELATION_STEP = {
    "step": 1,
    "subject": "Champions League winner",
    "relation_label": "league",
    "pid": "P118",
    "output_slot": "hop_1",
    "dependencies": [],
}
SUBQUERY_STEP = {
    "step": 1,
    "subquery_template": (
        "Which league had the most Champions League wins between 1992 and 2013?"
    ),
    "output_slot": "step_1",
    "dependencies": [],
}
PASSAGES = [
    {
        "id": "doc-1",
        "contents": (
            "UEFA Champions League\n"
            "Spanish clubs accumulated the most titles in this period. "
            "The winning domestic competition was La Liga."
        ),
        "source": "e5",
    },
    {
        "id": 22,
        "title": "La Liga",
        "text": "A La Liga season is contested by 20 teams over 38 games.",
    },
]


def _candidate(
    answer="La Liga", doc_id="doc-1", answer_type="entity", abstain=False
):
    return {
        "answer": answer,
        "cited_doc_ids": [] if abstain else [doc_id],
        "answer_type": answer_type,
        "abstain": abstain,
    }


def test_reader_messages_whitelist_inputs_and_never_serialize_label_metadata():
    hidden = "DO-NOT-LEAK-HIDDEN-LABEL-937541"
    step = dict(
        SUBQUERY_STEP,
        gold_answer=hidden,
        supporting_facts=[hidden],
        arbitrary_nested={"answer": hidden},
    )
    passages = [
        dict(
            PASSAGES[0],
            gold_answer=hidden,
            supporting_facts=[hidden],
            score=0.999,
        )
    ]
    original_step = json.loads(json.dumps(step))
    original_passages = json.loads(json.dumps(passages))

    messages = build_subanswer_reader_messages(
        QUESTION, step, passages, target_type="subquery_graph"
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    joined = json.dumps(messages, ensure_ascii=False)
    assert hidden not in joined
    assert "gold_answer" not in joined
    assert "supporting_facts" not in joined
    assert "exactly that one document's string doc_id" in messages[0]["content"]

    payload = json.loads(messages[1]["content"])
    assert payload["prompt_version"]
    assert payload["original_question"] == QUESTION
    assert payload["subquestion_to_answer"] == SUBQUERY_STEP["subquery_template"]
    assert payload["current_plan_step"] == SUBQUERY_STEP
    assert payload["retrieved_documents"] == [
        {
            "doc_id": "doc-1",
            "title": "UEFA Champions League",
            "text": PASSAGES[0]["contents"],
        }
    ]
    assert step == original_step
    assert passages == original_passages


def test_reader_renders_relation_step_and_stringifies_numeric_document_id():
    messages = build_subanswer_reader_messages(
        QUESTION, RELATION_STEP, PASSAGES, target_type="relation_graph"
    )
    payload = json.loads(messages[1]["content"])
    assert payload["subquestion_to_answer"] == (
        "What is the league of Champions League winner?"
    )
    assert [doc["doc_id"] for doc in payload["retrieved_documents"]] == [
        "doc-1",
        "22",
    ]


@pytest.mark.parametrize(
    ("passages", "error"),
    [
        ([], "must not be empty"),
        ([{"contents": "Title\nText"}], "no stable document id"),
        (
            [
                {"id": "same", "contents": "A\ntext"},
                {"id": "same", "contents": "B\ntext"},
            ],
            "duplicate document id",
        ),
        (
            [{"id": "a", "doc_id": "b", "contents": "A\ntext"}],
            "conflicting document ids",
        ),
    ],
)
def test_reader_rejects_ambiguous_or_unverifiable_document_identity(passages, error):
    with pytest.raises(SubanswerV7Error, match=error):
        build_subanswer_reader_messages(QUESTION, SUBQUERY_STEP, passages)


def test_strict_parser_accepts_only_the_exact_typed_contract():
    raw = json.dumps(_candidate(answer="  La   Liga  "))
    assert parse_subanswer_response(raw) == _candidate(answer="La Liga")

    abstention = {
        "answer": "",
        "cited_doc_ids": [],
        "answer_type": "yes_no",
        "abstain": True,
    }
    assert parse_subanswer_response(json.dumps(abstention)) == abstention


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("```json\n{}\n```", "invalid_json"),
        (json.dumps({**_candidate(), "explanation": "because"}), "field_set"),
        (json.dumps({"answer": "La Liga"}), "field_set"),
        (json.dumps({**_candidate(), "abstain": "false"}), "abstain_not_boolean"),
        (json.dumps({**_candidate(), "answer_type": "person"}), "invalid_answer_type"),
        (json.dumps({**_candidate(), "cited_doc_ids": [22]}), "citation_not_string"),
        (
            json.dumps({**_candidate(), "cited_doc_ids": ["doc-1", "doc-2"]}),
            "citation_count",
        ),
        (
            json.dumps(
                {
                    "answer": "La Liga",
                    "cited_doc_ids": [],
                    "answer_type": "entity",
                    "abstain": True,
                }
            ),
            "incoherent_abstention",
        ),
        (
            '{"answer":"La Liga","answer":"Bundesliga",'
            '"cited_doc_ids":["doc-1"],"answer_type":"entity","abstain":false}',
            "duplicate_key",
        ),
    ],
)
def test_strict_parser_rejects_wrappers_coercions_extra_fields_and_duplicates(raw, code):
    with pytest.raises(SubanswerParseError) as caught:
        parse_subanswer_response(raw)
    assert caught.value.code == code


def test_verified_entity_has_one_citation_and_mechanically_derived_support():
    result = verify_subanswer(
        _candidate(),
        QUESTION,
        SUBQUERY_STEP,
        PASSAGES,
        target_type="subquery_graph",
    )

    sentence = "The winning domestic competition was La Liga."
    assert result["verified"] is True
    assert result["verified_answer"] == "La Liga"
    assert result["reason"] == "verified"
    assert result["cited_doc_ids"] == ["doc-1"]
    assert result["supporting_doc_id"] == "doc-1"
    assert result["supporting_sentence"] == sentence
    assert result["supporting_sentence_sha256"] == hashlib.sha256(
        sentence.encode("utf-8")
    ).hexdigest()
    assert result["support_location"] == "text"
    assert result["gold_access"] is False
    assert result["verification_scope"] == "surface_locality_not_semantic_entailment"
    json.dumps(result, sort_keys=True)


def test_title_only_surface_is_allowed_but_is_labelled_as_locality_not_entailment():
    passages = [
        {
            "id": "title-only",
            "title": "Armie Hammer",
            "contents": "Actor entry\nNo occurrence of the person's name in this body.",
        }
    ]
    result = verify_subanswer(
        _candidate(answer="Armie Hammer", doc_id="title-only"),
        "Which actor provides the voice?",
        {"subject": "The Polar Bears", "relation_label": "voice actor"},
        passages,
    )
    assert result["verified"] is True
    assert result["support_location"] == "title"
    assert result["supporting_sentence"] == "Armie Hammer"
    assert result["verification_scope"] == "surface_locality_not_semantic_entailment"


def test_citation_must_exist_and_answer_must_be_in_that_specific_document():
    unknown = verify_subanswer(
        _candidate(doc_id="not-retrieved"), QUESTION, SUBQUERY_STEP, PASSAGES
    )
    assert unknown["verified"] is False
    assert unknown["reason"] == "cited_document_not_in_input"

    wrong_document = verify_subanswer(
        _candidate(answer="38", doc_id="doc-1", answer_type="number"),
        QUESTION,
        SUBQUERY_STEP,
        PASSAGES,
    )
    assert wrong_document["verified"] is False
    assert wrong_document["reason"] == "answer_surface_not_in_cited_document"

    right_document = verify_subanswer(
        _candidate(answer="38", doc_id="22", answer_type="number"),
        QUESTION,
        SUBQUERY_STEP,
        PASSAGES,
    )
    assert right_document["verified"] is True
    assert right_document["verified_answer"] == "38"


def test_entity_matching_is_nfkc_casefolded_but_obeys_word_boundaries():
    decomposed = "Man\u0303alac"
    passages = [
        {
            "id": "unicode",
            "contents": f"Bamboo {decomposed}\nBamboo {decomposed} is a singer.",
        },
        {"id": "boundary", "contents": "Yorkshire\nYorkshire is a county."},
    ]
    composed = verify_subanswer(
        _candidate(answer="Bamboo Mañalac", doc_id="unicode"),
        "Who fronted the band?",
        {"subject": "The band", "relation_label": "frontman"},
        passages,
    )
    assert composed["verified"] is True

    substring = verify_subanswer(
        _candidate(answer="York", doc_id="boundary"),
        "Which place is meant?",
        {"subject": "The clue", "relation_label": "place"},
        passages,
    )
    assert substring["verified"] is False
    assert substring["reason"] == "answer_surface_not_in_cited_document"


@pytest.mark.parametrize("answer", ["Champions League winner", "The Champions League winner"])
def test_subject_echo_is_rejected_even_with_article_or_disambiguator_variation(answer):
    passages = [
        {
            "id": "d",
            "contents": f"Champions League winner\nThe answer phrase is {answer}.",
        }
    ]
    result = verify_subanswer(
        _candidate(answer=answer, doc_id="d"),
        QUESTION,
        dict(RELATION_STEP, subject="Champions League winner (competition)"),
        passages,
        target_type="relation_graph",
    )
    assert result["verified"] is False
    assert result["reason"] == "subject_echo"


@pytest.mark.parametrize("answer", ["unknown", "N/A", "cannot determine"])
def test_null_like_answer_fails_closed_even_if_document_contains_the_words(answer):
    passages = [{"id": "d", "contents": f"Entry\nThe record says {answer}."}]
    result = verify_subanswer(
        _candidate(answer=answer, doc_id="d"), QUESTION, SUBQUERY_STEP, passages
    )
    assert result["verified"] is False
    assert result["reason"] == "empty_or_null_answer"


def test_number_and_date_require_matching_surface_shapes_and_cited_evidence():
    passages = [
        {
            "id": "facts",
            "contents": (
                "Biography\nShe was born on March 21, 1976. "
                "The league season contains 38 games."
            ),
        }
    ]
    number = verify_subanswer(
        _candidate(answer="38", doc_id="facts", answer_type="number"),
        QUESTION,
        SUBQUERY_STEP,
        passages,
    )
    date = verify_subanswer(
        _candidate(answer="March 21, 1976", doc_id="facts", answer_type="date"),
        QUESTION,
        SUBQUERY_STEP,
        passages,
    )
    wrong_number_shape = verify_subanswer(
        _candidate(answer="many", doc_id="facts", answer_type="number"),
        QUESTION,
        SUBQUERY_STEP,
        passages,
    )
    wrong_date_shape = verify_subanswer(
        _candidate(answer="recently", doc_id="facts", answer_type="date"),
        QUESTION,
        SUBQUERY_STEP,
        passages,
    )
    assert number["verified"] is True
    assert date["verified"] is True
    assert wrong_number_shape["reason"] == "answer_type_surface_mismatch:number"
    assert wrong_date_shape["reason"] == "answer_type_surface_mismatch:date"


def test_number_surface_is_not_accepted_inside_a_different_decimal_value():
    passages = [{"id": "d", "contents": "Statistic\nThe measured value was 38.5."}]
    result = verify_subanswer(
        _candidate(answer="38", doc_id="d", answer_type="number"),
        QUESTION,
        SUBQUERY_STEP,
        passages,
    )
    assert result["verified"] is False
    assert result["reason"] == "answer_surface_not_in_cited_document"


@pytest.mark.parametrize("answer_type", ["yes_no", "other"])
def test_nonextractive_types_are_never_promoted_to_verified_answers(answer_type):
    result = verify_subanswer(
        _candidate(answer="yes", answer_type=answer_type),
        QUESTION,
        SUBQUERY_STEP,
        PASSAGES,
    )
    assert result["verified"] is False
    assert result["reason"] == f"non_extractive_answer_type:{answer_type}"
    assert result["verified_answer"] is None


def test_boolean_surface_cannot_evade_abstention_by_claiming_entity_type():
    passages = [{"id": "d", "contents": "Answer\nThe document says yes."}]
    result = verify_subanswer(
        _candidate(answer="yes", doc_id="d", answer_type="entity"),
        QUESTION,
        SUBQUERY_STEP,
        passages,
    )
    assert result["verified"] is False
    assert result["reason"] == "non_extractive_boolean_answer"


def test_explicit_model_abstention_is_a_clean_fail_closed_outcome():
    result = verify_subanswer(
        _candidate(answer="", answer_type="entity", abstain=True),
        QUESTION,
        SUBQUERY_STEP,
        PASSAGES,
    )
    assert result["verified"] is False
    assert result["reason"] == "model_abstained"
    assert result["cited_doc_ids"] == []


def test_parse_and_verify_converts_bad_model_text_to_serializable_fallback_reason():
    raw = "Answer: La Liga"
    result = parse_and_verify_subanswer(
        raw, QUESTION, SUBQUERY_STEP, PASSAGES, target_type="subquery_graph"
    )
    assert result["verified"] is False
    assert result["verified_answer"] is None
    assert result["reason"] == "parse_error:invalid_json"
    assert result["response_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert result["parser_version"]
    json.dumps(result, sort_keys=True)


def test_parse_and_verify_does_not_hide_invalid_caller_input_behind_parse_failure():
    with pytest.raises(SubanswerV7Error, match="no stable document id"):
        parse_and_verify_subanswer(
            "not json",
            QUESTION,
            SUBQUERY_STEP,
            [{"contents": "Missing id\ntext"}],
        )
