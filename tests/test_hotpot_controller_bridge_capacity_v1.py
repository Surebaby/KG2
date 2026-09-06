from scripts.diagnose.audit_hotpot_controller_bridge_capacity_v1 import (
    audit_records,
)


def _row(
    qid,
    question,
    answer,
    *,
    title0="Film Alpha",
    sentences0=None,
    title1="Director Beta",
    sentences1=None,
    support_titles=None,
    support_indices=None,
    level="medium",
):
    sentences0 = sentences0 or ["Film Alpha was directed by Director Beta."]
    sentences1 = sentences1 or ["Director Beta was born in Paris."]
    return {
        "id": qid,
        "question": question,
        "golden_answers": [answer],
        "metadata": {
            "type": "bridge",
            "level": level,
            "supporting_facts": {
                "title": support_titles or [title0, title1],
                "sent_id": support_indices or [0, 0],
            },
            "context": {
                "title": [title0, title1, "Distractor"],
                "sentences": [
                    sentences0,
                    sentences1,
                    ["This sentence is irrelevant."],
                ],
            },
        },
    }


def test_clean_forward_chain_reaches_identity_hardened_capacity() -> None:
    row = _row(
        "clean",
        "Where was the director of Film Alpha born?",
        "Paris",
    )
    report = audit_records([row], example_limit=1)

    assert report["cross_title_mentions"]["annotated_support_sentences"][
        "unique_direction_count"
    ] == 1
    assert report["cross_title_mentions"]["support_sentence_binding"][
        "unique_bridge_support_sentence_qids"
    ] == 1
    assert report["answer_type"]["nonboolean_bridge_count"] == 1
    assert report["strict_forward_funnel"]["unique_second_hop_answer_binding"] == 1
    assert report["pilot_capacity"]["identity_hardened_eligible_qids"] == 1
    assert report["pilot_capacity"]["unique_answer_free_families"] == 1


def test_out_of_range_support_is_recorded_and_never_used() -> None:
    row = _row(
        "bad-index",
        "Where was the director of Film Alpha born?",
        "Paris",
        support_indices=[0, 9],
    )
    report = audit_records([row])

    integrity = report["support_integrity"]
    assert integrity["invalid_qid_count"] == 1
    assert integrity["sent_id_out_of_range_references"] == 1
    assert integrity["invalid_references"] == [{
        "qid": "bad-index",
        "row_index": 1,
        "reason": "sent_id_out_of_range",
        "title": "Director Beta",
        "sent_id": 9,
        "available_sentence_count": 1,
    }]
    assert report["pilot_capacity"].get("identity_hardened_eligible_qids", 0) == 0


def test_bidirectional_mentions_are_not_called_a_unique_chain() -> None:
    row = _row(
        "bidirectional",
        "Where was the director of Film Alpha born?",
        "Paris",
        sentences1=["Director Beta, director of Film Alpha, was born in Paris."],
    )
    report = audit_records([row])

    mentions = report["cross_title_mentions"]["annotated_support_sentences"]
    assert mentions["bidirectional"] == 1
    assert mentions["unique_direction_count"] == 0
    assert report["pilot_capacity"].get("identity_hardened_eligible_qids", 0) == 0


def test_direct_final_and_first_hop_future_surface_are_separate_leaks() -> None:
    direct = _row(
        "direct-final",
        "Who directed Film Alpha?",
        "Director Beta",
        sentences1=["Director Beta is a French filmmaker."],
    )
    early = _row(
        "early-final",
        "Where was the director of Film Alpha born?",
        "Paris",
        sentences0=["Film Alpha was set in Paris and directed by Director Beta."],
    )
    report = audit_records([direct, early])
    leakage = report["literal_future_leakage"]

    assert leakage["final_answer_equals_intermediate_title"] == 1
    assert leakage["final_surface_in_any_first_hop_support"] == 1
    assert leakage["final_surface_in_bound_first_hop_excerpt"] == 1
    assert report["pilot_capacity"].get("identity_hardened_eligible_qids", 0) == 0


def test_inverse_arrow_is_disclosed_instead_of_silently_reordered() -> None:
    row = _row(
        "inverse",
        "What nationality was the spouse of Director Beta?",
        "French",
        title0="Artist Alpha",
        sentences0=[
            "Artist Alpha was married to Director Beta.",
            "She was a French painter.",
        ],
        title1="Director Beta",
        sentences1=["Director Beta was a film director."],
        support_titles=["Artist Alpha", "Artist Alpha", "Director Beta"],
        support_indices=[0, 1, 0],
    )
    report = audit_records([row], example_limit=1)
    crosscheck = report["orientation"]["question_answer_document_crosscheck"]

    assert crosscheck["inverse_agreement"] == 1
    assert crosscheck["opposite_document_agreement"] == 1
    assert report["strict_forward_funnel"].get(
        "question_names_mention_source_only", 0
    ) == 0
    assert report["diagnostic_examples"][
        "inverse_chain_despite_mention_arrow"
    ][0]["qid"] == "inverse"


def test_identity_alias_and_answer_lead_are_precision_pool_exclusions() -> None:
    alias = _row(
        "alias",
        "Who was the first president connected to Film Alpha?",
        "Seretse Goitsebeng Khama",
        title1="Seretse Khama",
        sentences0=["Film Alpha featured Seretse Khama."],
        sentences1=[
            "He was also known as Seretse Goitsebeng Khama and served as president."
        ],
    )
    answer_lead = _row(
        "answer-lead",
        "What was the legal name of the performer in Film Alpha?",
        "Jaime Meline",
        title1="El-P",
        sentences0=["Film Alpha featured El-P."],
        sentences1=["Jaime Meline is better known by the stage name El-P."],
    )
    report = audit_records([alias, answer_lead])
    capacity = report["pilot_capacity"]

    assert capacity["precision_pool_before_identity_hardening"] == 2
    assert capacity["conservative_intermediate_final_alias_exclusions"] == 1
    assert capacity["after_conservative_alias_screen"] == 1
    assert capacity["answer_leading_second_hop_identity_risk_exclusions"] == 1
    assert capacity.get("identity_hardened_eligible_qids", 0) == 0


def test_boolean_bridge_is_counted_but_not_pilot_eligible() -> None:
    row = _row(
        "boolean",
        "Did the director of Film Alpha win the award?",
        "yes",
        sentences1=["Director Beta won the award; yes, this is documented."],
    )
    report = audit_records([row])

    assert report["answer_type"]["boolean_bridge_count"] == 1
    assert report["answer_type"]["nonboolean_bridge_count"] == 0
    assert report["pilot_capacity"].get("identity_hardened_eligible_qids", 0) == 0
