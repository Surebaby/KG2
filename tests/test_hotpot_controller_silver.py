"""CPU-only tests for the pure Hotpot Controller companion helpers."""

from __future__ import annotations

from copy import deepcopy
import json

import pytest

from kgproweight.data.hotpot_controller_silver import (
    FINAL_MASK,
    HOTPOT_BINDING_METHOD,
    INTERMEDIATE_MASK,
    PROPOSAL_SCHEMA_VERSION,
    SCHEMA_VERSION,
    HotpotSilverReject,
    ValidatedQueryProposal,
    build_hotpot_action_pair,
    build_masked_proposal_view,
    companion_compatibility_report,
    extract_hotpot_support_chain,
    validate_hotpot_action_pair,
    validate_query_proposal,
)
from kgproweight.eval.query_controller_v1 import (
    ActionValidationError,
    validate_action_record,
)


def _raw() -> dict:
    return {
        "id": "hotpot-train-1",
        "question": (
            "The Oberoi family is involved in a hotel company whose head office "
            "is in what city?"
        ),
        "golden_answers": ["Delhi"],
        "metadata": {
            "type": "bridge",
            "level": "medium",
            "supporting_facts": {
                "title": ["Oberoi family", "The Oberoi Group"],
                "sent_id": [0, 0],
            },
            "context": {
                "title": ["Oberoi family", "The Oberoi Group", "Distractor"],
                "sentences": [
                    [
                        "The Oberoi family is famous for its involvement in hotels "
                        "through The Oberoi Group."
                    ],
                    [
                        "The Oberoi Group is a hotel company with its head office "
                        "in Delhi."
                    ],
                    ["This paragraph is not supporting evidence."],
                ],
            },
        },
    }


def _proposal(**updates: object) -> dict:
    value = {
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "q1": "Which hotel company is the Oberoi family involved with?",
        "q2_template": "In what city does #1 have its head office?",
    }
    value.update(updates)
    return value


def _chain_and_pair() -> tuple:
    chain = extract_hotpot_support_chain(_raw())
    pair = build_hotpot_action_pair(chain, _proposal(), split="train")
    return chain, pair


def test_happy_path_masks_secrets_and_builds_nullable_text_action_pair() -> None:
    chain = extract_hotpot_support_chain(_raw())
    assert chain.root_title == "Oberoi family"
    assert chain.bridge_title == chain.intermediate == "The Oberoi Group"
    assert chain.first_hop.sentence_index == 0
    assert chain.second_hop.sentence_index == 0

    view = build_masked_proposal_view(chain)
    encoded_view = json.dumps(view, ensure_ascii=False)
    assert INTERMEDIATE_MASK in view["first_hop_evidence_masked"]
    assert INTERMEDIATE_MASK in view["second_hop_evidence_masked"]
    assert FINAL_MASK in view["second_hop_evidence_masked"]
    assert "The Oberoi Group" not in encoded_view
    assert "Delhi" not in encoded_view

    validated = validate_query_proposal(_proposal(), chain)
    assert validated.q2_query == (
        "In what city does The Oberoi Group have its head office?"
    )
    q1, q2 = build_hotpot_action_pair(chain, validated, split="train")
    assert {q1["schema_version"], q2["schema_version"]} == {SCHEMA_VERSION}
    assert [q1["target"]["pid"], q2["target"]["pid"]] == [None, None]
    assert {q1["target"]["source_action"], q2["target"]["source_action"]} == {
        "text"
    }
    assert q2["target"]["dependencies"] == ["q1"]
    assert q2["state"]["verified_observations"][0]["answer"] == (
        "The Oberoi Group"
    )
    provenance = q2["state"]["verified_observations"][0]["provenance"]
    assert provenance == {
        "source": "train_annotation_support",
        "annotation_path": "metadata.supporting_facts.title[1]",
        "binding_method": HOTPOT_BINDING_METHOD,
    }
    assert "Delhi" not in json.dumps([q1, q2], ensure_ascii=False)


def test_pair_is_deterministic_and_declares_central_validator_boundary() -> None:
    chain, pair = _chain_and_pair()
    assert pair == build_hotpot_action_pair(chain, _proposal(), split="train")
    assert validate_hotpot_action_pair(pair, chain=chain, expected_split="train") == pair

    with pytest.raises(ActionValidationError) as caught:
        validate_action_record(pair[1], expected_split="train")
    assert "schema_version" in caught.value.codes
    assert "observation_provenance_schema" in caught.value.codes

    report = companion_compatibility_report(pair, chain=chain)
    assert report["companion_pair_valid"] is True
    assert report["central_v1_direct_validation_supported"] is False
    assert report["central_v1_structural_projection_valid"] is True
    assert report["retrieval_or_reader_validation_performed"] is False


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda raw: raw["metadata"].update(type="comparison"), "not_bridge_type"),
        (
            lambda raw: raw["metadata"]["supporting_facts"]["title"].append(
                "Third support"
            ),
            "support_arrays_misaligned",
        ),
        (
            lambda raw: raw["metadata"]["supporting_facts"]["sent_id"].__setitem__(
                0, 99
            ),
            "support_sentence_pointer_out_of_range",
        ),
        (
            lambda raw: raw["metadata"]["context"]["sentences"][0].__setitem__(
                0, "The family is involved with a hotel company."
            ),
            "bridge_surface_in_root_support_not_unique",
        ),
        (
            lambda raw: raw["metadata"]["context"]["sentences"][1].__setitem__(
                0,
                "The Oberoi Group was created by the Oberoi family and is based "
                "in Delhi.",
            ),
            "bidirectional_support_title_link",
        ),
    ],
)
def test_chain_extraction_fails_closed(mutation, code: str) -> None:
    raw = _raw()
    mutation(raw)
    with pytest.raises(HotpotSilverReject) as caught:
        extract_hotpot_support_chain(raw)
    assert caught.value.code == code


def test_chain_requires_exactly_one_question_visible_root_title() -> None:
    raw = _raw()
    raw["question"] = "Which city contains the company's head office?"
    with pytest.raises(HotpotSilverReject) as caught:
        extract_hotpot_support_chain(raw)
    assert caught.value.code == "question_root_title_not_unique"


def test_chain_requires_exactly_one_support_fact_per_document() -> None:
    raw = _raw()
    raw["metadata"]["supporting_facts"]["title"].append("Oberoi family")
    raw["metadata"]["supporting_facts"]["sent_id"].append(0)
    with pytest.raises(HotpotSilverReject) as caught:
        extract_hotpot_support_chain(raw)
    assert caught.value.code == "support_fact_count_not_two"


def test_raw_horizontal_padding_is_normalized_but_multiline_text_is_rejected() -> None:
    raw = _raw()
    raw["question"] = f"  {raw['question']}  "
    raw["metadata"]["context"]["sentences"][0][0] = (
        f"  {raw['metadata']['context']['sentences'][0][0]}  "
    )
    chain = extract_hotpot_support_chain(raw)
    assert chain.question == _raw()["question"]
    assert chain.first_hop.evidence_excerpt == (
        _raw()["metadata"]["context"]["sentences"][0][0]
    )

    raw = _raw()
    raw["question"] += "\nmalicious second line"
    with pytest.raises(HotpotSilverReject) as caught:
        extract_hotpot_support_chain(raw)
    assert caught.value.code == "raw_identity_invalid"

    raw = _raw()
    raw["question"] = (
        "The Oberoi family is involved with The Oberoi Group, whose office is where?"
    )
    with pytest.raises(HotpotSilverReject) as caught:
        extract_hotpot_support_chain(raw)
    assert caught.value.code == "question_root_title_not_unique"


def test_final_alias_in_first_hop_is_rejected_even_with_intervening_word() -> None:
    raw = _raw()
    raw["golden_answers"] = ["US 60"]
    raw["metadata"]["context"]["sentences"][0][0] = (
        "The Oberoi family operates hotels through The Oberoi Group near U.S. "
        "Highway 60."
    )
    raw["metadata"]["context"]["sentences"][1][0] = (
        "The Oberoi Group has its head office along US 60."
    )
    with pytest.raises(HotpotSilverReject) as caught:
        extract_hotpot_support_chain(raw)
    assert caught.value.code == "final_alias_in_first_hop_support"


def test_final_alias_equivalent_to_bridge_is_rejected() -> None:
    raw = _raw()
    raw["question"] = "The Coy family belongs to a group headquartered where?"
    raw["golden_answers"] = ["Walter Darwin Coy"]
    raw["metadata"]["supporting_facts"]["title"] = [
        "Coy family",
        "Walter Coy",
    ]
    raw["metadata"]["context"]["title"][:2] = ["Coy family", "Walter Coy"]
    raw["metadata"]["context"]["sentences"][0][0] = (
        "The Coy family is represented by Walter Coy."
    )
    raw["metadata"]["context"]["sentences"][1][0] = (
        "Walter Coy used the full name Walter Darwin Coy."
    )
    with pytest.raises(HotpotSilverReject) as caught:
        extract_hotpot_support_chain(raw)
    assert caught.value.code == "final_alias_equals_chain_entity"


def test_accent_folded_final_alias_and_answer_leading_sentence_are_rejected() -> None:
    raw = _raw()
    raw["golden_answers"] = ["The Oberoi Groúp"]
    raw["metadata"]["context"]["sentences"][1][0] = (
        "The Oberoi Group is also written The Oberoi Groúp."
    )
    with pytest.raises(HotpotSilverReject) as caught:
        extract_hotpot_support_chain(raw)
    assert caught.value.code == "final_alias_equals_chain_entity"

    raw = _raw()
    raw["metadata"]["context"]["sentences"][1][0] = (
        "Delhi is where The Oberoi Group has its head office."
    )
    with pytest.raises(HotpotSilverReject) as caught:
        extract_hotpot_support_chain(raw)
    assert caught.value.code == "final_alias_leads_second_hop_support"


@pytest.mark.parametrize(
    ("updates", "code"),
    [
        (
            {"q1": "Which company is The Oberoi Group?"},
            "q1_secret_leak",
        ),
        (
            {"q1": "Which hotel company is connected to #1?"},
            "q1_contains_placeholder",
        ),
        (
            {"q2_template": "In what city is its head office?"},
            "q2_template_dependency_count",
        ),
        (
            {"q2_template": "How is #1 connected to #1?"},
            "q2_template_dependency_count",
        ),
        (
            {"q2_template": "Who is #1?"},
            "q2_template_no_relation_content",
        ),
        (
            {"q2_template": "Is #1 headquartered in Delhi?"},
            "q2_template_secret_leak",
        ),
    ],
)
def test_query_proposal_rejects_dependency_or_secret_defects(
    updates: dict, code: str
) -> None:
    chain = extract_hotpot_support_chain(_raw())
    with pytest.raises(HotpotSilverReject) as caught:
        validate_query_proposal(_proposal(**updates), chain)
    assert caught.value.code == code


def test_query_proposal_rejects_article_stripped_and_diacritic_secret_aliases() -> None:
    chain = extract_hotpot_support_chain(_raw())
    with pytest.raises(HotpotSilverReject) as caught:
        validate_query_proposal(
            _proposal(q1="Which hotel company is Oberoi Group?"), chain
        )
    assert caught.value.code == "q1_secret_leak"

    with pytest.raises(HotpotSilverReject) as caught:
        validate_query_proposal(
            _proposal(q2_template="Is #1 headquartered in Délhi?"), chain
        )
    assert caught.value.code == "q2_template_secret_leak"


def test_masked_view_masks_expanded_middle_token_alias() -> None:
    raw = _raw()
    raw["metadata"]["context"]["sentences"][1][0] = (
        "The Oberoi International Group is a hotel company with its head office in Delhi."
    )
    chain = extract_hotpot_support_chain(raw)
    view = build_masked_proposal_view(chain)
    assert "Oberoi" not in view["second_hop_evidence_masked"]
    assert "International" not in view["second_hop_evidence_masked"]
    assert "[INTERMEDIATE]" in view["second_hop_evidence_masked"]


def test_prevalidated_proposal_is_recomputed_instead_of_trusted() -> None:
    chain = extract_hotpot_support_chain(_raw())
    valid = validate_query_proposal(_proposal(), chain)
    forged = ValidatedQueryProposal(
        q1_query=valid.q1_query,
        q2_template=valid.q2_template,
        q2_query="Delhi",
        q1_relation_intent=valid.q1_relation_intent,
        q2_relation_intent=valid.q2_relation_intent,
        proposal_sha256=valid.proposal_sha256,
    )
    with pytest.raises(HotpotSilverReject) as caught:
        build_hotpot_action_pair(chain, forged, split="train")
    assert caught.value.code == "validated_proposal_integrity"


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda pair: pair[0]["target"].update(pid="P31"),
            "hotpot_target_pid_or_source_action",
        ),
        (
            lambda pair: pair[1]["target"].update(source_action="graph"),
            "hotpot_target_pid_or_source_action",
        ),
        (
            lambda pair: pair[1]["state"]["verified_observations"][0][
                "provenance"
            ].update(annotation_path="metadata.evidences.entity[0]"),
            "hotpot_observation_provenance_binding",
        ),
        (
            lambda pair: pair[1]["state"]["verified_observations"][0].update(
                answer="Fake bridge"
            ),
            "central_v1_structural_projection_failed",
        ),
    ],
)
def test_pair_validator_rejects_tampering(mutation, code: str) -> None:
    chain, pair = _chain_and_pair()
    mutable = [deepcopy(row) for row in pair]
    mutation(mutable)
    with pytest.raises(HotpotSilverReject) as caught:
        validate_hotpot_action_pair(mutable, chain=chain, expected_split="train")
    assert caught.value.code == code


def test_pair_validator_binds_q2_history_and_source_provenance_to_q1() -> None:
    chain, pair = _chain_and_pair()
    mutable = [deepcopy(row) for row in pair]
    mutable[1]["state"]["previous_actions"][0]["query"] = (
        "Which unrelated hotel company exists?"
    )
    with pytest.raises(HotpotSilverReject) as caught:
        validate_hotpot_action_pair(mutable, chain=chain)
    assert caught.value.code == "q2_previous_action_pair_mismatch"

    mutable = [deepcopy(row) for row in pair]
    mutable[1]["source_provenance"]["raw_record_sha256"] = "0" * 64
    with pytest.raises(HotpotSilverReject) as caught:
        validate_hotpot_action_pair(mutable, chain=chain)
    assert caught.value.code == "source_provenance_binding"


def test_pair_validator_recomputes_proposal_hash_and_action_binding() -> None:
    chain, pair = _chain_and_pair()
    mutable = [deepcopy(row) for row in pair]
    forged_q1 = "Which company did the Oberoi family establish?"
    mutable[0]["target"]["query"] = forged_q1
    mutable[0]["target"]["relation_intent"] = forged_q1[:-1]
    mutable[1]["state"]["previous_actions"][0]["query"] = forged_q1
    with pytest.raises(HotpotSilverReject) as caught:
        validate_hotpot_action_pair(mutable, chain=chain)
    assert caught.value.code == "source_provenance_proposal_binding"

    mutable = [deepcopy(row) for row in pair]
    mutable[1]["source_provenance"]["q2_template"] = (
        "In which country is #1 headquartered?"
    )
    with pytest.raises(HotpotSilverReject) as caught:
        validate_hotpot_action_pair(mutable, chain=chain)
    assert caught.value.code == "source_provenance_proposal_binding"


def test_extra_provenance_cannot_serialize_hidden_answers_or_bridge() -> None:
    chain = extract_hotpot_support_chain(_raw())
    with pytest.raises(HotpotSilverReject) as caught:
        build_hotpot_action_pair(
            chain,
            _proposal(),
            split="train",
            extra_source_provenance={"gold_answer": "redacted"},
        )
    assert caught.value.code == "extra_provenance_forbidden_key"

    with pytest.raises(HotpotSilverReject) as caught:
        build_hotpot_action_pair(
            chain,
            _proposal(),
            split="train",
            extra_source_provenance={"note": "candidate was The Oberoi Group"},
        )
    assert caught.value.code == "extra_provenance_secret_leak"
