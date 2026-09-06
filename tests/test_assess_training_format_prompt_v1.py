from copy import deepcopy

import pytest

from scripts.pilot.assess_training_format_prompt_v1 import (
    FIELDS, answer_scores, check_tokens, field_content, macro_and_intervals,
    paired_questions, repeated_nonempty, normalized,
    validate_cohort,
)


def test_canonical_aliases_and_invalid_outputs_stay_in_denominator():
    result = answer_scores("[Final Answer]\nNYC", "New York City", ["NYC"], False)
    assert result["em"] == result["f1"] == 1
    assert result["format_gated_em"] == result["format_gated_f1"] == 0
    assert result["ppo_em"] == result["ppo_f1"] == 0
    assert answer_scores("[Final Answer]\nyes maybe", "yes", [], True)["f1"] == 0


def test_primary_kept_and_missing_gold_rejected():
    assert answer_scores("[Final Answer]\nParis", "Paris", ["PARIS", ""], True)["em"] == 1
    with pytest.raises(ValueError, match="missing frozen"):
        answer_scores("[Final Answer]\nParis", "", [], True)


def test_repetition_is_nonempty_casefold_whitespace_proxy_with_field_boundaries():
    step = "Reasoning: A fact\n continuing here.\nKnowledge Used: []\nConclusion: Name A"
    assert field_content(step, "Reasoning") == "a fact continuing here."
    assert field_content(step, "Conclusion") == "name a"
    assert repeated_nonempty(["", ""]) is False
    assert repeated_nonempty([normalized("A  Fact"), normalized("a\nfact")]) is True
    assert repeated_nonempty(["first fact", "second fact"]) is False


def record(q, dataset, k, old, new):
    result = {"question_key": q, "dataset": dataset, "candidate_index": k, "family_sha256": q}
    for arm, value in [("legacy", old), ("prompt_v1", new)]:
        result[arm] = {f:value for f in FIELDS}
        result[arm].update(generation_sha256=f"trace{k}", normalized_answer_sha256="sameanswer")
    return result


def test_macro_is_dataset_equal_and_K2_not_best_of_two():
    records = [record("a", "large", 0, 0, 1), record("a", "large", 1, 0, 0),
               record("b", "large", 0, 0, 1), record("b", "large", 1, 0, 1),
               record("c", "small", 0, 0, 0), record("c", "small", 1, 0, 0)]
    questions = paired_questions(records)
    assert questions[0]["prompt_v1"]["em"] == .5
    assert questions[0]["prompt_v1"]["same_normalized_K2_answer"] is True
    assert questions[0]["prompt_v1"]["identical_K2_generation"] is False
    summary = macro_and_intervals(questions, replicates=100)
    assert summary["em"]["prompt_v1"] == .375
    assert summary == macro_and_intervals(questions, replicates=100)


@pytest.mark.parametrize("change", ["missing", "duplicate_k", "duplicate_family", "metadata"])
def test_incomplete_or_pseudoreplicated_question_pairs_rejected(change):
    rows = [record("a", "one", k, 0, 1) for k in (0, 1)]
    if change == "missing":
        rows.pop()
    elif change == "duplicate_k":
        rows[1]["candidate_index"] = 0
    elif change == "duplicate_family":
        extra = deepcopy(rows)
        for r in extra:
            r["question_key"] = "another question"
        rows.extend(extra)
    else:
        rows[1]["dataset"] = "two"
    with pytest.raises(ValueError):
        paired_questions(rows)


class Tokenizer:
    def decode(self, ids, skip_special_tokens):
        return "answer"


def prediction(ids):
    return {"response_token_ids":ids, "raw_response_token_ids":ids,
            "n_response_tokens":len(ids), "effective_eos_token_ids":[9],
            "generation":"answer", "reached_max_new_tokens":len(ids) == 384 and ids[-1] != 9}


def test_token_contract_allows_eos_or_cap_and_rejects_silent_truncation():
    check_tokens(prediction([1, 9]), [9], Tokenizer())
    check_tokens(prediction([1]*384), [9], Tokenizer())
    for ids in ([1, 2], [9, 1], [1]*385):
        with pytest.raises(ValueError):
            check_tokens(prediction(ids), [9], Tokenizer())


def test_token_contract_rejects_decoded_text_tampering():
    p = prediction([1, 9]); p["generation"] = "modified answer"
    with pytest.raises(ValueError, match="does not decode"):
        check_tokens(p, [9], Tokenizer())


def test_gold_free_identity_check_needs_no_score_fields():
    rows = [{"question_key":"q", "candidate_index":k, "dataset":"d", "family_sha256":"f", "question_sha256":"text"}
            for k in (0, 1)]
    validate_cohort(rows, {"d":1})
    with pytest.raises(ValueError, match="quota"):
        validate_cohort(rows, {"d":2})
    more = deepcopy(rows)
    for r in more:
        r["question_key"] = "other"
        r["family_sha256"] = "otherfamily"
    with pytest.raises(ValueError, match="repeated family or question"):
        validate_cohort(rows + more, {"d":2})


def test_gold_free_identity_check_rejects_duplicate_candidate_and_identity_disagreement():
    rows = [{"question_key":"q", "candidate_index":k, "dataset":"d", "family_sha256":"f", "question_sha256":"text"}
            for k in (0, 0)]
    with pytest.raises(ValueError, match="K2"):
        validate_cohort(rows, {"d":1})
    rows[1].update(candidate_index=1, family_sha256="changed")
    with pytest.raises(ValueError, match="inconsistent"):
        validate_cohort(rows, {"d":1})
