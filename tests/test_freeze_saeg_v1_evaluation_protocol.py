import pytest

from scripts.prepare.freeze_saeg_v1_evaluation_protocol import assert_answer_free


def test_answer_free_guard_accepts_question_and_passages():
    assert_answer_free({"question": "Q?", "passages": [{"contents": "text"}], "gold_access": False})


@pytest.mark.parametrize("field", ["answer", "golden_answers", "supporting_facts", "question_decomposition"])
def test_answer_free_guard_rejects_nested_gold_fields(field):
    with pytest.raises(ValueError, match="forbidden"):
        assert_answer_free({"nested": {field: "leak"}})
