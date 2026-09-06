from scripts.prepare.audit_2wiki_official_raw_plans_v1 import summarize_plans


def _input(key: str, question_type: str = "comparison") -> dict:
    return {
        "question_key": key,
        "dataset": "2wikimultihopqa",
        "qid": key.rsplit("::", 1)[-1],
        "question": f"Question for {key}",
        "question_sha256": f"hash-{key}",
        "question_type": question_type,
    }


def _prediction(source: dict, *, valid: bool = True) -> dict:
    return {
        "question_key": source["question_key"],
        "dataset": source["dataset"],
        "qid": source["qid"],
        "question": source["question"],
        "question_sha256": source["question_sha256"],
        "generated_text": "{}",
        "schema_valid": valid,
        "validation_errors": [] if valid else ["invalid_relation_step"],
        "gold_access": False,
    }


def test_summary_stratifies_and_retains_invalid_rows() -> None:
    inputs = [_input("2wikimultihopqa::1"), _input("2wikimultihopqa::2")]
    predictions = [_prediction(inputs[0]), _prediction(inputs[1], valid=False)]
    result = summarize_plans(inputs, predictions)
    assert result["n_predictions"] == 2
    assert result["schema_valid"] == 1
    assert result["schema_invalid"] == 1
    assert result["by_question_type"]["comparison"]["schema_valid_rate"] == 0.5
    assert result["validation_errors"] == {"invalid_relation_step": 1}


def test_summary_detects_missing_and_duplicate_predictions() -> None:
    inputs = [_input("2wikimultihopqa::1"), _input("2wikimultihopqa::2")]
    prediction = _prediction(inputs[0])
    result = summarize_plans(inputs, [prediction, prediction])
    assert result["integrity"]["duplicate_prediction_keys"] == 1
    assert result["integrity"]["missing_prediction_keys"] == 1


def test_summary_detects_identity_and_gold_access_violation() -> None:
    source = _input("2wikimultihopqa::1")
    prediction = _prediction(source)
    prediction["question"] = "drift"
    prediction["gold_access"] = True
    result = summarize_plans([source], [prediction])
    assert result["integrity"]["identity_mismatches"] == 1
    assert result["integrity"]["gold_access_violations"] == 1
