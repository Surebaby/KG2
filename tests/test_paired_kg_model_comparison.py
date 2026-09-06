from scripts.pilot.score_paired_kg_model_comparison import _model_comparison


def _row(qid: str, arm: str, em: float):
    return {"qid": qid, "arm": arm, "em": em}


def test_model_comparison_reports_proof_legacy_and_difference_in_differences():
    baseline = [
        _row("q1", "legacy", 0), _row("q1", "proof", 1),
        _row("q2", "legacy", 1), _row("q2", "proof", 1),
    ]
    candidate = [
        _row("q1", "legacy", 0), _row("q1", "proof", 1),
        _row("q2", "legacy", 0), _row("q2", "proof", 1),
    ]
    result = _model_comparison(baseline, candidate)
    assert result["candidate_minus_sft_proof_em"] == 0.0
    assert result["candidate_minus_sft_legacy_em"] == -0.5
    assert result["utilization_difference_in_differences"] == 0.5
    assert result["proof_arm_net_correct"] == 0


def test_model_comparison_rejects_different_qid_sets():
    baseline = [_row("q1", "legacy", 0), _row("q1", "proof", 0)]
    candidate = [_row("q2", "legacy", 0), _row("q2", "proof", 0)]
    try:
        _model_comparison(baseline, candidate)
    except SystemExit as exc:
        assert "identities differ" in str(exc)
    else:
        raise AssertionError("different qid sets must fail")
