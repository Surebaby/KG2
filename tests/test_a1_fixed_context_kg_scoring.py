from scripts.pilot.score_a1_fixed_context_kg import _mcnemar_exact, paired_metrics


def _row(qid, arm, em, prediction, cite=False, well=True):
    return {
        "qid": qid,
        "arm": arm,
        "model_label": "sft",
        "em": em,
        "f1": float(em),
        "prediction": prediction,
        "well_formed": well,
        "known_citation_response": cite,
        "citation_contract_error": False,
    }


def test_paired_metrics_tracks_gains_losses_and_citations():
    rows = [
        _row("q1", "legacy", 0, "x"), _row("q1", "proof", 1, "gold", True),
        _row("q2", "legacy", 1, "gold"), _row("q2", "proof", 0, "x"),
        _row("q3", "legacy", 0, "x"), _row("q3", "proof", 1, "gold", True),
    ]

    result = paired_metrics(rows, "sft")

    assert result["gained_correct"] == 2
    assert result["lost_correct"] == 1
    assert result["net_correct"] == 1
    assert result["prediction_changed"] == 3
    assert result["citation_utilization_gain"] == 2 / 3


def test_mcnemar_exact_handles_no_discordance():
    assert _mcnemar_exact(0, 0) == 1.0
