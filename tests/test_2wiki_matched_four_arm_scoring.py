from scripts.eval.score_2wiki_matched_four_arm import (
    _interaction,
    _mcnemar_exact,
    _paired_effect,
)


def _row(model: str, arm: str, qid: str, em: float, f1: float | None = None):
    return {
        "model": model,
        "arm": arm,
        "qid": qid,
        "em": em,
        "f1": em if f1 is None else f1,
    }


def test_paired_effect_reports_delta_and_discordant_counts():
    left = [_row("sft", "legacy", "q1", 0), _row("sft", "legacy", "q2", 1)]
    right = [_row("sft", "proof", "q1", 1), _row("sft", "proof", "q2", 0)]
    result = _paired_effect(left, right, seed=7, draws=100)
    assert result["em_delta"] == 0.0
    assert result["em_gained"] == 1
    assert result["em_lost"] == 1
    assert result["em_tied"] == 0
    assert result["mcnemar_exact_p"] == 1.0


def test_interaction_is_difference_of_within_model_supply_effects():
    cells = {
        ("sft", "legacy"): [_row("sft", "legacy", "q1", 0), _row("sft", "legacy", "q2", 1)],
        ("sft", "proof"): [_row("sft", "proof", "q1", 1), _row("sft", "proof", "q2", 1)],
        ("proofkg_ppo", "legacy"): [
            _row("proofkg_ppo", "legacy", "q1", 0),
            _row("proofkg_ppo", "legacy", "q2", 0),
        ],
        ("proofkg_ppo", "proof"): [
            _row("proofkg_ppo", "proof", "q1", 1),
            _row("proofkg_ppo", "proof", "q2", 1),
        ],
    }
    result = _interaction(cells, metric="em", seed=9, draws=100)
    # SFT supply = +0.5; PPO supply = +1.0; interaction = +0.5.
    assert result["delta"] == 0.5


def test_mcnemar_handles_no_discordant_predictions():
    assert _mcnemar_exact(0, 0) == 1.0
