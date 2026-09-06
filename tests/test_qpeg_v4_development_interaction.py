import pytest

from scripts.eval.score_qpeg_v4_development_interaction import score
from scripts.eval.select_qpeg_v4_development_checkpoint import select


GATES = {
    "macro_interaction_em_gt": 0.0,
    "macro_interaction_f1_gt": 0.0,
    "positive_interaction_datasets_ge": 2,
    "macro_C_minus_B_em_gt": 0.0,
    "macro_D_minus_A_em_ge": -0.01,
    "max_no_graph_net_loss_per_dataset": 1,
    "max_parse_rate_drop": 0.02,
}


def _rows(label, *, adapted=False, harmful_no_graph=False):
    rows = []
    datasets = ("hotpotqa", "2wikimultihopqa", "musique")
    for dataset in datasets:
        for index in range(50):
            common = {
                "row_id": f"{dataset}::{index}",
                "dataset": dataset,
                "qid": str(index),
                "question": f"q{index}",
                "gold_answers": ["x"],
                "model_label": label,
                "well_formed": True,
            }
            no_graph_em = 0.0 if not harmful_no_graph else (0.0 if index else 1.0)
            graph_em = 1.0 if adapted and index < 5 else 0.0
            rows.append({**common, "arm": "legacy", "em": no_graph_em, "f1": no_graph_em})
            rows.append({**common, "arm": "proof", "em": graph_em, "f1": graph_em})
    return rows


def test_positive_interaction_passes_all_gates():
    result = score(
        _rows("strong"), _rows("adapted", adapted=True),
        strong_label="strong", adapted_label="adapted", gates=GATES,
    )
    assert result["macro"]["interaction_em"] == pytest.approx(0.1)
    assert result["positive_interaction_datasets"] == 3
    assert result["all_development_gates_pass"] is True


def test_no_interaction_fails_strict_positive_gates():
    result = score(
        _rows("strong"), _rows("adapted"),
        strong_label="strong", adapted_label="adapted", gates=GATES,
    )
    assert result["all_development_gates_pass"] is False
    assert result["gate_checks"]["macro_interaction_em"] is False


def test_checkpoint_selection_is_earliest_passing_step():
    assert select([
        {"checkpoint_step": 75, "all_development_gates_pass": True},
        {"checkpoint_step": 25, "all_development_gates_pass": False},
        {"checkpoint_step": 50, "all_development_gates_pass": True},
    ]) == 50
    assert select([{"checkpoint_step": 25, "all_development_gates_pass": False}]) is None
