import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "pilot"))
from score_paired_kg_eval_models import (
    _mcnemar_exact,
    paired_metrics,
    stratified,
)


def _row(qid, arm, model_label, em, gold_in_passages, gold_in_kg_tail, known_cite=True):
    return {
        "qid": qid, "arm": arm, "model_label": model_label, "em": em, "f1": em,
        "well_formed": True, "known_citation_response": known_cite,
        "citation_contract_error": False, "gold_in_passages": gold_in_passages,
        "gold_in_kg_tail": gold_in_kg_tail, "prediction": str(em),
    }


def test_mcnemar_exact_extremes():
    assert _mcnemar_exact(0, 0) == 1.0
    assert _mcnemar_exact(1, 0) == 1.0  # 2 * (C(1,0)/2) = 1.0


def test_paired_metrics_reproduces_delta_and_net():
    rows = []
    # 2 qids where proof > legacy
    for q in ["q1", "q2"]:
        rows.append(_row(q, "legacy", "sft", 0.0, True, True))
        rows.append(_row(q, "proof", "sft", 1.0, True, True))
    # 1 qid where proof == legacy
    rows.append(_row("q3", "legacy", "sft", 1.0, False, False))
    rows.append(_row("q3", "proof", "sft", 1.0, False, False))
    m = paired_metrics(rows, "sft")
    assert m["n"] == 3
    assert abs(m["legacy_em"] - (0 + 0 + 1) / 3) < 1e-9
    assert abs(m["proof_em"] - (1 + 1 + 1) / 3) < 1e-9
    assert m["net_correct"] == 2
    assert m["gained_correct"] == 2 and m["lost_correct"] == 0


def test_stratified_splits_passages_and_tail():
    rows = [
        _row("q1", "legacy", "sft", 0.0, True, False),
        _row("q1", "proof", "sft", 1.0, True, False),
        _row("q2", "legacy", "sft", 0.0, False, True),
        _row("q2", "proof", "sft", 1.0, False, True),
        _row("q1", "legacy", "ppo", 0.0, True, False),
        _row("q1", "proof", "ppo", 0.0, True, False),
        _row("q2", "legacy", "ppo", 0.0, False, True),
        _row("q2", "proof", "ppo", 1.0, False, True),
    ]
    s = stratified(rows, "sft", "ppo")
    assert s["passages_visible"]["n"] == 1
    assert s["tail_visible"]["n"] == 1
    # q1 (passages visible, tail hidden): baseline delta +1, candidate delta 0
    assert abs(s["passages_visible"]["baseline_delta"] - 1.0) < 1e-9
    assert abs(s["passages_visible"]["candidate_delta"] - 0.0) < 1e-9
    # q2 (tail visible): baseline delta +1, candidate delta +1
    assert abs(s["tail_visible"]["baseline_delta"] - 1.0) < 1e-9
    assert abs(s["tail_visible"]["candidate_delta"] - 1.0) < 1e-9
