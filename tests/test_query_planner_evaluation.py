import json

from kgproweight.eval.query_planner import (
    build_scored_row,
    evaluate_gates,
    parse_plan,
    resolve_dev_gates,
    score_predictions,
)
from scripts.prepare.build_query_planner_supervision import build_2wiki_record, build_musique_record


def _two_wiki():
    return build_2wiki_record({
        "id": "q1", "question": "What country is the director of Film A from?",
        "golden_answers": ["Country X"],
        "metadata": {"evidences": {
            "fact": ["Film A", "Director X"],
            "relation": ["director", "country of citizenship"],
            "entity": ["Director X", "Country X"],
        }},
    })


def _musique():
    return build_musique_record({
        "id": "m1", "question": "When was the owner founded?", "golden_answers": ["1900"],
        "metadata": {"metadata": {"question_decomposition": [
            {"question": "Paper >> owned by", "answer": "Owner X"},
            {"question": "When was #1 founded?", "answer": "1900"},
        ]}},
    })


def test_parse_plan_is_strict_json_only():
    assert parse_plan('{"steps":[]}')[0] == {"steps": []}
    assert parse_plan('```json\n{"steps":[]}\n```')[0] is None


def test_perfect_predictions_pass_structural_metrics_and_gates():
    records = [_two_wiki(), _musique()]
    rows = [
        build_scored_row(record, json.dumps(record["target"]), source_row=None)
        for record in records
    ]
    metrics = score_predictions(rows)
    assert metrics["schema_valid_rate"] == 1.0
    assert metrics["2wikimultihopqa"]["pid_micro_f1"] == 1.0
    assert metrics["2wikimultihopqa"]["graph_exact"] == 1.0
    assert metrics["musique"]["dependency_graph_exact"] == 1.0
    gates = {
        "schema_valid_rate_min": 0.95,
        "answer_or_evidence_tail_leakage_rate_max": 0.0,
        "2wikimultihopqa": {
            "pid_micro_f1_min": 0.9, "pid_sequence_exact_min": 0.9,
            "dependency_edge_f1_min": 0.9, "graph_exact_min": 0.9,
        },
        "musique": {
            "dependency_edge_f1_min": 0.9, "dependency_graph_exact_min": 0.9,
            "operator_macro_f1_min": 0.9,
        },
    }
    assert evaluate_gates(metrics, gates)["pass"]


def test_invalid_prediction_counts_as_schema_failure():
    record = _two_wiki()
    row = build_scored_row(record, "not json", source_row=None)
    assert not row["schema_valid"]
    assert score_predictions([row])["schema_valid_rate"] == 0.0

    wrong_field = json.loads(json.dumps(record["target"]))
    wrong_field["steps"][0]["relation"] = wrong_field["steps"][0].pop("relation_label")
    row = build_scored_row(record, json.dumps(wrong_field), source_row=None)
    assert not row["schema_valid"]
    assert "invalid_relation_step_schema" in row["validation_errors"]


def test_source_loader_reads_both_dataset_files(tmp_path):
    from kgproweight.eval.query_planner import load_source_rows

    for dataset in ("2wikimultihopqa", "musique"):
        directory = tmp_path / dataset
        directory.mkdir()
        (directory / "train.jsonl").write_text(
            json.dumps({"id": "q1", "question": "q"}) + "\n", encoding="utf-8"
        )
    loaded = load_source_rows(tmp_path)
    assert set(loaded) == {"2wikimultihopqa::q1", "musique::q1"}


def test_dev_gate_protocol_key_is_backward_compatible():
    old = {"smoke_dev_gates": {"schema_valid_rate_min": 0.95}}
    new = {"dev_gates": {"schema_valid_rate_min": 0.97}}
    assert resolve_dev_gates(old)["schema_valid_rate_min"] == 0.95
    assert resolve_dev_gates(new)["schema_valid_rate_min"] == 0.97
