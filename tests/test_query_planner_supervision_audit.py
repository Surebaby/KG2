import json

from scripts.prepare.audit_query_planner_supervision import audit, validate_record
from scripts.prepare.build_query_planner_supervision import build_2wiki_record


def _source():
    return {
        "id": "q1",
        "question": "What country is the director of Film A from?",
        "golden_answers": ["SECRET_COUNTRY"],
        "metadata": {
            "evidences": {
                "fact": ["Film A", "SECRET_DIRECTOR"],
                "relation": ["director", "country of citizenship"],
                "entity": ["SECRET_DIRECTOR", "SECRET_COUNTRY"],
            }
        },
    }


def test_valid_record_has_prior_only_dependencies():
    record = build_2wiki_record(_source())
    assert validate_record(record) == []


def test_audit_detects_tail_leakage(tmp_path):
    source = _source()
    record = build_2wiki_record(source)
    record["target"]["anchors"].append("SECRET_COUNTRY")
    supervision = tmp_path / "supervision.jsonl"
    supervision.write_text(json.dumps(record) + "\n", encoding="utf-8")
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(json.dumps(source) + "\n", encoding="utf-8")
    result, leakage = audit(supervision, {"2wikimultihopqa": source_path})
    assert result["status"] == "FAIL"
    assert result["counts"]["answer_or_tail_leakage"] >= 1
    assert leakage[0]["leaked_value"] in {"SECRET_DIRECTOR", "SECRET_COUNTRY"}
