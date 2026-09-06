from __future__ import annotations

import json
from pathlib import Path

import pytest

from kgproweight.kg.question_kg import question_sha256
from scripts.prepare.freeze_2wiki_official_raw_canonical_retrieval_v1 import (
    FAMILY_VERSION,
    REQUEST_SCHEMA,
    RETRIEVAL_STACK,
    SCOPE_SCHEMA,
    SCOPE_STATUS,
    build_scope_requests,
    family_sha256,
    predicate_eligible,
)
from scripts.prepare.materialize_2wiki_official_raw_canonical_retrieval_v1 import (
    STATUS,
    materialize,
)
from scripts.prepare.materialize_2wiki_proofkg_reserve_retrieval_v1 import (
    STATUS as RESERVE50_STATUS,
)
from scripts.prepare.materialize_qpeg_v1_retrieval import _sha_json


QTYPES = ("bridge_comparison", "comparison", "compositional", "inference")


def _base(index: int, qtype: str) -> dict:
    qid = f"q-{qtype}-{index}"
    question = f"Where does {qtype} marker {index} lead?"
    return {
        "dataset": "2wikimultihopqa",
        "qid": qid,
        "question_key": f"2wikimultihopqa::{qid}",
        "question": question,
        "question_sha256": question_sha256(question),
        "family_version": FAMILY_VERSION,
        "family_sha256": family_sha256(question),
        "question_type": qtype,
        "gold_access": False,
    }


def _telemetry(base: dict, *, eligible: bool = True) -> dict:
    value = {
        **{key: base[key] for key in (
            "dataset", "qid", "question_key", "question_sha256",
            "family_sha256", "question_type",
        )},
        "m_graph": 1,
        "planner_schema_valid": True,
        "all_root_anchors_resolved": True,
        "all_hops_complete": True,
        "graph_nonempty": True,
        "gold_access_false": True,
        "runtime_error_zero": True,
        "provenance_complete": True,
        "retained_edges_traceable": True,
        "no_duplicate_edges": True,
        "runtime_record_sha256": "a" * 64,
        "kg_sha256": "b" * 64,
        "execution_sha256": "c" * 64,
        "routing_reason": "identity_safe_complete_traceable_graph",
    }
    if not eligible:
        value["m_graph"] = 0
        value["all_hops_complete"] = False
        value["routing_reason"] = "failed:complete"
    return value


def test_scope_is_all_and_only_predicate_rows_and_not_top800():
    cohort = []
    telemetry = []
    for qtype in QTYPES:
        for index in range(3):
            row = _base(index, qtype)
            cohort.append(row)
            telemetry.append(_telemetry(row, eligible=index != 2))
    requests, counts = build_scope_requests(cohort, telemetry, min_per_type=2)
    assert len(requests) == 8
    assert counts["strict_scope_by_question_type"] == {
        qtype: 2 for qtype in QTYPES
    }
    assert counts["rejected_by_routing_reason"] == {"failed:complete": 4}
    assert all(predicate_eligible(telemetry[index]) for index in (0, 1, 3, 4, 6, 7, 9, 10))
    assert all(row["schema_version"] == REQUEST_SCHEMA for row in requests)
    assert all("answer" not in row and row["gold_access"] is False for row in requests)


def test_scope_fails_if_any_hard_question_type_quota_is_short():
    cohort = [_base(0, qtype) for qtype in QTYPES]
    telemetry = [_telemetry(row) for row in cohort]
    telemetry[-1]["m_graph"] = 0
    with pytest.raises(RuntimeError, match="lacks strict capacity"):
        build_scope_requests(cohort, telemetry, min_per_type=1)


def _request(index: int, qtype: str) -> dict:
    row = _base(index, qtype)
    return {
        "schema_version": REQUEST_SCHEMA,
        **row,
        "role": "official_raw_proofkg_rollout_retrieval",
        "closure_runtime_record_sha256": "a" * 64,
        "closure_kg_sha256": "b" * 64,
        "closure_execution_sha256": "c" * 64,
        "evaluation_eligible": False,
    }


def _fake_retrieval(dataset: str, rows: list[dict], batch_size: int) -> list[dict]:
    del batch_size
    output = []
    for request in reversed(rows):
        passages = [
            {
                "id": f"{request['qid']}-{index}",
                "source": "canonical",
                "contents": f"Evidence {index}.",
            }
            for index in range(10)
        ]
        output.append(
            {
                "question_key": request["question_key"],
                "dataset": dataset,
                "qid": request["qid"],
                "question": request["question"],
                "question_sha256": request["question_sha256"],
                "family_sha256": request["family_sha256"],
                "role": request["role"],
                "gold_access": False,
                "passages": passages,
                "passages_sha256": _sha_json(passages),
                "retrieval_source": RETRIEVAL_STACK,
            }
        )
    return output


def _fake_manifest(directory, extra=None, *, status="COMPLETE"):
    path = Path(directory) / "manifest.json"
    path.write_text(
        json.dumps({"status": status, "run": dict(extra or {})}), encoding="utf-8"
    )
    return path


def test_materializer_is_dynamic_official_raw_release_not_reserve50(
    tmp_path: Path, monkeypatch
):
    import scripts.prepare.materialize_2wiki_official_raw_canonical_retrieval_v1 as module

    monkeypatch.setattr(module, "MIN_PER_TYPE", 1)
    monkeypatch.setattr(module, "dump_manifest", _fake_manifest)
    requests = [_request(0, qtype) for qtype in QTYPES]
    request_path = tmp_path / "requests.jsonl"
    request_path.write_text(
        "".join(json.dumps(row) + "\n" for row in requests), encoding="utf-8"
    )
    protocol = {
        "schema_version": SCOPE_SCHEMA,
        "status": SCOPE_STATUS,
        "population": {
            "strict_scope_total": 4,
            "strict_scope_by_question_type": {qtype: 1 for qtype in QTYPES},
        },
        "retrieval": {"stack": RETRIEVAL_STACK, "backend_fallback_allowed": False},
        "outputs": {
            "retrieval_requests": {
                "path": str(request_path),
                "sha256": module._sha256(request_path),
                "size_bytes": request_path.stat().st_size,
            }
        },
        "gates": {"scope": True},
        "scientific_boundary": {"retrieval_started": False, "training_started": False},
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    report = materialize(
        protocol_path=protocol_path,
        output_dir=tmp_path / "release",
        batch_size=2,
        experiment_id="TEST-OFFICIAL-RAW",
        retrieval_fn=_fake_retrieval,
        backend_attestation={
            "mode": "cross_encoder",
            "requested_backend": "bge-reranker-v2-m3",
            "load_succeeded": True,
            "backend_fallback": False,
        },
    )
    assert report["status"] == STATUS
    assert STATUS != RESERVE50_STATUS
    assert report["counts"]["contexts"] == 4
    assert all(report["gates"].values())
    contexts = [
        json.loads(line)
        for line in (tmp_path / "release/retrieval_contexts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["question_key"] for row in contexts] == [
        row["question_key"] for row in requests
    ]
    assert all(row["closure_kg_sha256"] == "b" * 64 for row in contexts)


def test_materializer_fails_before_output_when_backend_fallbacks(tmp_path, monkeypatch):
    import scripts.prepare.materialize_2wiki_official_raw_canonical_retrieval_v1 as module

    monkeypatch.setattr(module, "MIN_PER_TYPE", 1)
    requests = [_request(0, qtype) for qtype in QTYPES]
    request_path = tmp_path / "requests.jsonl"
    request_path.write_text(
        "".join(json.dumps(row) + "\n" for row in requests), encoding="utf-8"
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        json.dumps(
            {
                "schema_version": SCOPE_SCHEMA,
                "status": SCOPE_STATUS,
                "population": {
                    "strict_scope_total": 4,
                    "strict_scope_by_question_type": {qtype: 1 for qtype in QTYPES},
                },
                "retrieval": {"stack": RETRIEVAL_STACK, "backend_fallback_allowed": False},
                "outputs": {
                    "retrieval_requests": {
                        "path": str(request_path),
                        "sha256": module._sha256(request_path),
                        "size_bytes": request_path.stat().st_size,
                    }
                },
                "gates": {"scope": True},
                "scientific_boundary": {"retrieval_started": False, "training_started": False},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="BGE attestation"):
        materialize(
            protocol_path=protocol_path,
            output_dir=tmp_path / "bad",
            batch_size=2,
            experiment_id="TEST-BAD",
            retrieval_fn=_fake_retrieval,
            backend_attestation={"load_succeeded": False, "backend_fallback": True},
        )
    assert not (tmp_path / "bad").exists()
