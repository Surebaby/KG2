from __future__ import annotations

import json
from pathlib import Path

import pytest

from kgproweight.kg.question_kg import question_sha256
from scripts.prepare.freeze_2wiki_proofkg_extension_v1 import file_ref, write_jsonl
from scripts.prepare.freeze_2wiki_proofkg_reserve_retrieval_v1 import (
    EXPECTED_SOURCE_SCHEMA,
    EXPECTED_SOURCE_STATUS,
    RETRIEVAL_STACK,
    build_requests,
    freeze,
)
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256
from scripts.prepare.materialize_2wiki_proofkg_reserve_retrieval_v1 import (
    STATUS,
    materialize,
)
from scripts.prepare.materialize_2wiki_proofkg_unified_v2 import (
    _validate_retrieval_release,
)
from scripts.prepare.materialize_qpeg_v1_retrieval import _sha_json


def _cohort() -> list[dict]:
    qtypes = ["inference"] * 30 + ["compositional"] * 12 + ["comparison"] * 8
    rows = []
    for index, qtype in enumerate(qtypes):
        qid = f"r-{index:03d}"
        question = f"Where does reserve marker {index} lead?"
        rows.append(
            {
                "schema_version": (
                    "automatic-proofkg-extension-reserve-question-only-v1"
                ),
                "question_key": f"2wikimultihopqa::{qid}",
                "dataset": "2wikimultihopqa",
                "qid": qid,
                "question": question,
                "question_sha256": question_sha256(question),
                "family_version": FAMILY_VERSION,
                "family_sha256": family_sha256(question),
                "question_type": qtype,
                "source_role": "automatic_proofkg_extension_reserve_candidate",
                "gold_access": False,
                "evaluation_eligible": False,
            }
        )
    return rows


def _source_protocol(tmp_path: Path) -> Path:
    cohort = tmp_path / "source" / "cohort.question_only.jsonl"
    cohort.parent.mkdir(parents=True)
    write_jsonl(cohort, _cohort())
    protocol = tmp_path / "source" / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "schema_version": EXPECTED_SOURCE_SCHEMA,
                "status": EXPECTED_SOURCE_STATUS,
                "output_cohort": file_ref(cohort),
            }
        ),
        encoding="utf-8",
    )
    return protocol


def _fake_retrieval(dataset: str, rows: list[dict], batch_size: int) -> list[dict]:
    assert dataset == "2wikimultihopqa"
    assert batch_size == 8
    output = []
    for row in reversed(rows):
        passages = [
            {
                "id": f"{row['qid']}-{index}",
                "source": "canonical",
                "contents": f"Evidence {index}.",
            }
            for index in range(10)
        ]
        output.append(
            {
                "question_key": row["question_key"],
                "dataset": dataset,
                "qid": row["qid"],
                "question": row["question"],
                "question_sha256": row["question_sha256"],
                "family_sha256": row["family_sha256"],
                "role": row["role"],
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


def test_freeze_and_materialize_exact_answer_free_reserve50(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.prepare.freeze_2wiki_proofkg_reserve_retrieval_v1.dump_manifest",
        _fake_manifest,
    )
    monkeypatch.setattr(
        "scripts.prepare.materialize_2wiki_proofkg_reserve_retrieval_v1.dump_manifest",
        _fake_manifest,
    )
    prereg = tmp_path / "prereg"
    freeze(
        source_protocol_path=_source_protocol(tmp_path),
        output_dir=prereg,
        experiment_id="TEST-PREREG",
    )
    requests = [
        json.loads(line)
        for line in (prereg / "retrieval_requests.question_only.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(requests) == 50
    assert all("answer" not in row and row["gold_access"] is False for row in requests)

    out = tmp_path / "retrieval"
    attestation = {
        "mode": "cross_encoder",
        "requested_backend": "bge-reranker-v2-m3",
        "load_succeeded": True,
        "backend_fallback": False,
    }
    report = materialize(
        protocol_path=prereg / "protocol.json",
        output_dir=out,
        batch_size=8,
        experiment_id="TEST-RETRIEVAL",
        retrieval_fn=_fake_retrieval,
        backend_attestation=attestation,
    )
    assert report["status"] == STATUS
    assert all(report["gates"].values())
    contexts = [
        json.loads(line)
        for line in (out / "retrieval_contexts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["question_key"] for row in contexts] == [
        row["question_key"] for row in requests
    ]
    assert all(len(row["passages"]) == 10 for row in contexts)
    assert _validate_retrieval_release(out)[0] == out / "retrieval_contexts.jsonl"


def test_request_builder_rejects_gold_or_wrong_quota():
    rows = _cohort()
    rows[0]["answer"] = "forbidden"
    with pytest.raises(ValueError, match="forbidden fields"):
        build_requests(rows)
    rows = _cohort()
    rows[0]["question_type"] = "comparison"
    with pytest.raises(ValueError, match="quotas drifted"):
        build_requests(rows)


def test_materializer_fails_before_output_on_bad_passages(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.prepare.freeze_2wiki_proofkg_reserve_retrieval_v1.dump_manifest",
        _fake_manifest,
    )
    prereg = tmp_path / "prereg"
    freeze(
        source_protocol_path=_source_protocol(tmp_path),
        output_dir=prereg,
        experiment_id="TEST-PREREG",
    )

    def bad(dataset: str, rows: list[dict], batch_size: int):
        values = _fake_retrieval(dataset, rows, batch_size)
        values[0]["passages"] = values[0]["passages"][:9]
        values[0]["passages_sha256"] = _sha_json(values[0]["passages"])
        return values

    out = tmp_path / "bad"
    with pytest.raises(ValueError, match="passage/backend contract"):
        materialize(
            protocol_path=prereg / "protocol.json",
            output_dir=out,
            batch_size=8,
            experiment_id="TEST-BAD",
            retrieval_fn=bad,
        )
    assert not out.exists()
