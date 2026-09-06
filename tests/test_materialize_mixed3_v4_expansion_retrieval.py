from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from kgproweight.kg.question_kg import question_sha256
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256
from scripts.prepare.materialize_mixed3_v4_expansion_retrieval import (
    EXPECTED_PROTOCOL_SCHEMA,
    EXPECTED_PROTOCOL_STATUS,
    HM_PREREG_PROTOCOL_SCHEMA,
    HM_PREREG_PROTOCOL_STATUS,
    RETRIEVAL_STACK,
    _sha_json,
    materialize,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _request(dataset: str, qid: str) -> dict:
    question = f"Question for {dataset} {qid}?"
    return {
        "schema_version": "mixed-ppo-v4-retrieval-request-v1",
        "question_key": f"{dataset}::{qid}",
        "dataset": dataset,
        "qid": qid,
        "question": question,
        "question_sha256": question_sha256(question),
        "family_version": FAMILY_VERSION,
        "family_sha256": family_sha256(question),
        "role": "rollout_retrieval",
        "gold_access": False,
        "stratum": f"{dataset}_outcome",
    }


def _protocol(
    tmp_path: Path,
    rows: list[dict],
    *,
    frozen_counts: dict[str, int] | None = None,
    schema_version: str = EXPECTED_PROTOCOL_SCHEMA,
    status: str = EXPECTED_PROTOCOL_STATUS,
) -> Path:
    requests = tmp_path / "freeze" / "retrieval_requests.question_only.jsonl"
    _write_jsonl(requests, rows)
    counts = frozen_counts or dict(Counter(row["dataset"] for row in rows))
    protocol = {
        "schema_version": schema_version,
        "status": status,
        "population": {
            "retrieval_requests_by_dataset": counts,
        },
        "outputs": {
            "retrieval_requests": {
                "path": str(requests),
                "sha256": hashlib.sha256(requests.read_bytes()).hexdigest(),
                "size_bytes": requests.stat().st_size,
            }
        },
    }
    path = tmp_path / "freeze" / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    return path


def _fake_retrieval(dataset: str, rows: list[dict], batch_size: int) -> list[dict]:
    assert batch_size == 7
    contexts = []
    for request in rows:
        passages = [
            {
                "id": f"{request['qid']}-{index}",
                "source": "e5",
                "contents": f"Title {index}\nEvidence text {index}.",
            }
            for index in range(10)
        ]
        contexts.append(
            {
                "schema_version": "qpeg-retrieval-context-v1",
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
    return contexts


@pytest.mark.parametrize(
    ("schema_version", "status"),
    [
        (EXPECTED_PROTOCOL_SCHEMA, EXPECTED_PROTOCOL_STATUS),
        (HM_PREREG_PROTOCOL_SCHEMA, HM_PREREG_PROTOCOL_STATUS),
    ],
)
def test_materializes_protocol_counts_without_gpu(
    tmp_path: Path, monkeypatch, schema_version: str, status: str
):
    rows = [
        _request("hotpotqa", "h1"),
        _request("hotpotqa", "h2"),
        _request("musique", "m1"),
    ]
    protocol = _protocol(
        tmp_path,
        rows,
        schema_version=schema_version,
        status=status,
    )
    out = tmp_path / "retrieval"
    calls: list[tuple[str, int]] = []

    def fake(dataset: str, requests: list[dict], batch_size: int) -> list[dict]:
        calls.append((dataset, len(requests)))
        return _fake_retrieval(dataset, requests, batch_size)

    def fake_manifest(directory, extra=None, *, status="COMPLETE"):
        path = Path(directory) / "manifest.json"
        path.write_text(
            json.dumps({"status": status, "run": dict(extra or {})}),
            encoding="utf-8",
        )
        return path

    monkeypatch.setattr(
        "scripts.prepare.materialize_mixed3_v4_expansion_retrieval.dump_manifest",
        fake_manifest,
    )
    report = materialize(
        protocol_path=protocol,
        output_dir=out,
        batch_size=7,
        experiment_id="TEST-MIXED3-V4-RETRIEVAL",
        retrieval_fn=fake,
    )

    assert calls == [("hotpotqa", 2), ("musique", 1)]
    assert report["status"] == "COMPLETE_ANSWER_FREE_RETRIEVAL_NOT_TRAINED"
    assert report["counts"] == {
        "requests_total": 3,
        "contexts_total": 3,
        "by_dataset": {"hotpotqa": 2, "musique": 1},
    }
    assert all(report["gates"].values())
    assert (out / "hotpotqa.retrieval_contexts.jsonl").is_file()
    assert (out / "musique.retrieval_contexts.jsonl").is_file()
    assert (out / "retrieval_contexts.jsonl").is_file()
    assert (out / "report.json").is_file()
    assert (out / "manifest.json").is_file()
    combined = [
        json.loads(line)
        for line in (out / "retrieval_contexts.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [row["question_key"] for row in combined] == [
        "hotpotqa::h1", "hotpotqa::h2", "musique::m1"
    ]
    assert all(len(row["passages"]) == 10 for row in combined)
    assert all("answer" not in row for row in combined)
    assert report["backend_attestation"]["backend_fallback"] is False


def test_default_runtime_fails_before_writing_when_bge_attestation_fails(
    tmp_path: Path, monkeypatch
):
    rows = [_request("hotpotqa", "h1"), _request("musique", "m1")]
    protocol = _protocol(tmp_path, rows)
    out = tmp_path / "no-reranker-output"

    def fail_attestation():
        raise RuntimeError("BGE unavailable")

    monkeypatch.setattr(
        "scripts.prepare.materialize_mixed3_v4_expansion_retrieval._attest_cross_encoder_backend",
        fail_attestation,
    )
    with pytest.raises(RuntimeError, match="BGE unavailable"):
        materialize(
            protocol_path=protocol,
            output_dir=out,
            batch_size=7,
            experiment_id="TEST-BGE-FAIL-HARD",
        )
    assert not out.exists()


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda row: row.update(answer="Gold"), "forbidden Gold/evidence"),
        (lambda row: row.update(dataset="2wikimultihopqa"), "forbidden dataset"),
        (lambda row: row.update(question_sha256="0" * 64), "question_sha256"),
    ],
)
def test_rejects_non_answer_free_or_bad_identity_requests(
    tmp_path: Path, mutation, error: str
):
    rows = [_request("hotpotqa", "h1"), _request("musique", "m1")]
    mutation(rows[0])
    # Keep the protocol population valid so the unsupported-dataset case is
    # rejected by the row-level allow-list rather than by an earlier protocol
    # shape check.
    protocol = _protocol(
        tmp_path,
        rows,
        frozen_counts={"hotpotqa": 1, "musique": 1},
    )
    with pytest.raises(ValueError, match=error):
        materialize(
            protocol_path=protocol,
            output_dir=tmp_path / "out",
            batch_size=7,
            experiment_id="TEST-BAD-REQUEST",
            retrieval_fn=_fake_retrieval,
        )


def test_rejects_protocol_bound_request_hash_drift(tmp_path: Path):
    rows = [_request("hotpotqa", "h1"), _request("musique", "m1")]
    protocol = _protocol(tmp_path, rows)
    request_path = protocol.parent / "retrieval_requests.question_only.jsonl"
    request_path.write_text(request_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        materialize(
            protocol_path=protocol,
            output_dir=tmp_path / "out",
            batch_size=7,
            experiment_id="TEST-HASH-DRIFT",
            retrieval_fn=_fake_retrieval,
        )


def test_rejects_bad_retrieval_output_and_preserves_append_only_target(
    tmp_path: Path,
):
    rows = [_request("hotpotqa", "h1"), _request("musique", "m1")]
    protocol = _protocol(tmp_path, rows)

    def nine_passages(dataset: str, requests: list[dict], batch_size: int):
        contexts = _fake_retrieval(dataset, requests, batch_size)
        contexts[0]["passages"] = contexts[0]["passages"][:9]
        contexts[0]["passages_sha256"] = _sha_json(contexts[0]["passages"])
        return contexts

    with pytest.raises(ValueError, match="ten safe nonempty passages"):
        materialize(
            protocol_path=protocol,
            output_dir=tmp_path / "bad-output",
            batch_size=7,
            experiment_id="TEST-BAD-OUTPUT",
            retrieval_fn=nine_passages,
        )
    assert not (tmp_path / "bad-output").exists()

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materialize(
            protocol_path=protocol,
            output_dir=existing,
            batch_size=7,
            experiment_id="TEST-APPEND-ONLY",
            retrieval_fn=_fake_retrieval,
        )
