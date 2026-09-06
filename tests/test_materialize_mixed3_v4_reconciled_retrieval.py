from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kgproweight.kg.question_kg import question_sha256
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256
from scripts.prepare.materialize_mixed3_v4_expansion_retrieval import (
    RETRIEVAL_STACK,
    _sha_json,
)
from scripts.prepare.materialize_mixed3_v4_reconciled_retrieval import (
    RECONCILIATION_SCHEMA,
    RECONCILIATION_STATUS,
    _validate_reused_context,
    materialize,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _file_identity(path: Path) -> dict:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


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
        "stratum": "test",
    }


def _context(request: dict) -> dict:
    passages = [
        {
            "id": f"{request['qid']}-{index}",
            "source": "e5",
            "contents": f"Title {index}\nEvidence {index}.",
        }
        for index in range(10)
    ]
    return {
        "schema_version": "mixed3-v4-expansion-retrieval-v1",
        **{
            name: request[name]
            for name in (
                "question_key",
                "dataset",
                "qid",
                "question",
                "question_sha256",
                "family_sha256",
                "role",
                "gold_access",
            )
        },
        "passages": passages,
        "passages_sha256": _sha_json(passages),
        "retrieval_source": RETRIEVAL_STACK,
    }


def _binding(request: dict, context: dict) -> dict:
    return {
        "schema_version": "mixed-ppo-v4-reused-context-binding-v2",
        **{
            name: request[name]
            for name in (
                "dataset",
                "qid",
                "question",
                "question_sha256",
                "family_version",
                "family_sha256",
                "gold_access",
            )
        },
        "passages_sha256": context["passages_sha256"],
        "reuse_existing_context": True,
    }


def _frozen_fixture(tmp_path: Path) -> tuple[Path, list[dict]]:
    hotpot = [_request("hotpotqa", f"h{index:04d}") for index in range(417)]
    old_musique = [_request("musique", f"m{index:04d}") for index in range(401)]
    new_musique = [_request("musique", f"new{index:04d}") for index in range(11)]
    prior_requests = [*hotpot, *old_musique]
    prior_contexts = [_context(row) for row in prior_requests]
    reused_requests = [*hotpot, *old_musique[6:]]
    requirements = [*hotpot, *old_musique[6:], *new_musique]
    reused_bindings = [
        _binding(request, context)
        for request, context in zip(prior_requests, prior_contexts)
        if request in reused_requests
    ]
    retired = [
        {
            "schema_version": "mixed-ppo-v4-retired-context-identity-v2",
            **{
                name: request[name]
                for name in (
                    "dataset",
                    "qid",
                    "question",
                    "question_sha256",
                    "family_sha256",
                    "gold_access",
                )
            },
            "reason": "removed_by_complete_protected_ledger",
        }
        for request in old_musique[:6]
    ]

    freeze = tmp_path / "freeze"
    paths = {
        "retrieval_requirements": freeze / "requirements.jsonl",
        "reused_context_bindings": freeze / "reused.jsonl",
        "new_retrieval_requests": freeze / "new.jsonl",
        "retired_contexts": freeze / "retired.jsonl",
    }
    for path, rows in zip(
        paths.values(), (requirements, reused_bindings, new_musique, retired)
    ):
        _write_jsonl(path, rows)

    prior_dir = tmp_path / "prior"
    prior_contexts_path = prior_dir / "retrieval_contexts.jsonl"
    _write_jsonl(prior_contexts_path, prior_contexts)
    attestation = {
        "mode": "cross_encoder",
        "requested_backend": "bge-reranker-v2-m3",
        "config": {"sha256": "a" * 64},
        "weights": {"sha256": "b" * 64},
        "tokenizer": {"sha256": "c" * 64},
        "load_succeeded": True,
        "backend_fallback": False,
    }
    prior_report = {
        "status": "COMPLETE_ANSWER_FREE_RETRIEVAL_NOT_TRAINED",
        "retrieval": RETRIEVAL_STACK,
        "gates": {"complete": True},
        "backend_attestation": attestation,
        "outputs": {"combined": _file_identity(prior_contexts_path)},
    }
    prior_report_path = prior_dir / "report.json"
    prior_report_path.write_text(json.dumps(prior_report), encoding="utf-8")

    protocol = {
        "schema_version": RECONCILIATION_SCHEMA,
        "status": RECONCILIATION_STATUS,
        "gates": {"frozen": True},
        "retrieval_reconciliation": {
            "required_for_reconciled_population": 823,
            "reused": 812,
            "new_requests": 11,
            "retired": 6,
            "retrieval_executed_in_this_stage": False,
        },
        "inputs": {
            "completed_retrieval_report": _file_identity(prior_report_path),
            "completed_retrieval_contexts": _file_identity(prior_contexts_path),
        },
        "outputs": {name: _file_identity(path) for name, path in paths.items()},
    }
    protocol_path = freeze / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    (freeze / "manifest.json").write_text(
        json.dumps(
            {
                "status": RECONCILIATION_STATUS,
                "run": {
                    "protocol_sha256": hashlib.sha256(
                        protocol_path.read_bytes()
                    ).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )
    return protocol_path, new_musique


def test_materializes_exact_812_reuse_plus_11_new(tmp_path: Path, monkeypatch):
    protocol, expected_new = _frozen_fixture(tmp_path)
    calls: list[tuple[str, list[str], int]] = []

    def fake_retrieval(dataset: str, rows: list[dict], batch_size: int) -> list[dict]:
        calls.append((dataset, [row["qid"] for row in rows], batch_size))
        return [_context(row) for row in rows]

    def fake_manifest(directory, extra=None, *, status="COMPLETE"):
        path = Path(directory) / "manifest.json"
        path.write_text(
            json.dumps({"status": status, "run": dict(extra or {})}),
            encoding="utf-8",
        )
        return path

    monkeypatch.setattr(
        "scripts.prepare.materialize_mixed3_v4_reconciled_retrieval.dump_manifest",
        fake_manifest,
    )
    attestation = {
        "mode": "cross_encoder",
        "requested_backend": "bge-reranker-v2-m3",
        "config": {"sha256": "a" * 64},
        "weights": {"sha256": "b" * 64},
        "tokenizer": {"sha256": "c" * 64},
        "load_succeeded": True,
        "backend_fallback": False,
    }
    out = tmp_path / "release"
    report = materialize(
        protocol_path=protocol,
        output_dir=out,
        batch_size=5,
        experiment_id="TEST-HM-RECONCILED-RETRIEVAL",
        retrieval_fn=fake_retrieval,
        backend_attestation=attestation,
    )

    assert calls == [("musique", [row["qid"] for row in expected_new], 5)]
    assert report["counts"] == {
        "requests_total": 823,
        "contexts_total": 823,
        "by_dataset": {"hotpotqa": 417, "musique": 406},
    }
    assert report["reconciliation"] == {
        "reused_contexts": 812,
        "newly_retrieved_contexts": 11,
        "retired_contexts": 6,
        "prior_contexts": 818,
    }
    assert all(report["gates"].values())
    assert len((out / "retrieval_contexts.jsonl").read_text().splitlines()) == 823
    assert len((out / "newly_retrieved_contexts.jsonl").read_text().splitlines()) == 11


def test_reused_context_hash_drift_fails_closed():
    request = _request("hotpotqa", "h1")
    context = _context(request)
    binding = _binding(request, context)
    binding["passages_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="reused passage hash drifted"):
        _validate_reused_context(context, binding, request)
