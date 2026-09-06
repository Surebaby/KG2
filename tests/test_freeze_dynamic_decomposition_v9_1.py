"""CPU-only tests for the append-only v9.1 protocol freezer."""

from __future__ import annotations

import json

import pytest

from scripts.pilot import run_dynamic_decomposition_v9_1 as runner
from scripts.prepare import freeze_dynamic_decomposition_v9_1 as freezer


def test_freeze_writes_self_committed_gold_free_protocol(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "v91_protocol"
    monkeypatch.setattr(runner, "PROTOCOL_DIR", destination)

    protocol = freezer.freeze()
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))

    assert runner.v8_driver.implementation.verify_self_commitment(
        protocol, field="protocol_body_canonical_sha256"
    )
    assert protocol["gold_access"] is False
    assert protocol["answer_scoring"] is False
    assert protocol["prospective_opened_or_hashed"] is False
    assert protocol["gates"] == runner.expected_gates_v91()
    assert protocol["runtime_contract"] == runner.runtime_contract_v91()
    assert manifest["protocol"]["sha256"] == runner._sha256_file(
        destination / "protocol.json"
    )

    monkeypatch.setattr(runner, "PROTOCOL_PATH", destination / "protocol.json")
    monkeypatch.setattr(
        runner, "PROTOCOL_MANIFEST_PATH", destination / "manifest.json"
    )
    loaded, loaded_lock = runner._load_protocol()
    assert loaded == protocol
    assert loaded_lock == manifest["protocol"]

    with pytest.raises(FileExistsError, match="append-only"):
        freezer.freeze()


def test_frozen_cohorts_are_identity_only_and_exact_cardinality() -> None:
    for path, expected in (
        (freezer.SMOKE_COHORT, 12),
        (freezer.FRESH_PILOT_COHORT, 90),
    ):
        rows = [
            json.loads(line)
            for line in (freezer.PROJECT_ROOT / path)
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(rows) == expected
        assert all(tuple(row) == ("dataset", "qid", "question") for row in rows)
        assert all("gold" not in row and "answer" not in row for row in rows)
