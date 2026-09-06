from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare.freeze_query_controller_v4_4_eval_e1 import (
    EXPERIMENT_ID,
    GENERATION_EXPERIMENT_ID,
    STATUS,
    _canonical_sha256,
    _sha256_file,
    build_protocol,
    freeze,
)


def test_eval_e1_protocol_is_eval_only_and_binds_completed_v4_4() -> None:
    protocol, checks = build_protocol()
    body = dict(protocol)
    declared = body.pop("protocol_body_canonical_sha256")
    assert declared == _canonical_sha256(body)
    assert protocol["experiment_id"] == EXPERIMENT_ID
    assert protocol["status"] == STATUS
    assert protocol["scope"] == {
        "kind": "eval_only_successor",
        "parent_training_version": "v4.4",
        "cohort_role": "dev",
        "teacher_forced_q2_state": True,
        "runtime_reader_predicted": False,
        "training_authorized": False,
    }
    assert protocol["parent_training_lineage"]["probe"]["status"] == "COMPLETE"
    assert protocol["parent_training_lineage"]["probe"]["global_steps"] == 20
    assert protocol["authorized_generation"]["experiment_id"] == GENERATION_EXPERIMENT_ID
    assert protocol["authorized_generation"]["exact_actions"] == 240
    assert protocol["authorized_generation"]["outcome_metrics_authorized"] == {
        "em": False,
        "f1": False,
        "ihr": False,
    }
    assert protocol["change_control"]["retraining"] is False
    assert protocol["change_control"]["checkpoint_reused"] is True
    assert protocol["scientific_boundary"]["confirmation_access"] is False
    assert protocol["scientific_boundary"]["prospective_access"] is False
    assert checks["parent_training_complete_20_steps"] is True
    assert checks["parent_adapter_exact"] is True


def test_eval_e1_predecessor_failure_is_bound_and_pre_cuda() -> None:
    protocol, _ = build_protocol()
    failure = protocol["predecessor_eval_failure"]
    assert failure["status"] == "FAIL_PRE_CUDA_NO_GENERATION_NO_PREDICTIONS_RECORDED"
    assert failure["cuda_queried"] is False
    assert failure["generation_started"] is False
    assert failure["prediction_rows_written"] == 0
    assert len(failure["sha256"]) == 64


@pytest.mark.parametrize(
    "name",
    [
        "confirmation.jsonl",
        "confirmation.identity_only.jsonl",
        "prospective.identity_only.jsonl",
    ],
)
def test_eval_e1_freezer_refuses_to_open_held_out_files(
    tmp_path: Path, name: str
) -> None:
    path = tmp_path / name
    path.write_text("secret\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="must not open held-out"):
        _sha256_file(path)


def test_eval_e1_freeze_is_append_only_and_manifest_binds_outputs(
    tmp_path: Path,
) -> None:
    output = tmp_path / "eval_e1"
    hashes = freeze(output)
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert hashes == {
        name: hashlib.sha256((output / f"{name}.json").read_bytes()).hexdigest()
        for name in ("protocol", "report", "manifest")
    }
    assert manifest["outputs"] == {
        "protocol.json": hashes["protocol"],
        "report.json": hashes["report"],
    }
    assert manifest["training_started"] is False
    assert manifest["generation_started"] is False
    assert manifest["confirmation_opened_or_hashed"] is False
    assert manifest["prospective_opened_or_hashed"] is False
    assert report["parent_adapter_sha256"] == protocol["parent_training_lineage"]["probe"][
        "adapter_sha256"
    ]
    with pytest.raises(FileExistsError, match="append-only"):
        freeze(output)
