"""Formal training outputs must be unique and provenance-complete."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from kgproweight.training.phase3_ppo import Phase3PPOConfig, run_phase3_ppo
import kgproweight.utils.logging as logging_utils
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir


def test_prepare_new_run_dir_refuses_reuse(tmp_path: Path):
    out = tmp_path / "exp_001"
    created, exp_id = prepare_new_run_dir(out)
    assert created == out
    assert exp_id == "exp_001"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "RUNNING"
    assert manifest["run"]["experiment_id"] == "exp_001"
    assert manifest["source_tree_sha256"]
    assert manifest["source_tree_hash_mode"] in {"git_inventory", "filesystem_fallback"}
    assert manifest["source_tree_file_count"] > 0

    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        prepare_new_run_dir(out)


def test_complete_manifest_preserves_start_and_hashes_file(tmp_path: Path):
    source = tmp_path / "silver.jsonl"
    source.write_text('{"qid":"q"}\n', encoding="utf-8")
    out, _ = prepare_new_run_dir(tmp_path / "exp_002")
    identity = artifact_identity(source)
    assert identity["md5"]

    dump_manifest(out, extra={"experiment_id": "exp_002", "source": identity})
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETE"
    assert manifest["started_at"]
    assert manifest["completed_at"]
    assert manifest["run"]["source"]["md5"] == identity["md5"]


def test_fail_stop_manifest_records_terminal_completion(tmp_path: Path):
    out, _ = prepare_new_run_dir(tmp_path / "eval_001")
    running = json.loads((out / "manifest.json").read_text())
    assert running["completed_at"] is None

    dump_manifest(out, status="FAIL_STOP", extra={"experiment_id": "eval_001"})
    stopped = json.loads((out / "manifest.json").read_text())
    assert stopped["status"] == "FAIL_STOP"
    assert stopped["completed_at"]


def test_source_hash_falls_back_to_filesystem_without_git(monkeypatch, tmp_path: Path):
    (tmp_path / "kgproweight").mkdir()
    (tmp_path / "configs").mkdir()
    source = tmp_path / "kgproweight" / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "configs" / "run.yaml").write_text("seed: 42\n", encoding="utf-8")

    def no_git(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0] if args else "git")

    monkeypatch.setattr(logging_utils.subprocess, "check_output", no_git)
    monkeypatch.setenv("KGPW_PROJECT_ROOT", str(tmp_path))

    first = logging_utils._source_tree_provenance()
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second = logging_utils._source_tree_provenance()

    assert first["mode"] == "filesystem_fallback"
    assert first["root"] == str(tmp_path)
    assert first["sha256"]
    assert first["file_count"] == 2
    assert first["sha256"] != second["sha256"]


def _dummy_ppo_inputs(tmp_path: Path):
    silver = tmp_path / "silver.jsonl"
    silver.write_text("{}\n", encoding="utf-8")
    sft = tmp_path / "sft"
    sft.mkdir()
    index = tmp_path / "kg.json"
    index.write_text("[]\n", encoding="utf-8")
    return silver, sft, index


def test_ppo_refuses_missing_alpha_before_reserving_output(tmp_path: Path):
    silver, sft, index = _dummy_ppo_inputs(tmp_path)
    out = tmp_path / "ppo"
    cfg = Phase3PPOConfig(
        silver_path=str(silver), output_dir=str(out), split="train",
        sft_checkpoint=str(sft), alpha_gate_path=str(tmp_path / "missing.pt"),
        question_kg_index_path=str(index),
    )
    with pytest.raises(FileNotFoundError, match="randomly initialised gate"):
        run_phase3_ppo(cfg)
    assert not out.exists()
