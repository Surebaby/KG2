from __future__ import annotations

from pathlib import Path

import pytest

from scripts.prepare.materialize_light20_ppo_config import (
    EXPECTED_EXPERIMENT,
    EXPECTED_THRESHOLDS,
    resolve_selected_checkpoint,
)


def _report(*, status="PASS", selected="step40", passes=True):
    return {
        "experiment_id": EXPECTED_EXPERIMENT,
        "status": status,
        "thresholds": EXPECTED_THRESHOLDS,
        "selected": selected,
        "candidates": [{"label": selected, "passes_gate": passes}],
    }


def _adapter(root: Path, name: str) -> None:
    path = root / name
    path.mkdir(parents=True)
    (path / "adapter_model.safetensors").write_bytes(b"safe")
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")


def test_resolves_passing_adapter(tmp_path: Path):
    _adapter(tmp_path, "checkpoint-40")
    label, path = resolve_selected_checkpoint(_report(), tmp_path)
    assert label == "step40"
    assert path == tmp_path / "checkpoint-40"


@pytest.mark.parametrize("report", [
    _report(status="FAIL_STOP"),
    _report(passes=False),
    _report(selected="unknown"),
])
def test_refuses_unapproved_selection(tmp_path: Path, report):
    _adapter(tmp_path, "checkpoint-40")
    with pytest.raises(ValueError):
        resolve_selected_checkpoint(report, tmp_path)
