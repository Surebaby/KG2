from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare import (
    freeze_dependent_retrieval_v7_recursive_trajectory_addendum as addendum,
)


def test_real_recursive_trajectory_addendum_build_is_gold_free() -> None:
    protocol = addendum.build_protocol(addendum.DEFAULT_PATHS)

    assert protocol["schema_version"] == addendum.SCHEMA_VERSION
    assert protocol["status"] == addendum.STATUS
    assert protocol["scope"] == addendum.SCOPE
    assert protocol["effective_invariants"] == addendum.EXPECTED_INVARIANTS
    assert protocol["gold_access"] is False
    assert protocol["gpu_calls"] == 0
    assert protocol["planner_calls"] == 0
    assert protocol["retrieval_calls"] == 0
    assert protocol["execution_authorization"].startswith("BLOCKED_")


def test_recursive_trajectory_addendum_is_append_only(tmp_path: Path) -> None:
    protocol = addendum.build_protocol(addendum.DEFAULT_PATHS)
    output = tmp_path / "freeze"

    first = addendum.write_protocol(protocol, output)
    assert Path(first["protocol"]["path"]).is_file()
    with pytest.raises(FileExistsError):
        addendum.write_protocol(protocol, output)


def test_recursive_trajectory_source_tamper_fails_content_lock(tmp_path: Path) -> None:
    paths = dict(addendum.DEFAULT_PATHS)
    source = json.loads(paths["design_trajectory_addendum"].read_text(encoding="utf-8"))
    source["effective_invariants"][
        "divergent_upstream_bridges_may_induce_arm_specific_producer_passages"
    ] = False
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(source), encoding="utf-8")
    paths["design_trajectory_addendum"] = tampered

    with pytest.raises(ValueError, match="SHA256 drift"):
        addendum.build_protocol(paths)
