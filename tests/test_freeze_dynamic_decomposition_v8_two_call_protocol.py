from __future__ import annotations

from pathlib import Path

import pytest

from kgproweight.retrieval.dynamic_decomposition_v8_cohort import (
    COHORT_LOADER_VERSION,
    EXPECTED_DEVELOPMENT_SHA256,
    EXPECTED_MANIFEST_SHA256,
)
from scripts.prepare import freeze_dynamic_decomposition_v8_two_call_protocol as freeze


def _cohort_lock() -> dict:
    return {
        "loader_version": COHORT_LOADER_VERSION,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "development_sha256": EXPECTED_DEVELOPMENT_SHA256,
        "row_count": 90,
        "prospective_opened_or_hashed_by_this_command": False,
    }


def test_protocol_freezes_approved_two_call_fallback_matrix():
    protocol = freeze.build_protocol(
        generated_at_utc="2026-09-04T00:00:00+00:00",
        cohort_lock=_cohort_lock(),
    )
    assert protocol["researcher_authorization"]["evaluation_protocol_change_approved"] is True
    assert protocol["arm_contract"]["B"]["logical_calls"]["controller"] == 2
    assert protocol["arm_contract"]["C"]["logical_calls"]["controller"] == 2
    assert protocol["arm_contract"]["C"]["third_controller_call_allowed"] is False
    dynamic_invalid = next(
        row
        for row in protocol["fallback_matrix"]
        if row["condition"] == "C_a1_admissible_and_dynamic_output_invalid"
    )
    assert dynamic_invalid["selected_query"] == "original_question"
    assert dynamic_invalid["read_B_static_artifact"] is False
    assert dynamic_invalid["third_controller_call"] is False


def test_protocol_distinguishes_logical_and_joint_physical_calls():
    protocol = freeze.build_protocol(
        generated_at_utc="2026-09-04T00:00:00+00:00",
        cohort_lock=_cohort_lock(),
    )
    accounting = protocol["call_and_cache_accounting"]
    assert accounting["logical_requests_equal_cache_hits_plus_cache_misses"] is True
    assert accounting["logical_budget_B_C_identical"] is True
    assert accounting["joint_physical_budget_B_C_identity_required"] is False
    assert set(accounting["cache_key_forbidden_fields"]) == {"arm", "outcome", "gold"}


def test_protocol_keeps_prospective_and_gold_closed():
    protocol = freeze.build_protocol(
        generated_at_utc="2026-09-04T00:00:00+00:00",
        cohort_lock=_cohort_lock(),
    )
    assert protocol["scope"]["prospective_unlocked"] is False
    assert protocol["gold_and_seal_boundary"]["materializer_gold_access"] is False
    assert protocol["gold_and_seal_boundary"]["official_support_source"] == "UNKNOWN"
    assert protocol["researcher_authorization"]["training_authorized"] is False


def test_protocol_rejects_wrong_development_lock():
    bad = _cohort_lock()
    bad["row_count"] = 89
    with pytest.raises(ValueError, match="development90"):
        freeze.build_protocol(
            generated_at_utc="2026-09-04T00:00:00+00:00",
            cohort_lock=bad,
        )


def test_freeze_is_append_only(monkeypatch, tmp_path: Path):
    cohort = {
        "loader_version": COHORT_LOADER_VERSION,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "cohort_sha256": EXPECTED_DEVELOPMENT_SHA256,
        "rows": [{} for _ in range(90)],
    }
    monkeypatch.setattr(freeze, "load_frozen_v8_cohort", lambda **_: cohort)
    output = tmp_path / "protocol"
    freeze.freeze_protocol(
        project_root=freeze.PROJECT_ROOT,
        output_dir=output,
        generated_at_utc="2026-09-04T00:00:00+00:00",
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze.freeze_protocol(
            project_root=freeze.PROJECT_ROOT,
            output_dir=output,
            generated_at_utc="2026-09-04T00:00:00+00:00",
        )
