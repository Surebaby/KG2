"""CPU-only source-credit fitting and immutable parent-view regression tests."""
from copy import deepcopy
import json

import numpy as np
import pytest

from kgproweight.reward.source_credit_gate_v1 import FrozenSourceCreditMask, SourceCreditGateV1
from kgproweight.reward.source_quality_gate_v1 import (
    SourceQualityGateV1, assign_family_splits, canonical_sha256,
)
from scripts.train import calibrate_source_credit_gate_v1 as credit
from scripts.train import calibrate_source_quality_gate_v1 as legacy
from tests.test_source_quality_gate_v1 import _fit_rows, _write_bank


def _mask(tmp_path, rows):
    """Construct a synthetic in-memory mask; production load verification is separate."""
    mask = FrozenSourceCreditMask()
    mask.manifest_path = tmp_path / "synthetic-mask.json"
    mask.manifest_path.write_text('{"synthetic_unit_fixture":true}\n')
    mask.manifest_sha256 = legacy._identity(mask.manifest_path)["sha256"]
    mask.payload_sha256 = canonical_sha256({"synthetic_unit_fixture": True})
    mask._entries = {}
    for index, row in enumerate(rows):
        status = "FAIL" if index % 4 == 0 else "UNVERIFIED" if index % 4 == 1 else "PASS"
        record = row["source_quality_record"]
        mask._entries[f"{row['dataset']}::{row['qid']}"] = {
            "status": status, "clearance": status == "PASS", "original_m_graph": 1,
            "question_sha256": row["question_sha256"],
            "record_sha256": canonical_sha256(record),
            "graph_sha256": canonical_sha256(record["kg_subgraph"]),
        }
    return mask


@pytest.fixture
def rows():
    result = _fit_rows(80)
    for row in result:
        row["format_validation"]["contract_version"] = "source-gate-runtime-v2-format-v2"
    return result


@pytest.fixture
def binding():
    return {"bank_source": "synthetic", "format_contract_version": "source-gate-runtime-v2-format-v2",
            "source_integrity_clearance": False, "source_integrity_status": "LABEL_PROJECTION_REPAIR_PENDING"}


def test_mask_keeps_every_candidate_and_original_text_without_promoting_graphs(tmp_path, rows):
    snapshot = deepcopy(rows)
    mask = _mask(tmp_path, rows)
    masked = credit.apply_credit_view(rows, mask)
    assert rows == snapshot
    assert len(masked) == len(rows) == 80
    assert [row["candidate_id"] for row in masked] == [row["candidate_id"] for row in rows]
    for old, new in zip(rows, masked):
        assert new["schema_version"] == credit.ROW_SCHEMA
        for key in ("generation", "retrieved_passages", "source_quality_record", "raw_graph", "raw_text", "trajectory_valid"):
            assert old[key] == new[key]
        assert old["features"]["values"] == new["features"]["values"]
        assert new["features"]["m_graph"] == int(new["source_credit"]["status"] == "PASS")
        if not new["features"]["m_graph"]:
            assert new["quality"]["target"] is None
            assert new["quality"]["abstain_reason"] == "graph_ineligible"


def test_missing_identity_never_obtains_graph_credit(tmp_path, rows):
    mask = _mask(tmp_path, rows)
    mask._entries.pop(f"{rows[0]['dataset']}::{rows[0]['qid']}")
    result = credit.apply_credit_view(rows, mask)
    assert result[0]["features"]["m_graph"] == 0
    assert result[0]["source_credit"]["status"] == "MISSING"


def test_new_fit_keeps_splits_fit_parameters_and_text_population_excludes_masked_graphs(tmp_path, rows, binding):
    # Vary graph scores to make population exclusion observable in the unit fit.
    for index, row in enumerate(rows):
        row["raw_graph"] = 0.1 + 0.7 * (index % 7) / 6
    mask = _mask(tmp_path, rows)
    masked = credit.apply_credit_view(rows, mask)
    artifact, report, assignments = credit.fit_source_credit(masked, binding, mask, experiment_id="UNIT-CREDIT-FIT")
    splits = assign_family_splits(row["family_sha256"] for row in rows)
    train = [row for row in masked if splits[row["family_sha256"]] == "train"]
    graph_train = [row for row in train if row["features"]["m_graph"]]
    assert artifact["normalization"]["graph_center"] == pytest.approx(np.mean([row["raw_graph"] for row in graph_train]))
    assert artifact["normalization"]["text_center"] == pytest.approx(np.mean([np.mean(row["raw_text"]) for row in train]))
    assert artifact["normalization"]["graph_fit_population"].startswith("valid_credit_eligible")
    assert artifact["fit"] == {"optimizer": "numpy_full_batch_logistic_soft_bce", "seed": 42, "epochs": 800,
        "learning_rate": 0.05, "l2": 0.001, "model_selection": "fixed_final_no_confirmation_selection"}
    assert all(item["split"] == splits[item["family_sha256"]] for item in assignments)
    assert len(assignments) == 80
    assert artifact["source_credit_clearance"] is True
    assert artifact["source_integrity_clearance"] is False
    assert artifact["reader_inputs_repaired"] is False
    assert report["ppo_launch_clearance"] is False
    assert artifact["training_clearance"] is False  # Synthetic fit remains diagnostic.
    loader = SourceCreditGateV1(artifact, mask=mask, allow_synthetic=True, allow_unvalidated=True)
    assert all(loader.predict(row["features"]) == 0 for row in masked if not row["features"]["m_graph"])
    with pytest.raises(ValueError, match="legacy/unknown"):
        SourceQualityGateV1(artifact, allow_synthetic=True, allow_unvalidated=True)
    with pytest.raises(ValueError, match="processed by its frozen mask"):
        loader.predict(rows[0]["features"])


def test_mask_population_summary_counts_all_statuses_and_text(tmp_path, rows, binding):
    mask = _mask(tmp_path, rows)
    masked = credit.apply_credit_view(rows, mask)
    _artifact, _report, assignments = credit.fit_source_credit(masked, binding, mask, experiment_id="UNIT-CREDIT-SUMMARY")
    summary = credit.summarize_mask(rows, masked, assignments)
    assert summary["candidate_count"] == summary["question_count"] == 80
    assert summary["candidate_status_counts"] == {"FAIL": 20, "UNVERIFIED": 20, "PASS": 40}
    assert summary["parent_graph_questions"] == 80 and summary["graph_credit_questions"] == 40
    assert summary["all_valid_text_scores_retained"] is True
    assert summary["reader_inputs_and_generations_unchanged"] is True
    dataset = summary["by_dataset"]["2wikimultihopqa"]
    assert dataset["valid_text_steps_retained"] == 160
    assert dataset["graph_credit_removed"] == 40


def _parent_bank(tmp_path):
    manifest_path, isolation_path = _write_bank(tmp_path / "parent", n=80)
    manifest = json.loads(manifest_path.read_text())
    manifest.update(format_contract_version="source-gate-runtime-v2-format-v2",
        source_integrity_clearance=False, source_integrity_status="LABEL_PROJECTION_REPAIR_PENDING")
    bank_path = manifest_path.parent / "rows.jsonl"
    rows = [json.loads(line) for line in bank_path.read_text().splitlines()]
    for row in rows:
        row["format_validation"]["contract_version"] = "source-gate-runtime-v2-format-v2"
    bank_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest["bank"]["sha256"] = legacy._identity(bank_path)["sha256"]
    isolation = json.loads(isolation_path.read_text())
    isolation["bank_sha256"] = manifest["bank"]["sha256"]
    isolation_path.write_text(json.dumps(isolation))
    manifest["isolation_proof"]["sha256"] = legacy._identity(isolation_path)["sha256"]
    archived = manifest_path.parent / "runtime_code" / "frozen_backend.py"
    archived.parent.mkdir(parents=True)
    archived.write_text("# original frozen implementation\n")
    live = tmp_path / "live_backend.py"
    live.write_text("# subsequently authorized successor implementation\n")
    manifest["source_bindings"]["code:frozen_backend.py"] = {"path": str(live), "sha256": legacy._identity(archived)["sha256"]}
    manifest["outputs"] = {"rows.jsonl": {"path": "rows.jsonl", "sha256": manifest["bank"]["sha256"]}}
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, isolation_path, rows


def test_parent_view_relocates_exact_snapshot_then_validates_original_rows(tmp_path):
    parent, isolation, _rows = _parent_bank(tmp_path)
    original_bytes = parent.read_bytes()
    view, proof, relocation = credit.create_parent_view(parent, tmp_path / "view")
    validated, _binding = legacy.validate_bank(view, proof, synthetic_test_only=True)
    assert len(validated) == 80
    assert proof == isolation.resolve()
    assert parent.read_bytes() == original_bytes
    assert relocation["bank_bytes_unchanged"] is True
    assert len(relocation["source_code_relocations"]) == 1
    assert "runtime_code/frozen_backend.py" in relocation["source_code_relocations"][0]["archived_path"]
    assert all(row["schema_version"] == legacy.ROW_SCHEMA for row in validated)


@pytest.mark.parametrize("target", ["snapshot", "bank", "parent_failed"])
def test_tampered_parent_or_snapshot_is_rejected(tmp_path, target):
    parent, _isolation, _rows = _parent_bank(tmp_path)
    if target == "snapshot":
        (parent.parent / "runtime_code/frozen_backend.py").write_text("# changed archived bytes\n")
    elif target == "bank":
        (parent.parent / "rows.jsonl").write_text("{}\n")
    else:
        (parent.parent / "FAILED.json").write_text("{}\n")
    with pytest.raises(ValueError, match="hash mismatch|failed parent"):
        credit.create_parent_view(parent, tmp_path / "view")


def test_calibration_is_append_only_and_preserves_parent_bytes(tmp_path, monkeypatch):
    parent, _isolation, rows = _parent_bank(tmp_path)
    original = {path: path.read_bytes() for path in parent.parent.rglob("*") if path.is_file()}
    mask = _mask(tmp_path, rows)
    monkeypatch.setattr(FrozenSourceCreditMask, "load", lambda path: mask)
    output = tmp_path / "credit-fit"
    result = credit.calibrate(parent, mask.manifest_path, output, experiment_id="UNIT-CREDIT-CALIBRATE",
                              synthetic_test_only=True)
    assert result["source_integrity_clearance"] is False
    assert result["source_credit_clearance"] is True
    assert result["training_clearance"] is result["ppo_launch_clearance"] is False
    assert all(path.read_bytes() == contents for path, contents in original.items())
    masked = [json.loads(line) for line in (output / "candidates.credit_masked.jsonl").read_text().splitlines()]
    assert len(masked) == 80 and all(row["schema_version"] == credit.ROW_SCHEMA for row in masked)
    parent_validation = json.loads((output / "parent_validation.json").read_text())
    assert parent_validation["masked_rows_were_not_validated_as_legacy_rows"] is True
    before = (output / "gate.json").read_bytes()
    with pytest.raises(FileExistsError):
        credit.calibrate(parent, mask.manifest_path, output, experiment_id="UNIT-NO-OVERWRITE", synthetic_test_only=True)
    assert (output / "gate.json").read_bytes() == before
