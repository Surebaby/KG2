from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from kgproweight.data.parsers import parse_steps
from kgproweight.kg.question_kg import make_question_kg_record, question_sha256
from kgproweight.reward.proofkg_process_v2_3 import SCORER_VERSION, build_execution_trace_v2_3, score_proofkg_v2_3
from kgproweight.reward.source_quality_gate_v1 import (
    FEATURE_NAMES, FEATURE_VERSION, SourceQualityGateV1, assign_family_splits,
    canonical_sha256, compute_gate_features, heuristic_ratio_target,
)
from kgproweight.training.reward_function import validate_source_gate_trajectory_v1
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256
from scripts.train.calibrate_source_quality_gate_v1 import (
    BANK_SCHEMA, ISOLATION_SCHEMA, ROW_SCHEMA, TEXT_CONTRACT,
    calibrate, fit_gate, validate_bank,
)
from scripts.train import calibrate_source_quality_gate_v1 as calibration_module


def _row(index=0, dataset="2wikimultihopqa"):
    word = "token" + chr(97 + index // 26) + chr(97 + index % 26)
    question, qid = f"where does alpha lead for {word}?", f"synthetic-{index}"
    confidence = (index % 11 + 1) / 12
    triples = [("Alpha", "links to", "Beta"), ("Beta", "links to", "Gamma")]
    plan = {"recognized": True, "hops": [
        {"subject": "Alpha", "output_slot": "hop_1", "pids": ["P1"], "relation_role": "bridge"},
        {"subject": "$hop_1", "output_slot": "hop_2", "pids": ["P2"], "relation_role": "answer_operand"},
    ]}
    record = make_question_kg_record(
        dataset=dataset, qid=qid, question=question, triples=triples, query_plan=plan,
        provenance={"builder_version": "synthetic-test-builder", "gold_access": False,
                    "complete_plan_execution": True, "historical_cutoff": "2020-12-09T23:59:59Z"},
    )
    record["execution"] = {"complete_plan_execution": True, "hops": [
        {"hop_index": i, "input_entities": [{"qid": f"Q{i}", "score": confidence}], "matches": [triple]}
        for i, triple in enumerate(triples, 1)
    ]}
    generation = "\n".join(
        f"[Step {i}]\nReasoning: The frozen source edge supplies this required relation.\n"
        f"Knowledge Used: [({head}, {relation}, {tail})]\nConclusion: The indicated entity is {tail}."
        for i, (head, relation, tail) in enumerate(triples, 1)
    ) + "\n[Final Answer]\nGamma"
    spec = SimpleNamespace(query=question, kg_subgraph=triples, retrieved_passages=[{"text": "Synthetic evidence only."}], metadata={"dataset": dataset, "qid": qid, "source_quality_record": record})
    steps = parse_steps(generation, known_kg=triples)
    proof = score_proofkg_v2_3(question=question, generation=generation, kg_triples=triples, execution_trace=build_execution_trace_v2_3(plan, record["execution"]), planned_hops=2)
    features = compute_gate_features(spec, steps, proof)
    validation = validate_source_gate_trajectory_v1(spec, generation)
    raw_text = [2 * (.1 + .8 * confidence) - 1] * len(steps)
    return {
        "schema_version": ROW_SCHEMA, "candidate_id": f"{dataset}::{qid}::0", "dataset": dataset,
        "qid": qid, "question": question, "question_sha256": question_sha256(question),
        "family_sha256": family_sha256(question), "family_version": FAMILY_VERSION,
        "generation": generation, "retrieved_passages": spec.retrieved_passages,
        "source_quality_record": record, "proof_result": proof, "features": features,
        "raw_graph": proof["score"], "raw_text": raw_text,
        "trajectory_valid": validation["valid"],
        "format_validation": {key: validation[key] for key in ("valid", "violations", "all_step_count", "required_steps", "contract_version")},
        "source_bindings": {"synthetic_input": "inline_test_fixture"},
    }, spec, steps


@pytest.mark.parametrize("dataset", ["hotpotqa", "2wikimultihopqa", "musique"])
def test_complete_source_hard_gate_is_dataset_agnostic_and_has_four_stable_features(dataset):
    row, spec, steps = _row(dataset=dataset)
    features = compute_gate_features(spec, steps, row["proof_result"])
    assert features["m_graph"] == 1
    assert tuple(features["values"]) == FEATURE_NAMES
    assert features["values"]["cite_match"] == 1
    assert features["values"]["cite_any"] == 1
    assert features["telemetry"]["policy_entropy_used"] is False
    spec.current_policy_entropy = 999
    assert features == compute_gate_features(spec, steps, {})


@pytest.mark.parametrize("mutation", ["qid", "hash", "cutoff", "gold", "view", "compact"])
def test_identity_provenance_and_prompt_view_mismatch_fail_closed(mutation):
    row, spec, steps = _row()
    record = spec.metadata["source_quality_record"]
    if mutation == "qid":
        record["qid"] = "wrong"
    elif mutation == "hash":
        record["question_sha256"] = "0" * 64
    elif mutation == "cutoff":
        record["provenance"].pop("historical_cutoff")
    elif mutation == "gold":
        record["provenance"]["gold_access"] = True
    elif mutation == "view":
        spec.kg_subgraph = spec.kg_subgraph[:1]
    else:
        spec.metadata["source_quality_record"] = {key: record[key] for key in ("question_key", "query_plan", "provenance", "execution")}
    assert compute_gate_features(spec, steps, row["proof_result"])["m_graph"] == 0


def test_unknown_citations_and_missing_link_confidence_are_not_silently_perfect():
    row, spec, steps = _row()
    steps[0].unknown_citation_surfaces.append("(Bad, link, Unseen)")
    spec.metadata["source_quality_record"]["execution"]["hops"][0]["input_entities"][0].pop("score")
    result = compute_gate_features(spec, steps, row["proof_result"])
    assert result["values"]["cite_match"] == pytest.approx(2 / 3)
    assert result["telemetry"]["confidence_observed"] == 1
    assert result["values"]["link_confidence"] < row["features"]["values"]["link_confidence"]
    with pytest.raises(ValueError, match="v2.3"):
        compute_gate_features(spec, steps, {"scorer_version": "proofkg-process-v2-2-frozen-1"})


@pytest.mark.parametrize(("graph", "text", "reason"), [(0, [-1], "zero_denominator"), (.02, [-.96], "both_quality_low")])
def test_degenerate_low_quality_targets_abstain(graph, text, reason):
    result = heuristic_ratio_target(graph, text, m_graph=1, trajectory_valid=True)
    assert result["target"] is None
    assert result["abstain_reason"] == reason


def test_heuristic_ratio_and_noneligible_target_contract():
    assert heuristic_ratio_target(.85, [0, 0], m_graph=1, trajectory_valid=True)["target"] == pytest.approx(2 / 3)
    assert heuristic_ratio_target(-1, [], m_graph=1, trajectory_valid=False)["abstain_reason"] == "invalid_trajectory"
    assert heuristic_ratio_target(0, [], m_graph=0, trajectory_valid=True)["abstain_reason"] == "graph_ineligible"
    with pytest.raises(ValueError, match="finite"):
        heuristic_ratio_target(.85, [float("nan")], m_graph=1, trajectory_valid=True)


def _fit_rows(n=80):
    rows = []
    for i in range(n):
        row, _spec, _steps = _row(i)
        row["quality"] = heuristic_ratio_target(row["raw_graph"], row["raw_text"], m_graph=1, trajectory_valid=True)
        row["trajectory_valid"] = True
        rows.append(row)
    return rows


@pytest.fixture(scope="module")
def fitted():
    rows = _fit_rows()
    artifact, report, assignments = fit_gate(rows, {"bank_source": "synthetic"}, experiment_id="UNIT-SYNTHETIC-FIT", epochs=400)
    return rows, artifact, report, assignments


def test_family_split_has_exact_quota_and_no_candidate_or_order_leakage(fitted):
    rows, _artifact, _report, assignments = fitted
    families = [row["family_sha256"] for row in rows]
    splits = assign_family_splits(families + families)
    assert splits == assign_family_splits(reversed(families))
    assert [sum(value == name for value in splits.values()) for name in ("train", "calibration", "confirmation")] == [48, 16, 16]
    assert all(item["split"] == splits[item["family_sha256"]] for item in assignments)


def test_synthetic_fit_never_produces_production_clearance(fitted, tmp_path):
    rows, artifact, report, _assignments = fitted
    assert artifact["training_clearance"] is False
    assert report["status"] == "SYNTHETIC_FIT_NOT_TRAINING_CLEARANCE"
    assert report["metrics"]["calibration"]["brier"] < report["metrics"]["calibration"]["constant_brier"]
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="synthetic"):
        SourceQualityGateV1.load(path)
    with pytest.raises(ValueError, match="clearance"):
        SourceQualityGateV1.load(path, allow_synthetic=True)
    loaded = SourceQualityGateV1.load(path, allow_synthetic=True, allow_unvalidated=True)
    assert 0 < loaded.predict(rows[0]["features"]) < 1
    masked = deepcopy(rows[0]["features"])
    masked["m_graph"] = 0
    assert loaded.predict(masked) == 0
    assert loaded.normalization["graph_scale"] >= .1


def test_loader_refuses_tampering_legacy_checkpoints_and_nonfinite_parameters(fitted, tmp_path):
    _rows, original, _report, _assignments = fitted
    edited = deepcopy(original)
    edited["weights"][0] += 1
    with pytest.raises(ValueError, match="hash"):
        SourceQualityGateV1(edited, allow_synthetic=True, allow_unvalidated=True)
    legacy = tmp_path / "old.pt"
    legacy.write_bytes(b"\x80\x04legacy-pytorch-pickle")
    with pytest.raises(ValueError, match="legacy"):
        SourceQualityGateV1.load(legacy)
    edited = deepcopy(original)
    edited["normalization"]["graph_scale"] = 0
    edited.pop("payload_sha256")
    edited["payload_sha256"] = canonical_sha256(edited)
    with pytest.raises(ValueError, match="scale"):
        SourceQualityGateV1(edited, allow_synthetic=True, allow_unvalidated=True)


def test_heldout_changes_cannot_change_training_standardization_scaling_or_fixed_alpha(fitted):
    rows, original, _report, _assignments = fitted
    changed = deepcopy(rows)
    splits = assign_family_splits(row["family_sha256"] for row in rows)
    for row in changed:
        if splits[row["family_sha256"]] != "train":
            row["features"]["values"]["link_confidence"] = 1.0
            row["raw_text"] = [-1.0, -1.0]
            row["raw_graph"] = 0
            row["quality"]["target"] = 0.0
    artifact, _report, _assignments = fit_gate(changed, {"bank_source": "synthetic"}, experiment_id="UNIT-HELDOUT-MUTATION", epochs=400)
    assert artifact["weights"] == original["weights"]
    assert artifact["bias"] == original["bias"]
    assert artifact["feature_standardization"] == original["feature_standardization"]
    assert artifact["normalization"] == original["normalization"]


def _write_bank(directory, n=20):
    directory.mkdir()
    rows = [_row(i)[0] for i in range(n)]
    bank = directory / "rows.jsonl"
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    ledger = directory / "ledger.jsonl"
    ledger.write_text(json.dumps({"qid": "protected-unrelated", "question": "why does unrelated protected evidence disagree?"}) + "\n")
    bind = lambda path: {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    sources = {}
    for name in ("source_evidence", "source_rows", "policy"):
        path = directory / f"{name}.json"
        path.write_text(json.dumps({"synthetic_only": True, "name": name}))
        sources[name] = bind(path)
    isolation = directory / "isolation.json"
    isolation.write_text(json.dumps({"schema_version": ISOLATION_SCHEMA, "status": "PASS", "bank_sha256": bind(bank)["sha256"], "family_version": FAMILY_VERSION, "protected_ledger_binding": bind(ledger), "overlap_counts": {"qid": 0, "question_sha256": 0, "family_sha256": 0}}))
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": BANK_SCHEMA, "experiment_id": "UNIT-SYNTHETIC-BANK", "status": "SYNTHETIC_TEST_ONLY", "bank_source": "synthetic", "bank": bind(bank), "feature_version": FEATURE_VERSION, "graph_scorer_version": SCORER_VERSION, "text_score_contract": TEXT_CONTRACT, "gold_access_for_gate_target": False, "source_bindings": sources, "isolation_proof": bind(isolation)}))
    return manifest, isolation


def test_bank_validation_recomputes_proof_features_and_isolation_before_fit(tmp_path):
    manifest, isolation = _write_bank(tmp_path / "bank")
    rows, binding = validate_bank(manifest, isolation, synthetic_test_only=True)
    assert len(rows) == 20
    assert binding["recomputed_overlap_counts"] == {"qid": 0, "question_sha256": 0, "family_sha256": 0}
    with pytest.raises(ValueError, match="synthetic"):
        validate_bank(manifest, isolation)
    # Updating just a bank file without updating its release bindings is refused.
    bank = manifest.parent / "rows.jsonl"
    bank.write_text(bank.read_text().replace('"raw_graph": 0.85', '"raw_graph": 0.0', 1))
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_bank(manifest, isolation, synthetic_test_only=True)


def test_synthetic_end_to_end_calibration_is_append_only_and_diagnostic(tmp_path):
    manifest, isolation = _write_bank(tmp_path / "bank")
    output = tmp_path / "synthetic_fit"
    result = calibrate(manifest, isolation, output, experiment_id="UNIT-END-TO-END-SYNTHETIC", epochs=80, synthetic_test_only=True)
    assert result["status"] == "SYNTHETIC_FIT_NOT_TRAINING_CLEARANCE"
    assert result["training_clearance"] is False
    with pytest.raises(ValueError, match="synthetic"):
        SourceQualityGateV1.load(output / "gate.json")
    previous = (output / "gate.json").read_bytes()
    with pytest.raises(FileExistsError):
        calibrate(manifest, isolation, output, experiment_id="UNIT-RETRY", synthetic_test_only=True)
    assert (output / "gate.json").read_bytes() == previous


def _refresh_bindings(manifest_path, isolation_path):
    """Rebind synthetic corruptions so the semantic audit, not only SHA, runs."""
    manifest = json.loads(manifest_path.read_text())
    isolation = json.loads(isolation_path.read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["bank"]["sha256"] = sha(manifest_path.parent / manifest["bank"]["path"])
    isolation["bank_sha256"] = manifest["bank"]["sha256"]
    isolation["protected_ledger_binding"]["sha256"] = sha(isolation_path.parent / isolation["protected_ledger_binding"]["path"])
    isolation_path.write_text(json.dumps(isolation))
    manifest["isolation_proof"]["sha256"] = sha(isolation_path)
    manifest_path.write_text(json.dumps(manifest))


def test_isolation_claim_cannot_override_actual_protected_ledger_overlap(tmp_path):
    manifest, isolation = _write_bank(tmp_path / "bank")
    row = json.loads((manifest.parent / "rows.jsonl").read_text().splitlines()[0])
    (manifest.parent / "ledger.jsonl").write_text(json.dumps({"qid": row["qid"], "question": row["question"]}) + "\n")
    _refresh_bindings(manifest, isolation)
    with pytest.raises(ValueError, match="actual protected"):
        validate_bank(manifest, isolation, synthetic_test_only=True)


@pytest.mark.parametrize("field", ["raw_graph", "features", "family_sha256", "gold_target"])
def test_frozen_bank_claim_cannot_override_recomputed_identity_proof_or_features(tmp_path, field):
    manifest, isolation = _write_bank(tmp_path / "bank")
    path = manifest.parent / "rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if field == "raw_graph":
        rows[0][field] = .0
    elif field == "features":
        rows[0][field]["values"]["link_confidence"] = 1
    elif field == "family_sha256":
        rows[0][field] = "0" * 64
    else:
        rows[0][field] = 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _refresh_bindings(manifest, isolation)
    with pytest.raises(ValueError):
        validate_bank(manifest, isolation, synthetic_test_only=True)


def _ordinary_row(index=90):
    row, spec, _steps = _row(index, dataset="hotpotqa")
    record = spec.metadata["source_quality_record"]
    record["kg_subgraph"], spec.kg_subgraph = [], []
    record["execution"] = {}
    record["query_plan"] = {}
    record["provenance"]["complete_plan_execution"] = False
    generation = "\n".join(
        f"[Step {i}]\nReasoning: This conclusion follows from the frozen passages provided.\n"
        "Knowledge Used: []\nConclusion: The passage supplies the required fact."
        for i in range(1, 4)
    ) + "\n[Final Answer]\nGamma"
    validation = validate_source_gate_trajectory_v1(spec, generation)
    assert validation["valid"] and validation["source_features"]["m_graph"] == 0
    row.update({"generation": generation, "source_quality_record": record,
                "proof_result": {"trajectory_valid": False, "score": 0, "scorer_version": SCORER_VERSION},
                "raw_graph": 0.0, "raw_text": [.99] * 3,
                "features": validation["source_features"], "trajectory_valid": True,
                "format_validation": {key: validation[key] for key in ("valid", "violations", "all_step_count", "required_steps", "contract_version")}})
    return row


def test_valid_passage_only_control_uses_shared_format_and_enters_train_text_statistics(tmp_path):
    manifest, isolation = _write_bank(tmp_path / "bank", n=80)
    rows = [json.loads(line) for line in (manifest.parent / "rows.jsonl").read_text().splitlines()]
    controls = [_ordinary_row(i) for i in range(80, 100)]
    rows += controls
    (manifest.parent / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    _refresh_bindings(manifest, isolation)
    validated, binding = validate_bank(manifest, isolation, synthetic_test_only=True)
    control_rows = [row for row in validated if not row["features"]["m_graph"]]
    assert len(control_rows) == 20
    assert all(row["trajectory_valid"] and row["quality"]["abstain_reason"] == "graph_ineligible" for row in control_rows)
    artifact, report, assignments = fit_gate(validated, binding, experiment_id="UNIT-PASSAGE-CONTROLS", epochs=100)
    splits = assign_family_splits(row["family_sha256"] for row in validated)
    train = [row for row in validated if splits[row["family_sha256"]] == "train"]
    expected = sum(sum(row["raw_text"]) / len(row["raw_text"]) for row in train) / len(train)
    assert artifact["normalization"]["text_center"] == pytest.approx(expected)
    assert any(not row["features"]["m_graph"] for row in train)
    assert report["passage_only_fail_closed"] is True
    assert all(row["alpha"] == 0 for row in assignments if not row["m_graph"])


def test_caller_cannot_claim_invalid_control_is_valid(tmp_path):
    manifest, isolation = _write_bank(tmp_path / "bank")
    row = _ordinary_row()
    row["generation"] = row["generation"].replace("[Step 3]", "[Step 8]")
    path = manifest.parent / "rows.jsonl"
    path.write_text(path.read_text() + json.dumps(row) + "\n")
    _refresh_bindings(manifest, isolation)
    with pytest.raises(ValueError, match="shared PPO format"):
        validate_bank(manifest, isolation, synthetic_test_only=True)


def test_gitless_release_finishes_calibration_with_actual_code_hashes(tmp_path, monkeypatch):
    manifest, isolation = _write_bank(tmp_path / "bank")
    gitless_root = tmp_path / "portable_release"
    gitless_root.mkdir()
    # A portable release contains actual source files but deliberately no .git.
    source_paths = (
        "kgproweight/reward/source_quality_gate_v1.py",
        "kgproweight/reward/proofkg_process_v2_3.py",
        "kgproweight/training/reward_function.py",
        "scripts/prepare/freeze_qpeg_v1_protocol.py",
    )
    for relative in source_paths:
        target = gitless_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((calibration_module.ROOT / relative).read_bytes())
    monkeypatch.setattr(calibration_module, "ROOT", gitless_root)
    output = tmp_path / "portable_fit"
    result = calibrate(
        manifest, isolation, output, experiment_id="UNIT-GITLESS-PORTABLE-FIT",
        epochs=80, synthetic_test_only=True,
    )
    stored = json.loads((output / "manifest.json").read_text())
    assert stored["git_head"] is None
    assert result["status"] == stored["status"] == "SYNTHETIC_FIT_NOT_TRAINING_CLEARANCE"
    assert stored["training_clearance"] is False
    assert not (output / "FAILED_CALIBRATION.json").exists()
    assert len(stored["code_bindings"]) == len(source_paths) + 1
    for binding in stored["code_bindings"].values():
        assert calibration_module._identity(calibration_module.Path(binding["path"])) == binding
    for filename, binding in stored["outputs"].items():
        assert calibration_module._identity(output / filename) == binding
    with pytest.raises(ValueError, match="synthetic"):
        SourceQualityGateV1.load(output / "gate.json")
