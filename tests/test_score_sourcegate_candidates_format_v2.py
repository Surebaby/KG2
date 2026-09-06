"""Synthetic CPU coverage for immutable-parent format-v2 scoring and guards."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kgproweight.kg.question_kg import make_question_kg_record
from kgproweight.training.reward_function import validate_source_gate_source_integrity
from scripts.prepare import score_sourcegate_candidates_format_v2 as revision

bank = revision.parent
TRACE = """[Step 1]
Reasoning: Follow the first documented edge.
Knowledge Used: [(Alpha, links to, Beta)]
Conclusion: Beta is reached.
[Step 2]
Reasoning: Follow the second documented edge.
Knowledge Used: [(Beta, links to, Gamma)]
Conclusion: Gamma is reached.
[Final Answer]
Gamma"""


class FakeScorer:
    max_length = 4096

    def __init__(self, value=0.25):
        self.calls = []
        self.value = value

    def tokenizer(self, text, **kwargs):
        return {"input_ids": [0] * len(text.split())}

    def score_step(self, prompt, text):
        self.calls.append((prompt, text))
        return self.value


@pytest.fixture
def row():
    identity = {"dataset": "2wikimultihopqa", "qid": "synthetic-train", "question": "Where do two documented synthetic links lead?"}
    triples = [["Alpha", "links to", "Beta"], ["Beta", "links to", "Gamma"]]
    plan = {"recognized": True, "hops": [
        {"subject": "Alpha", "pids": ["P1"], "output_slot": "hop_1"},
        {"subject": "$hop_1", "pids": ["P2"], "output_slot": "hop_2"},
    ]}
    record = make_question_kg_record(**identity, triples=triples, query_plan=plan, provenance={
        "builder_version": "synthetic-unit-test-only", "gold_access": False,
        "complete_plan_execution": True, "historical_cutoff": "2020-12-09T23:59:59Z",
    })
    record["runtime_error"] = None
    record["execution"] = {"complete_plan_execution": True, "hops": [
        {"hop_index": i + 1, "input_entities": [{"qid": f"Q{i + 1}", "score": 1.0}], "matches": [triple]}
        for i, triple in enumerate(triples)
    ]}
    passages = [{"id": str(i), "contents": f"Frozen synthetic passage {i}"} for i in range(10)]
    result = {**identity, "question_key": bank.key(identity), "kg_subgraph": triples,
              "retrieved_passages": passages, "source_quality_record": record,
              "fullsource_record": record, "source_bindings": {},
              "spec": {"query": identity["question"], "retrieved_passages": passages,
                       "kg_subgraph": triples, "metadata": {"dataset": identity["dataset"],
                       "qid": identity["qid"], "source_quality_record": record}}}
    result["input_sha256"] = bank.input_hash(result)
    return result


def test_empty_final_is_retained_invalid_without_scorer_call(row):
    prediction = {"candidate_id": "synthetic::k0", "generation": TRACE.rsplit("Gamma", 1)[0]}
    historical = bank.score_candidate(row, prediction, FakeScorer())
    scorer = FakeScorer()
    repaired = revision.score_candidate_v2(row, prediction, scorer)
    assert historical["trajectory_valid"] is True  # Reproduces the actual v1 boundary bug.
    assert repaired["trajectory_valid"] is False
    assert "final_answer_empty_or_decoration_only" in repaired["format_validation"]["violations"]
    assert scorer.calls == []
    assert repaired["raw_text"] == [] and repaired["raw_text_step_mean"] is None
    assert repaired["generation"] == prediction["generation"]


def test_valid_candidate_preserves_v1_scores_inputs_features_and_generation(row):
    before = deepcopy(row)
    prediction = {"candidate_id": "synthetic::k0", "generation": TRACE}
    old_scorer, new_scorer = FakeScorer(), FakeScorer()
    old = bank.score_candidate(row, prediction, old_scorer)
    new = revision.score_candidate_v2(row, prediction, new_scorer)
    assert old["trajectory_valid"] is new["trajectory_valid"] is True
    assert old_scorer.calls == new_scorer.calls and len(new_scorer.calls) == 2
    assert old["format_validation"]["contract_version"].endswith("format-v1")
    assert new["format_validation"]["contract_version"].endswith("format-v2")
    old["format_validation"]["contract_version"] = new["format_validation"]["contract_version"]
    assert old == new and row == before


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1.01, 1.01])
def test_nonfinite_or_out_of_range_real_scorer_values_rejected(row, value):
    with pytest.raises(ValueError, match="invalid original BF16 ReaRAG score"):
        revision.score_candidate_v2(row, {"candidate_id": "synthetic::k0", "generation": TRACE}, FakeScorer(value))


@pytest.fixture
def parents(tmp_path, monkeypatch, row):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(bank, "ROOT", root)
    monkeypatch.setattr(revision, "EXTRA_CODE", [])
    source_files = ["kgproweight/training/reward_function.py", "kgproweight/reward/text_reward_model.py"]
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    bindings = {}
    saved_sources = {}
    for name in source_files:
        current = root / name
        saved = snapshot / name
        current.parent.mkdir(parents=True, exist_ok=True)
        saved.parent.mkdir(parents=True, exist_ok=True)
        current.write_text("# immutable synthetic parent source\n")
        saved.write_bytes(current.read_bytes())
        bindings["code:" + name] = bank.identity(current)
        saved_sources[name] = bank.identity(saved)
    policy = root / "adapter.safetensors"
    policy.write_bytes(b"unit-test-policy")
    ledger = root / "ledger.jsonl"
    bank.write_rows(ledger, [{"dataset": "hotpotqa", "qid": "reserve", "question": "An unrelated protected example?"}])
    bindings.update(policy=bank.identity(policy), protected_ledger=bank.identity(ledger))
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    bank.write_rows(prepared / "inputs.jsonl", [row])
    report = {"schema_version": bank.PREPARE_VERSION, "source_bindings": bindings,
              "qid_order": [row["question_key"]], "seed": 42, "generation": {"candidates_per_question": 2},
              "scoring": {"format_contract": "source-gate-runtime-v2-format-v1", "rearag_max_tokens": 4096},
              "base_model": {"path": "models/base", "files": {}},
              "rearag_model": {"path": "models/rearag", "files": {}},
              "policy_tokenizer": {"path": "models/tokenizer", "files": {}}}
    bank.write_json(prepared / "score_config.json", report["scoring"])
    bindings["score_config"] = bank.identity(prepared / "score_config.json")
    bank.finish(prepared, report, ["inputs.jsonl", "score_config.json"])
    bank_sha = bank.file_sha(prepared / "manifest.json")
    bank.write_json(snapshot / "manifest.json", {"parent_bank_manifest_sha256": bank_sha, "source_files": saved_sources})
    predictions = [{"candidate_id": f"{row['question_key']}::k{index}", "dataset": row["dataset"],
                    "qid": row["qid"], "candidate_index": index,
                    "seed": bank.candidate_seed(42, row["question_key"], index),
                    "input_sha256": row["input_sha256"], "bank_manifest_sha256": bank_sha,
                    "generation_contract_sha256": bank.digest(report["generation"]),
                    "policy_sha256": bindings["policy"]["sha256"],
                    "base_model_identity_sha256": bank.digest(report["base_model"]),
                    "generation": TRACE} for index in range(2)]
    generated = tmp_path / "generated"
    generated.mkdir()
    bank.write_rows(generated / "generations.jsonl", predictions)
    bank.finish(generated, {"schema_version": bank.GENERATION_VERSION, "bank_manifest_sha256": bank_sha,
                           "n_candidates": 2}, ["generations.jsonl"])
    return prepared, generated, snapshot


def test_parent_release_verified_and_only_allowed_changes_recorded(parents):
    prepared, generated, snapshot = parents
    verified = revision.verify_parents(*parents)
    assert len(verified[1]) == 1 and len(verified[2]) == 2 and verified[4] == {}
    changed = bank.ROOT / "kgproweight/training/reward_function.py"
    changed.write_text("# authorized synthetic format repair\n")
    verified = revision.verify_parents(*parents)
    assert set(verified[4]) == {"kgproweight/training/reward_function.py"}
    assert (snapshot / "kgproweight/training/reward_function.py").read_text().startswith("# immutable")


@pytest.mark.parametrize("target", ["inputs", "generations", "snapshot", "unexpected_code", "manifest_parent"])
def test_parent_tampering_is_rejected_before_model_loading(parents, target):
    prepared, generated, snapshot = parents
    if target == "inputs":
        path = prepared / "inputs.jsonl"
    elif target == "generations":
        path = generated / "generations.jsonl"
    elif target == "snapshot":
        path = snapshot / "kgproweight/training/reward_function.py"
    elif target == "unexpected_code":
        path = bank.ROOT / "kgproweight/reward/text_reward_model.py"
    else:
        path = generated / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["bank_manifest_sha256"] = "0" * 64
        path.write_text(json.dumps(manifest))
    if target != "manifest_parent":
        path.write_text(path.read_text() + "\n# altered\n")
    with pytest.raises(ValueError, match="mismatch|mutation"):
        revision.verify_parents(*parents)


def test_completed_scoring_manifest_cannot_authorize_ppo(parents, tmp_path, monkeypatch):
    from kgproweight.reward.text_reward_model import RearagPromptScorer
    from scripts.train import calibrate_source_quality_gate_v1 as calibration

    fake_cuda = SimpleNamespace(is_bf16_supported=lambda: True, get_device_name=lambda index: "SYNTHETIC-CPU-MOCK",
        reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 0,
        max_memory_reserved=lambda: 0, empty_cache=lambda: None)
    monkeypatch.setattr(bank, "require_cuda", lambda device: SimpleNamespace(cuda=fake_cuda, __version__="test-only"))
    monkeypatch.setattr(RearagPromptScorer, "from_pretrained", lambda *args, **kwargs: FakeScorer())
    calibrated = []

    def fake_calibrate(manifest_path, isolation_path, output, **kwargs):
        manifest = json.loads(manifest_path.read_text())
        assert manifest["source_integrity_clearance"] is False
        assert manifest["source_integrity_status"] == "LABEL_PROJECTION_REPAIR_PENDING"
        assert manifest["format_contract_version"].endswith("format-v2")
        with pytest.raises(ValueError, match="source_integrity_clearance=true"):
            validate_source_gate_source_integrity(SimpleNamespace(artifact=manifest), "v2")
        calibrated.append(manifest)
        return {"status": "SYNTHETIC_FIT_ONLY", "training_clearance": True}

    monkeypatch.setattr(calibration, "calibrate", fake_calibrate)
    output = tmp_path / "scored"
    revision.score(bank_dir=parents[0], generation_dir=parents[1], snapshot_dir=parents[2],
                   output_dir=output, calibration_dir=tmp_path / "fit", experiment_id="SYNTHETIC-FORMAT-V2-SCORE")
    assert len(calibrated) == 1
    status = json.loads((output / "status.json").read_text())
    assert status["source_integrity_clearance"] is False and status["ppo_started"] is False
    assert status["heuristic_calibration_clearance"] is True  # Fit quality cannot override the source guard.
    scored = bank.read_rows(output / "candidates.scored.jsonl")
    assert len(scored) == 2 and all(item["generation"] == TRACE for item in scored)
