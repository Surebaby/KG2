"""Synthetic CPU checks: no fresh examples, labels, CUDA, or gate fitting."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from scripts.prepare import score_source_credit_v2_fresh_confirmation_v1 as score
from kgproweight.reward.source_quality_gate_v1 import compute_gate_features
from kgproweight.reward.proofkg_process_v2_3 import SCORER_VERSION
from kgproweight.reward.source_trajectory_features_v2 import compute_gate_features_v2
from kgproweight.reward.source_reward_normalization_v2 import fit_text_normalization_v2, normalize_text_steps_v2
from tests.test_source_trajectory_features_v2 import _fixture


class Backend:
    max_length = 4096

    def __init__(self, value=.4):
        self.value, self.calls = value, []

    def tokenizer(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split())))}

    def score_step(self, prompt, text):
        self.calls.append((prompt, text))
        return self.value


class Gate:
    def __init__(self, *, six=False, source_pass=True):
        self.six, self.source_pass, self.predictions = six, source_pass, 0
        normalizer = fit_text_normalization_v2([{
            "dataset": "synthetic", "qid": "normalization-only", "candidate_id": "n0", "split": "train",
            "trajectory_valid": True, "raw_text": [-.4, .2, .8],
        }])
        self.normalization = {"text_v2": normalizer, "fixed_alpha": .4, "graph_center": .4, "graph_scale": .2}

    def compute_features(self, spec, steps, proof):
        return (compute_gate_features_v2 if self.six else compute_gate_features)(spec, steps, proof)

    def mask_features(self, spec, features):
        result = deepcopy(features)
        result["source_credit_mask"] = {"parent_m_graph": result["m_graph"], "status": "PASS" if self.source_pass else "FAIL"}
        result["m_graph"] *= int(self.source_pass)
        return result

    def predict(self, features):
        self.predictions += 1
        return (.6 if self.six else .5) if features["m_graph"] else 0.


def fixture(*, graph=True, steps=2, source_pass=True):
    spec, _, _ = _fixture()
    if not graph:
        spec.kg_subgraph = []
        spec.metadata["source_quality_record"] = {}
    spec.retrieved_passages = [{"id": i, "title": "Synthetic", "text": f"Visible passage {i}."} for i in range(1, 11)]
    record = spec.metadata["source_quality_record"]
    row = {"question": spec.query, "dataset": spec.metadata["dataset"], "qid": spec.metadata["qid"],
           "kg_subgraph": deepcopy(spec.kg_subgraph), "source_quality_record": deepcopy(record),
           "retrieved_passages": deepcopy(spec.retrieved_passages), "spec": deepcopy(vars(spec)),
           "source_bindings": {}, "input_sha256": "i" * 64,
           "source_record_sha256": score.digest(record)}
    row.update(score.bank.row_identity(row))
    row["question_key"] = f"{row['dataset']}::{row['qid']}"
    trace = []
    for index in range(1, steps + 1):
        citation = str(tuple(spec.kg_subgraph[(index - 1) % len(spec.kg_subgraph)])) if graph else "[]"
        trace.append(f"[Step {index}]\nReasoning: The visible passage supports independent reasoning connection number {index}.\n"
                     f"Knowledge Used: {citation}\nConclusion: This gives connection number {index}.")
    generation = "\n".join(trace) + "\n[Final Answer]\nGamma"
    pred = {"candidate_id": row["question_key"] + "::k0", "candidate_index": 0,
            "dataset": row["dataset"], "qid": row["qid"],
            "generation_kind": "sampled", "generation": generation, "seed": 42,
            "n_response_tokens": 100, "reached_max_new_tokens": False}
    gates = {"norm_only": Gate(source_pass=source_pass), "features_v2": Gate(six=True, source_pass=source_pass)}
    return row, pred, Backend(), gates


def test_real_shared_candidate_scorer_evaluates_text_once_for_all_six_views():
    row, pred, backend, gates = fixture()
    before = deepcopy((row, pred))
    result = score.score_one(row, pred, backend, gates, protocol_sha256="p" * 64)
    assert result["trajectory_valid"]
    assert result["format_validation"]["required_steps"] == 2
    assert len(backend.calls) == len(result["raw_text"]) == 2
    assert result["proof_result"]["scorer_version"] == SCORER_VERSION
    assert len(result["features"]["features_v2"]["original"]["values"]) == 6
    assert len(result["features"]["norm_only"]["original"]["values"]) == 4
    for name in score.VARIANTS:
        assert set(result["variants"][name]) == set(score.ARMS)
        assert result["variants"][name]["T"]["alpha_effective"] == 0
        assert gates[name].predictions == 1
    assert result["gold_access"] is result["outcome_in_process"] is False
    assert result["generation"] == pred["generation"]
    assert (row, pred) == before
    assert result["process_row_sha256"] == score.digest({k: v for k, v in result.items() if k != "process_row_sha256"})


def test_ordinary_complete_two_steps_remains_invalid_and_never_scores_process():
    row, pred, backend, gates = fixture(graph=False)
    result = score.score_one(row, pred, backend, gates, protocol_sha256="p")
    assert result["trajectory_valid"] is False
    assert result["format_validation"]["required_steps"] == 3
    assert result["shortfall_salvage"]["shortfall_salvage_eligible"] is True
    assert backend.calls == []
    assert result["raw_text"] == []
    assert result["raw_graph_invalid_is_diagnostic_only"] is True
    for name in score.VARIANTS:
        assert gates[name].predictions == 0
        assert all(term["process"] == 0 and not term["rank_eligible"] for term in result["variants"][name].values())


def test_source_excluded_graph_keeps_existing_legal_two_step_format_but_zero_alpha():
    row, pred, backend, gates = fixture(source_pass=False)
    result = score.score_one(row, pred, backend, gates, protocol_sha256="p")
    assert result["trajectory_valid"] and len(backend.calls) == 2
    for name in score.VARIANTS:
        assert result["features"][name]["original"]["m_graph"] == 1
        assert result["features"][name]["masked"]["m_graph"] == 0
        terms = result["variants"][name]
        assert all(term["alpha_effective"] == term["graph_component"] == 0 for term in terms.values())
        assert terms["A"]["process"] == terms["F"]["process"] == terms["T"]["process"]


def test_ordinary_three_step_is_scored_and_never_receives_graph_credit():
    row, pred, backend, gates = fixture(graph=False, steps=3)
    result = score.score_one(row, pred, backend, gates, protocol_sha256="p")
    assert result["trajectory_valid"] and len(backend.calls) == 3
    assert result["variants"]["features_v2"]["A"]["graph_component"] == 0


def test_gold_field_is_rejected_before_any_scoring():
    row, pred, backend, gates = fixture()
    row["spec"]["gold_answer"] = "never-consumed"
    with pytest.raises(ValueError, match="gold"):
        score.score_one(row, pred, backend, gates, protocol_sha256="p")
    assert backend.calls == []


def test_overflow_fails_before_first_rearag_call():
    row, pred, backend, gates = fixture()
    backend.max_length = 5
    with pytest.raises(RuntimeError, match="implicit truncation forbidden"):
        score.score_one(row, pred, backend, gates, protocol_sha256="p")
    assert backend.calls == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1.01, -1.01])
def test_invalid_rearag_values_never_publish(value):
    row, pred, backend, gates = fixture()
    backend.value = value
    with pytest.raises(ValueError, match="ReaRAG score"):
        score.score_one(row, pred, backend, gates, protocol_sha256="p")


def test_process_arithmetic_uses_step_softsign_before_mean_and_one_alpha():
    gate = Gate(six=True)
    raw = [-.8, .7]
    features = {"m_graph": 1, "source_credit_mask": {"parent_m_graph": 1}}
    terms = score.process_terms(valid=True, raw_text=raw, raw_graph=.85, features=features, gate=gate)
    steps = normalize_text_steps_v2(raw, gate.normalization["text_v2"])["bounded_step_scores"]
    for arm, alpha in (("A", .6), ("F", .4), ("T", 0)):
        expected_text = .3 * (1 - alpha) * sum(steps) / len(steps)
        assert terms[arm]["text_component"] == pytest.approx(expected_text)
        assert terms[arm]["graph_component"] == pytest.approx(.2 * alpha)
        assert terms[arm]["process"] == pytest.approx(expected_text + .2 * alpha)
    assert gate.predictions == 1


def rank_rows():
    return [{"question_key": "synthetic::q", "candidate_id": f"synthetic::q::k{i}", "candidate_index": i,
             "generation_kind": "sampled" if i < 4 else "greedy", "trajectory_valid": i != 0,
             "variants": {variant: {arm: {"process": [0, -.1, -.1, -.2, 99][i]} for arm in score.ARMS}
                          for variant in score.VARIANTS}} for i in range(5)]


def test_rank_excludes_invalid_and_greedy_and_breaks_exact_ties_by_index():
    result = score.rank_questions(list(reversed(rank_rows())))[0]
    for name in score.VARIANTS:
        for arm in score.ARMS:
            ranked = result["rankings"][name][arm]
            assert ranked["selected_candidate_id"].endswith("::k1")
            assert ranked["ordered_process_scores"] == [-.1, -.1, -.2]
    assert result["invalid_sampled_candidate_ids"] == ["synthetic::q::k0"]


def test_all_invalid_sampled_has_no_selection_even_when_greedy_valid():
    rows = rank_rows()
    for row in rows[:4]: row["trajectory_valid"] = False
    result = score.rank_questions(rows)[0]
    assert result["all_sampled_invalid"]
    assert all(value["selected_candidate_id"] is None for variants in result["rankings"].values() for value in variants.values())


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "kind", "nan"])
def test_rank_malformed_population_rejected(mutation):
    rows = rank_rows()
    if mutation == "missing": rows.pop()
    elif mutation == "duplicate": rows[-1] = deepcopy(rows[0])
    elif mutation == "kind": rows[4]["generation_kind"] = "sampled"
    else: rows[1]["variants"]["features_v2"]["A"]["process"] = float("nan")
    with pytest.raises(ValueError): score.rank_questions(rows)


def test_per_step_resume_reuses_only_exact_bound_prompt_target_and_run(tmp_path):
    backend = Backend()
    kwargs = dict(candidate_id="synthetic::q::k0", generation_sha256="g", scoring_binding_sha256="s")
    first = score.StepCacheScorer(backend, tmp_path, **kwargs)
    assert first.score_step("prompt", "step one") == .4
    resumed = score.StepCacheScorer(backend, tmp_path, **kwargs)
    assert resumed.score_step("prompt", "step one") == .4
    assert resumed.score_step("prompt plus step one", "step two") == .4
    assert len(backend.calls) == 2 and resumed.cache_hits == resumed.new_calls == 1
    changed = score.StepCacheScorer(backend, tmp_path, **kwargs)
    with pytest.raises(ValueError, match="binding changed"):
        changed.score_step("changed prompt", "step one")
    changed_run = score.StepCacheScorer(backend, tmp_path, **{**kwargs, "scoring_binding_sha256": "other"})
    with pytest.raises(ValueError, match="binding changed"):
        changed_run.score_step("prompt", "step one")
    assert len(backend.calls) == 2


@pytest.mark.parametrize("value", [True, float("nan"), 2])
def test_invalid_step_cache_values_rejected_without_writing(tmp_path, value):
    cached = score.StepCacheScorer(Backend(value), tmp_path, candidate_id="q", generation_sha256="g", scoring_binding_sha256="s")
    with pytest.raises(ValueError): cached.score_step("prompt", "step")
    assert list(tmp_path.iterdir()) == []


def test_scoring_protocol_requires_registered_code_and_fixed_no_greedy_contract():
    config = {"format_version": "v2", "max_steps": 5, "ordinary_min_steps": 3, "min_reasoning_chars": 20,
              "text_backend": "rearag", "dtype": "bf16", "max_text_length": 4096,
              "process_weights": {"text": .3, "graph": .2}, "rank_samples": 4,
              "greedy_in_ranking": False, "rank_tie_break": "candidate_index_ascending",
              "gates": {key: {"path": key, "sha256": "x"} for key in score.VARIANTS}, "rearag_model": {"path": "synthetic"}}
    protocol = {"scoring": config, "code_bindings": dict.fromkeys(score.SCORING_CODE_FILES, {})}
    assert score.validate_scoring_protocol(protocol) == config
    for key, value in (("ordinary_min_steps", 2), ("greedy_in_ranking", True), ("greedy_in_ranking", 0), ("rank_samples", 2)):
        changed = deepcopy(protocol); changed["scoring"][key] = value
        with pytest.raises(ValueError): score.validate_scoring_protocol(changed)
    protocol["code_bindings"].pop(score.SCORING_CODE_FILES[0])
    with pytest.raises(ValueError, match="dependency code binding"):
        score.validate_scoring_protocol(protocol)


def test_atomic_publication_preserves_existing_scientific_record(tmp_path):
    target = tmp_path / "record.json"
    score.write_json(target, {"original": True})
    before = target.read_bytes()
    with pytest.raises(FileExistsError): score.write_json(target, {"replacement": True})
    assert target.read_bytes() == before
    attempts = list(tmp_path.glob("record.json.attempt-*"))
    assert len(attempts) == 1 and json.loads(attempts[0].read_text()) == {"replacement": True}


def test_interrupted_atomic_publication_can_resume_without_half_final(tmp_path, monkeypatch):
    original = score.os.link
    def interrupt(*args): raise InterruptedError("synthetic interruption")
    monkeypatch.setattr(score.os, "link", interrupt)
    target = tmp_path / "record.json"
    with pytest.raises(InterruptedError): score.write_json(target, {"valid": True})
    assert not target.exists() and len(list(tmp_path.glob("*.attempt-*"))) == 1
    monkeypatch.setattr(score.os, "link", original)
    score.write_json(target, {"valid": True})
    assert json.loads(target.read_text()) == {"valid": True}
    assert len(list(tmp_path.glob("*.attempt-*"))) == 1


def test_complete_cpu_double_run_interruption_resume_and_seal(tmp_path, monkeypatch):
    """Exercise the real orchestration with synthetic transport/model doubles."""
    from scripts.prepare import generate_source_credit_v2_fresh_confirmation_v1 as producer
    from kgproweight.reward.text_reward_model import RearagPromptScorer
    from transformers import AutoTokenizer
    row, prediction, backend, gates = fixture()
    original_call = backend.score_step
    calls = 0
    def interrupted_backend(prompt, text):
        nonlocal calls
        calls += 1
        if calls == 2: raise RuntimeError("synthetic CUDA runtime interruption")
        return original_call(prompt, text)
    backend.score_step = interrupted_backend
    model_info = {"path": str(tmp_path), "files": {}}
    gate_bindings = {}
    for name, gate in gates.items():
        path = tmp_path / (name + ".json")
        score.write_json(path, {"synthetic_gate": name})
        gate_bindings[name] = score.binding(path)
        gate.artifact = {"training_clearance": False,
                         "feature_version": "source-quality-trajectory-features-v2" if gate.six else "source-quality-trajectory-features-v1"}
        gate.mask = SimpleNamespace(payload_sha256="same-synthetic-mask")
    config = {"format_version": "v2", "max_steps": 5, "ordinary_min_steps": 3, "min_reasoning_chars": 20,
              "text_backend": "rearag", "dtype": "bf16", "max_text_length": 4096,
              "process_weights": {"text": .3, "graph": .2}, "rank_samples": 4,
              "greedy_in_ranking": False, "rank_tie_break": "candidate_index_ascending",
              "gates": gate_bindings, "rearag_model": model_info}
    protocol = tmp_path / "protocol.json"
    p = {"experiment_id": "SYNTHETIC-SCORING-TEST", "scoring": config,
         "code_bindings": dict.fromkeys(score.SCORING_CODE_FILES, {})}
    score.write_json(protocol, p)
    context = {"protocol": p, "protocol_sha256": score.sha(protocol), "inputs": [row],
               "policy_path": tmp_path, "input_manifest": {"models": {"rearag_model": model_info}}}
    generation = tmp_path / "generated"; generation.mkdir()
    predictions = [{**prediction, "candidate_id": row["question_key"] + f"::k{i}", "candidate_index": i,
                    "generation_kind": "sampled" if i < 4 else "greedy"} for i in range(5)]
    score.write_text(generation / "generations.jsonl", "".join(score.canonical(value) + "\n" for value in predictions))
    score.write_json(generation / "manifest.json", {
        "schema_version": "source-credit-v2-fresh-confirmation-generations-v1", "status": "COMPLETE_GENERATED_NOT_SCORED",
        "protocol_sha256": context["protocol_sha256"], "outputs": {"generations.jsonl": score.binding(generation / "generations.jsonl")}})
    tokenizer = SimpleNamespace(pad_token_id=None, eos_token_id=128001)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(producer, "verify_protocol", lambda *a, **k: context)
    def validate(preds, ctx, *, tokenizer):
        assert preds == predictions and tokenizer.pad_token_id == 128001
    monkeypatch.setattr(producer, "verify_generation_rows", validate)
    monkeypatch.setattr(score.SourceCreditGateV2, "load", lambda path, **k: gates[path.stem])
    monkeypatch.setattr(RearagPromptScorer, "from_pretrained", lambda *a, **k: backend)
    cuda = SimpleNamespace(is_bf16_supported=lambda: True, reset_peak_memory_stats=lambda: None,
                           get_device_name=lambda i: "SYNTHETIC CPU DOUBLE", max_memory_allocated=lambda: 0,
                           max_memory_reserved=lambda: 0, empty_cache=lambda: None)
    monkeypatch.setattr(score.bank, "require_cuda", lambda device: SimpleNamespace(cuda=cuda))
    out = tmp_path / "scoring"
    with pytest.raises(RuntimeError, match="synthetic CUDA runtime interruption"):
        score.run(protocol=protocol, generation=generation, out=out)
    assert not (out / "manifest.json").exists()
    assert len(list((out / "raw_steps").glob("*.json"))) == 1
    with pytest.raises(FileExistsError): score.run(protocol=protocol, generation=generation, out=out)
    report = score.run(protocol=protocol, generation=generation, out=out, resume=True)
    assert report["n_candidates"] == 5 and report["n_questions"] == 1
    assert report["reused_step_calls_this_attempt"] == 1 and report["new_step_calls_this_attempt"] == 9
    assert report["experiment_id"] == "SYNTHETIC-SCORING-TEST-SCORING"
    assert calls == 11 and len(backend.calls) == 10
    before = {p.name: p.read_bytes() for p in out.glob("*.json*")}
    assert score.run(protocol=protocol, generation=generation, out=out, resume=True) == report
    assert calls == 11 and {p.name: p.read_bytes() for p in out.glob("*.json*")} == before
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETE_GOLD_FREE_SCORED_NOT_ANALYZED"
    assert manifest["gold_access"] is manifest["gate_fitting"] is manifest["ppo_launch_clearance"] is False
    assert all(score.resolve(ref).is_file() for ref in manifest["outputs"].values())
