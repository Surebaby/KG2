"""Independent CPU replay counterexamples; no real candidates or Gold reads."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.pilot import analyze_source_credit_v2_fresh_confirmation_v1 as audit
from scripts.prepare import score_source_credit_v2_fresh_confirmation_v1 as scorer
from tests.test_score_source_credit_v2_fresh_confirmation_v1 import fixture


def example(*, graph=True, steps=2, source_pass=True):
    original, pred, backend, gates = fixture(graph=graph, steps=steps, source_pass=source_pass)
    original["m_graph"] = int(graph)
    for gate in gates.values():
        gate.normalization.update({k: gate.normalization["text_v2"][k] for k in ("text_center", "text_scale")})
    row = scorer.score_one(original, pred, backend, gates, protocol_sha256="synthetic-protocol")
    return row, original, pred, backend, gates


@pytest.mark.parametrize("graph,steps,source_pass", [(True, 2, True), (True, 2, False), (False, 2, True), (False, 3, True)])
def test_independent_cpu_replay_matches_actual_shared_scorer(graph, steps, source_pass):
    row, original, _, backend, gates = example(graph=graph, steps=steps, source_pass=source_pass)
    old_calls = len(backend.calls)
    validity = audit.verify_cpu_components(row, original, gates, backend.tokenizer)
    assert validity["valid"] is row["trajectory_valid"]
    assert len(backend.calls) == old_calls


@pytest.mark.parametrize("mutation", ["raw_graph", "proof_score", "proof_version", "format_min", "format_violations",
                                    "raw_mean", "budget_truncated", "budget_count", "diagnostic_flag"])
def test_independent_cpu_replay_rejects_self_signed_component_drift(mutation):
    row, original, _, backend, gates = example()
    if mutation == "raw_graph": row["raw_graph"] += .001
    elif mutation == "proof_score": row["proof_result"]["score"] += .001
    elif mutation == "proof_version": row["proof_result"]["scorer_version"] = "unregistered-version"
    elif mutation == "format_min": row["format_validation"]["required_steps"] = 3
    elif mutation == "format_violations": row["format_validation"]["violations"] = ["invented"]
    elif mutation == "raw_mean": row["raw_text_step_mean"] += .001
    elif mutation == "budget_truncated": row["text_token_budget"]["truncated_tokens"] = 1
    elif mutation == "budget_count": row["text_token_budget"]["step_lengths"][0]["prompt_tokens"] += 1
    else: row["raw_graph_invalid_is_diagnostic_only"] = True
    row["process_row_sha256"] = scorer.digest({k: v for k, v in row.items() if k != "process_row_sha256"})
    with pytest.raises(ValueError):
        audit.verify_cpu_components(row, original, gates, backend.tokenizer)


def test_omitting_text_step_fails_even_after_all_means_and_terms_are_recomputed():
    row, original, _, backend, gates = example()
    row["raw_text"].pop()
    row["raw_text_step_mean"] = sum(row["raw_text"]) / len(row["raw_text"])
    row["text_token_budget"]["step_lengths"].pop()
    for variant, gate in gates.items():
        row["variants"][variant] = scorer.process_terms(valid=True, raw_text=row["raw_text"], raw_graph=row["raw_graph"],
                                                        features=row["features"][variant]["masked"], gate=gate)
    with pytest.raises(ValueError, match="every and only valid step"):
        audit.verify_cpu_components(row, original, gates, backend.tokenizer)


def test_valid_text_budget_replay_requires_tokenizer_and_no_truncation():
    row, original, _, backend, gates = example()
    with pytest.raises(ValueError, match="tokenizer required"):
        audit.verify_cpu_components(row, original, gates)
    def overflowing_tokenizer(*args, **kwargs): return {"input_ids": [0] * 4097}
    with pytest.raises(RuntimeError, match="implicit truncation forbidden"):
        audit.verify_cpu_components(row, original, gates, overflowing_tokenizer)


@pytest.mark.parametrize("mutation", ["text", "mean", "budget", "graph_flag"])
def test_invalid_two_step_replay_cannot_be_presented_as_scored(mutation):
    row, original, _, backend, gates = example(graph=False)
    if mutation == "text": row["raw_text"] = [.5, .5]
    elif mutation == "mean": row["raw_text_step_mean"] = 0.
    elif mutation == "budget": row["text_token_budget"] = {"truncated_tokens": 0}
    else: row["raw_graph_invalid_is_diagnostic_only"] = False
    with pytest.raises(ValueError): audit.verify_cpu_components(row, original, gates, backend.tokenizer)


@pytest.mark.parametrize("pad", [None, 128001, 128009])
def test_pad_fallback_exactly_matches_producer_and_scorer(pad):
    tokenizer = SimpleNamespace(pad_token_id=pad, eos_token_id=128001)
    assert audit.policy_tokenizer_with_frozen_pad(tokenizer) is tokenizer
    assert tokenizer.pad_token_id == (128001 if pad is None else pad)


def population(*, source_pass=True, graph=True, steps=2):
    """Expand an artificial trace to exercise membership and all six formulas."""
    base, original, pred, backend, gates = example(graph=graph, steps=steps, source_pass=source_pass)
    inputs, rows, predictions, checks = [], [], [], {}
    for question in range(132):
        source = deepcopy(original)
        source["qid"] = f"synthetic-only-{question}"
        source["question_key"] = f"{source['dataset']}::{source['qid']}"
        source["family_sha256"] = f"synthetic-family-{question}"
        inputs.append(source)
        if graph: checks[source["question_key"]] = {"status": "PASS" if source_pass else "FAIL"}
        for index in range(5):
            generated = {**pred, "qid": source["qid"], "candidate_id": source["question_key"] + f"::k{index}",
                         "candidate_index": index, "generation_kind": "sampled" if index < 4 else "greedy"}
            predictions.append(generated)
            process = deepcopy(base)
            process.update({k: source[k] for k in ("qid", "question_key", "family_sha256")})
            process.update({k: generated[k] for k in ("candidate_id", "candidate_index", "generation_kind")})
            process["generation_sha256"] = scorer.digest(generated)
            process["process_row_sha256"] = scorer.digest({k: v for k, v in process.items() if k != "process_row_sha256"})
            rows.append(process)
    return rows, predictions, inputs, checks, backend, gates


def test_full_660_synthetic_process_math_and_six_view_join_matches():
    rows, predictions, inputs, checks, backend, gates = population()
    grouped = audit.verify_process_rows(rows, predictions, inputs, {}, checks, protocol_sha256="synthetic-protocol",
                                       gates=gates, rearag_tokenizer=backend.tokenizer)
    assert len(grouped) == 132 and all(len(values) == 5 for values in grouped.values())
    assert len(backend.calls) == 2


@pytest.mark.parametrize("mutation", ["alpha", "normalization", "identity", "generation", "missing_candidate", "mask", "process", "tiny_tie_shift"])
def test_process_validation_rejects_self_signed_scientific_and_identity_drift(mutation):
    rows, predictions, inputs, checks, backend, gates = population()
    row = rows[0]
    if mutation == "alpha":
        for arm in ("A", "F"):
            item = row["variants"]["features_v2"][arm]
            item["alpha_effective"] = .01
            item["text_step_components"] = [.3 * .99 * value / len(row["raw_text"]) for value in item["text_normalized_steps"]]
            item["text_component"] = sum(item["text_step_components"])
            item["graph_component"] = .2 * .01 * item["graph_normalized"]
            item["process"] = item["text_component"] + item["graph_component"]
    elif mutation == "normalization": row["variants"]["features_v2"]["A"]["text_normalized_steps"][0] += .1
    elif mutation == "identity": row["qid"] = "other-question"
    elif mutation == "generation": row["generation"] += " changed"
    elif mutation == "missing_candidate": rows.pop()
    elif mutation == "mask": row["features"]["features_v2"]["masked"]["m_graph"] = 0
    # This moves originally tied k0 below k1 and changes the frozen selection.
    elif mutation == "tiny_tie_shift": row["variants"]["features_v2"]["A"]["process"] -= 1e-13
    else: row["variants"]["features_v2"]["A"]["process"] += .1
    row["process_row_sha256"] = scorer.digest({k: v for k, v in row.items() if k != "process_row_sha256"})
    with pytest.raises(ValueError):
        audit.verify_process_rows(rows, predictions, inputs, {}, checks, protocol_sha256="synthetic-protocol",
                                  gates=gates, rearag_tokenizer=backend.tokenizer)


@pytest.mark.parametrize("graph,steps", [(True, 2), (False, 3), (False, 2)])
def test_source_excluded_ordinary_and_invalid_cannot_receive_alpha(graph, steps):
    rows, predictions, inputs, checks, backend, gates = population(source_pass=False, graph=graph, steps=steps)
    row = rows[0]
    row["variants"]["features_v2"]["A"]["alpha_effective"] = .1
    row["process_row_sha256"] = scorer.digest({k: v for k, v in row.items() if k != "process_row_sha256"})
    with pytest.raises(ValueError):
        audit.verify_process_rows(rows, predictions, inputs, {}, checks, protocol_sha256="synthetic-protocol",
                                  gates=gates, rearag_tokenizer=backend.tokenizer)


def test_legacy_source_binding_without_bytes_is_verified_and_upgraded_for_seal(tmp_path):
    path = tmp_path / "historical-source.json"
    path.write_text("synthetic immutable source")
    legacy = {"path": str(path), "sha256": audit.sha(path)}
    assert audit.checked(legacy) == path.resolve()
    sealed = audit.binding(audit.checked(legacy))
    assert sealed["bytes"] == path.stat().st_size and sealed["sha256"] == legacy["sha256"]
    with pytest.raises(ValueError): audit.checked({**sealed, "bytes": sealed["bytes"] + 1})
    path.write_text("changed synthetic immutable source")
    with pytest.raises(ValueError): audit.checked(legacy)
