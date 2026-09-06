from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import pytest

from kgproweight.kg.question_kg import make_question_kg_record
from scripts.prepare.freeze_mixed_ppo_three_dataset_v4_proof800 import (
    DEFAULT_REPLAY,
    HOTPOT_TARGET_CELLS,
    MUSIQUE_TARGET_HOPS,
    PROOF_TARGET_TYPES,
    IdentityIndex,
    _identity,
    build_groups,
    build_hotpot_population,
    build_musique_population,
    build_weights,
    expand_k4,
    identity_overlap_counts,
    load_hm_reconciliation_release,
    load_ordinary200_release,
    normalise_proof_candidates,
    select_proof800,
    PROTECTED_LEDGER_SCHEMA,
    PROTECTED_LEDGER_STATUS,
    validate_protected_ledger_release,
)


def test_v4_freezer_defaults_to_clean_replay_v2():
    assert str(DEFAULT_REPLAY).endswith(
        "sft_replay_legacy_strong_train_rendered3to5_n2000_seed42_v2/"
        "silver_train.jsonl"
    )


def _letters(index: int) -> str:
    value = index + 1
    output = []
    while value:
        value, remainder = divmod(value - 1, 26)
        output.append(chr(ord("a") + remainder))
    return "".join(reversed(output))


def _hotpot_raw(qid: str, stratum: str, serial: int) -> dict:
    qtype, level = stratum.split("/")
    return {
        "id": qid,
        "question": f"which item has marker zeta{_letters(serial)}?",
        "metadata": {"type": qtype, "level": level},
    }


def _musique_raw(qid: str, hop: str, serial: int) -> dict:
    n_hops = int(hop[0])
    return {
        "id": qid,
        "question": f"what follows marker theta{_letters(serial)}?",
        "metadata": {
            "metadata": {
                "question_decomposition": [
                    {"support_paragraph": {"paragraph_text": "unused"}}
                    for _ in range(n_hops)
                ]
            }
        },
    }


def test_hotpot_replaces_family_overlaps_and_hits_both_marginals():
    retained_counts = {
        "bridge/easy": 96,
        "bridge/medium": 325,
        "bridge/hard": 86,
        "comparison/easy": 15,
        "comparison/medium": 46,
        "comparison/hard": 15,
    }
    removed_counts = {
        "bridge/easy": 3,
        "bridge/medium": 8,
        "bridge/hard": 2,
        "comparison/easy": 0,
        "comparison/medium": 4,
        "comparison/hard": 0,
    }
    raw = []
    parent = []
    blocked = IdentityIndex()
    serial = 0
    for stratum in HOTPOT_TARGET_CELLS:
        for local in range(retained_counts[stratum] + removed_counts[stratum]):
            row = _hotpot_raw(f"h-parent-{serial}", stratum, serial)
            raw.append(row)
            identity = _identity(row, dataset="hotpotqa")
            parent.append(identity)
            if local >= retained_counts[stratum]:
                alias = dict(identity)
                alias["qid"] = f"train-alias-{serial}"
                alias["question_key"] = f"hotpotqa::{alias['qid']}"
                blocked.add(alias)
            serial += 1
        required_new = HOTPOT_TARGET_CELLS[stratum] - retained_counts[stratum]
        for _ in range(required_new + 2):
            raw.append(_hotpot_raw(f"h-new-{serial}", stratum, serial))
            serial += 1

    final, selected, reserve, stats = build_hotpot_population(
        parent, raw, externally_blocked=blocked, reserve_per_stratum=2
    )
    assert len(final) == 1000
    assert len(selected) == 417
    assert len(reserve) == 12
    assert stats["retained_parent"] == 583
    assert stats["removed_parent"] == 17
    assert stats["removed_overlap_reasons"] == {
        "family_sha256": 17,
        "question_sha256": 17,
    }
    assert Counter(row["stratum"] for row in final) == Counter(HOTPOT_TARGET_CELLS)
    assert Counter(row["question_type"] for row in final) == Counter(
        {"bridge": 750, "comparison": 250}
    )
    levels = Counter(row["stratum"].split("/")[1] for row in final)
    assert levels == Counter({"easy": 200, "medium": 600, "hard": 200})


def test_musique_retains_599_and_adds_exact_hop_quotas():
    retained_counts = {"2hop": 423, "3hop": 146, "4hop": 30}
    raw = []
    parent = []
    serial = 0
    for hop in MUSIQUE_TARGET_HOPS:
        for _ in range(retained_counts[hop]):
            row = _musique_raw(f"m-parent-{serial}", hop, serial)
            raw.append(row)
            parent.append(_identity(row, dataset="musique"))
            serial += 1
        required_new = MUSIQUE_TARGET_HOPS[hop] - retained_counts[hop]
        for _ in range(required_new + 2):
            raw.append(_musique_raw(f"m-new-{serial}", hop, serial))
            serial += 1

    final, selected, reserve, stats = build_musique_population(
        parent, raw, externally_blocked=IdentityIndex(), reserve_per_stratum=2
    )
    assert len(final) == 1000
    assert len(selected) == 401
    assert len(reserve) == 6
    assert stats["retained_parent"] == 599
    assert Counter(row["stratum"] for row in final) == Counter(MUSIQUE_TARGET_HOPS)


def test_musique_replaces_protected_parent_instead_of_weakening_isolation():
    retained_counts = {"2hop": 423, "3hop": 146, "4hop": 30}
    raw = []
    parent = []
    serial = 0
    for hop in MUSIQUE_TARGET_HOPS:
        for _ in range(retained_counts[hop]):
            row = _musique_raw(f"m-parent-{serial}", hop, serial)
            raw.append(row)
            parent.append(_identity(row, dataset="musique"))
            serial += 1
        required_new = MUSIQUE_TARGET_HOPS[hop] - retained_counts[hop]
        for _ in range(required_new + 3):
            raw.append(_musique_raw(f"m-new-{serial}", hop, serial))
            serial += 1

    blocked = IdentityIndex()
    blocked.add(parent[0])
    final, selected, reserve, stats = build_musique_population(
        parent, raw, externally_blocked=blocked, reserve_per_stratum=2
    )

    assert len(final) == 1000
    assert len(selected) == 402
    assert len(reserve) == 6
    assert stats["retained_parent"] == 598
    assert stats["removed_parent"] == 1
    assert stats["removed_overlap_reasons"] == {
        "family_sha256": 1,
        "qid": 1,
        "question_sha256": 1,
    }
    assert not any(row["qid"] == parent[0]["qid"] for row in final)
    assert Counter(row["stratum"] for row in final) == Counter(MUSIQUE_TARGET_HOPS)


def _valid_proof_wrapper(qid: str = "q1", qtype: str = "inference") -> dict:
    question = f"what does marker omega{qid} ultimately link to?"
    record = make_question_kg_record(
        dataset="2wikimultihopqa",
        qid=qid,
        question=question,
        triples=[("Alpha", "links to", "Beta"), ("Beta", "links to", "Gamma")],
        query_plan={
            "recognized": True,
            "hops": [
                {"subject": "Alpha", "pids": ["P1"], "output_slot": "hop_1"},
                {"subject": "$hop_1", "pids": ["P2"], "output_slot": "hop_2"},
            ],
        },
        provenance={
            "builder_version": "synthetic-proof-builder",
            "gold_access": False,
            "complete_plan_execution": True,
        },
    )
    record["runtime_error"] = None
    record["execution"] = {
        "complete_plan_execution": True,
        "hops": [
            {
                "hop_index": 1,
                "input_entities": [{"qid": "Q1"}],
                "matches": [["Alpha", "links to", "Beta"]],
            },
            {
                "hop_index": 2,
                "input_entities": [{"qid": "Q2"}],
                "matches": [["Beta", "links to", "Gamma"]],
            },
        ],
    }
    return {
        "question_type": qtype,
        "proof_passages_sha256": "a" * 64,
        "question_kg_record": record,
    }


def test_unified_proof_candidates_are_rechecked_by_hard_gate():
    valid = _valid_proof_wrapper()
    invalid = copy.deepcopy(_valid_proof_wrapper("q2"))
    invalid["question_kg_record"]["runtime_error"] = "failed"
    rows, reasons = normalise_proof_candidates(
        [valid, invalid], historical_cutoff="2020-12-09T23:59:59Z"
    )
    assert len(rows) == 1
    assert rows[0]["qid"] == "q1"
    assert rows[0]["process_reward_eligible"] is True
    assert reasons["eligible"] == 1
    assert reasons["failed:runtime_error_zero"] == 1


def _schedule_row(dataset: str, qid: str, eligible: bool) -> dict:
    return {
        "dataset": dataset,
        "qid": qid,
        "question_sha256": f"hash-{dataset}-{qid}",
        "family_sha256": f"family-{dataset}-{qid}",
        "route": "proof" if eligible else "outcome",
        "process_reward_eligible": eligible,
    }


def test_schedule_is_balanced_unique_k4_and_has_800_proof_groups():
    population = [
        *(_schedule_row("hotpotqa", f"h{i}", False) for i in range(1000)),
        *(_schedule_row("musique", f"m{i}", False) for i in range(1000)),
        *(_schedule_row("2wikimultihopqa", f"p{i}", True) for i in range(800)),
        *(_schedule_row("2wikimultihopqa", f"o{i}", False) for i in range(200)),
    ]
    groups = build_groups(population)
    schedule = expand_k4(groups)
    weights = build_weights(population, groups)
    assert len(groups) == 3000
    assert Counter(row["dataset"] for row in groups) == Counter(
        {"hotpotqa": 1000, "2wikimultihopqa": 1000, "musique": 1000}
    )
    assert len({(row["dataset"], row["qid"]) for row in groups}) == 3000
    assert sum(row["process_reward_eligible"] for row in groups) == 800
    assert len(schedule) == 12000
    assert all(
        len({(row["dataset"], row["qid"]) for row in schedule[start : start + 4]}) == 1
        for start in range(0, len(schedule), 4)
    )
    assert abs(sum(row["sampling_probability"] for row in weights) - 1.0) < 1e-12


def test_proof800_prefers_distinct_families_then_allows_template_repeats():
    candidates = []
    for qtype in PROOF_TARGET_TYPES:
        for index in range(205):
            candidates.append(
                {
                    "dataset": "2wikimultihopqa",
                    "qid": f"{qtype}-{index}",
                    "question_sha256": f"hash-{qtype}-{index}",
                    "family_sha256": f"family-{qtype}-{index % 40}",
                    "question_type": qtype,
                    "stratum": qtype,
                    "route": f"2wiki_proof_{qtype}",
                    "process_reward_eligible": True,
                }
            )
    selected, reserve, stats = select_proof800(
        candidates, blocked=IdentityIndex(), reserve_per_stratum=0
    )
    assert len(selected) == 800
    assert reserve == []
    assert Counter(row["question_type"] for row in selected) == Counter(PROOF_TARGET_TYPES)
    assert stats["selected_unique_families"] == 160


def test_identity_isolation_checks_qid_exact_question_and_family():
    base = {
        "dataset": "hotpotqa",
        "qid": "a",
        "question_sha256": "hash-a",
        "family_sha256": "family-a",
    }
    right = [base]
    left = [
        {**base, "question_sha256": "hash-x", "family_sha256": "family-x"},
        {**base, "qid": "b", "family_sha256": "family-y"},
        {**base, "qid": "c", "question_sha256": "hash-z"},
    ]
    assert identity_overlap_counts(left, right) == {
        "qid": 1,
        "question_sha256": 1,
        "family_sha256": 1,
    }


def test_freezer_binds_complete_ledger_report_and_manifest(tmp_path: Path):
    ledger = tmp_path / "protected_identities.question_only.jsonl"
    ledger.write_text("{}\n", encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "schema_version": PROTECTED_LEDGER_SCHEMA,
        "status": PROTECTED_LEDGER_STATUS,
        "complete": True,
        "identity_scope": "dataset-scoped",
        "current_family_recomputed": True,
        "output": {"rows": 1, "sha256": digest(ledger)},
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": PROTECTED_LEDGER_STATUS,
                "run": {
                    "protected_identities_sha256": digest(ledger),
                    "report_sha256": digest(report_path),
                },
            }
        ),
        encoding="utf-8",
    )
    path, binding = validate_protected_ledger_release(tmp_path)
    assert path == ledger
    assert set(binding) == {"ledger", "report", "manifest"}

    report["current_family_recomputed"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="completeness"):
        validate_protected_ledger_release(tmp_path)


def test_freezer_binds_successor_ordinary200_release(tmp_path: Path):
    rows_path = tmp_path / "ordinary.jsonl"
    rows = [
        {
            "dataset": "2wikimultihopqa",
            "qid": f"o{i}",
            "question": f"Which marker belongs to ordinary item {i}?",
        }
        for i in range(200)
    ]
    rows_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    protocol_path = tmp_path / "protocol.json"
    protocol = {
        "schema_version": "2wiki-ordinary200-full-ledger-protocol-v2",
        "status": "FROZEN_ANSWER_FREE_OUTCOME_SOURCE_BOUND_NOT_TRAINED",
        "selection": {"counts": {"gates": {"all_safe": True}}},
        "outputs": {
            "ordinary200": {
                "path": str(rows_path),
                "sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
            }
        },
    }
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    loaded, bound = load_ordinary200_release(protocol_path)
    assert len(loaded) == 200
    assert bound["selection"]["counts"]["gates"]["all_safe"] is True

    protocol["selection"]["counts"]["gates"]["all_safe"] = False
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="gates"):
        load_ordinary200_release(protocol_path)


def test_freezer_binds_hm_reconciliation_and_delta_contract(tmp_path: Path):
    population = []
    for dataset in ("hotpotqa", "musique"):
        for index in range(1000):
            is_new = index < (417 if dataset == "hotpotqa" else 406)
            population.append(
                {
                    "dataset": dataset,
                    "qid": f"{dataset}-{index}",
                    "source_role": "new_retrieval" if is_new else "retained_parent",
                }
            )
    requirements = [
        {"dataset": row["dataset"], "qid": row["qid"]}
        for row in population
        if row["source_role"] == "new_retrieval"
    ]
    new_requests = [
        row for row in requirements if row["dataset"] == "musique"
    ][:11]
    outputs = {}
    for name, rows in (
        ("hm_population", population),
        ("retrieval_requirements", requirements),
        ("new_retrieval_requests", new_requests),
        ("reserve", []),
    ):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        outputs[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    protocol = {
        "schema_version": "mixed-ppo-v4-hm-full-ledger-reconciliation-v2",
        "status": "FROZEN_HM_FULL_LEDGER_DELTA_RETRIEVAL_NOT_RUN_NOT_TRAINED",
        "gates": {"all_safe": True},
        "outputs": outputs,
    }
    protocol_path = tmp_path / "hm_protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    loaded_population, loaded_requirements, loaded_new, reserve, _ = (
        load_hm_reconciliation_release(protocol_path)
    )
    assert len(loaded_population) == 2000
    assert len(loaded_requirements) == 823
    assert len(loaded_new) == 11
    assert reserve == []

    protocol["gates"]["all_safe"] = False
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="gates"):
        load_hm_reconciliation_release(protocol_path)
