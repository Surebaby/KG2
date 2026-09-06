from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from kgproweight.kg.question_kg import validate_question_kg_record
from kgproweight.reward.proofkg_process import is_automatic_proofkg
from scripts.prepare.freeze_mixed_ppo_three_dataset_v1 import sha256_file
from scripts.prepare.freeze_mixed_ppo_three_dataset_v2_proof400 import (
    FAMILY_VERSION,
    build_groups,
    build_weights,
    expand_k4,
    select_proof400,
)
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256
from scripts.prepare.resolve_phase3_ppo_runtime_config import resolve_phase3_ppo_runtime_config


PROTOCOL_DIR = Path("outputs/audits/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42_protocol")
DATA_DIR = Path("data/silver_data/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42")
ADDENDUM_DIR = Path("outputs/audits/mixed_ppo_three_dataset_v2_proof400_family_scope_addendum_v2")
CONFIG_COMPARISON_DIR = Path("outputs/audits/mixed3_rearag_ppo_pair_proof400_7200_seed42_config_comparison_v2")
QTYPES = ("inference", "comparison", "compositional", "bridge_comparison")


def _read(path: Path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _row(dataset: str, qid: str, *, eligible: bool, route: str, qtype: str = "unknown"):
    return {
        "dataset": dataset, "qid": qid, "question": f"Question {qid}?",
        "question_sha256": f"hash-{qid}", "family_version": FAMILY_VERSION,
        "family_sha256": f"family-{qid}", "route": route,
        "proof_source": "test" if eligible else "none",
        "question_type": qtype, "process_reward_eligible": eligible,
    }


def test_schedule_has_exact_dataset_and_proof_exposures():
    population = [
        *(_row("hotpotqa", f"h{i}", eligible=False, route="hotpotqa_outcome") for i in range(600)),
        *(_row("musique", f"m{i}", eligible=False, route="musique_outcome") for i in range(599)),
        *(_row("2wikimultihopqa", f"o{i}", eligible=False, route="2wiki_ordinary_outcome") for i in range(200)),
        *(_row("2wikimultihopqa", f"p{i}", eligible=True, route="2wiki_proof", qtype=QTYPES[i % 4]) for i in range(400)),
    ]
    groups = build_groups(population)
    schedule = expand_k4(groups)
    weights = build_weights(population, groups)
    assert Counter(row["dataset"] for row in groups) == Counter({
        "hotpotqa": 600, "2wikimultihopqa": 600, "musique": 600,
    })
    assert len(groups) == 1800 and len(schedule) == 7200
    assert sum(row["process_reward_eligible"] for row in groups) == 400
    assert sum(row["process_reward_eligible"] for row in schedule) == 1600
    exposures = Counter((row["dataset"], row["qid"]) for row in groups)
    assert Counter(exposures.values()) == Counter({1: 1798, 2: 1})
    assert abs(sum(row["sampling_probability"] for row in weights) - 1.0) < 1e-12


def test_proof_selection_retains_only_safe_hard_and_fills_four_types():
    hard_counts = {"inference": 34, "comparison": 20, "compositional": 32, "bridge_comparison": 39}
    hard = []
    for qtype, count in hard_counts.items():
        hard.extend(_row("2wikimultihopqa", f"hard-{qtype}-{i}", eligible=True,
                         route="2wiki_hard_stability", qtype=qtype) for i in range(count))
    # Add 83 rows blocked by protected family, reproducing the real 208->125 boundary.
    blocked = []
    for i in range(83):
        item = _row("2wikimultihopqa", f"blocked-{i}", eligible=True,
                    route="2wiki_hard_stability", qtype=QTYPES[i % 4])
        item["family_sha256"] = f"protected-{i}"
        blocked.append(item)
    complete = []
    for qtype in QTYPES:
        complete.extend(_row("2wikimultihopqa", f"auto-{qtype}-{i}", eligible=True,
                             route=f"2wiki_proof_expansion_{qtype}", qtype=qtype) for i in range(120))
    selected, stats = select_proof400(
        [*hard, *blocked], complete,
        protected_qids=set(),
        protected_families={("2wikimultihopqa", f"protected-{i}") for i in range(83)},
    )
    assert len(selected) == 400
    assert stats["safe_hard"] == 125 and stats["fill_selected"] == 275
    assert Counter(row["question_type"] for row in selected) == Counter({q: 100 for q in QTYPES})


def test_frozen_v2_assets_have_zero_protected_a_overlap_and_exact_counts():
    protocol = json.loads((PROTOCOL_DIR / "protocol.json").read_text(encoding="utf-8"))
    population = _read(PROTOCOL_DIR / "population.question_only.jsonl")
    proof = _read(PROTOCOL_DIR / "proof400.question_only.jsonl")
    main = _read(PROTOCOL_DIR / "protected_a_canonical_main.question_only.jsonl")
    confirmation = _read(PROTOCOL_DIR / "protected_a_unopened_confirmation.question_only.jsonl")
    protected = [*main, *confirmation]
    protected_qids = {(row["dataset"], row["qid"]) for row in protected}
    protected_families = {(row["dataset"], family_sha256(row["question"])) for row in protected}
    population_qids = {(row["dataset"], row["qid"]) for row in population}
    population_families = {(row["dataset"], family_sha256(row["question"])) for row in population}
    assert len(population) == len(population_qids) == 1799
    assert Counter(row["dataset"] for row in population) == Counter({
        "hotpotqa": 600, "2wikimultihopqa": 600, "musique": 599,
    })
    assert len(proof) == 400
    assert Counter(row["question_type"] for row in proof) == Counter({q: 100 for q in QTYPES})
    assert population_qids.isdisjoint(protected_qids)
    assert population_families.isdisjoint(protected_families)
    assert all(row["family_version"] == FAMILY_VERSION for row in population + protected)
    assert protocol["protected_a_class"]["population_qid_overlap"] == 0
    assert protocol["protected_a_class"]["population_family_overlap"] == 0
    assert protocol["scientific_boundary"]["v1_family_gate_status"] == "SUPERSEDED_NAMESPACE_INCOMPARABLE"
    for identity in protocol["outputs"].values():
        assert sha256_file(Path(identity["path"])) == identity["sha256"]


def test_materialized_v2_has_400_complete_traces_and_1399_empty_records():
    silver = _read(DATA_DIR / "silver_train.jsonl")
    records = _read(DATA_DIR / "question_kg_records.jsonl")
    schedule = _read(DATA_DIR / "fixed_rollout_schedule.jsonl")
    by_key = {row["question_key"]: row for row in records}
    assert len(silver) == len(records) == len(by_key) == 1799
    eligible = 0
    empty = 0
    qtypes = Counter()
    for row in silver:
        key = f"{row['dataset']}::{row['qid']}"
        record = by_key[key]
        validate_question_kg_record(record)
        assert row["steps"] == []
        assert len(row["retrieved_passages"]) == 10
        assert record["question"] == row["question"]
        if record.get("process_reward_eligible"):
            eligible += 1
            qtypes[row["metadata"]["question_type"]] += 1
            assert is_automatic_proofkg(record, record["kg_subgraph"])
            assert record["runtime_error"] is None
            assert record["execution"]["complete_plan_execution"] is True
            assert len(record["execution"]["hops"]) >= len(record["query_plan"]["hops"])
            assert all(hop["matches"] for hop in record["execution"]["hops"][:len(record["query_plan"]["hops"])])
        else:
            empty += 1
            assert record["kg_subgraph"] == []
    assert eligible == 400 and empty == 1399
    assert qtypes == Counter({q: 100 for q in QTYPES})
    assert len(schedule) == 7200
    assert sum(row["process_reward_eligible"] for row in schedule) == 1600
    report = json.loads((DATA_DIR / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "COMPLETE_DATA_NOT_TRAINED"
    assert all(report["gates"].values())


def test_family_scope_addendum_locks_code_and_separates_cross_dataset_telemetry():
    addendum = json.loads((ADDENDUM_DIR / "addendum.json").read_text(encoding="utf-8"))
    scope = addendum["clarification"]
    assert scope["family_isolation_scope"] == "dataset-scoped (dataset, family_sha256)"
    assert scope["protected_a_qid_overlap"] == 0
    assert scope["protected_a_dataset_scoped_family_overlap"] == 0
    assert scope["population_internal_cross_dataset_same_template_family_count"] == 0
    assert scope["population_to_A_cross_dataset_same_template_family_count"] == 2
    assert scope["population_to_A_cross_dataset_same_template_row_count"] == 6
    assert addendum["v1_supersession_evidence"]["v1_recomputed_dataset_scoped_overlap_family_count"] == 51
    assert addendum["v1_supersession_evidence"]["v1_recomputed_overlap_row_count"] == 83
    for identity in addendum["code_lock"].values():
        assert sha256_file(Path(identity["path"])) == identity["sha256"]
    for identity in addendum["bound_assets"].values():
        assert sha256_file(Path(identity["path"])) == identity["sha256"]


def test_v2_runtime_configs_share_everything_except_process_switch_and_output():
    text_path = Path("configs/training/phase3_ppo_mixed3_rearag_v2_proof400_text7200_seed42.yaml")
    kg_path = Path("configs/training/phase3_ppo_mixed3_rearag_v2_proof400_text_kg_v2_1_7200_seed42.yaml")
    text = resolve_phase3_ppo_runtime_config(text_path)
    kg = resolve_phase3_ppo_runtime_config(kg_path)
    differences = {key for key in set(text) | set(kg) if text.get(key) != kg.get(key)}
    assert differences == {"output_dir", "proofkg_process_reward"}
    assert text["total_steps"] == kg["total_steps"] == 7200
    assert text["proofkg_process_reward"] is False
    assert kg["proofkg_process_reward"] is True
    assert text["silver_path"] == kg["silver_path"] == str(DATA_DIR / "silver_train.jsonl")
    assert text["fixed_rollout_schedule_path"] == kg["fixed_rollout_schedule_path"] == str(
        DATA_DIR / "fixed_rollout_schedule.jsonl"
    )
    assert text["alpha_gate_path"] is None and kg["alpha_gate_path"] is None
    assert text["text_reward_backend"] == kg["text_reward_backend"] == "rearag"


def test_current_config_hashes_are_bound_by_append_only_v2_comparison():
    report = json.loads((CONFIG_COMPARISON_DIR / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS_CONFIG_ONLY_NOT_GPU_PROBED_NOT_TRAINED"
    assert report["real_cli"]["pair_differences"].keys() == {"output_dir", "proofkg_process_reward"}
    assert report["supersession"]["v1_report_or_configs_overwritten"] is False
    assert all(report["gates"].values())
    for identity in report["configs"].values():
        if isinstance(identity, dict) and "path" in identity:
            assert sha256_file(Path(identity["path"])) == identity["sha256"]
