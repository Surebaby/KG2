from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from kgproweight.eval.query_controller_v1 import validate_action_record
from kgproweight.kg.question_kg import question_key, question_sha256
import scripts.prepare.build_query_controller_action_supervision_v1 as builder_module
from scripts.prepare.build_query_controller_action_supervision_v1 import (
    BuildReject,
    PlannerCandidate,
    build_action_pair,
    build_candidate_pool,
    build_release,
    canonical_action_pair_sha256,
    is_clean_linear_two_step,
    normalize_adjacent_duplicate_articles,
)
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256


def _candidate(
    dataset: str,
    qid: str,
    question: str,
    *,
    relation_one: str = "parent",
    relation_two: str = "place of birth",
) -> PlannerCandidate:
    if dataset == "2wikimultihopqa":
        plan = {
            "target_type": "relation_graph",
            "target": {
                "anchors": ["Alpha"],
                "steps": [
                    {
                        "step": 1,
                        "subject": "Alpha",
                        "relation_label": relation_one,
                        "pid": "P22",
                        "output_slot": "hop_1",
                        "dependencies": [],
                    },
                    {
                        "step": 2,
                        "subject": "$hop_1",
                        "relation_label": relation_two,
                        "pid": "P19",
                        "output_slot": "hop_2",
                        "dependencies": ["hop_1"],
                    },
                ],
            },
        }
    else:
        plan = {
            "target_type": "subquery_graph",
            "target": {
                "steps": [
                    {
                        "step": 1,
                        "subquery_template": "The Collegian >> owned by",
                        "dependencies": [],
                        "output_slot": "step_1",
                    },
                    {
                        "step": 2,
                        "subquery_template": "When was #1 founded?",
                        "dependencies": ["step_1"],
                        "output_slot": "step_2",
                    },
                ]
            },
        }
    return PlannerCandidate(
        dataset=dataset,
        qid=qid,
        question=question,
        family=family_sha256(question),
        plan=plan,
        planner_path="planner.jsonl",
        planner_sha256="a" * 64,
    )


def _2wiki_raw(qid: str, question: str, *, first_sentence: str | None = None) -> dict:
    return {
        "id": qid,
        "question": question,
        "golden_answers": ["Finaltown"],
        "metadata": {
            "context": {
                "title": ["Alpha", "Bridge"],
                "content": [
                    [first_sentence or "Alpha's parent was Bridge."],
                    ["Bridge was born in Finaltown."],
                ],
            },
            "supporting_facts": {"title": ["Alpha", "Bridge"], "sent_id": [0, 0]},
            "evidences": {
                "fact": ["Alpha", "Bridge"],
                "relation": ["parent", "place of birth"],
                "entity": ["Bridge", "Finaltown"],
            },
        },
    }


def _musique_raw(qid: str, question: str) -> dict:
    return {
        "id": qid,
        "question": question,
        "golden_answers": ["1960"],
        "metadata": {
            "metadata": {
                "question_decomposition": [
                    {
                        "question": "The Collegian >> owned by",
                        "answer": "Houston Baptist University",
                        "paragraph_support_idx": 5,
                        "support_paragraph": {
                            "idx": 5,
                            "title": "The Collegian",
                            "paragraph_text": (
                                "The Collegian is the official student publication of "
                                "Houston Baptist University. It is issued twice a month."
                            ),
                        },
                    },
                    {
                        "question": "When was #1 founded?",
                        "answer": "1960",
                        "paragraph_support_idx": 9,
                        "support_paragraph": {
                            "idx": 9,
                            "title": "Houston Baptist University",
                            "paragraph_text": "The university was founded in 1960.",
                        },
                    },
                ]
            }
        },
    }


def _planner_row(candidate: PlannerCandidate) -> dict:
    return {
        "schema_version": "query-planner-supervision-1",
        "question_key": question_key(candidate.dataset, candidate.qid),
        "dataset": candidate.dataset,
        "qid": candidate.qid,
        "question": candidate.question,
        "question_sha256": question_sha256(candidate.question),
        **candidate.plan,
        "provenance": {"split": "train"},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_2wiki_pair_is_valid_and_future_tail_is_absent() -> None:
    question = "Where was the parent of Alpha born?"
    q1, q2 = build_action_pair(
        _candidate("2wikimultihopqa", "w1", question),
        _2wiki_raw("w1", question),
        split="train",
    )
    validate_action_record(q1, expected_split="train")
    validate_action_record(q2, expected_split="train")
    assert q1["target"]["query"] == "Alpha parent"
    assert q2["target"]["query"] == "Bridge place of birth"
    assert q2["state"]["verified_observations"][0]["answer"] == "Bridge"
    assert q2["gold_boundary"]["train_intermediate_annotation_used"] is True
    assert "Finaltown" not in json.dumps([q1, q2], ensure_ascii=False)
    assert {q1["target"]["source_action"], q2["target"]["source_action"]} == {"text"}


def test_musique_pair_substitutes_verified_intermediate() -> None:
    question = "When was the institute that owned The Collegian founded?"
    q1, q2 = build_action_pair(
        _candidate("musique", "m1", question),
        _musique_raw("m1", question),
        split="dev",
    )
    assert q1["target"]["query"] == "The Collegian owned by"
    assert q2["target"]["query"] == "When was Houston Baptist University founded?"
    assert "#1" not in q2["target"]["query"]
    assert q2["target"]["dependencies"] == ["q1"]
    assert q2["state"]["verified_observations"][0]["provenance"]["binding_method"] == (
        "decomposition_step_support_answer_surface"
    )
    validate_action_record(q2, expected_split="dev")


def test_future_answer_in_first_hop_evidence_is_rejected() -> None:
    question = "Where was the parent of Alpha born?"
    with pytest.raises(BuildReject, match="final_or_future_tail_leak"):
        build_action_pair(
            _candidate("2wikimultihopqa", "w1", question),
            _2wiki_raw("w1", question, first_sentence="Alpha's parent was Bridge in Finaltown."),
            split="train",
        )


def test_two_character_final_alias_inside_intermediate_is_rejected() -> None:
    question = "When was the institute that owned The Collegian founded?"
    raw = _musique_raw("m-short", question)
    raw["golden_answers"] = ["WB"]
    decomposition = raw["metadata"]["metadata"]["question_decomposition"]
    decomposition[0]["answer"] = "The WB"
    decomposition[0]["support_paragraph"]["paragraph_text"] = (
        "The Collegian is the official publication of The WB."
    )
    decomposition[1]["answer"] = "WB"
    with pytest.raises(BuildReject, match="final_or_future_tail_leak"):
        build_action_pair(
            _candidate("musique", "m-short", question),
            raw,
            split="train",
        )


def test_single_digit_final_is_scanned_in_semantic_text() -> None:
    question = "Where was the parent of Alpha born?"
    raw = _2wiki_raw("w-digit", question, first_sentence="Alpha's parent was Bridge in district 1.")
    raw["golden_answers"] = ["1"]
    raw["metadata"]["evidences"]["entity"][1] = "1"
    with pytest.raises(BuildReject, match="final_or_future_tail_leak"):
        build_action_pair(
            _candidate("2wikimultihopqa", "w-digit", question), raw, split="train"
        )


def test_single_letter_final_exact_intermediate_is_rejected() -> None:
    question = "Where was the parent of Alpha born?"
    raw = _2wiki_raw("w-letter", question, first_sentence="Alpha's parent was A.")
    raw["golden_answers"] = ["A"]
    raw["metadata"]["evidences"]["entity"] = ["A", "A"]
    with pytest.raises(BuildReject, match="final_or_future_tail_leak"):
        build_action_pair(
            _candidate("2wikimultihopqa", "w-letter", question), raw, split="train"
        )


def test_short_final_in_identifier_hash_like_field_is_not_a_false_positive() -> None:
    question = "When was the institute that owned The Collegian founded?"
    raw = _musique_raw("m-hash", question)
    raw["golden_answers"] = ["11"]
    decomposition = raw["metadata"]["metadata"]["question_decomposition"]
    decomposition[0]["paragraph_support_idx"] = 11
    decomposition[0]["support_paragraph"]["idx"] = 11
    decomposition[1]["answer"] = "11"
    q1, q2 = build_action_pair(
        _candidate("musique", "m-hash", question), raw, split="train"
    )
    assert q2["state"]["verified_observations"][0]["document_id"].endswith("::11")
    validate_action_record(q1, expected_split="train")
    validate_action_record(q2, expected_split="train")


def test_adjacent_duplicate_article_normalization_is_generic() -> None:
    assert normalize_adjacent_duplicate_articles("the The WB") == ("The WB", 1)
    assert normalize_adjacent_duplicate_articles("AN an example") == ("an example", 1)
    assert normalize_adjacent_duplicate_articles("a the example") == ("a the example", 0)


def test_article_normalization_applies_to_target_query_and_relation_intent() -> None:
    question = "When was the institute that owned The Collegian founded?"
    candidate = _candidate("musique", "m-article", question)
    candidate.plan["target"]["steps"][0]["subquery_template"] = (
        "The The Last Sleep of Arthur in Avalon was made by whom?"
    )
    q1, _ = build_action_pair(
        candidate, _musique_raw("m-article", question), split="train"
    )
    assert q1["target"]["query"].startswith("The Last")
    assert q1["target"]["relation_intent"].startswith("The Last")
    assert q1["source_provenance"]["adjacent_duplicate_article_normalization_count"] == 2
    assert set(q1["source_provenance"]["adjacent_duplicate_article_normalized_fields"]) == {
        "query", "relation_intent"
    }


def test_missing_intermediate_evidence_binding_is_rejected() -> None:
    question = "Where was the parent of Alpha born?"
    with pytest.raises(BuildReject, match="ambiguous_or_missing_intermediate_support"):
        build_action_pair(
            _candidate("2wikimultihopqa", "w1", question),
            _2wiki_raw("w1", question, first_sentence="Alpha had one parent."),
            split="train",
        )


def test_strict_linear_predicate_rejects_parallel_or_placeholder_free_second_step() -> None:
    candidate = _candidate("2wikimultihopqa", "w1", "Where was the parent of Alpha born?")
    row = {"dataset": candidate.dataset, **candidate.plan}
    assert is_clean_linear_two_step(row)
    row["target"]["steps"][1]["dependencies"] = []
    row["target"]["steps"][1]["subject"] = "Bridge"
    assert not is_clean_linear_two_step(row)


def test_unknown_candidate_exception_is_fail_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    question = "Where was the parent of Alpha born?"
    candidate = _candidate("2wikimultihopqa", "w1", question)

    def explode(*args, **kwargs):
        raise RuntimeError("unexpected implementation defect")

    monkeypatch.setattr(builder_module, "build_action_pair", explode)
    with pytest.raises(RuntimeError, match="unexpected implementation defect"):
        builder_module._select_pairs(
            groups={candidate.family: [candidate]},
            raw={candidate.qid: _2wiki_raw(candidate.qid, question)},
            dataset=candidate.dataset,
            split="train",
            quota=1,
            seed=42,
            additionally_excluded_families=set(),
        )


def test_tiny_release_is_deterministic_isolated_and_refuses_overwrite(tmp_path: Path) -> None:
    split_root = tmp_path / "splits"
    data_root = tmp_path / "data"
    questions = {
        # Cross-dataset qid collisions are legal and must still count as two
        # actions per dataset::qid rather than four actions per bare qid.
        ("2wikimultihopqa", "train"): ("same_train", "Where was the parent of Alpha born?"),
        ("2wikimultihopqa", "dev"): ("same_dev", "Where was Alpha's parent born?"),
        ("musique", "train"): ("same_train", "When was the institute owning The Collegian founded?"),
        ("musique", "dev"): ("same_dev", "When was The Collegian's owning institute founded?"),
    }
    split_rows = {"train": [], "dev": []}
    raw_rows = {dataset: [] for dataset in ("2wikimultihopqa", "musique")}
    for (dataset, split), (qid, question) in questions.items():
        candidate = _candidate(dataset, qid, question)
        split_rows[split].append(_planner_row(candidate))
        raw_rows[dataset].append(
            _2wiki_raw(qid, question) if dataset == "2wikimultihopqa" else _musique_raw(qid, question)
        )
    _write_jsonl(split_root / "train.jsonl", split_rows["train"])
    _write_jsonl(split_root / "dev.jsonl", split_rows["dev"])
    _write_jsonl(split_root / "confirmation.jsonl", [])
    _write_jsonl(split_root / "seen_diagnostics.jsonl", [])
    for dataset, rows in raw_rows.items():
        _write_jsonl(data_root / dataset / "train.jsonl", rows)

    candidate_report = build_candidate_pool(
        project_root=tmp_path,
        data_root=data_root,
        split_root=split_root,
        output_dir=tmp_path / "candidate_pool",
        experiment_id="TEST-CONTROLLER-V1-CANDIDATES",
        seed=42,
        consumed_cohorts=(),
    )
    for split in ("train", "dev"):
        for dataset in ("2wikimultihopqa", "musique"):
            assert candidate_report["capacity"][split][dataset]["valid_qids"] == 1
            assert candidate_report["capacity"][split][dataset]["action_rows"] == 2

    first = tmp_path / "release_a"
    second = tmp_path / "release_b"
    common = dict(
        project_root=tmp_path,
        data_root=data_root,
        split_root=split_root,
        experiment_id="TEST-CONTROLLER-V1",
        train_per_dataset=1,
        dev_per_dataset=1,
        seed=42,
        consumed_cohorts=(),
        allow_unfrozen_selection_for_tests=True,
    )
    report = build_release(output_dir=first, **common)
    build_release(output_dir=second, **common)
    assert report["all_release_gates_pass"] is True
    assert report["gold_boundary"]["hotpotqa"].startswith("UNKNOWN")
    assert (first / "train.jsonl").read_bytes() == (second / "train.jsonl").read_bytes()
    assert (first / "dev.jsonl").read_bytes() == (second / "dev.jsonl").read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_release(output_dir=first, **common)


def test_formal_release_exact_joins_all_three_protocol_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_root = tmp_path / "splits"
    data_root = tmp_path / "data"
    protocol_dir = tmp_path / "protocol"
    specs = {
        "train": {
            "2wikimultihopqa": ("wt", "Where was the parent of Alpha born?"),
            "musique": ("mt", "When was the institute owning The Collegian founded?"),
        },
        "confirmation": {
            "2wikimultihopqa": ("wc", "Which birthplace is linked through Alpha's parent?"),
            "musique": ("mc", "Which year saw the owner of The Collegian established?"),
        },
        "dev": {
            "2wikimultihopqa": ("wd", "Name the birthplace of Alpha's parent."),
            "musique": ("md", "Give the founding year of the institution behind The Collegian."),
        },
    }
    planner = {"train": [], "dev": []}
    raw = {dataset: [] for dataset in ("2wikimultihopqa", "musique")}
    locks = {split: [] for split in specs}
    for lock_split, datasets in specs.items():
        source_split = "train" if lock_split == "confirmation" else lock_split
        for dataset, (qid, question) in datasets.items():
            candidate = _candidate(dataset, qid, question)
            planner[source_split].append(_planner_row(candidate))
            raw[dataset].append(
                _2wiki_raw(qid, question)
                if dataset == "2wikimultihopqa"
                else _musique_raw(qid, question)
            )
            raw_row = (
                _2wiki_raw(qid, question)
                if dataset == "2wikimultihopqa"
                else _musique_raw(qid, question)
            )
            locks[lock_split].append(
                {
                    "dataset": dataset,
                    "qid": qid,
                    "question_key": question_key(dataset, qid),
                    "question": question,
                    "question_sha256": question_sha256(question),
                    "family_sha256": family_sha256(question),
                    "split": lock_split,
                    "action_pair_sha256": "PENDING_SOURCE_FILE_HASH",
                }
            )
    _write_jsonl(split_root / "train.jsonl", planner["train"])
    _write_jsonl(split_root / "dev.jsonl", planner["dev"])
    _write_jsonl(split_root / "confirmation.jsonl", [])
    _write_jsonl(split_root / "seen_diagnostics.jsonl", [])
    for dataset, rows in raw.items():
        _write_jsonl(data_root / dataset / "train.jsonl", rows)
    raw_by_qid = {
        dataset: {row["id"]: row for row in rows} for dataset, rows in raw.items()
    }
    for lock_split, rows in locks.items():
        source_split = "train" if lock_split == "confirmation" else lock_split
        planner_path = (split_root / f"{source_split}.jsonl").resolve()
        planner_sha = _file_sha256(planner_path)
        for lock in rows:
            base = _candidate(lock["dataset"], lock["qid"], lock["question"])
            candidate = PlannerCandidate(
                dataset=base.dataset,
                qid=base.qid,
                question=base.question,
                family=base.family,
                plan=base.plan,
                planner_path=str(planner_path),
                planner_sha256=planner_sha,
            )
            lock["action_pair_sha256"] = canonical_action_pair_sha256(
                build_action_pair(
                    candidate,
                    raw_by_qid[lock["dataset"]][lock["qid"]],
                    split=lock_split,
                )
            )
    cohort_locks = {}
    for split, rows in locks.items():
        path = protocol_dir / f"{split}.identity_only.jsonl"
        _write_jsonl(path, rows)
        cohort_locks[split] = {
            "path": path.name,
            "sha256": _file_sha256(path),
            "rows": len(rows),
        }
    (protocol_dir / "protocol.json").write_text(
        json.dumps({"cohort": {"cohort_locks": cohort_locks}}), encoding="utf-8"
    )

    # This test isolates exact action-pair materialization.  The formal bundle
    # validator has its own real-asset/tamper checks; do not weaken production
    # validation merely to accept this deliberately tiny synthetic protocol.
    synthetic_protocol = {
        "schema_version": builder_module.FORMAL_PROTOCOL_SCHEMA_VERSION,
        "experiment_id": builder_module.FORMAL_PROTOCOL_EXPERIMENT_ID,
        "status": builder_module.FORMAL_PROTOCOL_STATUS,
        "protocol_body_canonical_sha256": "b" * 64,
        "cohort": {"cohort_locks": cohort_locks},
        "implementation_locks": {},
        "probe_evaluation_contract": {},
    }

    def accept_synthetic_protocol(*args, **kwargs):
        return synthetic_protocol, [
            {
                "role": "protocol",
                "path": str(protocol_dir / "protocol.json"),
                "sha256": "a" * 64,
            }
        ]

    monkeypatch.setattr(
        builder_module, "_validate_formal_protocol", accept_synthetic_protocol
    )

    output = tmp_path / "locked_release"
    report = build_release(
        project_root=tmp_path,
        data_root=data_root,
        split_root=split_root,
        output_dir=output,
        experiment_id="TEST-LOCKED-CONTROLLER-V1",
        train_per_dataset=1,
        dev_per_dataset=1,
        confirmation_per_dataset=1,
        seed=42,
        consumed_cohorts=(),
        protocol_dir=protocol_dir,
        expected_protocol_sha256="a" * 64,
        allow_unfrozen_selection_for_tests=True,
    )
    assert report["selection"]["identity_authority"] == "frozen_protocol_exact_join"
    assert report["checks"]["cross_split_qid_overlap"] == 0
    assert report["checks"]["cross_split_family_overlap"] == 0
    for split in ("train", "dev", "confirmation"):
        rows = [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]
        assert len(rows) == 4
        assert {row["split"] for row in rows} == {split}
        assert {row["qid"] for row in rows} == {item["qid"] for item in locks[split]}


def test_formal_release_requires_external_protocol_sha256(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="external expected_protocol_sha256"):
        build_release(
            project_root=tmp_path,
            data_root=tmp_path / "data",
            split_root=tmp_path / "splits",
            output_dir=tmp_path / "release",
            protocol_dir=tmp_path / "protocol",
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"experiment_id": "WRONG"}, "Experiment ID mismatch"),
        ({"train_per_dataset": 599}, "quotas must be exactly"),
        ({"seed": 7}, "selection seed must be 42"),
    ],
)
def test_formal_release_rejects_runtime_identity_drift(
    tmp_path: Path, override: dict, message: str
) -> None:
    kwargs = {
        "project_root": tmp_path,
        "data_root": tmp_path / "data",
        "split_root": tmp_path / "splits",
        "output_dir": tmp_path / "release",
        "protocol_dir": tmp_path / "protocol",
        "expected_protocol_sha256": "a" * 64,
        **override,
    }
    with pytest.raises(ValueError, match=message):
        build_release(**kwargs)
