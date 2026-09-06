import json
from pathlib import Path

import pytest

from scripts.prepare import freeze_sft_v3_candidates_v1 as pool
from scripts.prepare import freeze_sft_v3_protected_ledger_v1 as ledger


def write_source(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def letters(number):
    return chr(97 + number // 26) + chr(97 + number % 26)


def raw_sources(tmp_path, n=150):
    result = {}
    for dataset in pool.DATASETS:
        path = tmp_path / (dataset + ".jsonl")
        write_source(path, [{"id": str(i), "question": f"what does {dataset} {letters(i)} tell us about rivers?",
                            "golden_answers": [f"SECRET_{dataset}_{i}", "A distinct alias"],
                            "metadata": {"supporting_facts": "DO_NOT_EMIT"}} for i in range(n)])
        result[dataset] = path
    return result


def fixture_ledger(tmp_path, sources):
    protected_path = tmp_path / "protected.jsonl"
    write_source(protected_path, [{"dataset": "hotpotqa", "qid": "p", "question": "where is this hidden observatory?"}])
    spec = ledger.SourceSpec(protected_path, "heldout", ledger.sha_file(protected_path))
    output = tmp_path / "ledger"
    ledger.freeze_ledger(output_dir=output, specs=[spec], experiment_id="test-ledger", raw_train_sources=sources)
    return output


def test_selection_is_order_invariant_and_balanced(tmp_path):
    sources = raw_sources(tmp_path)
    candidates, _ = pool.collect_safe_candidates(raw_sources=sources, protected=[], seed=42)
    first, _ = pool.select_candidates(candidates, train_per_dataset=3, validation_per_dataset=2)
    second, _ = pool.select_candidates(list(reversed(candidates)), train_per_dataset=3, validation_per_dataset=2)
    assert first == second
    assert len(first) == 15
    assert [r["dataset"] for r in first[:6]] == list(pool.DATASETS) * 2
    assert [r["split"] for r in first] == ["train"] * 9 + ["validation"] * 6
    assert all(pool.family_split(r["family_sha256"], 42) == r["split"] for r in first)
    assert all(pool.verify_isolation(first, [])["gates"].values())


def test_changing_answers_does_not_change_identity_selection(tmp_path):
    sources = raw_sources(tmp_path)
    candidates, _ = pool.collect_safe_candidates(raw_sources=sources, protected=[], seed=42)
    before, _ = pool.select_candidates(candidates, train_per_dataset=3, validation_per_dataset=2)
    for path in sources.values():
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["golden_answers"] = ["COMPLETELY_CHANGED_CHECKER_LABEL"]
        write_source(path, rows)
    candidates, _ = pool.collect_safe_candidates(raw_sources=sources, protected=[], seed=42)
    after, _ = pool.select_candidates(candidates, train_per_dataset=3, validation_per_dataset=2)
    assert [(r["question_key"], r["split"]) for r in before] == [(r["question_key"], r["split"]) for r in after]


def test_cross_domain_template_has_one_owner(tmp_path):
    sources = raw_sources(tmp_path)
    for dataset, path in sources.items():
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows.append({"id": "shared", "question": "who wrote Silver Lake?", "golden_answers": ["author"]})
        write_source(path, rows)
    candidates, _ = pool.collect_safe_candidates(raw_sources=sources, protected=[], seed=42)
    selected, stats = pool.select_candidates(candidates, train_per_dataset=3, validation_per_dataset=2)
    assert stats["cross_dataset_family_groups_before_owner_assignment"] == 1
    assert sum(stats["by_dataset"][ds].get("family_owned_by_other_dataset", 0) for ds in pool.DATASETS) == 2
    assert len({r["family_sha256"] for r in selected}) == len(selected)


def test_shortage_never_redraws_partition(tmp_path):
    sources = raw_sources(tmp_path, n=2)
    candidates, _ = pool.collect_safe_candidates(raw_sources=sources, protected=[], seed=42)
    with pytest.raises(ValueError, match="no redraw"):
        pool.select_candidates(candidates, train_per_dataset=4, validation_per_dataset=4)


def test_gold_free_requests_are_frozen_before_faithful_same_line_labels(tmp_path):
    sources = raw_sources(tmp_path)
    protected = fixture_ledger(tmp_path, sources)
    out = tmp_path / "pool"
    report = pool.freeze_candidates(output_dir=out, protected_ledger_dir=protected, raw_sources=sources,
                                    experiment_id="test-pool", train_per_dataset=3, validation_per_dataset=2)
    assert report["candidate_questions"] == 15
    assert all(report["gates"].values())
    requests_text = (out / "retrieval_requests.question_only.jsonl").read_text()
    assert "SECRET_" not in requests_text
    assert "DO_NOT_EMIT" not in requests_text
    before = json.loads((out / "before_gold_labels.json").read_text())
    assert before["status"] == "QUESTION_ONLY_SELECTION_FROZEN_BEFORE_LABEL_COPY"
    for name, binding in before["selection_outputs"].items():
        assert ledger.sha_file(out / name) == binding["sha256"]
    labels = [json.loads(line) for line in (out / "labels.checker_only.jsonl").read_text().splitlines()]
    for label in labels:
        raw = json.loads(sources[label["dataset"]].read_text().splitlines()[label["source"]["line_number"] - 1])
        assert label["golden_answers"] == raw["golden_answers"]
        assert label["qid"] == raw["id"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert all(ledger.sha_file(out / name) == binding["sha256"] for name, binding in manifest["outputs"].items())
    with pytest.raises(FileExistsError):
        pool.freeze_candidates(output_dir=out, protected_ledger_dir=protected, raw_sources=sources,
                               experiment_id="test-pool", train_per_dataset=3, validation_per_dataset=2)


def test_label_copy_rejects_same_qid_modified_raw_content(tmp_path):
    sources = raw_sources(tmp_path)
    candidates, _ = pool.collect_safe_candidates(raw_sources=sources, protected=[], seed=42)
    selected, _ = pool.select_candidates(candidates, train_per_dataset=1, validation_per_dataset=1)
    first = selected[0]
    path = sources[first["dataset"]]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[first["source"]["line_number"] - 1]["golden_answers"] = ["unauthorized change"]
    write_source(path, rows)
    with pytest.raises(ValueError, match="same-line"):
        pool.copy_selected_labels(selected=selected, raw_sources=sources)


def test_incomplete_or_failed_ledger_cannot_be_used(tmp_path):
    sources = raw_sources(tmp_path)
    protected = fixture_ledger(tmp_path, sources)
    (protected / "FAILED.json").write_text("{}")
    with pytest.raises(ValueError, match="incomplete or failed"):
        pool.load_ledger(protected)


def test_ledger_source_hash_blocks_raw_mutation_before_candidate_freeze(tmp_path):
    sources = raw_sources(tmp_path)
    protected = fixture_ledger(tmp_path, sources)
    path = sources[pool.DATASETS[0]]
    path.write_text(path.read_text() + "\n")
    out = tmp_path / "failed"
    with pytest.raises(ValueError, match="differs from protected-ledger"):
        pool.freeze_candidates(output_dir=out, protected_ledger_dir=protected, raw_sources=sources,
                               experiment_id="test-pool", train_per_dataset=1, validation_per_dataset=1)
    assert (out / "FAILED.json").exists()
    assert not (out / "manifest.json").exists()


def test_duplicate_or_protected_selected_family_fails():
    item = ledger.identity({"dataset": "hotpotqa", "qid": "a", "question": "who directed Silver Lake?"})
    selected = [{**item, "split": "train", "source_split": "train", "evaluation_eligible": False},
                {**item, "qid": "b", "split": "validation", "source_split": "train", "evaluation_eligible": False}]
    with pytest.raises(ValueError, match="isolation"):
        pool.verify_isolation(selected, [])
    with pytest.raises(ValueError, match="isolation"):
        pool.verify_isolation(selected[:1], [item])
