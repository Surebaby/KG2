"""Iterative prefetch closure regression tests."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare.run_inference_proofkg_closure import (
    _extract_new_property_requests,
    _seed_requested,
    _sort_cache,
)


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _hop(entity_qid, pid, matches, hop_index=1):
    return {
        "hop_index": hop_index,
        "pids": [pid],
        "input_entities": [{"qid": entity_qid, "abstained": False}],
        "matches": matches,
    }


def test_seed_requested_is_dataset_scoped(tmp_path):
    reqs = tmp_path / "missing_requests.jsonl"
    _write_jsonl(reqs, [
        {"request_type": "historical_property", "dataset": "2wikimultihopqa", "entity_qid": "Q1", "pid": "P17"},
        {"request_type": "historical_property", "dataset": "musique", "entity_qid": "Q2", "pid": "P17"},
        {"request_type": "title_or_qid_resolution", "dataset": "2wikimultihopqa", "anchor_surface": "X"},
    ])
    requested = _seed_requested(reqs, "2wikimultihopqa")
    assert requested == {("Q1", "P17")}  # musique + title requests excluded


def test_extract_negative_not_repeated(tmp_path):
    rd = tmp_path / "rd.jsonl"
    _write_jsonl(rd, [{"qid": "dev_1", "execution": {"hops": [_hop("Q1", "P17", [])]}}])
    requested = {("Q1", "P17")}
    assert _extract_new_property_requests(rd, requested, "2wikimultihopqa") == []


def test_extract_exposes_next_hop_after_previous_filled(tmp_path):
    rd = tmp_path / "rd.jsonl"
    _write_jsonl(rd, [{"qid": "dev_1", "execution": {"hops": [_hop("Q2", "P569", [])]}}])
    requested = {("Q1", "P17")}  # only first hop was requested before
    new = _extract_new_property_requests(rd, requested, "2wikimultihopqa")
    assert len(new) == 1
    assert new[0]["entity_qid"] == "Q2" and new[0]["pid"] == "P569"
    assert ("Q2", "P569") in requested


def test_extract_skips_hops_with_matches(tmp_path):
    rd = tmp_path / "rd.jsonl"
    _write_jsonl(rd, [{"qid": "dev_1", "execution": {"hops": [_hop("Q1", "P17", [["X", "country", "Y"]])]}}])
    assert _extract_new_property_requests(rd, set(), "2wikimultihopqa") == []


def test_extract_skips_hops_without_resolved_input(tmp_path):
    rd = tmp_path / "rd.jsonl"
    _write_jsonl(rd, [{"qid": "dev_1", "execution": {"hops": [{
        "hop_index": 1, "pids": ["P17"],
        "input_entities": [{"qid": None, "abstained": True}],
        "matches": [],
    }]}}])
    assert _extract_new_property_requests(rd, set(), "2wikimultihopqa") == []


def test_extract_dataset_parameterized(tmp_path):
    rd = tmp_path / "rd.jsonl"
    _write_jsonl(rd, [{"qid": "dev_1", "execution": {"hops": [_hop("Q2", "P569", [])]}}])
    new = _extract_new_property_requests(rd, set(), "hotpotqa")
    assert new[0]["dataset"] == "hotpotqa"


def test_sort_cache_deterministic(tmp_path):
    p = tmp_path / "cache.jsonl"
    _write_jsonl(p, [
        {"qid": "Q3", "key": "v::2020::Q3", "revision": {}},
        {"qid": "Q1", "key": "v::2020::Q1", "revision": {}},
        {"qid": "Q2", "key": "v::2020::Q2", "revision": {}},
    ])
    _sort_cache(p)
    lines = [json.loads(l)["qid"] for l in p.read_text().splitlines() if l.strip()]
    assert lines == ["Q1", "Q2", "Q3"]
