#!/usr/bin/env python
"""Validate Gold-free v6 materialization, then freeze scorer inputs.

Every input/code/model/Wiki18 lock and every materialization/mechanism gate is
checked before scorer Gold is opened.  Failed runs remain append-only and do
not produce answer-evaluation inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.utils.logging import dump_manifest, prepare_new_run_dir
from scripts.prepare import freeze_dependent_retrieval_v6 as v6_freeze
from scripts.prepare.finalize_dependent_retrieval_pilot import (
    _index_raw_gold,
    _read_jsonl,
    _sha256_text,
    _validate_and_enrich,
    _write_jsonl,
)


FINALIZER_VERSION = "dependent-retrieval-v6-finalizer-1"
EXPECTED_REPORT_SCHEMA = "plan-once-dependent-retrieval-v6-report-1"
EXPECTED_REPORT_STATUS = "COMPLETE_INPUTS_NOT_ANSWER_EVALUATED"
EVAL_PROTOCOL_SCHEMA = "dependent-retrieval-v6-eval-protocol-1"
DATASETS = ("hotpotqa", "musique")
EXPECTED_PER_DATASET = 30
EXPECTED_TOTAL = 60
EXPECTED_PASSAGES = 10
PROTECTED_PREFIX = 8

FORBIDDEN_KEYS = frozenset({
    "answer", "answers", "gold_answer", "gold_answers", "golden_answers",
    "supporting_facts", "supporting_titles", "question_decomposition",
    "decomposition", "evidence", "evidences", "target",
})


def _canonical_sha256(value: Any) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _passages_sha256(value: Any) -> str:
    """Match the historical v4/v5/v6 runner JSON identity exactly."""

    blob = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _assert_gold_free(value: Any, where: str) -> None:
    if isinstance(value, Mapping):
        bad = {str(key).casefold() for key in value} & FORBIDDEN_KEYS
        if bad:
            raise ValueError(f"Gold/support field before gate at {where}: {sorted(bad)}")
        for key, child in value.items():
            _assert_gold_free(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_gold_free(child, f"{where}[{index}]")


def _passage_key(value: Mapping[str, Any]) -> str:
    if value.get("id") is not None:
        return f"id:{value['id']}"
    normalize = lambda text: " ".join(re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold()).split())
    blob = f"{normalize(value.get('title'))}\n{normalize(value.get('contents') or value.get('text'))}"
    return "text:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _row_key(row: Mapping[str, Any]) -> str:
    return f"{str(row.get('dataset') or '').strip()}::{str(row.get('qid') or '').strip()}"


def _artifact_from_report(report: Mapping[str, Any], name: str) -> Path:
    item = (report.get("outputs") or {}).get(name)
    if not isinstance(item, Mapping):
        raise ValueError(f"retrieval report has no output lock: {name}")
    path = Path(str(item.get("path") or "")).expanduser().resolve()
    current = v6_freeze.file_lock(path)
    if current["sha256"] != str(item.get("sha256") or ""):
        raise ValueError(f"retrieval output bytes differ from report: {name}")
    if item.get("size_bytes") is not None and int(item["size_bytes"]) != current["size_bytes"]:
        raise ValueError(f"retrieval output size differs from report: {name}")
    return path


def _query_records(detail: Mapping[str, Any]) -> list[dict[str, Any]]:
    direct = detail.get("dependent_query_variants")
    if isinstance(direct, list):
        return [dict(row) for row in direct if isinstance(row, Mapping)]
    result: list[dict[str, Any]] = []
    for hop in detail.get("hops") or []:
        if not isinstance(hop, Mapping):
            continue
        if not bool(hop.get("dependencies")):
            continue
        rows = hop.get("query_variants") or hop.get("dependent_queries") or []
        if not isinstance(rows, list):
            continue
        for raw in rows:
            if isinstance(raw, Mapping):
                row = dict(raw)
                row.setdefault("hop_id", hop.get("hop_id"))
                result.append(row)
    return result


def _final_ce_pairs(detail: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = detail.get("final_ce_pairs")
    return [dict(row) for row in rows] if isinstance(rows, list) and all(isinstance(row, Mapping) for row in rows) else []


def _validate_query_trace(
    detail: Mapping[str, Any], question: str
) -> tuple[int, int, int, bool, bool]:
    """Return query count, duplicates, max variants, prefix-ok, atomic-ok."""

    rows = _query_records(detail)
    by_hop: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    duplicates = 0
    prefix_ok = True
    atomic_ok = True
    for index, row in enumerate(rows):
        query = str(row.get("query") or "")
        hop_id = str(row.get("hop_id") or "").strip()
        if not query or not hop_id:
            raise ValueError(f"dependent query telemetry lacks query/hop_id at variant {index}")
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
        if str(row.get("query_sha256") or "") != digest:
            raise ValueError("dependent query hash mismatch")
        if not query.startswith(question + "\n"):
            prefix_ok = False
        identity = (hop_id, digest)
        if identity in seen:
            duplicates += 1
        seen.add(identity)
        by_hop[hop_id] += 1
        hints = row.get("hints")
        hint_count = row.get("hint_count")
        if isinstance(hints, list):
            atomic_ok = atomic_ok and len(hints) <= 1
        elif hint_count is not None:
            atomic_ok = atomic_ok and int(hint_count) <= 1
        elif isinstance(row.get("hint"), Mapping):
            matched = row["hint"].get("matched_dependency_values") or []
            atomic_ok = atomic_ok and isinstance(matched, list) and len(matched) <= 1
        else:
            atomic_ok = atomic_ok and row.get("hint") is None
    return len(rows), duplicates, max(by_hop.values(), default=0), prefix_ok, atomic_ok


def _validate_ce_trace(detail: Mapping[str, Any], question: str, changed: bool) -> tuple[int, bool]:
    rows = _final_ce_pairs(detail)
    declared = int(detail.get("final_ce_pair_count") or 0)
    if rows and declared != len(rows):
        raise ValueError("final CE pair count differs from pair telemetry")
    if changed and declared <= 0:
        raise ValueError("changed row has no final CE pair count")
    # The v6 runner builds the global CE pair list in code but stores only the
    # count plus this invariant.  Exact code bytes are preregistered; if a
    # future runner also stores pairs, validate every pair below.
    if not rows:
        return declared, bool(detail.get("all_final_ce_pairs_use_exact_original_question"))
    keys: set[str] = set()
    exact = True
    for row in rows:
        query = str(row.get("query") or row.get("question") or "")
        if query != question:
            exact = False
        query_hash = row.get("query_sha256") or row.get("question_sha256")
        if query_hash is not None and str(query_hash) != hashlib.sha256(query.encode("utf-8")).hexdigest():
            raise ValueError("final CE question hash mismatch")
        document_key = str(row.get("document_key") or "")
        if not document_key or document_key in keys:
            raise ValueError("final CE pair has missing/duplicate document key")
        keys.add(document_key)
    return len(rows), exact


def audit_materialization(
    arm_a: Sequence[Mapping[str, Any]],
    arm_b: Sequence[Mapping[str, Any]],
    details_raw: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(arm_a) != EXPECTED_TOTAL or len(arm_b) != EXPECTED_TOTAL or len(details_raw) != EXPECTED_TOTAL:
        raise ValueError("v6 materialization must contain 60 A/B/detail rows")
    details = {_row_key(row): dict(row) for row in details_raw}
    if len(details) != EXPECTED_TOTAL or "::" in details:
        raise ValueError("execution-detail identity join is not one-to-one")
    counts = Counter(str(row.get("dataset") or "") for row in arm_a)
    if counts != Counter({dataset: EXPECTED_PER_DATASET for dataset in DATASETS}):
        raise ValueError("v6 materialization is not HotpotQA30 + MuSiQue30")

    aggregate = Counter()
    max_query_variants_observed = 0
    per_dataset: dict[str, Counter[str]] = {dataset: Counter() for dataset in DATASETS}
    question_keys: list[str] = []
    qids: list[str] = []
    common_fields = (
        "row_id", "question_key", "dataset", "qid", "question", "question_sha256",
        "split", "gold_access", "kg_subgraph", "legacy_kg_sha256",
    )
    for index, (left_raw, right_raw) in enumerate(zip(arm_a, arm_b)):
        left, right = dict(left_raw), dict(right_raw)
        _assert_gold_free(left, f"arm_a[{index}]")
        _assert_gold_free(right, f"arm_b[{index}]")
        key = _row_key(left)
        if key != _row_key(right) or key not in details or key in question_keys:
            raise ValueError(f"A/B/detail identity join failure at row {index}")
        question_keys.append(key)
        qids.append(str(left.get("qid") or ""))
        dataset = str(left.get("dataset") or "")
        for field in common_fields:
            if field not in left or left[field] != right.get(field):
                raise ValueError(f"paired common field differs: {key} {field}")
        if left.get("gold_access") is not False:
            raise ValueError(f"Gold access is not false: {key}")

        a_passages, b_passages = left.get("retrieved_passages"), right.get("retrieved_passages")
        if not isinstance(a_passages, list) or not isinstance(b_passages, list):
            raise ValueError(f"passages are not lists: {key}")
        if len(a_passages) != EXPECTED_PASSAGES or len(b_passages) != EXPECTED_PASSAGES:
            raise ValueError(f"top10 gate failed: {key}")
        if str(left.get("passages_sha256") or "") != _passages_sha256(a_passages):
            raise ValueError(f"Arm A passage hash mismatch: {key}")
        if str(right.get("passages_sha256") or "") != _passages_sha256(b_passages):
            raise ValueError(f"Arm B passage hash mismatch: {key}")
        a_keys = [_passage_key(row) for row in a_passages]
        b_keys = [_passage_key(row) for row in b_passages]
        duplicate_output = len(b_keys) - len(set(b_keys))
        if len(set(a_keys)) != EXPECTED_PASSAGES or duplicate_output:
            raise ValueError(f"duplicate passage gate failed: {key}")
        if a_passages[:PROTECTED_PREFIX] != b_passages[:PROTECTED_PREFIX]:
            raise ValueError(f"protected Arm-A prefix differs: {key}")

        detail = details[key]
        _assert_gold_free(detail, f"details[{key}]")
        if detail.get("gold_access") is not False:
            raise ValueError(f"detail Gold access is not false: {key}")
        if detail.get("error") is not None or detail.get("execution_status") == "fallback_execution_error":
            raise ValueError(f"runtime/execution error before Gold: {key}")
        exact = a_passages == b_passages
        if bool(right.get("fallback_to_a")) != exact:
            raise ValueError(f"fallback_to_a differs from exact passage equality: {key}")
        added, removed = set(b_keys) - set(a_keys), set(a_keys) - set(b_keys)
        if len(added) != len(removed) or len(added) > 2:
            raise ValueError(f"replacement budget/inventory failure: {key}")

        merge = detail.get("merge")
        if merge is None:
            if not exact or added or removed:
                raise ValueError(f"changed row has no merge telemetry: {key}")
            selected, evicted = [], []
        elif isinstance(merge, Mapping):
            selected = list(merge.get("selected_new") or [])
            evicted = list(merge.get("evicted_originals") or [])
        else:
            raise ValueError(f"invalid merge telemetry: {key}")
        selected_keys = {str(row.get("document_key") or "") for row in selected}
        evicted_keys = {str(row.get("document_key") or "") for row in evicted}
        if selected_keys != added or evicted_keys != removed or len(selected_keys) != len(selected) or len(evicted_keys) != len(evicted):
            raise ValueError(f"merge displacement inventory differs: {key}")
        for row in evicted:
            rank = int(row.get("original_rank", -1))
            if rank <= PROTECTED_PREFIX or rank > EXPECTED_PASSAGES:
                raise ValueError(f"protected/invalid displacement: {key}")
            if not float(row.get("replacement_score")) > float(row.get("score")):
                raise ValueError(f"non-improving replacement: {key}")

        query_count, query_duplicates, max_variants, prefix_ok, atomic_ok = _validate_query_trace(
            detail, str(left["question"])
        )
        if query_duplicates or max_variants > 2 or not prefix_ok or not atomic_ok:
            raise ValueError(f"dependent-query safety gate failed: {key}")
        max_query_variants_observed = max(max_query_variants_observed, max_variants)
        ce_count, ce_exact = _validate_ce_trace(detail, str(left["question"]), not exact)
        if not ce_exact:
            raise ValueError(f"final CE did not use exact original question: {key}")
        dependent_hops = {str(row.get("hop_id") or "") for row in _query_records(detail)}
        if any(str(row.get("hop_id") or "") not in dependent_hops for row in selected):
            raise ValueError(f"root-only or untraced document was selected: {key}")

        current = per_dataset[dataset]
        current.update({
            "n": 1,
            "plan_executable": int(bool(detail.get("plan_executable"))),
            "dependent_eligible": int(bool(detail.get("has_dependent_step"))),
            "dependent_nonempty": int(bool(detail.get("has_dependent_step")) and query_count > 0),
            "changed": int(not exact),
            "query_count": query_count,
            "final_ce_pair_count": ce_count,
        })
        aggregate.update({
            "n": 1, "changed": int(not exact), "fallback": int(exact),
            "selected": len(added), "queries": query_count, "ce_pairs": ce_count,
        })

    by_dataset: dict[str, dict[str, Any]] = {}
    for dataset, value in per_dataset.items():
        n, eligible = int(value["n"]), int(value["dependent_eligible"])
        by_dataset[dataset] = {
            **dict(value),
            "plan_executable_rate": int(value["plan_executable"]) / n,
            "dependent_hop_query_nonempty_rate": (
                int(value["dependent_nonempty"]) / eligible if eligible else None
            ),
            "retained_new_dependent_document_question_rate": int(value["changed"]) / n,
        }
    return {
        "n": EXPECTED_TOTAL,
        "identity_join_rate": 1.0,
        "runtime_errors": 0,
        "fallback_execution_error": 0,
        "all_rows_top10": True,
        "a_prefix8_exact_when_changed": True,
        "unauthorized_displacement": 0,
        "root_only_injection": 0,
        "duplicate_output_documents": 0,
        "duplicate_dependent_queries": 0,
        "all_dependent_queries_start_with_exact_question": True,
        "max_query_variants_per_logical_hop": max_query_variants_observed,
        "all_final_ce_pairs_use_exact_original_question": True,
        "fallback_exact": True,
        "changed_questions": int(aggregate["changed"]),
        "fallback_questions": int(aggregate["fallback"]),
        "selected_new_documents": int(aggregate["selected"]),
        "dependent_query_count": int(aggregate["queries"]),
        "final_ce_pair_count": int(aggregate["ce_pairs"]),
        "qid_order_sha256": _sha256_text("\n".join(qids)),
        "question_key_order_sha256": _sha256_text("\n".join(question_keys)),
        "by_dataset": by_dataset,
    }


def _validate_current_protocol_locks(protocol: Mapping[str, Any]) -> None:
    for section in ("inputs", "code"):
        for name, lock in (protocol.get(section) or {}).items():
            if v6_freeze.file_lock(Path(str(lock.get("path") or ""))) != dict(lock):
                raise ValueError(f"current {section} lock drifted: {name}")
    for name, identity in (protocol.get("models") or {}).items():
        path = Path(str(identity.get("path") or ""))
        if v6_freeze.artifact_identity(path) != identity:
            raise ValueError(f"current model identity drifted: {name}")
    for name, lock in (protocol.get("model_content_locks") or {}).items():
        path = Path(str(lock.get("path") or ""))
        current = v6_freeze.file_lock(path) if path.is_file() else v6_freeze.tree_lock(path)
        if current != lock:
            raise ValueError(f"current model content lock drifted: {name}")
    assets = protocol.get("retrieval_asset_content_locks") or {}
    if v6_freeze.file_lock(Path(str((assets.get("corpus") or {}).get("path") or ""))) != assets.get("corpus"):
        raise ValueError("current Wiki18 corpus lock drifted")
    if v6_freeze.file_lock(Path(str((assets.get("dense_index") or {}).get("path") or ""))) != assets.get("dense_index"):
        raise ValueError("current Wiki18 dense lock drifted")
    if v6_freeze.tree_lock(Path(str((assets.get("bm25_index") or {}).get("path") or ""))) != assets.get("bm25_index"):
        raise ValueError("current Wiki18 BM25 lock drifted")


def validate_preregistration(prereg_path: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    protocol = v6_freeze.read_json(prereg_path)
    if protocol.get("schema_version") != v6_freeze.SCHEMA_VERSION:
        raise ValueError("unexpected v6 preregistration schema")
    if protocol.get("status") != v6_freeze.STATUS or protocol.get("scope") != v6_freeze.SCOPE:
        raise ValueError("v6 preregistration is not the frozen same60 combination")
    if protocol.get("experiment_ids") != v6_freeze.EXPERIMENT_IDS:
        raise ValueError("v6 Experiment IDs differ")
    if str(report.get("experiment_id") or "") != v6_freeze.EXPERIMENT_IDS["materialization"]:
        raise ValueError("materialization Experiment ID differs from preregistration")
    lock = v6_freeze.file_lock(prereg_path)
    if report.get("preregistration") != lock:
        raise ValueError("report preregistration lock differs")
    runtime = report.get("runtime_locks") or {}
    for name in ("inputs", "code", "models", "settings"):
        if runtime.get(name) != protocol.get(name):
            raise ValueError(f"runtime {name} locks differ from preregistration")
    _validate_current_protocol_locks(protocol)
    return {
        "document": protocol,
        "lock": lock,
        "section_sha256": {
            name: _canonical_sha256(protocol.get(name))
            for name in (
                "inputs", "code", "models", "model_content_locks",
                "retrieval_assets", "retrieval_asset_content_locks", "settings",
                "decision_gates",
            )
        },
    }


def _enforce_report(report: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    if report.get("schema_version") != EXPECTED_REPORT_SCHEMA or report.get("status") != EXPECTED_REPORT_STATUS:
        raise ValueError("unexpected/incomplete v6 retrieval report")
    if report.get("gold_access") is not False or report.get("development_only") is not True:
        raise ValueError("v6 report violates Gold/development boundary")
    if report.get("settings") is None:
        raise ValueError("v6 report has no settings")
    summary = report.get("safety_summary") or {}
    report_names = {
        "all_rows_top10": "all_top10",
        "a_prefix8_exact_when_changed": "prefix8_exact",
        "unauthorized_displacement": "unauthorized_original_displacements",
        "root_only_injection": "root_passages_injected",
    }
    derived_only = {"fallback_execution_error", "identity_join_rate"}
    for name, expected in v6_freeze.MATERIALIZATION_GATES.items():
        report_name = report_names.get(name, name)
        if name == "max_query_variants_per_logical_hop":
            report_ok = int(summary.get(report_name, -1)) <= int(expected)
            observed_ok = int(observed.get(name, -1)) <= int(expected)
        else:
            report_ok = name in derived_only or summary.get(report_name) == expected
            observed_ok = observed.get(name) == expected
        if not report_ok or not observed_ok:
            raise ValueError(f"Gold-free materialization gate failed: {name}")
    by_dataset = report.get("by_dataset") or {}
    for dataset in DATASETS:
        reported, computed = by_dataset.get(dataset) or {}, observed["by_dataset"][dataset]
        if int(reported.get("n", -1)) != EXPECTED_PER_DATASET:
            raise ValueError(f"dataset count differs: {dataset}")
        for name, minimum in {
            "plan_executable_rate": 0.8,
            "dependent_hop_query_nonempty_rate": 0.8,
            "retained_new_dependent_document_question_rate": 0.5,
        }.items():
            if reported.get(name) != computed.get(name):
                raise ValueError(f"reported/recomputed mechanism metric differs: {dataset} {name}")
            if reported.get(name) is None or float(reported[name]) < minimum:
                raise ValueError(f"Gold-free mechanism gate failed: {dataset} {name}")


def validate_gold_free_materialization(
    *, report_path: Path, prereg_path: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Complete every pre-Gold check and return verified A/B rows."""

    report = v6_freeze.read_json(report_path)
    prereg = validate_preregistration(prereg_path, report)
    arm_a_path, arm_b_path, details_path = (
        _artifact_from_report(report, "arm_a"),
        _artifact_from_report(report, "arm_b"),
        _artifact_from_report(report, "execution_details"),
    )
    arm_a, arm_b, details = _read_jsonl(arm_a_path), _read_jsonl(arm_b_path), _read_jsonl(details_path)
    observed = audit_materialization(arm_a, arm_b, details)
    if report.get("settings") != prereg["document"].get("settings"):
        raise ValueError("runtime settings differ from preregistration")
    _enforce_report(report, observed)
    return report, prereg, arm_a, arm_b


def _validate_v4_controls(
    protocol: Mapping[str, Any], prereg: Mapping[str, Any], arm_a_path: Path, observed: Mapping[str, Any]
) -> None:
    if protocol.get("decision_gates") != {
        "pooled_net_correct_gain_min": 3,
        "max_net_correct_loss_per_dataset": 1,
        "parse_count_delta_min": 0,
        "plan_executable_rate_min_each_dataset": 0.8,
        "second_hop_query_nonempty_rate_min_each_dataset": 0.8,
        "new_dependent_candidate_question_rate_min_each_dataset": 0.5,
    }:
        raise ValueError("v4 evaluation gates drifted")
    if str(((protocol.get("inputs") or {}).get("retrieval_arm_a_no_gold") or {}).get("sha256") or "") != v6_freeze.sha256_file(arm_a_path):
        raise ValueError("v6 Arm A bytes differ from frozen v4")
    if protocol.get("qid_order_sha256") != observed.get("qid_order_sha256") or protocol.get("question_key_order_sha256") != observed.get("question_key_order_sha256"):
        raise ValueError("v6 population order differs from frozen v4")
    models = prereg.get("models") or {}
    if models.get("strong_sft") != (protocol.get("models") or {}).get("strong_sft"):
        raise ValueError("strong SFT differs from frozen v4")
    if models.get("base_model") != protocol.get("base_model"):
        raise ValueError("base model differs from frozen v4")
    if (prereg.get("settings") or {}).get("generation") != protocol.get("generation"):
        raise ValueError("generation controls differ from frozen v4")


def _scorer_gold_locks(v4: Mapping[str, Any], paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    locks = (v4.get("inputs") or {}).get("scorer_gold") or {}
    result: dict[str, dict[str, Any]] = {}
    for dataset, raw_path in zip(DATASETS, paths):
        path = raw_path.expanduser().resolve()
        expected = locks.get(dataset)
        current = v6_freeze.file_lock(path)
        if not isinstance(expected, Mapping) or str(expected.get("path")) != str(path) or expected.get("sha256") != current["sha256"]:
            raise ValueError(f"scorer Gold differs from frozen v4: {dataset}")
        result[dataset] = current
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval_report", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--v4_frozen_protocol", type=Path, default=Path("outputs/audits/plan_once_dependent_retrieval_pilot30x2_freeze_v4/protocol.json"))
    parser.add_argument("--hotpot_dev", type=Path, default=Path("data/hotpotqa/dev.jsonl"))
    parser.add_argument("--musique_dev", type=Path, default=Path("data/musique/dev.jsonl"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--evaluation_experiment_id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out, freeze_id = prepare_new_run_dir(args.output_dir, experiment_id=args.experiment_id, extra={
        "phase": "dependent_retrieval_v6_post_materialization_freeze",
        "scope": v6_freeze.SCOPE, "gold_access_during_retrieval": False,
    })
    try:
        report_path, prereg_path = args.retrieval_report.resolve(), args.preregistration.resolve()
        # Complete all content hashes and materialization/mechanism gates before
        # any raw scorer-Gold file is opened.
        report, prereg_bundle, arm_a, arm_b = validate_gold_free_materialization(
            report_path=report_path, prereg_path=prereg_path
        )
        protocol = prereg_bundle["document"]
        ids = protocol["experiment_ids"]
        if (
            args.experiment_id != ids["post_materialization_freeze"]
            or args.evaluation_experiment_id != ids["answer_evaluation"]
        ):
            raise ValueError("finalizer/evaluator Experiment IDs differ from preregistration")
        arm_a_path = _artifact_from_report(report, "arm_a")
        arm_b_path = _artifact_from_report(report, "arm_b")
        details_path = _artifact_from_report(report, "execution_details")
        observed = audit_materialization(arm_a, arm_b, _read_jsonl(details_path))
        v4_path = args.v4_frozen_protocol.resolve()
        v4 = v6_freeze.read_json(v4_path)
        _validate_v4_controls(v4, protocol, arm_a_path, observed)

        # Gold is opened only below this line, after every prior check passed.
        gold_paths = [args.hotpot_dev.resolve(), args.musique_dev.resolve()]
        scorer_locks = _scorer_gold_locks(v4, gold_paths)
        gold_index = _index_raw_gold(gold_paths)
        final_a, final_b = _validate_and_enrich(arm_a, arm_b, gold_index)
        # Preserve the upstream v6 arm label as provenance while retaining the
        # frozen evaluator's A/B arm vocabulary and telemetry field names.
        detail_index = {
            _row_key(row): row for row in _read_jsonl(details_path)
        }
        for row in final_b:
            row["source_arm"] = row.get("arm")
            row["arm"] = "B_dependent"
            trace = dict(row.get("retrieval_trace") or {})
            trace["second_hop_query_count"] = int(
                detail_index[_row_key(row)].get("second_hop_query_count") or 0
            )
            row["retrieval_trace"] = trace

        a_path, b_path = out / "arm_a.scored.jsonl", out / "arm_b.scored.jsonl"
        _write_jsonl(a_path, final_a)
        _write_jsonl(b_path, final_b)
        eval_protocol = {
            "schema_version": EVAL_PROTOCOL_SCHEMA,
            "status": "FROZEN_AFTER_RETRIEVAL_BEFORE_ANSWER_GENERATION",
            "scope": "ADAPTIVE_DEVELOPMENT_COMBINATION; SAME60_CONSUMED",
            "experiment_id": args.evaluation_experiment_id,
            "freeze_experiment_id": freeze_id,
            "materialization_experiment_id": report["experiment_id"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "gold_access_during_retrieval": False,
            "n": EXPECTED_TOTAL,
            "qid_order_sha256": observed["qid_order_sha256"],
            "question_key_order_sha256": observed["question_key_order_sha256"],
            "inputs": {
                "arm_a": v6_freeze.file_lock(a_path), "arm_b": v6_freeze.file_lock(b_path),
                "retrieval_report": v6_freeze.file_lock(report_path),
                "retrieval_arm_a_no_gold": v6_freeze.file_lock(arm_a_path),
                "retrieval_arm_b_no_gold": v6_freeze.file_lock(arm_b_path),
                "execution_details_no_gold": v6_freeze.file_lock(details_path),
                "preregistration": prereg_bundle["lock"],
                "v4_frozen_protocol": v6_freeze.file_lock(v4_path),
                "scorer_gold": scorer_locks,
            },
            "models": {"strong_sft": protocol["models"]["strong_sft"]},
            "base_model": protocol["models"]["base_model"],
            "generation": protocol["settings"]["generation"],
            "decision_gates": {
                "pooled_net_correct_gain_min": 3,
                "max_net_correct_loss_per_dataset": 1,
                "parse_count_delta_min": 0,
                "plan_executable_rate_min_each_dataset": 0.8,
                "second_hop_query_nonempty_rate_min_each_dataset": 0.8,
                "new_dependent_candidate_question_rate_min_each_dataset": 0.5,
            },
            "answer_utility_gates_unchanged": dict(v6_freeze.ANSWER_UTILITY_GATES),
            "materialization_safety_observed": observed,
            "preregistration_section_sha256": prereg_bundle["section_sha256"],
            "code": {
                "finalizer_v6": v6_freeze.file_lock(Path(__file__)),
                "answer_evaluator": protocol["code"]["answer_evaluator"],
            },
            "scientific_boundary": (
                "Same consumed development 60 only; combined-v6 utility only. A pass "
                "permits one fresh confirmation with a budget-matched expansion arm."
            ),
        }
        protocol_path = out / "protocol.json"
        protocol_path.write_text(json.dumps(eval_protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        dump_manifest(out, status="FROZEN_READY_FOR_LOCAL_ANSWER_EVALUATION", extra={
            "experiment_id": freeze_id,
            "phase": "dependent_retrieval_v6_post_materialization_freeze",
            "protocol_sha256": v6_freeze.sha256_file(protocol_path),
        })
        print(json.dumps({"status": "FROZEN_READY_FOR_LOCAL_ANSWER_EVALUATION", "protocol": str(protocol_path)}, indent=2))
    except Exception as exc:
        dump_manifest(out, status="FAILED_RUNTIME", extra={
            "experiment_id": freeze_id,
            "failure": {"type": type(exc).__name__, "message": str(exc)},
        })
        raise


if __name__ == "__main__":
    main()
