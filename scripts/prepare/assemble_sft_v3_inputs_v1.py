#!/usr/bin/env python3
"""Versioned SFT-v3 random-plus-verified-graph inputs; never a teacher generator.

freeze preserves the original random candidates and appends family-disjoint
source-backed graph candidates.  materialize joins immutable real retrieval
batches to that full pool, exports a ready snapshot and a deterministic pending
request list.  Gold labels are copied to a separate checker-only file after the
model-input/identity snapshot is written.  Old graph-supply passages remain
provenance only: their older release lacks full corpus/index SHA attestation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from kgproweight.data.sft_v3_contract import build_sft_v3_messages
from kgproweight.kg.question_kg import question_sha256
from kgproweight.reward.source_integrity_v1 import validate_source_integrity_v1
from kgproweight.reward.source_quality_gate_v1 import canonical_sha256
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256
from scripts.prepare.freeze_sft_v3_candidates_v1 import family_split
from scripts.prepare.freeze_sft_v3_protected_ledger_v1 import identity as question_identity, make_index, overlap_reasons
from scripts.prepare.project_sft_v3_retrieval_requests_v1 import project
from scripts.prepare import materialize_sft_v3_retrieval_v1 as retrieval

ROOT = Path(__file__).resolve().parents[2]
VERSION = "sft-v3-combined-inputs-v1"
STATUS = "COMPLETE_COMBINED_IDENTITIES_AND_CHECKER_LABELS_NOT_TEACHER_DATA_NOT_TRAINED"
IDENTITY_FIELDS = ("question_key", "dataset", "qid", "question", "question_sha256", "family_sha256")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def bind(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": retrieval.sha(path), "bytes": path.stat().st_size}


def verify_binding(value: dict) -> Path:
    path = Path(value["path"]).resolve()
    if retrieval.sha(path) != value["sha256"] or ("bytes" in value and path.stat().st_size != value["bytes"]) or ("size_bytes" in value and path.stat().st_size != value["size_bytes"]):
        raise ValueError(f"bound artifact differs: {path}")
    return path


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(retrieval.canonical(row) + "\n")


def output_path(manifest: dict, name: str) -> Path:
    return verify_binding(manifest["outputs"][name])


def manifest(path: Path) -> dict:
    if (path.parent / "FAILED.json").exists():
        raise ValueError("failed release cannot be consumed")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_evidence(evidence: dict) -> None:
    if evidence.get("schema_version") != "qid-source-evidence-v1" or evidence.get("gold_used") is not False or not evidence.get("bindings"):
        raise ValueError("typed source evidence missing or not Gold-free")
    for path, digest in evidence["bindings"].items():
        verify_binding({"path": str(ROOT / path), "sha256": digest})


def validate_graph_assignments(rows: list[dict], evidence: dict, *, check_bindings: bool = True) -> None:
    if check_bindings:
        verify_evidence(evidence)
    seen = set()
    for row in rows:
        if row["question_key"] in seen:
            raise ValueError("duplicate graph assignment")
        seen.add(row["question_key"])
        record = row["fullsource_record"]
        if any(record.get(k) != row.get(k) for k in IDENTITY_FIELDS if k != "family_sha256"):
            raise ValueError("KG record assigned to another question")
        if canonical_sha256(record) != row["source_record_sha256"]:
            raise ValueError("original full source record digest mismatch")
        kg = record.get("kg_subgraph")
        if not isinstance(kg, list) or not 1 <= len(kg) <= 12:
            raise ValueError("nonempty original graph must fit max12 without truncation")
        result = validate_source_integrity_v1(record, evidence)
        if result != row["source_check"] or result["status"] != "PASS" or result["clearance"] is not True:
            raise ValueError("source-integrity PASS must reproduce from exact typed evidence")
        if row["family_sha256"] != family_sha256(row["question"]) or row["split"] != family_split(row["family_sha256"], 42):
            raise ValueError("graph question/family split differs from shared frozen rule")


def combine_identities(original: list[dict], graph_attached: list[dict], graph_supplement: list[dict]) -> tuple[list[dict], list[dict]]:
    """No quality/Gold draw: preserve original order; append fixed supplement."""
    by_key = {row["question_key"]: row for row in original}
    if len(by_key) != len(original):
        raise ValueError("duplicate original question identity")
    combined = deepcopy(original)
    keys = set(by_key); hashes = {r["question_sha256"] for r in original}; families = {r["family_sha256"] for r in original}
    if len(hashes) != len(original) or len(families) != len(original):
        raise ValueError("original pool is not question/global-family unique")
    for graph in graph_attached:
        original_row = by_key.get(graph["question_key"])
        if original_row is None or any(original_row[k] != graph[k] for k in IDENTITY_FIELDS) or original_row["split"] != graph["split"]:
            raise ValueError("attached KG must match the exact original question and split")
    # This deterministic ordering is declared before checker labels are read.
    supplements = sorted(graph_supplement, key=lambda r: (r["split"] != "train", canonical_sha256(["sft-v3-graph-supplement-order-v1", 42, r["family_sha256"]]), r["question_key"]))
    counts = Counter((r["split"], r["dataset"]) for r in original)
    for g in supplements:
        if g["question_key"] in keys or g["question_sha256"] in hashes or g["family_sha256"] in families:
            raise ValueError("supplement overlaps original or another supplement by qid/question/global family")
        item = {k: deepcopy(g[k]) for k in IDENTITY_FIELDS}
        item.update(schema_version=VERSION, family_version=FAMILY_VERSION, split=g["split"], source_split="train", gold_access=False,
                    role="sft_v3_source_backed_graph_supplement", evaluation_eligible=False, teacher_acceptance_pending=True,
                    graph_capacity_provenance="pass_disjoint_graph_supplement.inputs.jsonl")
        counts[(item["split"], item["dataset"])] += 1
        item["within_split_dataset_rank"] = counts[(item["split"], item["dataset"])]
        item["consumption_order"] = len(combined) + 1
        item["selection_rank"] = canonical_sha256(["sft-v3-graph-supplement-order-v1", 42, item["family_sha256"]])
        combined.append(item)
        keys.add(item["question_key"]); hashes.add(item["question_sha256"]); families.add(item["family_sha256"])
    requests = [project(r) for r in combined]
    retrieval.validate_requests(requests)
    return combined, requests


def labels_for_supplement(raw_path: Path, raw_binding: dict, supplements: list[dict]) -> list[dict]:
    """Read only train answers AFTER the full identity pool is frozen."""
    verify_binding(raw_binding)
    wanted = {r["qid"]: r for r in supplements}; result = {}
    with raw_path.open("rb") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            raw = json.loads(line); qid = str(raw.get("id", ""))
            if qid not in wanted:
                continue
            q = wanted[qid]
            if not isinstance(raw.get("question"), str) or raw["question"].strip() != q["question"] or question_sha256(raw["question"]) != q["question_sha256"] or qid in result:
                raise ValueError("supplement/raw-train identity is missing, duplicate or conflicting")
            answers = raw.get("golden_answers")
            if not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
                raise ValueError("raw train answer list is invalid; never repair labels")
            result[qid] = {"schema_version": VERSION + "-labels", "question_key": q["question_key"], "dataset": q["dataset"], "qid": qid,
                "question_sha256": q["question_sha256"], "family_sha256": q["family_sha256"], "split": q["split"],
                "role": "post_generation_checker_only_never_teacher_input", "golden_answers": deepcopy(answers),
                "source_question_outer_whitespace_normalized": raw["question"] != q["question"],
                "source": {**raw_binding, "line_number": lineno, "line_bytes_sha256": hashlib.sha256(line).hexdigest()}}
    if set(result) != set(wanted):
        raise ValueError("missing raw train labels for graph supplement")
    verify_binding(raw_binding)
    return [result[r["qid"]] for r in supplements]


def freeze(*, candidate_manifest: Path, graph_capacity_manifest: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError("combined release must be a new directory")
    parent = manifest(candidate_manifest); capacity = manifest(graph_capacity_manifest)
    if parent.get("complete") is not True or parent.get("status") != "COMPLETE_CANDIDATE_IDENTITIES_AND_CHECKER_LABELS_FROZEN_NOT_SFT_DATA" or not all(parent.get("gates", {}).values()):
        raise ValueError("original random candidate release is not complete")
    if capacity.get("status") != "COMPLETE_OFFLINE_SOURCE_CAPACITY_NOT_ACCEPTED_TEACHER_DATA_NOT_TRAINED":
        raise ValueError("unexpected graph-capacity release")
    parent_rows_path = output_path(parent, "candidates.question_only.jsonl")
    parent_protocol_path = output_path(parent, "protocol.json")
    parent_protocol = json.loads(parent_protocol_path.read_text())
    evidence_path = output_path(capacity, "qid_source_evidence.json"); evidence = json.loads(evidence_path.read_text()); verify_evidence(evidence)
    attached = read_rows(output_path(capacity, "pass_exact_random_pool_attachment.inputs.jsonl"))
    supplement = read_rows(output_path(capacity, "pass_disjoint_graph_supplement.inputs.jsonl"))
    source_checks = {r["question_key"]: r for r in read_rows(output_path(capacity, "source_checks.all335.jsonl"))}
    assignments = []
    for g in [*attached, *supplement]:
        pure = validate_source_integrity_v1(g["fullsource_record"], evidence)
        cached = source_checks[g["question_key"]]
        if any(cached.get(k) != v for k, v in pure.items()) or cached.get("new_ledger_blocked") is not False:
            raise ValueError("capacity source check did not reproduce or was ledger-blocked")
        assignments.append({**{k: deepcopy(g[k]) for k in IDENTITY_FIELDS}, "split": g["split"], "fullsource_record": deepcopy(g["fullsource_record"]),
            "source_record_sha256": g["source_record_sha256"], "source_check": pure,
            "assignment_kind": "exact_random_question" if g in attached else "disjoint_supplement", "gold_access": False,
            "historical_passages_provenance_only": {"retrieved_passages": g["retrieved_passages"], "passages_sha256": g["original_passages_sha256"]}})
    validate_graph_assignments(assignments, evidence)
    original = read_rows(parent_rows_path)
    if len(original) != parent["candidate_questions"]:
        raise ValueError("parent candidate count mismatch")
    combined, requests = combine_identities(original, attached, supplement)
    protected_path = verify_binding(parent_protocol["protected_identities"])
    protected_index = make_index(read_rows(protected_path))
    if any(overlap_reasons(question_identity(r), protected_index) for r in combined):
        raise ValueError("combined pool overlaps the complete SFT protected ledger")
    out.mkdir(parents=True, exist_ok=False)
    try:
        write_rows(out / "candidates.question_only.jsonl", combined)
        write_rows(out / "requests.question_only.jsonl", requests)
        write_rows(out / "graph_supplement.question_only.jsonl", combined[len(original):])
        write_rows(out / "graph_supplement.requests.question_only.jsonl", requests[len(original):])
        write_json(out / "qid_source_evidence.json", evidence)
        evidence_binding = bind(out / "qid_source_evidence.json")
        for row in assignments:
            row["source_evidence_binding"] = deepcopy(evidence_binding)
        write_rows(out / "graph_assignments.jsonl", assignments)
        protocol = {"schema_version": VERSION, "experiment_id": "SFT-V3-COMBINED-RANDOM-PLUS-VERIFIED-GRAPH-20260906-V1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "parent_manifest": bind(candidate_manifest), "graph_capacity_manifest": bind(graph_capacity_manifest), "parent_protocol": bind(parent_protocol_path),
            "protected_identities": parent_protocol["protected_identities"], "original_candidates": len(original), "supplement_candidates": len(supplement),
            "total_candidates": len(combined), "graph_assignments": len(assignments), "exact_random_graph_attachments": len(attached),
            "ordering": "exact original random pool order unchanged; append graph supplement sorted by train-before-validation then fixed family hash", "seed": 42,
            "combination_experiment": True, "combination_variables": ["unchanged random three-domain candidate pool", "source-backed KG on exact existing qids", "family-disjoint graph supplement"],
            "teacher_input": {"passages": 10, "kg_max_triples": 12, "step_contract": [2, 5], "graph_admission": "exact original fullsource record, reexecuted source-integrity PASS with bound typed evidence"},
            "retrieval": retrieval.STACK, "historical_retrieval_reuse": False,
            "historical_retrieval_note": "same declared stack and BGE attestation exist, but the older graph supply lacks full corpus/index asset hashes; every combined candidate requires the new real canonical backend or a verified completed prefix batch from it",
            "labels": "original parent checker labels plus exact original raw-train answers for fixed supplement; no labels in question requests or teacher kwargs",
            "raw_question_label_join": "existing identity contract strips outer whitespace; full question SHA must also match; raw source line bytes are bound and normalization is flagged",
            "code": [bind(Path(__file__)), bind(Path(retrieval.__file__)), bind(ROOT / "kgproweight/data/sft_v3_contract.py"), bind(ROOT / "kgproweight/reward/source_integrity_v1.py"), bind(ROOT / "scripts/prepare/freeze_sft_v3_candidates_v1.py")],
            "frozen_before_labels": {name: bind(out / name) for name in ("candidates.question_only.jsonl", "requests.question_only.jsonl", "graph_assignments.jsonl", "qid_source_evidence.json")},
            "teacher_calls": 0, "model_updates": 0}
        write_json(out / "protocol.json", protocol)
        write_json(out / "before_checker_labels.json", {"protocol": bind(out / "protocol.json"), "identity_pool_frozen": True, "label_values_read_yet": False})
        labels = read_rows(output_path(parent, "labels.checker_only.jsonl"))
        by_key = {r["question_key"]: r for r in labels}
        if len(by_key) != len(original) or any(r["question_key"] not in by_key or any(by_key[r["question_key"]][k] != r[k] for k in ("dataset", "qid", "question_sha256", "family_sha256", "split")) for r in original):
            raise ValueError("parent labels do not exactly join frozen original identities")
        labels = [by_key[r["question_key"]] for r in original]
        raw_bound = parent_protocol["raw_train_sources"]["2wikimultihopqa"]
        labels += labels_for_supplement(Path(raw_bound["path"]), raw_bound, combined[len(original):])
        write_rows(out / "labels.checker_only.jsonl", labels)
        report = {"schema_version": VERSION, "experiment_id": protocol["experiment_id"], "status": STATUS, "complete": True,
            "candidate_questions": len(combined), "original_candidates": len(original), "supplement_candidates": len(supplement),
            "by_dataset_split": dict(Counter(r["dataset"] + "::" + r["split"] for r in combined)),
            "graph_by_split": dict(Counter(r["split"] for r in assignments)), "graph_assignments": len(assignments),
            "original_random_rows_unchanged": combined[:len(original)] == original, "global_families_unique": len({r["family_sha256"] for r in combined}) == len(combined),
            "protected_overlap": 0, "typed_source_PASS_reexecuted": True, "no_graph_truncation": True, "historical_passages_used_as_new_teacher_input": False,
            "teacher_calls": 0, "model_updates": 0, "qualified_sft_examples": 0, "fullmethod_training_ready": False}
        write_json(out / "report.json", report)
        write_json(out / "manifest.json", {**report, "outputs": {p.name: bind(p) for p in sorted(out.iterdir()) if p.is_file()}})
        return report
    except BaseException as exc:
        write_json(out / "FAILED.json", {"status": "FAILED_NOT_TRAINING_DATA", "type": type(exc).__name__, "error": str(exc)})
        raise


def load_real_retrieval_batches(directory: Path) -> tuple[list[tuple[dict, dict]], list[dict]]:
    """Only sealed real-backend batches; ignore currently unsealed writes."""
    protocol, requests = retrieval.verify(directory)
    if protocol.get("test_double_only") is not False or protocol.get("retrieval") != retrieval.STACK or protocol.get("full_wiki18_documents") != 21015324 or not protocol.get("assets"):
        raise ValueError("real full-asset-bound Wiki18 retrieval required")
    found, seals = [], []
    for seal_path in sorted((directory / "batches").glob("*.seal.json")):
        seal = json.loads(seal_path.read_text()); path = verify_binding(seal["batch"]); b = json.loads(path.read_text())
        try:
            offset = int(path.stem)
        except ValueError as exc:
            raise ValueError("invalid batch offset") from exc
        source_requests = requests[offset:offset + protocol["batch_size"]]
        if (offset < 0 or offset % protocol["batch_size"] or not source_requests
                or seal.get("protocol_sha256") != retrieval.sha(directory / "protocol.json")
                or b.get("protocol_sha256") != seal["protocol_sha256"]
                or b.get("request_sha256") != canonical_sha256(source_requests)
                or b.get("contexts_sha256") != canonical_sha256(b.get("contexts"))):
            raise ValueError("batch source request/protocol/content hash mismatch")
        attestation = b.get("backend_attestation") or {}
        if attestation.get("mode") != "real_full_wiki18" or attestation.get("fallback") is not False or attestation.get("load_succeeded") is not True or attestation.get("corpus_documents") != 21015324:
            raise ValueError("batch backend is not the declared real full Wiki18 backend")
        retrieval.validate_contexts(source_requests, b["contexts"])
        found.extend((row, bind(path)) for row in b["contexts"]); seals.append(bind(seal_path))
    return found, seals


def join_ready_inputs(requests: list[dict], graph_rows: list[dict], contexts: list[tuple[dict, dict]], *, graph_binding: dict, evidence_binding: dict) -> tuple[list[dict], list[dict]]:
    wanted = {r["question_key"]: r for r in requests}; available = {}; graphs = {r["question_key"]: r for r in graph_rows}
    for context, source_binding in contexts:
        key = context["question_key"]
        if key not in wanted or any(context.get(f) != wanted[key].get(f) for f in IDENTITY_FIELDS):
            raise ValueError("retrieval context outside exact frozen combined identities")
        if key in available:
            if available[key][0]["passages"] != context["passages"]:
                raise ValueError("same question has conflicting frozen retrieval contexts")
            continue
        # Ignore role/rank/schema from an older identical-question request;
        # identity, actual passages, source protocol and batch were validated.
        available[key] = context, source_binding
    ready, pending = [], []
    for r in requests:
        key = r["question_key"]
        if key not in available:
            pending.append(deepcopy(r)); continue
        context, source_binding = available[key]; graph = graphs.get(key)
        kg = deepcopy(graph["fullsource_record"]["kg_subgraph"]) if graph else []
        item = {**deepcopy(r), "schema_version": VERSION + "-teacher-input", "retrieved_passages": deepcopy(context["passages"]),
            "kg_subgraph": kg, "retrieval_binding": deepcopy(source_binding), "passages_sha256": canonical_sha256(context["passages"]),
            "gold_access": False, "graph_input": bool(kg)}
        if graph:
            item["kg_source_verification"] = {"status": "PASS", "binding": deepcopy(graph_binding), "source_evidence_binding": deepcopy(evidence_binding), "source_record_sha256": graph["source_record_sha256"]}
        build_sft_v3_messages(question=item["question"], retrieved_passages=item["retrieved_passages"], kg_triples=kg)
        item["input_sha256"] = canonical_sha256(item)
        ready.append(item)
    return ready, pending


def materialize(*, combined_manifest: Path, retrieval_runs: list[Path], out: Path) -> dict:
    if out.exists():
        raise FileExistsError("ready snapshot must be append-only in a new directory")
    combined = manifest(combined_manifest)
    if combined.get("status") != STATUS or combined.get("complete") is not True:
        raise ValueError("combined identity release incomplete")
    requests_path = output_path(combined, "requests.question_only.jsonl"); requests = read_rows(requests_path); retrieval.validate_requests(requests)
    graphs_path = output_path(combined, "graph_assignments.jsonl"); graphs = read_rows(graphs_path)
    evidence_path = output_path(combined, "qid_source_evidence.json"); evidence = json.loads(evidence_path.read_text())
    validate_graph_assignments(graphs, evidence)
    all_contexts, seals = [], []
    for directory in retrieval_runs:
        contexts, batch_seals = load_real_retrieval_batches(directory)
        all_contexts.extend(contexts); seals.extend(batch_seals)
    ready, pending = join_ready_inputs(requests, graphs, all_contexts, graph_binding=bind(graphs_path), evidence_binding=bind(evidence_path))
    if not ready:
        raise ValueError("no exact real retrieval contexts available yet")
    out.mkdir(parents=True, exist_ok=False)
    try:
        write_rows(out / "inputs.jsonl", ready)
        write_rows(out / "pending_retrieval_requests.question_only.jsonl", pending)
        ready_keys = {v["question_key"] for v in ready}
        write_rows(out / "ready.question_only.jsonl", [r for r in requests if r["question_key"] in ready_keys])
        protocol = {"schema_version": VERSION + "-ready-snapshot", "experiment_id": out.name, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "combined_manifest": bind(combined_manifest), "full_requests": bind(requests_path), "graph_assignments": bind(graphs_path), "typed_source_evidence": bind(evidence_path),
            "real_retrieval_protocols": [bind(d / "protocol.json") for d in retrieval_runs], "sealed_batches": seals,
            "ready_inputs": bind(out / "inputs.jsonl"), "pending_requests": bind(out / "pending_retrieval_requests.question_only.jsonl"),
            "scope": "ready subset snapshot; full frozen pool unchanged, no label-driven inclusion or redrawing",
            "code": [bind(Path(__file__))], "model_inputs_frozen_before_checker_label_read": True, "teacher_calls": 0, "model_updates": 0}
        write_json(out / "protocol.json", protocol)
        write_json(out / "before_checker_labels.json", {"protocol": bind(out / "protocol.json"), "model_inputs_frozen": True, "labels_read_yet": False})
        all_labels = read_rows(output_path(combined, "labels.checker_only.jsonl")); labels = {r["question_key"]: r for r in all_labels}
        selected_labels = []
        for row in ready:
            label = labels.get(row["question_key"])
            if not label or any(label.get(k) != row[k] for k in ("dataset", "qid", "question_sha256", "family_sha256", "split")):
                raise ValueError("ready input/checker label exact join mismatch")
            selected_labels.append(label)
        write_rows(out / "labels.checker_only.jsonl", selected_labels)
        report = {"schema_version": VERSION + "-ready-snapshot", "experiment_id": out.name,
            "status": "ALL_INPUTS_PREPARED_TEACHER_NOT_STARTED" if not pending else "PARTIAL_INPUTS_PREPARED_TEACHER_NOT_STARTED",
            "all_combined_inputs_prepared": not pending, "full_frozen_candidates": len(requests), "ready_inputs": len(ready), "pending_inputs": len(pending),
            "ready_by_dataset_split": dict(Counter(r["dataset"] + "::" + r["split"] for r in ready)), "ready_graph_by_split": dict(Counter(r["split"] for r in ready if r["kg_subgraph"])),
            "source_integrity_PASS_reexecuted": True, "original_random_order_preserved": True, "full_pool_redrawn": False,
            "qualified_sft_examples": 0, "fullmethod_training_ready": False, "teacher_calls": 0, "model_updates": 0}
        write_json(out / "report.json", report)
        write_json(out / "manifest.json", {**report, "outputs": {p.name: bind(p) for p in sorted(out.iterdir()) if p.is_file()}})
        return report
    except BaseException as exc:
        write_json(out / "FAILED.json", {"status": "FAILED_NOT_TRAINING_DATA", "type": type(exc).__name__, "error": str(exc)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze"); f.add_argument("--candidate_manifest", type=Path, required=True); f.add_argument("--graph_capacity_manifest", type=Path, required=True); f.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("materialize"); m.add_argument("--combined_manifest", type=Path, required=True); m.add_argument("--retrieval_run", type=Path, action="append", required=True); m.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(candidate_manifest=args.candidate_manifest, graph_capacity_manifest=args.graph_capacity_manifest, out=args.out) if args.command == "freeze" else materialize(combined_manifest=args.combined_manifest, retrieval_runs=args.retrieval_run, out=args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
