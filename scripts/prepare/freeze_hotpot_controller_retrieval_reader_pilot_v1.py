#!/usr/bin/env python
"""Freeze the Hotpot Controller Gold-free retrieval/Reader pilot.

This CPU-only freezer may run only after the generation/dual-review stage has
materialised its append-only answer-free producer projection.  It reads only
``producer_proposals.jsonl``, ``report.json``, and ``manifest.json`` from that
stage.  It never opens ``accepted_actions.jsonl``, reviewer-2 output, raw
HotpotQA annotations, or a Gold-bound q2 query.

Accepted proposal rows are projected again to the minimal runtime schema and
bound to the already verified model/index hashes from the formal v8 full-
content lock.  The freezer makes no retrieval, model, GPU, network, scoring,
or training call.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgproweight.kg.question_kg import question_sha256  # noqa: E402
from scripts.pilot import run_hotpot_controller_retrieval_reader_pilot_v1 as runner  # noqa: E402


SCHEMA_VERSION = runner.PROTOCOL_SCHEMA_VERSION
REPORT_SCHEMA_VERSION = "hotpot-controller-retrieval-reader-freeze-report-v1"
MANIFEST_SCHEMA_VERSION = "hotpot-controller-retrieval-reader-freeze-manifest-v1"
STATUS = runner.PROTOCOL_STATUS
EXPERIMENT_ID = runner.EXPERIMENT_ID
GENERATION_STATUS = "COMPLETE_GENERATION_DUAL_REVIEW_RETRIEVAL_NOT_RUN_NOT_TRAINED"
GENERATION_ROWS = 30
GENERATION_ACCEPTED_MIN = 24

# No formal generation directory or experiment ID is hard-coded here.  The
# first generation protocol was superseded before calls because its masked
# second-hop subject was not explicit on every item.  A caller must supply the
# append-only successor directory and ID after that run exists.
DEFAULT_PARENT_V8_PROTOCOL = Path(
    "outputs/audits/subquestion_decomposition_v8_development90_"
    "implementation_freeze_rrf100_seed42_v2/protocol.json"
)
DEFAULT_PARENT_V8_MANIFEST = DEFAULT_PARENT_V8_PROTOCOL.with_name("manifest.json")
DEFAULT_OUTPUT_DIR = runner.DEFAULT_PROTOCOL_DIR
SOURCE_FILES_READ_EXACT = (
    "producer_proposals.jsonl",
    "report.json",
    "manifest.json",
)
SOURCE_FILES_FORBIDDEN = (
    "accepted_actions.jsonl",
    "reviewer_2_reviews.jsonl",
    "reviewer_1_reviews.jsonl",
    "failures.jsonl",
    "semantic_call_ledger.jsonl",
    "api_transport_attempt_ledger.jsonl",
    "api_call_wal.jsonl",
)

SOURCE_REQUIRED_FIELDS = {
    "schema_version",
    "dataset",
    "qid",
    "question",
    "question_sha256",
    "stage",
    "status",
    "semantic_request_id",
    "requested_model",
    "response_model",
    "finish_reason",
    "raw_response_content",
    "raw_response_sha256",
    "parsed_response",
    "parsed_response_sha256",
    "nonce_echo_count",
    "reject_code",
    "detail_code",
    "final_item_status",
    "dual_review_unanimous_pass",
    "q1_query",
    "q2_template",
    "proposal_sha256",
    "runtime_projection_gold_or_observation_fields_present",
}
ACCEPTED_STATUS = "accepted_generation_and_dual_review"

CODE_PATHS = {
    "freezer": Path(
        "scripts/prepare/freeze_hotpot_controller_retrieval_reader_pilot_v1.py"
    ),
    "runner": Path("scripts/pilot/run_hotpot_controller_retrieval_reader_pilot_v1.py"),
    "v8_runtime": Path("scripts/pilot/materialize_dynamic_decomposition_v8.py"),
    "v8_production_driver": Path("scripts/pilot/run_dynamic_decomposition_v8.py"),
    "canonical_subqa_v9": Path("kgproweight/retrieval/canonical_subqa_v9.py"),
    "rank_first_binder_v9_1": Path("kgproweight/retrieval/canonical_subqa_v9_1.py"),
    "dependency_substitution": Path("kgproweight/retrieval/dependent.py"),
    "canonical_prompts": Path("kgproweight/data/prompts.py"),
    "canonical_parsers": Path("kgproweight/data/parsers.py"),
}


class HotpotRuntimeFreezeError(ValueError):
    """A prerequisite or frozen artifact violates the intended protocol."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_lock(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"required nonempty file missing: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _assert_file_lock(lock: Mapping[str, Any], *, label: str) -> None:
    if not isinstance(lock, Mapping):
        raise HotpotRuntimeFreezeError(f"{label} lock malformed")
    current = _file_lock(Path(str(lock.get("path") or "")))
    if any(current[key] != lock.get(key) for key in current):
        raise HotpotRuntimeFreezeError(f"{label} content drift")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HotpotRuntimeFreezeError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n") or not line.strip():
                raise HotpotRuntimeFreezeError(
                    f"invalid JSONL framing at {path}:{line_number}"
                )
            value = json.loads(line)
            if not isinstance(value, dict):
                raise HotpotRuntimeFreezeError(
                    f"JSONL row is not an object at {path}:{line_number}"
                )
            rows.append(value)
    return rows


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as handle:
        handle.write(_canonical_bytes(dict(value), newline=True))


def _write_jsonl_new(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("xb") as handle:
        for row in rows:
            handle.write(_canonical_bytes(dict(row), newline=True))


def _resolve(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _manifest_output_lock(manifest: Mapping[str, Any], filename: str) -> Mapping[str, Any]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise HotpotRuntimeFreezeError("generation manifest outputs missing")
    matches = [
        item
        for item in outputs
        if isinstance(item, Mapping) and item.get("path") == filename
    ]
    if len(matches) != 1:
        raise HotpotRuntimeFreezeError(
            f"generation manifest has no unique lock for {filename}"
        )
    return matches[0]


def _validate_generation_artifacts(
    *,
    generation_dir: Path,
    generation_experiment_id: str,
    generation_status: str,
    expected_rows: int,
    accepted_min: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = {name: generation_dir / name for name in SOURCE_FILES_READ_EXACT}
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(
            "generation answer-free projection/report/manifest are not all materialised"
        )
    report = _load_json(paths["report.json"])
    manifest = _load_json(paths["manifest.json"])
    if (
        report.get("experiment_id") != generation_experiment_id
        or report.get("status") != generation_status
        or int(report.get("fixed_denominator", -1)) != expected_rows
        or int(report.get("dual_review_accepted", -1)) < accepted_min
        or (report.get("scientific_boundary") or {}).get("retrieval_or_reader_calls") != 0
        or (report.get("scientific_boundary") or {}).get("training_started") is not False
    ):
        raise HotpotRuntimeFreezeError("generation report has not passed its frozen gate")
    if (
        manifest.get("experiment_id") != generation_experiment_id
        or manifest.get("status") != generation_status
        or int(manifest.get("fixed_denominator", -1)) != expected_rows
        or manifest.get("retrieval_calls") != 0
        or manifest.get("training_started") is not False
    ):
        raise HotpotRuntimeFreezeError("generation manifest boundary mismatch")
    for filename in ("producer_proposals.jsonl", "report.json"):
        bound = _manifest_output_lock(manifest, filename)
        actual = _file_lock(paths[filename])
        if (
            bound.get("sha256") != actual["sha256"]
            or int(bound.get("size_bytes", -1)) != actual["size_bytes"]
        ):
            raise HotpotRuntimeFreezeError(f"generation output lock drift: {filename}")

    source_rows = _load_jsonl(paths["producer_proposals.jsonl"])
    if len(source_rows) != expected_rows:
        raise HotpotRuntimeFreezeError("producer projection fixed denominator drift")
    if len({str(row.get("qid")) for row in source_rows}) != expected_rows:
        raise HotpotRuntimeFreezeError("producer projection qids are not unique")
    accepted: list[dict[str, Any]] = []
    observed_source_fields: set[str] | None = None
    for index, source in enumerate(source_rows):
        if not SOURCE_REQUIRED_FIELDS.issubset(source):
            raise HotpotRuntimeFreezeError(
                f"producer projection lacks required runtime fields at row {index}"
            )
        current_fields = set(source)
        if observed_source_fields is None:
            observed_source_fields = current_fields
        elif current_fields != observed_source_fields:
            raise HotpotRuntimeFreezeError("producer projection row schemas are inconsistent")
        if (
            source.get("dataset") != "hotpotqa"
            or source.get("stage") != "producer"
            or source.get("runtime_projection_gold_or_observation_fields_present")
            is not False
        ):
            raise HotpotRuntimeFreezeError(
                f"producer projection boundary violation at row {index}"
            )
        question = source.get("question")
        if not isinstance(question, str) or question_sha256(question) != source.get(
            "question_sha256"
        ):
            raise HotpotRuntimeFreezeError(
                f"producer projection question hash mismatch at row {index}"
            )
        is_accepted = (
            source.get("final_item_status") == ACCEPTED_STATUS
            and source.get("dual_review_unanimous_pass") is True
            and source.get("nonce_echo_count") == 0
        )
        if not is_accepted:
            if any(
                source.get(field) is not None
                for field in ("q1_query", "q2_template", "proposal_sha256")
            ):
                raise HotpotRuntimeFreezeError(
                    "rejected producer row exposes a runtime query projection"
                )
            continue
        proposal = source.get("parsed_response")
        if (
            not isinstance(proposal, Mapping)
            or set(proposal) != {"schema_version", "q1", "q2_template"}
            or proposal.get("schema_version")
            != "hotpot-controller-query-proposal-v1"
            or source.get("q1_query") != proposal.get("q1")
            or source.get("q2_template") != proposal.get("q2_template")
        ):
            raise HotpotRuntimeFreezeError("accepted producer proposal join mismatch")
        proposal_hash = _canonical_sha256(proposal)
        if (
            source.get("proposal_sha256") != proposal_hash
            or source.get("parsed_response_sha256") != proposal_hash
        ):
            raise HotpotRuntimeFreezeError("accepted producer proposal hash mismatch")
        projected = {
            "schema_version": runner.INPUT_SCHEMA_VERSION,
            "dataset": "hotpotqa",
            "qid": str(source["qid"]),
            "question": question,
            "question_sha256": str(source["question_sha256"]),
            "q1_query": str(source["q1_query"]),
            "q1_query_sha256": _sha256_text(str(source["q1_query"])),
            "q2_template": str(source["q2_template"]),
            "q2_template_sha256": _sha256_text(str(source["q2_template"])),
            "proposal_sha256": proposal_hash,
            "source_projection_row_sha256": _canonical_sha256(source),
        }
        # Reuse the production runner's exact answer-free schema checks before
        # committing the reduced projection.
        accepted.append(runner._validate_runtime_input(projected))
    if len(accepted) != int(report.get("dual_review_accepted", -1)):
        raise HotpotRuntimeFreezeError(
            "accepted producer projection count differs from generation report"
        )
    if len(accepted) < accepted_min:
        raise HotpotRuntimeFreezeError("accepted runtime projection is below pilot gate")
    return accepted, {
        "generation_dir": str(generation_dir.resolve()),
        "fixed_denominator": expected_rows,
        "accepted_rows": len(accepted),
        "rejected_rows": expected_rows - len(accepted),
        "source_locks": {name: _file_lock(path) for name, path in paths.items()},
        "source_files_read_exact": list(SOURCE_FILES_READ_EXACT),
        "source_files_forbidden_and_not_read": list(SOURCE_FILES_FORBIDDEN),
        "observed_source_fields": sorted(observed_source_fields or ()),
    }


def _asset_snapshot(parent_protocol: Mapping[str, Any]) -> dict[str, Any]:
    content_section = parent_protocol.get("content_reverification") or {}
    content = content_section.get("content") or {}
    models = content.get("models") or {}
    wiki18 = content.get("wiki18") or {}
    required_models = ("base_model", "strong_sft", "retrieval_encoder", "cross_encoder")
    if any(not isinstance(models.get(name), Mapping) for name in required_models):
        raise HotpotRuntimeFreezeError("parent v8 model asset locks missing")
    if any(not isinstance(wiki18.get(name), Mapping) for name in ("corpus", "dense_index", "bm25_index")):
        raise HotpotRuntimeFreezeError("parent v8 Wiki18 asset locks missing")
    role = parent_protocol.get("generation_role_identity") or {}
    model_identity = {
        "base_model_tree_sha256": str(models["base_model"]["tree_sha256"]),
        "adapter_tree_sha256": str(models["strong_sft"]["tree_sha256"]),
        "tokenizer_tree_sha256": str(models["base_model"]["tree_sha256"]),
    }
    retrieval_identity = {
        "corpus_sha256": str(wiki18["corpus"]["sha256"]),
        "dense_index_sha256": str(wiki18["dense_index"]["sha256"]),
        "bm25_tree_sha256": str(wiki18["bm25_index"]["tree_sha256"]),
        "e5_tree_sha256": str(models["retrieval_encoder"]["tree_sha256"]),
        "bge_tree_sha256": str(models["cross_encoder"]["tree_sha256"]),
    }
    if (
        role.get("same_base_tree_sha256") != model_identity["base_model_tree_sha256"]
        or role.get("same_adapter_tree_sha256") != model_identity["adapter_tree_sha256"]
    ):
        raise HotpotRuntimeFreezeError("parent v8 role/model identity mismatch")
    return {
        "model_asset_identity": model_identity,
        "retrieval_asset_identity": retrieval_identity,
        "content_locks": deepcopy(content),
        "tokenizer_and_chat_template": deepcopy(
            parent_protocol.get("tokenizer_and_chat_template") or {}
        ),
    }


def freeze_protocol(
    *,
    project_root: Path = PROJECT_ROOT,
    generation_dir: Path,
    generation_experiment_id: str,
    generation_status: str = GENERATION_STATUS,
    parent_v8_protocol_path: Path = DEFAULT_PARENT_V8_PROTOCOL,
    parent_v8_manifest_path: Path = DEFAULT_PARENT_V8_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    expected_rows: int = GENERATION_ROWS,
    accepted_min: int = GENERATION_ACCEPTED_MIN,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    if not isinstance(generation_experiment_id, str) or not generation_experiment_id.strip():
        raise HotpotRuntimeFreezeError("generation_experiment_id must be explicit")
    if not isinstance(generation_status, str) or not generation_status.strip():
        raise HotpotRuntimeFreezeError("generation_status must be explicit")
    generation_dir = _resolve(project_root, generation_dir)
    parent_v8_protocol_path = _resolve(project_root, parent_v8_protocol_path)
    parent_v8_manifest_path = _resolve(project_root, parent_v8_manifest_path)
    output_dir = _resolve(project_root, output_dir)
    if output_dir.exists():
        raise FileExistsError(f"append-only protocol directory exists: {output_dir}")
    accepted, generation = _validate_generation_artifacts(
        generation_dir=generation_dir,
        generation_experiment_id=generation_experiment_id,
        generation_status=generation_status,
        expected_rows=expected_rows,
        accepted_min=accepted_min,
    )
    parent_protocol = _load_json(parent_v8_protocol_path)
    parent_manifest = _load_json(parent_v8_manifest_path)
    parent_protocol_lock = _file_lock(parent_v8_protocol_path)
    if (
        parent_protocol.get("status")
        != "AUTHORIZED_SMOKE_THEN_CONDITIONAL_GOLD_FREE_DEVELOPMENT90"
        or parent_manifest.get("status") != parent_protocol.get("status")
        or parent_protocol.get("gold_access") is not False
        or parent_protocol.get("content_reverification", {}).get(
            "full_hash_verification_performed_by_this_command"
        )
        is not True
        or not isinstance(parent_manifest.get("protocol"), Mapping)
        or any(
            parent_manifest["protocol"].get(field) != parent_protocol_lock[field]
            for field in ("size_bytes", "sha256")
        )
    ):
        raise HotpotRuntimeFreezeError("parent v8 full-content protocol is invalid")
    assets = _asset_snapshot(parent_protocol)
    code_locks = {
        name: _file_lock(_resolve(project_root, path)) for name, path in CODE_PATHS.items()
    }
    parent_locks = {
        "generation_producer_projection": generation["source_locks"][
            "producer_proposals.jsonl"
        ],
        "generation_report": generation["source_locks"]["report.json"],
        "generation_manifest": generation["source_locks"]["manifest.json"],
        "v8_full_content_protocol": parent_protocol_lock,
        "v8_full_content_manifest": _file_lock(parent_v8_manifest_path),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    runtime_input_path = output_dir / "runtime_inputs.answer_free.jsonl"
    _write_jsonl_new(runtime_input_path, accepted)
    runtime_input_lock = {
        **_file_lock(runtime_input_path),
        "row_count": len(accepted),
        "schema_version": runner.INPUT_SCHEMA_VERSION,
        "fields_exact": list(runner._INPUT_FIELDS),
        "source_fixed_denominator": expected_rows,
        "gold_access": False,
    }
    created = generated_at_utc or _utc_now()
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": created,
        "status": STATUS,
        "gold_access": False,
        "scope": "HOTPOTQA_TRAIN_SIDE_SILVER_PILOT_ACCEPTED_ROWS_ONLY",
        "authorization": {
            "runtime_inference": True,
            "retrieval": True,
            "reader_generation": True,
            "answer_scoring": False,
            "gold_or_train_annotation_join": False,
            "accepted_actions_or_annotation_observation_read": False,
            "training": False,
            "reward_or_loss_change": False,
        },
        "runtime_input": runtime_input_lock,
        "runtime_contract": runner.runtime_contract(),
        "runtime_decision_gate": {
            "source_fixed_denominator": expected_rows,
            "candidate_min": accepted_min,
            "generation_rejected_and_runtime_failed_rows_retained": True,
            "runtime_candidate_definition": (
                "generation_dual_review_pass AND q1_prediction_v9_1_bound AND "
                "predicted_observation_q2_executed AND final_reader_output_parsed"
            ),
            "final_release_decision": (
                "pending_separate_train_annotation_retrieval_support_scorer; "
                "final accepted must be >=24/30 and pass every gate"
            ),
        },
        "model_asset_identity": assets["model_asset_identity"],
        "retrieval_asset_identity": assets["retrieval_asset_identity"],
        "frozen_full_content_locks": assets["content_locks"],
        "tokenizer_and_chat_template": assets["tokenizer_and_chat_template"],
        "code_locks": code_locks,
        "parent_locks": parent_locks,
        "source_generation_accounting": generation,
        "source_generation_experiment_id": generation_experiment_id,
        "source_generation_status": generation_status,
        "gates": {
            "all_runtime_inputs_and_failures_accounted_on_fixed_denominator": True,
            "q1_top10_exact": True,
            "q1_binding_failure_has_no_q2_or_final": True,
            "q2_observation_source_exact": (
                "strong_sft_reader_prediction_verified_by_v9_1_rank_first_binding"
            ),
            "gold_bridge_runtime_correction_allowed": False,
            "final_passages_exact": (
                "one_bound_q1_then_up_to_nine_q2_novel_then_q1_backfill_to_ten"
            ),
            "same_strong_sft_runtime_for_q1_and_final": True,
            "runtime_pass_candidates_min": accepted_min,
            "runtime_failure_replacement_allowed": False,
        },
        "scientific_boundary": (
            "This protocol freezes a Gold-free runtime mechanics pilot over labels that "
            "were selected and reviewed using train annotations. It authorizes no Gold "
            "join, retrieval-support scoring, EM/F1/IHR, training, reward, or loss change."
        ),
    }
    protocol["protocol_body_canonical_sha256"] = _canonical_sha256(protocol)
    protocol_path = output_dir / "protocol.json"
    _write_json_new(protocol_path, protocol)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": created,
        "status": STATUS,
        "source_fixed_denominator": expected_rows,
        "runtime_input_rows": len(accepted),
        "source_rejected_rows": expected_rows - len(accepted),
        "generation_gate_accepted_min": accepted_min,
        "generation_gate_pass": len(accepted) >= accepted_min,
        "source_files_read_exact": list(SOURCE_FILES_READ_EXACT),
        "source_files_forbidden_and_not_read": list(SOURCE_FILES_FORBIDDEN),
        "gpu_calls": 0,
        "retrieval_calls": 0,
        "model_calls": 0,
        "training_started": False,
        "runtime_status": "NOT_RUN",
    }
    report_path = output_dir / "report.json"
    _write_json_new(report_path, report)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": created,
        "status": STATUS,
        "outputs": {
            "runtime_input": _file_lock(runtime_input_path),
            "protocol": _file_lock(protocol_path),
            "report": _file_lock(report_path),
        },
        "source_files_read_exact": list(SOURCE_FILES_READ_EXACT),
        "source_files_forbidden_and_not_read": list(SOURCE_FILES_FORBIDDEN),
        "gold_access": False,
        "gpu_calls": 0,
        "retrieval_calls": 0,
        "model_calls": 0,
        "training_started": False,
    }
    _write_json_new(output_dir / "manifest.json", manifest)
    return {
        "protocol": protocol,
        "report": report,
        "manifest": manifest,
        "output_dir": output_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation_dir", type=Path, required=True)
    parser.add_argument("--generation_experiment_id", required=True)
    parser.add_argument("--generation_status", default=GENERATION_STATUS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = freeze_protocol(
        generation_dir=args.generation_dir,
        generation_experiment_id=args.generation_experiment_id,
        generation_status=args.generation_status,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
