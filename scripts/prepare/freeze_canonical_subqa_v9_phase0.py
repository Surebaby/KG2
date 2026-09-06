#!/usr/bin/env python
"""Freeze the consumed-development, Gold-free canonical-subanswer v9 diagnostic."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pilot import run_canonical_subqa_v9_phase0 as runner  # noqa: E402


SCHEMA_VERSION = "canonical-subqa-v9-phase0-protocol-1"
MANIFEST_SCHEMA_VERSION = "canonical-subqa-v9-phase0-protocol-manifest-1"
STATUS = "AUTHORIZED_GOLD_FREE_CONSUMED_DEV90_DIAGNOSTIC"
OUTPUT_DIR = runner.PROTOCOL_DIR
PARENT_IMPLEMENTATION_PROTOCOL = Path(
    "outputs/audits/"
    "subquestion_decomposition_v8_development90_implementation_freeze_"
    "rrf100_seed42_v2/protocol.json"
)
PARENT_IMPLEMENTATION_MANIFEST = PARENT_IMPLEMENTATION_PROTOCOL.with_name("manifest.json")
EXPECTED_PARENT_PROTOCOL_SHA256 = "3539d956a893cad119d583bdc875919b2349c47e503af9f05a83c887ebc039be"
EXPECTED_PARENT_MANIFEST_SHA256 = "289d6ef40db546221a33af2fba31c016e15091a748ae8a85d87b4dd648ffb491"


class V9FreezeError(RuntimeError):
    pass


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_lock(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"append-only output already exists: {path}")
    path.write_bytes(_canonical_json_bytes(dict(value)))


def freeze() -> dict[str, Any]:
    if Path.cwd().resolve() != PROJECT_ROOT.resolve():
        raise V9FreezeError(f"run from project root: {PROJECT_ROOT}")
    destination = (PROJECT_ROOT / OUTPUT_DIR).resolve()
    if destination.exists():
        raise FileExistsError(f"append-only protocol directory exists: {destination}")

    source_paths = {
        "v8_consumed_rows": runner.SOURCE_ROWS_PATH,
        "v8_failed_report": runner.SOURCE_REPORT_PATH,
        "v8_failed_manifest": runner.SOURCE_FAILED_MANIFEST_PATH,
        "v8_parent_protocol": PARENT_IMPLEMENTATION_PROTOCOL,
        "v8_parent_manifest": PARENT_IMPLEMENTATION_MANIFEST,
    }
    source_locks = {
        name: _file_lock(PROJECT_ROOT / path) for name, path in source_paths.items()
    }
    expected_sources = {
        "v8_consumed_rows": runner.EXPECTED_SOURCE_ROWS_SHA256,
        "v8_failed_report": runner.EXPECTED_SOURCE_REPORT_SHA256,
        "v8_failed_manifest": runner.EXPECTED_SOURCE_FAILED_MANIFEST_SHA256,
        "v8_parent_protocol": EXPECTED_PARENT_PROTOCOL_SHA256,
        "v8_parent_manifest": EXPECTED_PARENT_MANIFEST_SHA256,
    }
    for name, expected in expected_sources.items():
        if source_locks[name]["sha256"] != expected:
            raise V9FreezeError(f"source lock mismatch: {name}")

    code_paths = {
        "runner": Path("scripts/pilot/run_canonical_subqa_v9_phase0.py"),
        "canonical_subqa": Path("kgproweight/retrieval/canonical_subqa_v9.py"),
        "canonical_prompts": Path("kgproweight/data/prompts.py"),
        "canonical_parser": Path("kgproweight/data/parsers.py"),
        "unchanged_v8_binder": Path("kgproweight/retrieval/dynamic_decomposition_v8.py"),
        "shared_hf_runtime": Path("scripts/pilot/materialize_dynamic_decomposition_v8.py"),
    }
    code_locks = {name: _file_lock(PROJECT_ROOT / path) for name, path in code_paths.items()}

    parent = json.loads((PROJECT_ROOT / PARENT_IMPLEMENTATION_PROTOCOL).read_text(encoding="utf-8"))
    models = (((parent.get("content_reverification") or {}).get("content") or {}).get("models") or {})
    if not isinstance(models.get("base_model"), dict) or not isinstance(
        models.get("strong_sft"), dict
    ):
        raise V9FreezeError("parent model locks are missing")
    model = {
        "base_model": models["base_model"],
        "strong_sft": models["strong_sft"],
        "checkpoint_role": "same strong legacy SFT used by v8 and canonical baseline",
        "decoding": {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "max_new_tokens": 512,
            "seed": 42,
        },
        "chat_template_sha256": (parent.get("generation_role_identity") or {}).get(
            "same_chat_template_sha256"
        ),
    }
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": runner.EXPERIMENT_ID,
        "status": STATUS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CONSUMED_V8_DEVELOPMENT90_POSTHOC_SINGLE_VARIABLE_DIAGNOSTIC",
        "gold_access": False,
        "answer_scoring": False,
        "retrieval_calls": 0,
        "training": False,
        "reward_or_loss_change": False,
        "prospective_opened_or_hashed": False,
        "authorization_basis": (
            "Researcher approved a low-cost test of evaluation-time decomposition on 2026-09-04."
        ),
        "research_question": (
            "Did v8 primarily fail because the strong final-answer SFT was forced into a bespoke "
            "one-line subanswer interface rather than its trained canonical QA interface?"
        ),
        "single_variable": {
            "old": "bespoke one-line q1 answer or NO_RELEVANT_ANSWER",
            "new": "canonical build_inference_messages trace with [Final Answer] extraction",
        },
        "held_fixed": [
            "consumed v8 development90 identities",
            "v8-selected q1 query bytes",
            "v8 q1 reranked top-10 document bytes",
            "strong SFT adapter and base model",
            "base tokenizer chat template",
            "greedy decoding",
            "v8 unique-document lexical provenance binder",
        ],
        "canonical_prompt": {
            "builder": "kgproweight.data.prompts.build_inference_messages",
            "question": "frozen q1 selected_query",
            "top_k": 10,
            "kg_triples": [],
            "max_kg_triples": 0,
            "answer_parser": "kgproweight.data.parsers.extract_final_answer",
            "raw_generation_fallback_if_marker_missing": False,
        },
        "gates": {
            "row_count_exact": 90,
            "per_dataset_count_exact": 30,
            "final_answer_parse_rate_min_each_dataset": runner.FINAL_PARSE_RATE_MIN,
            "step_trace_rate_min_each_dataset": runner.TRACE_RATE_MIN,
            "admissible_rate_min_each_dataset": runner.ADMISSIBLE_RATE_MIN,
            "runtime_error_count": 0,
            "gold_access": False,
            "retrieval_calls": 0,
            "prospective_opened_or_hashed": False,
        },
        "decision": {
            "all_gates_pass": (
                "freeze a fresh train-side family-disjoint pilot30x3, then test the full v9 chain"
            ),
            "any_gate_fails": (
                "retain this diagnostic and stop canonical-subanswer v9 before fresh cohort use"
            ),
        },
        "model": model,
        "source_locks": source_locks,
        "code_locks": code_locks,
        "scientific_boundary": (
            "This is a Gold-free posthoc counterfactual on an already consumed development set. "
            "It can diagnose interface fit but cannot establish EM/F1 improvement, generalization, "
            "or PPO benefit. The sealed prospective900 is not opened or hashed."
        ),
    }
    destination.mkdir(parents=True)
    protocol_path = destination / "protocol.json"
    _write_json_new(protocol_path, protocol)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": runner.EXPERIMENT_ID,
        "status": STATUS,
        "gold_access": False,
        "prospective_opened_or_hashed": False,
        "protocol": _file_lock(protocol_path),
    }
    _write_json_new(destination / "manifest.json", manifest)
    return protocol


def main() -> None:
    protocol = freeze()
    print(json.dumps({
        "status": protocol["status"],
        "experiment_id": protocol["experiment_id"],
        "output_dir": str((PROJECT_ROOT / OUTPUT_DIR).resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

