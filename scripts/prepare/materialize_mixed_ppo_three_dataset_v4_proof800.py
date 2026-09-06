#!/usr/bin/env python3
"""Materialise the frozen mixed3-v4 Proof800 PPO release (CPU only).

The v4 protocol is answer-free and already fixes every selected identity and
every K=4 rollout exposure.  This stage performs only deterministic joins:

* retained v2 rows -> their frozen ten-passage contexts;
* 2Wiki ordinary replacements -> their line/hash-bound curriculum contexts;
* HotpotQA/MuSiQue context-required rows -> one reconciled 823-row release
  (812 immutable reused contexts plus 11 newly retrieved contexts);
* selected 2Wiki Proof rows -> the official-raw unified-v3 ProofKG supply;
* raw train rows -> Gold aliases used only as PPO outcome labels.

No source reasoning step is copied.  Every non-Proof row receives an explicit,
identity-safe empty question-KG record.  Every Proof row is rechecked with the
same dataset-agnostic hard Graph gate used at PPO runtime.  The existing Strong
SFT replay release is bound, not regenerated, and must have zero dataset-scoped
qid, exact-question-hash, and current-family overlap with the rollout pool.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.data.parsers import extract_final_answer, parse_steps
from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.kg.question_kg import (
    load_question_kg_index,
    question_key,
    question_sha256,
    validate_question_kg_record,
)
from kgproweight.kg.training_question_kg import (
    apply_training_question_kg,
    read_question_kg_records,
)
from kgproweight.reward.trajectory_source_gate import make_source_gate_record
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_mixed_ppo_three_dataset_v1 import sha256_file
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256
from scripts.prepare.audit_auto1500_v4_clean_reproducibility import (
    DEFAULT_PROTECTED_LEDGER_DIR,
    PROTECTED_LEDGER_SCHEMA_VERSION,
    validate_protected_ledger_release,
)
from scripts.prepare.materialize_mixed_ppo_three_dataset_v1 import (
    build_silver_row,
    load_selected_raw,
    make_outcome_only_kg_record,
)
from kgproweight.training.phase3_sft import _render_assistant_trace


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
EXPECTED_PROTOCOL_SCHEMA = "mixed-ppo-three-dataset-protocol-v4-proof800"
EXPECTED_PROTOCOL_STATUS = "FROZEN_ANSWER_FREE_NOT_MATERIALIZED_NOT_TRAINED"
EXPECTED_PROTOCOL_EXPERIMENT = (
    "MIXED-PPO-THREE-DATASET-V4-PROOF800-N3000-K4-12000-SEED42-PROTOCOL"
)
EXPECTED_EXPANSION_STATUS = "COMPLETE_ANSWER_FREE_RETRIEVAL_NOT_TRAINED"
# Final-v4 is intentionally bound to the official-raw unified-v3 release.  Do
# not accept the legacy unified-v2 candidate supply here: that release used a
# different source contract and is not interchangeable with the frozen
# official-raw scope/closure/retrieval chain.
EXPECTED_PROOF_SUPPLY_SCHEMA = (
    "2wiki-unified-proofkg-official-raw-candidate-supply-v3"
)
EXPECTED_PROOF_SUPPLY_STATUS = (
    "COMPLETE_STRICT_OFFICIAL_RAW_CANDIDATE_SUPPLY_NOT_SELECTED_NOT_TRAINED"
)
EXPECTED_PROOF_SUPPLY_OUTPUTS = {
    "silver_train": "silver_train.jsonl",
    "question_kg_records": "question_kg_records.jsonl",
    "source_gate_records": "source_gate_records.jsonl",
    "proof_candidates": "proof_candidates.jsonl",
}
EXPECTED_PARENT_SCHEMA = "mixed-ppo-three-dataset-materialization-v2-proof400"
EXPECTED_PARENT_EXPERIMENT = (
    "MIXED-PPO-THREE-DATASET-V2-PROOF400-N1799-K4-7200-SEED42-DATA"
)
EXPECTED_PARENT_MANIFEST_PHASE = "mixed_ppo_v2_data_materialization"
EXPECTED_REPLAY_SCHEMA = "sft-replay-rendered-3to5-v2-clean-isolated"
EXPECTED_REPLAY_EXPERIMENT = "SFT-REPLAY-STRONG-LEGACY-TRAIN-3TO5-N2000-SEED42-V2"
EXPECTED_REPLAY_MANIFEST_PHASE = "strong_sft_replay_v4_clean_refreeze"
EXPECTED_REPLAY_SELECTION_SCHEMA = "sft-replay-selection-v2-clean-isolated"
EXPECTED_RETRIEVAL_STACK = (
    "E5@100+BM25@100->RRF60@50->bge-reranker-v2-m3@10->pack3860"
)
REPORT_SCHEMA = "mixed-ppo-three-dataset-materialization-v4-proof800"
STATUS = "COMPLETE_DATA_NOT_TRAINED"
EXPERIMENT_ID = (
    "MIXED-PPO-THREE-DATASET-V4-PROOF800-N3000-K4-12000-SEED42-DATA"
)
DEFAULT_PROTOCOL = Path(
    "outputs/audits/mixed_ppo_three_dataset_v4_proof800_n3000_k4_seed42_protocol/"
    "protocol.json"
)
DEFAULT_PARENT = Path(
    "data/silver_data/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42"
)
DEFAULT_EXPANSION_RETRIEVAL = Path(
    "outputs/audits/mixed3_v4_expansion_retrieval_h417_m406_seed42_v2"
)
DEFAULT_PROOF_SUPPLY = Path(
    "data/derived/2wiki_unified_proofkg_official_raw_v3"
)
DEFAULT_REPLAY = Path(
    "data/silver_data/sft_replay_legacy_strong_train_rendered3to5_n2000_seed42_v2"
)
DEFAULT_OUTPUT = Path(
    "data/silver_data/mixed_ppo_three_dataset_v4_proof800_n3000_k4_seed42"
)
FORBIDDEN_PROMPT_FIELDS = {
    "answer",
    "answers",
    "gold_answer",
    "golden_answers",
    "supporting_facts",
    "decomposition",
}
ORDINARY_PROTOCOL_SCHEMA = "2wiki-ordinary200-full-ledger-protocol-v2"
ORDINARY_PROTOCOL_STATUS = "FROZEN_ANSWER_FREE_OUTCOME_SOURCE_BOUND_NOT_TRAINED"
HM_RECONCILIATION_SCHEMA = "mixed-ppo-v4-hm-full-ledger-reconciliation-v2"
HM_RECONCILIATION_STATUS = (
    "FROZEN_HM_FULL_LEDGER_DELTA_RETRIEVAL_NOT_RUN_NOT_TRAINED"
)
ORDINARY_SOURCE_CONTRACT = {
    "retained_parent_ordinary": (
        "mixed_ppo_v2_materialized",
        "parent_materialized_outcome_passages",
    ),
    "replacement_ordinary": (
        "proofkg_curriculum_mix_v1",
        "replacement_curriculum_outcome_passages",
    ),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _passages_sha256(value: Any) -> str:
    """Match the historical canonical-retrieval passage hash contract."""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _resolve_bound_file(identity: Mapping[str, Any], *, label: str) -> Path:
    raw = str(identity.get("path") or "").strip()
    expected = str(identity.get("sha256") or "").strip()
    if not raw or not expected:
        raise ValueError(f"{label} must bind path and sha256")
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen {label}: {path}")
    if _sha256(path) != expected:
        raise ValueError(f"frozen {label} SHA256 mismatch: {path}")
    if identity.get("size_bytes") is not None and path.stat().st_size != int(
        identity["size_bytes"]
    ):
        raise ValueError(f"frozen {label} size mismatch: {path}")
    return path


def _index(rows: Iterable[Mapping[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for value in rows:
        row = dict(value)
        dataset = str(row.get("dataset") or "").strip().lower()
        qid = str(row.get("qid") or "").strip()
        key = str(row.get("question_key") or question_key(dataset, qid))
        if not dataset or not qid or key in output:
            raise ValueError(f"{label} has empty/duplicate identity: {key!r}")
        if key != question_key(dataset, qid):
            raise ValueError(f"{label} has malformed question_key: {key}")
        output[key] = row
    return output


def _ten_safe_passages(passages: Any) -> bool:
    if not isinstance(passages, list) or len(passages) != 10:
        return False
    for passage in passages:
        if not isinstance(passage, Mapping):
            return False
        if not str(passage.get("id") or "").strip():
            return False
        if not str(passage.get("source") or "").strip():
            return False
        if not str(passage.get("contents") or "").strip():
            return False
        if FORBIDDEN_PROMPT_FIELDS & set(passage):
            return False
    return True


def _identity_fields_match(
    identity: Mapping[str, Any], row: Mapping[str, Any], *, label: str
) -> None:
    expected = {
        "dataset": str(identity["dataset"]).strip().lower(),
        "qid": str(identity["qid"]).strip(),
        "question": str(identity["question"]).strip(),
        "question_sha256": str(identity["question_sha256"]),
    }
    actual = {
        "dataset": str(row.get("dataset") or "").strip().lower(),
        "qid": str(row.get("qid") or "").strip(),
        "question": str(row.get("question") or "").strip(),
        "question_sha256": str(
            row.get("question_sha256") or question_sha256(str(row.get("question") or ""))
        ),
    }
    for field in expected:
        if actual[field] != expected[field]:
            raise ValueError(
                f"{label} identity mismatch at {field}: {identity['question_key']}"
            )


def _current_identity(row: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    dataset = str(row.get("dataset") or "").strip().lower()
    qid = str(row.get("qid") or "").strip()
    question = str(row.get("question") or "").strip()
    if dataset not in DATASETS or not qid or not question:
        raise ValueError(f"incomplete overlap identity: {dataset!r}::{qid!r}")
    return (
        ("qid", f"{dataset}::{qid}"),
        ("question_sha256", f"{dataset}::{question_sha256(question)}"),
        ("family_sha256", f"{dataset}::{family_sha256(question)}"),
    )


def identity_overlap_counts(
    left: Iterable[Mapping[str, Any]], right: Iterable[Mapping[str, Any]]
) -> dict[str, int]:
    """Count dataset-scoped overlap under all current identity definitions."""

    right_sets: dict[str, set[str]] = {
        "qid": set(),
        "question_sha256": set(),
        "family_sha256": set(),
    }
    for row in right:
        for name, value in _current_identity(row):
            right_sets[name].add(value)
    counts = Counter()
    for row in left:
        for name, value in _current_identity(row):
            counts[name] += int(value in right_sets[name])
    return {name: counts[name] for name in right_sets}


def _outcome_record(identity: Mapping[str, Any]) -> dict[str, Any]:
    record = make_outcome_only_kg_record(identity)
    provenance = dict(record.get("provenance") or {})
    provenance.update(
        {
            "builder_version": "mixed-ppo-outcome-only-empty-kg-v4-proof800",
            "mixed_ppo_data_version": "mixed-ppo-three-dataset-v4-proof800",
            "family_version": FAMILY_VERSION,
        }
    )
    record["provenance"] = provenance
    return record


def _proof_record(
    identity: Mapping[str, Any],
    source_record: Mapping[str, Any],
    *,
    cutoff: str,
) -> dict[str, Any]:
    validate_question_kg_record(source_record)
    _identity_fields_match(identity, source_record, label="ProofKG")
    expected_record_hash = str(identity.get("proof_record_sha256") or "")
    if not expected_record_hash:
        raise ValueError(f"frozen Proof identity lacks record hash: {identity['question_key']}")
    if _canonical_sha256(source_record) != expected_record_hash:
        raise ValueError(f"Proof record hash mismatch: {identity['question_key']}")
    provenance = dict(source_record.get("provenance") or {})
    if str(provenance.get("historical_cutoff") or "") != cutoff:
        raise ValueError(f"Proof historical cutoff mismatch: {identity['question_key']}")
    gate = make_source_gate_record(
        source_record,
        dataset=str(identity["dataset"]),
        qid=str(identity["qid"]),
        question=str(identity["question"]),
        text_evidence_available=True,
        historical_cutoff=cutoff,
    )
    if gate["m_graph"] != 1 or not all(gate["eligibility_checks"].values()):
        raise ValueError(
            f"selected Proof fails strict Graph gate: {identity['question_key']} "
            f"({gate['routing_reason']})"
        )
    record = json.loads(json.dumps(source_record, ensure_ascii=False))
    provenance = dict(record.get("provenance") or {})
    provenance.update(
        {
            "mixed_ppo_data_version": "mixed-ppo-three-dataset-v4-proof800",
            "family_version": FAMILY_VERSION,
            "process_reward_eligible": True,
            "failed_qpeg_or_saeg_p_edges_included": False,
        }
    )
    record["provenance"] = provenance
    record["process_reward_eligible"] = True
    return record


def assemble_materialized_rows(
    *,
    population: Sequence[Mapping[str, Any]],
    raw_by_key: Mapping[str, Mapping[str, Any]],
    parent_silver_by_key: Mapping[str, Mapping[str, Any]],
    ordinary_context_by_key: Mapping[str, Mapping[str, Any]],
    expansion_retrieval_by_key: Mapping[str, Mapping[str, Any]],
    proof_silver_by_key: Mapping[str, Mapping[str, Any]],
    proof_record_by_key: Mapping[str, Mapping[str, Any]],
    cutoff: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Join frozen identities to prompt evidence, outcome labels, and KG records."""

    silver_rows: list[dict[str, Any]] = []
    kg_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    passage_sources = Counter()
    for value in population:
        identity = dict(value)
        key = str(identity["question_key"])
        raw = raw_by_key.get(key)
        if raw is None:
            raise ValueError(f"raw train join miss: {key}")
        eligible = bool(identity.get("process_reward_eligible"))
        if eligible:
            if str(identity.get("dataset")) != "2wikimultihopqa":
                raise ValueError(f"v4 Proof eligibility outside 2Wiki: {key}")
            source_silver = proof_silver_by_key.get(key)
            source_record = proof_record_by_key.get(key)
            if source_silver is None or source_record is None:
                raise ValueError(f"unified Proof supply join miss: {key}")
            _identity_fields_match(identity, source_silver, label="Proof silver")
            record = _proof_record(identity, source_record, cutoff=cutoff)
            if list(source_silver.get("kg_subgraph") or []) != list(
                record.get("kg_subgraph") or []
            ):
                raise ValueError(f"Proof silver/record KG mismatch: {key}")
            passages = [dict(row) for row in source_silver.get("retrieved_passages") or []]
            expected_passages_hash = str(
                identity.get("proof_passages_sha256") or ""
            )
            if not expected_passages_hash:
                raise ValueError(
                    f"frozen Proof identity lacks passage hash: {key}"
                )
            if _passages_sha256(passages) != expected_passages_hash:
                raise ValueError(f"Proof passage hash mismatch: {key}")
            passage_sources["unified_2wiki_strict_proof_supply"] += 1
        else:
            record = _outcome_record(identity)
            source_role = str(identity.get("source_role") or "")
            if source_role == "new_retrieval":
                if str(identity.get("dataset")) not in {"hotpotqa", "musique"}:
                    raise ValueError(f"unexpected new retrieval route: {key}")
                context = expansion_retrieval_by_key.get(key)
                if context is None:
                    raise ValueError(f"H/M expansion retrieval join miss: {key}")
                _identity_fields_match(identity, context, label="expansion retrieval")
                passages = [dict(row) for row in context.get("passages") or []]
                passage_sources["canonical_expansion_wiki18_rrf_reranked_top10"] += 1
            elif source_role in ORDINARY_SOURCE_CONTRACT:
                if str(identity.get("dataset")) != "2wikimultihopqa":
                    raise ValueError(f"ordinary source route outside 2Wiki: {key}")
                context = ordinary_context_by_key.get(key)
                if context is None:
                    raise ValueError(f"ordinary200 source-provenance join miss: {key}")
                _identity_fields_match(identity, context, label="ordinary200 source")
                if context.get("source_role") != source_role:
                    raise ValueError(f"ordinary200 context role mismatch: {key}")
                passages = [dict(row) for row in context.get("passages") or []]
                passage_sources[
                    "ordinary200_"
                    + str(context.get("source_origin") or "unknown")
                ] += 1
            else:
                source_silver = parent_silver_by_key.get(key)
                if source_silver is None:
                    raise ValueError(f"retained parent context join miss: {key}")
                _identity_fields_match(identity, source_silver, label="retained parent")
                frozen_passages_hash = _passages_sha256(
                    source_silver.get("retrieved_passages") or []
                )
                passages = [dict(row) for row in source_silver.get("retrieved_passages") or []]
                if _passages_sha256(passages) != frozen_passages_hash:
                    raise ValueError(f"retained parent passage hash drifted: {key}")
                passage_sources["retained_v2_frozen_context"] += 1
        if not _ten_safe_passages(passages):
            raise ValueError(f"expected exactly ten safe passages: {key}")
        row = build_silver_row(
            identity,
            raw=raw,
            retrieved_passages=passages,
            kg_subgraph=record.get("kg_subgraph") or [],
        )
        row["metadata"].update(
            {
                "question_type": str(identity.get("question_type") or "unknown"),
                "family_version": FAMILY_VERSION,
                "family_sha256": str(identity["family_sha256"]),
                "mixed_ppo_data_version": "mixed-ppo-three-dataset-v4-proof800",
                "mixed_ppo_route": str(identity["route"]),
                "proof_source": str(identity.get("proof_source") or "none"),
                "process_reward_eligible": eligible,
                "source_gold_trace_removed": True,
                "failed_qpeg_or_saeg_p_edges_included": False,
                "gold_use": "outcome_reward_label_only",
            }
        )
        gate = make_source_gate_record(
            record,
            dataset=str(identity["dataset"]),
            qid=str(identity["qid"]),
            question=str(identity["question"]),
            text_evidence_available=True,
            historical_cutoff=cutoff,
        )
        if gate["m_graph"] != int(eligible):
            raise ValueError(
                f"protocol/strict Graph gate mismatch: {key} "
                f"eligible={eligible} m_graph={gate['m_graph']}"
            )
        silver_rows.append(row)
        kg_rows.append(record)
        gate_rows.append(gate)
    return silver_rows, kg_rows, gate_rows, dict(sorted(passage_sources.items()))


def _load_protocol(
    protocol_path: Path,
) -> tuple[dict[str, Any], dict[str, Path], Path]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": EXPECTED_PROTOCOL_SCHEMA,
        "status": EXPECTED_PROTOCOL_STATUS,
        "experiment_id": EXPECTED_PROTOCOL_EXPERIMENT,
    }
    for field, value in expected.items():
        if protocol.get(field) != value:
            raise ValueError(
                f"unexpected v4 protocol {field}: {protocol.get(field)!r} != {value!r}"
            )
    protocol_gates = protocol.get("gates") or {}
    if not protocol_gates or not all(bool(value) for value in protocol_gates.values()):
        raise ValueError("v4 protocol has empty or failed frozen gates")
    manifest_path = protocol_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"v4 protocol manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = manifest.get("run") or {}
    if (
        manifest.get("status") != EXPECTED_PROTOCOL_STATUS
        or run.get("phase") != "mixed_ppo_v4_answer_free_protocol_freeze"
        or run.get("experiment_id") != EXPECTED_PROTOCOL_EXPERIMENT
        or run.get("protocol_sha256") != _sha256(protocol_path)
        or run.get("training_started") is not False
    ):
        raise ValueError("v4 protocol manifest/status/hash binding drifted")
    population = protocol.get("population") or {}
    if int(population.get("unique_total", -1)) != 3000 or population.get(
        "unique_by_dataset"
    ) != {"2wikimultihopqa": 1000, "hotpotqa": 1000, "musique": 1000}:
        raise ValueError("v4 protocol population is not the frozen balanced n=3000 release")
    schedule = protocol.get("schedule") or {}
    if not (
        int(schedule.get("prompt_groups", -1)) == 3000
        and int(schedule.get("rollouts_per_prompt", -1)) == 4
        and int(schedule.get("trajectories", -1)) == 12000
        and int(schedule.get("proof_groups", -1)) == 800
    ):
        raise ValueError("v4 protocol schedule is not the frozen Proof800 K=4 schedule")
    paths: dict[str, Path] = {}
    for name in (
        "population",
        "sampling_weights",
        "prompt_groups",
        "fixed_rollout_schedule",
        "retrieval_requests",
    ):
        identity = (protocol.get("outputs") or {}).get(name)
        if not isinstance(identity, Mapping):
            raise ValueError(f"v4 protocol does not bind outputs.{name}")
        paths[name] = _resolve_bound_file(identity, label=name)
    return protocol, paths, manifest_path


def _resolve_release_file(path: Path, filename: str) -> Path:
    value = path / filename if path.is_dir() else path
    if not value.is_file():
        raise FileNotFoundError(value)
    return value.resolve()


def _release_binding(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {name: _identity(path) for name, path in paths.items()}


def _same_bound_identity(left: Any, right: Any, *, label: str) -> Path:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise ValueError(f"{label} must provide two bound identities")
    left_path = _resolve_bound_file(left, label=f"{label} left")
    right_path = _resolve_bound_file(right, label=f"{label} right")
    if (
        left_path != right_path
        or str(left.get("sha256") or "") != str(right.get("sha256") or "")
    ):
        raise ValueError(f"{label} identity drifted")
    return left_path


def _validate_parent_release(
    directory: Path, *, final_protocol: Mapping[str, Any]
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    """Validate the immutable v2 parent chain before reusing its passages."""

    files = {
        name: _resolve_release_file(directory, name)
        for name in ("silver_train.jsonl", "report.json", "manifest.json")
    }
    report = json.loads(files["report.json"].read_text(encoding="utf-8"))
    manifest = json.loads(files["manifest.json"].read_text(encoding="utf-8"))
    gates = report.get("gates") or {}
    if (
        report.get("schema_version") != EXPECTED_PARENT_SCHEMA
        or report.get("experiment_id") != EXPECTED_PARENT_EXPERIMENT
        or report.get("status") != STATUS
        or not gates
        or not all(bool(value) for value in gates.values())
        or int((report.get("counts") or {}).get("unique_population", -1)) != 1799
    ):
        raise ValueError("parent v2 report schema/status/counts/gates drifted")
    parent_output = (report.get("outputs") or {}).get("silver_train")
    if not isinstance(parent_output, Mapping) or _resolve_bound_file(
        parent_output, label="parent outputs.silver_train"
    ) != files["silver_train.jsonl"]:
        raise ValueError("parent report does not bind the live silver_train")
    run = manifest.get("run") or {}
    if (
        manifest.get("status") != STATUS
        or run.get("phase") != EXPECTED_PARENT_MANIFEST_PHASE
        or run.get("experiment_id") != EXPECTED_PARENT_EXPERIMENT
        or run.get("report_sha256") != _sha256(files["report.json"])
    ):
        raise ValueError("parent manifest/report binding drifted")
    frozen_parent_protocol = (final_protocol.get("inputs") or {}).get(
        "parent_protocol"
    )
    report_parent_protocol = (report.get("inputs") or {}).get("protocol")
    _same_bound_identity(
        frozen_parent_protocol,
        report_parent_protocol,
        label="final/parent protocol",
    )
    rows = _read_jsonl(files["silver_train.jsonl"])
    if len(rows) != 1799:
        raise ValueError("parent silver_train is not the frozen n=1799 release")
    return files, rows


def _valid_rendered_replay(row: Any) -> bool:
    text = _render_assistant_trace(row)
    steps = parse_steps(text, known_kg=getattr(row, "kg_subgraph", []))
    return bool(
        getattr(row, "accepted", False)
        and 3 <= len(steps) <= 5
        and [step.index for step in steps] == list(range(1, len(steps) + 1))
        and extract_final_answer(text)
        and all(
            not step.unknown_citation_surfaces
            and all(
                field in str(step.raw_text or "").casefold()
                for field in ("reasoning:", "knowledge used:", "conclusion:")
            )
            for step in steps
        )
    )


def _validate_replay_release(
    directory: Path,
    *,
    protected_ledger_binding: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    """Validate Strong-SFT replay payload, selection, and release manifest."""

    files = {
        name: _resolve_release_file(directory, name)
        for name in (
            "silver_train.jsonl",
            "selection_records.jsonl",
            "report.json",
            "manifest.json",
        )
    }
    report = json.loads(files["report.json"].read_text(encoding="utf-8"))
    manifest = json.loads(files["manifest.json"].read_text(encoding="utf-8"))
    gates = report.get("gates") or {}
    selection_summary = report.get("selection") or {}
    if (
        report.get("schema_version") != EXPECTED_REPLAY_SCHEMA
        or report.get("experiment_id") != EXPECTED_REPLAY_EXPERIMENT
        or report.get("status") != STATUS
        or not gates
        or not all(bool(value) for value in gates.values())
        or int(selection_summary.get("n_samples", -1)) != 2000
        or selection_summary.get("dataset_counts") != {"hotpotqa": 2000}
    ):
        raise ValueError("replay report schema/status/selection/gates drifted")
    for output_name, filename in (
        ("silver_train", "silver_train.jsonl"),
        ("selection_records", "selection_records.jsonl"),
    ):
        identity = (report.get("outputs") or {}).get(output_name)
        if not isinstance(identity, Mapping) or _resolve_bound_file(
            identity, label=f"replay outputs.{output_name}"
        ) != files[filename]:
            raise ValueError(f"replay report output drifted: {output_name}")
    expected_protected = {
        "protected_identities.question_only.jsonl": protected_ledger_binding["ledger"],
        "report.json": protected_ledger_binding["report"],
        "manifest.json": protected_ledger_binding["manifest"],
    }
    for name, expected in expected_protected.items():
        _same_bound_identity(
            (report.get("protected_ledger") or {}).get(name),
            expected,
            label=f"replay protected ledger {name}",
        )
    run = manifest.get("run") or {}
    if (
        manifest.get("status") != STATUS
        or run.get("phase") != EXPECTED_REPLAY_MANIFEST_PHASE
        or run.get("experiment_id") != EXPECTED_REPLAY_EXPERIMENT
        or run.get("training_started") is not False
        or run.get("report_sha256") != _sha256(files["report.json"])
        or run.get("silver_train_sha256") != _sha256(files["silver_train.jsonl"])
        or run.get("selection_records_sha256")
        != _sha256(files["selection_records.jsonl"])
        or run.get("protected_ledger_sha256")
        != str(protected_ledger_binding["ledger"].get("sha256") or "")
    ):
        raise ValueError("replay manifest/report/output binding drifted")

    silver_lines = [
        line
        for line in files["silver_train.jsonl"].read_bytes().splitlines()
        if line.strip()
    ]
    rows = [json.loads(line) for line in silver_lines]
    selections = _read_jsonl(files["selection_records.jsonl"])
    parsed = SilverDatasetReader(files["silver_train.jsonl"]).accepted()
    if len(rows) != len(selections) or len(parsed) != 2000:
        raise ValueError("replay silver/selection/accepted count is not exactly 2000")
    qids: set[tuple[str, str]] = set()
    hashes: set[tuple[str, str]] = set()
    families: set[tuple[str, str]] = set()
    for raw_line, row, selected in zip(silver_lines, rows, selections):
        dataset = str(row.get("dataset") or "").strip().lower()
        qid = str(row.get("qid") or "").strip()
        question = str(row.get("question") or "").strip()
        qhash = question_sha256(question)
        family = family_sha256(question)
        if (
            dataset != "hotpotqa"
            or not qid
            or row.get("accepted") is not True
            or selected.get("schema_version") != EXPECTED_REPLAY_SELECTION_SCHEMA
            or str(selected.get("dataset") or "").strip().lower() != dataset
            or str(selected.get("qid") or "").strip() != qid
            or str(selected.get("question") or "").strip() != question
            or selected.get("question_sha256") != qhash
            or selected.get("family_version") != FAMILY_VERSION
            or selected.get("family_sha256") != family
            or not 3 <= int(selected.get("rendered_steps", -1)) <= 5
            or selected.get("source_row_sha256")
            != hashlib.sha256(raw_line).hexdigest()
        ):
            raise ValueError(f"replay silver/selection identity drifted: {dataset}::{qid}")
        qids.add((dataset, qid))
        hashes.add((dataset, qhash))
        families.add((dataset, family))
    if len(qids) != 2000 or len(hashes) != 2000 or len(families) != 2000:
        raise ValueError("replay qid/question/current-family identities are not unique")
    if not all(_valid_rendered_replay(row) for row in parsed):
        raise ValueError("replay contains invalid accepted rendered 3-5-step traces")
    return files, rows


def _read_source_rows_at_lines(
    path: Path,
    requested: Mapping[int, Mapping[str, Any]],
    *,
    label: str,
) -> dict[int, dict[str, Any]]:
    """Read exactly the physical JSONL lines frozen by an ordinary identity release."""

    if any(int(line_number) < 1 for line_number in requested):
        raise ValueError(f"{label} contains a non-positive source line number")
    found: dict[int, dict[str, Any]] = {}
    if not requested:
        return found
    largest = max(int(value) for value in requested)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number > largest:
                break
            if line_number not in requested:
                continue
            if not line.strip():
                raise ValueError(f"{label} frozen source line is blank: {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid frozen ordinary source JSON at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"expected frozen ordinary source object at {path}:{line_number}"
                )
            found[line_number] = value
    missing = sorted(set(requested) - set(found))
    if missing:
        raise ValueError(
            f"{label} misses {len(missing)} frozen physical lines: {missing[:5]}"
        )
    return found


def _load_ordinary_context_release(
    protocol: Mapping[str, Any],
    *,
    population: Sequence[Mapping[str, Any]],
    parent_silver_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path], dict[str, int]]:
    """Resolve the full-ledger ordinary200 source contract without copying traces/KG.

    The successor release binds each retained or replacement ordinary identity to
    one physical source line and to hashes of both that complete source record and
    its ten passages.  This loader verifies that chain and returns passages only;
    answer labels continue to come from the immutable raw train split.
    """

    identity = (protocol.get("inputs") or {}).get("ordinary200_successor_protocol")
    if not isinstance(identity, Mapping):
        raise ValueError("v4 protocol does not bind the ordinary200 successor")
    ordinary_protocol_path = _resolve_bound_file(
        identity, label="ordinary200 successor protocol"
    )
    ordinary_protocol = json.loads(
        ordinary_protocol_path.read_text(encoding="utf-8")
    )
    if (
        ordinary_protocol.get("schema_version") != ORDINARY_PROTOCOL_SCHEMA
        or ordinary_protocol.get("status") != ORDINARY_PROTOCOL_STATUS
    ):
        raise ValueError("ordinary200 successor schema/status mismatch")
    counts = ((ordinary_protocol.get("selection") or {}).get("counts") or {})
    if int((ordinary_protocol.get("selection") or {}).get("target_n", -1)) != 200 or not all(
        bool(value) for value in (counts.get("gates") or {}).values()
    ):
        raise ValueError("ordinary200 successor target/gates failed")

    ordinary_identity = (ordinary_protocol.get("outputs") or {}).get("ordinary200")
    if not isinstance(ordinary_identity, Mapping):
        raise ValueError("ordinary200 successor does not bind its identity output")
    ordinary_path = _resolve_bound_file(
        ordinary_identity, label="ordinary200 identity provenance"
    )
    ordinary_rows = _read_jsonl(ordinary_path)
    ordinary_by_key = _index(ordinary_rows, label="ordinary200 successor")
    population_rows = [
        dict(row)
        for row in population
        if str(row.get("dataset")) == "2wikimultihopqa"
        and not bool(row.get("process_reward_eligible"))
    ]
    population_by_key = _index(population_rows, label="v4 ordinary population")
    if len(ordinary_rows) != 200 or set(ordinary_by_key) != set(population_by_key):
        raise ValueError(
            "ordinary200 successor/population identity set mismatch: "
            f"successor={len(ordinary_by_key)} population={len(population_by_key)}"
        )
    provenance_fields = (
        "dataset",
        "qid",
        "question",
        "question_sha256",
        "family_version",
        "family_sha256",
        "question_type",
        "route",
        "source_role",
        "source_origin",
        "source_line_number",
        "source_record_sha256",
        "source_passages_sha256",
        "process_reward_eligible",
        "gold_access",
    )
    for key, frozen in ordinary_by_key.items():
        selected = population_by_key[key]
        for field in provenance_fields:
            if selected.get(field) != frozen.get(field):
                raise ValueError(
                    f"ordinary200 population provenance mismatch at {field}: {key}"
                )

    manifest_path = ordinary_protocol_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = manifest.get("run") or {}
    manifest_ordinary = run.get("ordinary200") or {}
    if (
        manifest.get("status") != ORDINARY_PROTOCOL_STATUS
        or run.get("protocol_sha256") != _sha256(ordinary_protocol_path)
        or not isinstance(manifest_ordinary, Mapping)
        or manifest_ordinary.get("sha256") != _sha256(ordinary_path)
        or run.get("training_started") is not False
    ):
        raise ValueError("ordinary200 successor manifest/protocol binding drifted")

    inputs = ordinary_protocol.get("inputs") or {}
    source_paths: dict[str, Path] = {}
    requested_by_origin: dict[str, dict[int, Mapping[str, Any]]] = {}
    for source_role, (source_origin, input_name) in ORDINARY_SOURCE_CONTRACT.items():
        source_identity = inputs.get(input_name)
        if not isinstance(source_identity, Mapping):
            raise ValueError(f"ordinary200 successor does not bind {input_name}")
        source_path = _resolve_bound_file(source_identity, label=input_name)
        source_paths[input_name] = source_path
        requested_by_origin[source_origin] = {}
        for row in ordinary_rows:
            if row.get("source_role") != source_role:
                continue
            if row.get("source_origin") != source_origin:
                raise ValueError(
                    f"ordinary source role/origin mismatch: {row.get('qid')}"
                )
            line_number = int(row.get("source_line_number", 0))
            if line_number in requested_by_origin[source_origin]:
                raise ValueError(
                    f"ordinary source line reused within {source_origin}: {line_number}"
                )
            requested_by_origin[source_origin][line_number] = row
    if _sha256(source_paths["parent_materialized_outcome_passages"]) != _sha256(
        parent_silver_path
    ):
        raise ValueError("ordinary retained-parent source differs from v2 parent silver")

    contexts: dict[str, dict[str, Any]] = {}
    source_counts = Counter()
    for source_role, (source_origin, input_name) in ORDINARY_SOURCE_CONTRACT.items():
        requested = requested_by_origin[source_origin]
        source_rows = _read_source_rows_at_lines(
            source_paths[input_name], requested, label=source_origin
        )
        for line_number, frozen in requested.items():
            source = source_rows[line_number]
            key = question_key(frozen["dataset"], frozen["qid"])
            _identity_fields_match(frozen, source, label=f"ordinary source {source_origin}")
            if _canonical_sha256(source) != str(frozen["source_record_sha256"]):
                raise ValueError(f"ordinary source record hash mismatch: {key}")
            passages = source.get("retrieved_passages")
            if _canonical_sha256(passages) != str(frozen["source_passages_sha256"]):
                raise ValueError(f"ordinary source passages hash mismatch: {key}")
            if not _ten_safe_passages(passages):
                raise ValueError(f"ordinary source lacks ten safe passages: {key}")
            if source.get("accepted") is not True:
                raise ValueError(f"ordinary source row is not accepted: {key}")
            contexts[key] = {
                "dataset": str(frozen["dataset"]),
                "qid": str(frozen["qid"]),
                "question": str(frozen["question"]),
                "question_sha256": str(frozen["question_sha256"]),
                "passages": [dict(value) for value in passages],
                "source_role": source_role,
                "source_origin": source_origin,
            }
            source_counts[source_origin] += 1
    if set(contexts) != set(ordinary_by_key):
        raise ValueError("ordinary200 source-line materialization is incomplete")
    metadata_paths = {
        "protocol": ordinary_protocol_path,
        "manifest": manifest_path.resolve(),
        "identities": ordinary_path,
        **source_paths,
    }
    return contexts, metadata_paths, dict(sorted(source_counts.items()))


def _load_hm_reconciliation_contract(
    protocol: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Path],
]:
    """Load the frozen 812-reuse + 11-new H/M context reconciliation."""

    identity = (protocol.get("inputs") or {}).get("hm_reconciliation_protocol")
    if not isinstance(identity, Mapping):
        raise ValueError("v4 protocol does not bind the H/M reconciliation protocol")
    protocol_path = _resolve_bound_file(identity, label="H/M reconciliation protocol")
    reconciliation = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        reconciliation.get("schema_version") != HM_RECONCILIATION_SCHEMA
        or reconciliation.get("status") != HM_RECONCILIATION_STATUS
        or not all(bool(value) for value in (reconciliation.get("gates") or {}).values())
    ):
        raise ValueError("H/M reconciliation schema/status/gates failed")
    manifest_path = protocol_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != HM_RECONCILIATION_STATUS
        or (manifest.get("run") or {}).get("protocol_sha256")
        != _sha256(protocol_path)
    ):
        raise ValueError("H/M reconciliation manifest/protocol binding drifted")

    output_paths: dict[str, Path] = {}
    output_rows: dict[str, list[dict[str, Any]]] = {}
    for name in (
        "retrieval_requirements",
        "reused_context_bindings",
        "new_retrieval_requests",
        "retired_contexts",
    ):
        output_identity = (reconciliation.get("outputs") or {}).get(name)
        if not isinstance(output_identity, Mapping):
            raise ValueError(f"H/M reconciliation does not bind outputs.{name}")
        output_paths[name] = _resolve_bound_file(
            output_identity, label=f"H/M reconciliation {name}"
        )
        output_rows[name] = _read_jsonl(output_paths[name])
    requirements = _index(
        output_rows["retrieval_requirements"], label="H/M reconciliation requirements"
    )
    reused = _index(
        output_rows["reused_context_bindings"], label="H/M reused contexts"
    )
    new = _index(
        output_rows["new_retrieval_requests"], label="H/M new retrieval requests"
    )
    retired = _index(
        output_rows["retired_contexts"], label="H/M retired contexts"
    )
    if (
        len(requirements) != 823
        or Counter(row["dataset"] for row in requirements.values())
        != Counter({"hotpotqa": 417, "musique": 406})
        or len(reused) != 812
        or len(new) != 11
        or len(retired) != 6
        or set(reused) & set(new)
        or set(reused) | set(new) != set(requirements)
    ):
        raise ValueError("H/M reconciliation is not exact 812 reuse + 11 new = 823")
    metadata_paths = {
        "protocol": protocol_path,
        "manifest": manifest_path.resolve(),
        **output_paths,
    }
    return requirements, reused, new, metadata_paths


def _validate_hm_reconciled_contexts(
    expansion: Mapping[str, Mapping[str, Any]],
    *,
    requirements: Mapping[str, Mapping[str, Any]],
    reused: Mapping[str, Mapping[str, Any]],
    new: Mapping[str, Mapping[str, Any]],
) -> None:
    """Verify every reused passage block and every new passage hash fail-closed."""

    if set(expansion) != set(requirements) or set(reused) | set(new) != set(requirements):
        raise ValueError("H/M reconciled context identity sets drifted")
    for key, context in expansion.items():
        supplied_hash = str(context.get("passages_sha256") or "")
        actual_hash = _passages_sha256(context.get("passages"))
        if supplied_hash != actual_hash:
            raise ValueError(f"H/M context passage hash mismatch: {key}")
        if key in reused and supplied_hash != str(reused[key].get("passages_sha256") or ""):
            raise ValueError(f"H/M reused passage block drifted: {key}")


def _validate_protocol_protected_ledger(
    protocol: Mapping[str, Any],
    *,
    live_binding: Mapping[str, Mapping[str, Any]],
) -> None:
    frozen = protocol.get("protected_ledger") or {}
    if not (
        frozen.get("version") == PROTECTED_LEDGER_SCHEMA_VERSION
        and frozen.get("complete") is True
        and frozen.get("current_family_recomputed") is True
    ):
        raise ValueError("v4 protocol does not bind the complete protected ledger")
    for name in ("ledger", "report", "manifest"):
        expected = live_binding.get(name)
        actual = frozen.get(name)
        if (
            not isinstance(expected, Mapping)
            or not isinstance(actual, Mapping)
            or actual.get("sha256") != expected.get("sha256")
        ):
            raise ValueError(f"v4 protocol protected-ledger hash mismatch: {name}")


def _validate_expansion_release(
    directory: Path,
    contexts_path: Path,
    *,
    reconciliation_protocol_path: Path | None = None,
) -> dict[str, Path]:
    """Require runtime proof that the canonical CE reranker did not fall back."""

    if not directory.is_dir():
        raise ValueError(
            "formal H/M expansion retrieval must be a versioned directory with "
            "report.json, manifest.json, and backend_attestation"
        )
    metadata = {
        name: _resolve_release_file(directory, name)
        for name in ("report.json", "manifest.json")
    }
    report = json.loads(metadata["report.json"].read_text(encoding="utf-8"))
    manifest = json.loads(metadata["manifest.json"].read_text(encoding="utf-8"))
    if report.get("status") != EXPECTED_EXPANSION_STATUS:
        raise ValueError("H/M expansion retrieval is not a complete frozen release")
    if report.get("retrieval") != EXPECTED_RETRIEVAL_STACK:
        raise ValueError("H/M expansion retrieval stack differs from canonical protocol")
    if not all(bool(value) for value in (report.get("gates") or {}).values()):
        raise ValueError("H/M expansion retrieval report contains a failed gate")
    attestation = report.get("backend_attestation")
    if not isinstance(attestation, Mapping):
        raise ValueError("H/M expansion retrieval lacks backend_attestation")
    if not (
        attestation.get("mode") == "cross_encoder"
        and attestation.get("requested_backend") == "bge-reranker-v2-m3"
        and attestation.get("load_succeeded") is True
        and attestation.get("backend_fallback") is False
    ):
        raise ValueError(
            "H/M expansion retrieval lacks an exact no-fallback BGE backend attestation"
        )
    for asset in ("config", "weights", "tokenizer"):
        identity = attestation.get(asset)
        if not isinstance(identity, Mapping) or not str(identity.get("sha256") or ""):
            raise ValueError(f"H/M expansion reranker attestation lacks {asset} identity")
    combined = (report.get("outputs") or {}).get("combined")
    if not isinstance(combined, Mapping):
        raise ValueError("H/M expansion report does not bind outputs.combined")
    if str(combined.get("sha256") or "") != _sha256(contexts_path):
        raise ValueError("H/M expansion report/context SHA256 mismatch")
    if reconciliation_protocol_path is not None:
        binding = (report.get("inputs") or {}).get("hm_reconciliation_protocol")
        reconciliation = report.get("reconciliation") or {}
        if (
            not isinstance(binding, Mapping)
            or binding.get("sha256") != _sha256(reconciliation_protocol_path)
            or int(reconciliation.get("reused_contexts", -1)) != 812
            or int(reconciliation.get("newly_retrieved_contexts", -1)) != 11
            or int(reconciliation.get("retired_contexts", -1)) != 6
        ):
            raise ValueError(
                "H/M expansion release does not bind the frozen 812/11/6 reconciliation"
            )
    run = manifest.get("run") or {}
    manifest_combined = ((run.get("outputs") or {}).get("combined") or {})
    manifest_report = ((run.get("outputs") or {}).get("report") or {})
    if (
        manifest.get("status") != EXPECTED_EXPANSION_STATUS
        or not isinstance(manifest_combined, Mapping)
        or manifest_combined.get("sha256") != _sha256(contexts_path)
        or not isinstance(manifest_report, Mapping)
        or manifest_report.get("sha256") != _sha256(metadata["report.json"])
        or run.get("training_started") is not False
    ):
        raise ValueError("H/M expansion manifest/report/context binding drifted")
    return metadata


def _validate_expansion_requirement_join(
    population: Sequence[Mapping[str, Any]],
    retrieval_request_rows: Sequence[Mapping[str, Any]],
    expansion: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    """Require the reconciled release to cover all 823 H/M contexts exactly."""

    retrieval_requests = _index(
        retrieval_request_rows, label="frozen H/M retrieval requirements"
    )
    expected_population = {
        str(row["question_key"]): dict(row)
        for row in population
        if row.get("source_role") == "new_retrieval"
    }
    expected_counts = Counter(row["dataset"] for row in retrieval_request_rows)
    if (
        len(retrieval_requests) != 823
        or expected_counts != Counter({"hotpotqa": 417, "musique": 406})
        or set(retrieval_requests) != set(expected_population)
    ):
        raise ValueError(
            "frozen v4 retrieval-requirement contract is not exact H417/M406=823"
        )
    if set(expansion) != set(retrieval_requests):
        raise ValueError(
            "expansion retrieval does not exactly cover all frozen H/M context "
            f"requirements: got={len(expansion)} expected={len(retrieval_requests)}"
        )
    for key, request in retrieval_requests.items():
        _identity_fields_match(request, expansion[key], label="expansion requirement")
    return retrieval_requests, expected_counts


def _validate_proof_supply_release(
    directory: Path,
    *,
    proof_files: Mapping[str, Path],
    protected_ledger_binding: Mapping[str, Mapping[str, Any]],
) -> dict[str, Path]:
    """Bind selected Proof passages/KG to one complete unified release."""

    if not directory.is_dir():
        raise ValueError("unified Proof supply must be a versioned release directory")
    if set(proof_files) != set(EXPECTED_PROOF_SUPPLY_OUTPUTS):
        raise ValueError("unified Proof supply payload set is not exact official-v3")
    report_path = _resolve_release_file(directory, "report.json")
    manifest_path = _resolve_release_file(directory, "manifest.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = report.get("checks") or {}
    if (
        report.get("schema_version") != EXPECTED_PROOF_SUPPLY_SCHEMA
        or report.get("status") != EXPECTED_PROOF_SUPPLY_STATUS
        or manifest.get("status") != EXPECTED_PROOF_SUPPLY_STATUS
        or report.get("training_started") is not False
        or not checks
        or not all(bool(value) for value in checks.values())
    ):
        raise ValueError("unified Proof supply status/schema/checks failed")
    ledger = report.get("protected_ledger") or {}
    if not (
        ledger.get("version") == PROTECTED_LEDGER_SCHEMA_VERSION
        and ledger.get("complete") is True
        and ledger.get("current_family_recomputed") is True
    ):
        raise ValueError("unified Proof supply lacks the complete protected ledger")
    for name in ("ledger", "report", "manifest"):
        actual = ledger.get(name)
        expected = protected_ledger_binding.get(name)
        _same_bound_identity(
            actual,
            expected,
            label=f"unified Proof supply protected ledger {name}",
        )
    outputs = report.get("outputs") or {}
    if set(outputs) != set(EXPECTED_PROOF_SUPPLY_OUTPUTS):
        raise ValueError("unified Proof supply report outputs are not exact official-v3")
    for name, path in proof_files.items():
        identity = outputs.get(name)
        if not isinstance(identity, Mapping) or _resolve_bound_file(
            identity, label=f"unified Proof supply outputs.{name}"
        ) != path.resolve():
            raise ValueError(f"unified Proof supply output hash mismatch: {name}")
    run = manifest.get("run") or {}
    manifest_report = run.get("report") or {}
    if (
        not isinstance(manifest_report, Mapping)
        or _resolve_bound_file(
            manifest_report, label="unified Proof supply manifest report"
        )
        != report_path
        or run.get("phase")
        != "unified_2wiki_proofkg_official_raw_v3_candidate_supply"
        or run.get("experiment_id") != report.get("experiment_id")
        or run.get("training_started") is not False
    ):
        raise ValueError("unified Proof supply manifest/report binding drifted")
    return {"report.json": report_path, "manifest.json": manifest_path}


def _validate_population(population: Sequence[Mapping[str, Any]]) -> None:
    by_dataset = Counter(str(row.get("dataset")) for row in population)
    keys = {str(row.get("question_key")) for row in population}
    hashes = {
        (str(row.get("dataset")), str(row.get("question_sha256"))) for row in population
    }
    if len(population) != len(keys) or len(hashes) != len(population):
        raise ValueError("v4 population has duplicate key or exact question hash")
    if by_dataset != Counter({dataset: 1000 for dataset in DATASETS}):
        raise ValueError(f"v4 dataset counts drifted: {dict(by_dataset)}")
    if sum(bool(row.get("process_reward_eligible")) for row in population) != 800:
        raise ValueError("v4 population is not exactly Proof800")
    for row in population:
        question = str(row.get("question") or "").strip()
        if row.get("question_key") != question_key(row.get("dataset"), row.get("qid")):
            raise ValueError(f"population question_key mismatch: {row.get('question_key')}")
        if row.get("question_sha256") != question_sha256(question):
            raise ValueError(f"population question hash mismatch: {row.get('question_key')}")
        if row.get("family_sha256") != family_sha256(question):
            raise ValueError(f"population family hash mismatch: {row.get('question_key')}")
        if row.get("gold_access") is not False:
            raise ValueError(f"population is not answer-free: {row.get('question_key')}")


def validate_schedule_assets(
    population: Sequence[Mapping[str, Any]],
    weights: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    """Validate all frozen exposure assets against the materialised population."""

    pop = {str(row["question_key"]): row for row in population}
    weight_keys = {question_key(row["dataset"], row["qid"]) for row in weights}
    group_keys = [question_key(row["dataset"], row["qid"]) for row in groups]
    weights_match = len(weights) == len(weight_keys) == len(pop) and weight_keys == set(pop)
    if weights_match:
        weights_match = all(
            str(row.get("question_sha256")) == str(pop[question_key(row["dataset"], row["qid"])]["question_sha256"])
            and bool(row.get("process_reward_eligible"))
            == bool(pop[question_key(row["dataset"], row["qid"])]["process_reward_eligible"])
            for row in weights
        )
    groups_match = len(groups) == len(group_keys) == len(pop) and set(group_keys) == set(pop)
    if groups_match:
        groups_match = all(
            int(row.get("prompt_group_index", -1)) == index
            and str(row.get("question_sha256")) == str(pop[key]["question_sha256"])
            and bool(row.get("process_reward_eligible"))
            == bool(pop[key]["process_reward_eligible"])
            for index, (row, key) in enumerate(zip(groups, group_keys), start=1)
        )
    chunks_ok = len(schedule) == 4 * len(groups)
    if chunks_ok:
        for group_index, group in enumerate(groups, start=1):
            chunk = schedule[4 * (group_index - 1) : 4 * group_index]
            expected_key = question_key(group["dataset"], group["qid"])
            if [int(row.get("within_group_rollout", -1)) for row in chunk] != [1, 2, 3, 4]:
                chunks_ok = False
                break
            if any(
                int(row.get("prompt_group_index", -1)) != group_index
                or question_key(row["dataset"], row["qid"]) != expected_key
                or str(row.get("question_sha256")) != str(group["question_sha256"])
                or bool(row.get("process_reward_eligible"))
                != bool(group["process_reward_eligible"])
                for row in chunk
            ):
                chunks_ok = False
                break
    return {
        "weights_identity_join_1": weights_match,
        "weights_sum_1": abs(
            sum(float(row.get("sampling_probability", 0.0)) for row in weights) - 1.0
        )
        <= 1e-12,
        "groups_identity_join_1_and_once": groups_match,
        "schedule_k4_identity_exact": chunks_ok,
    }


def materialize(
    *,
    protocol_path: Path,
    parent_dir: Path,
    expansion_retrieval: Path,
    proof_supply_dir: Path,
    protected_ledger_dir: Path,
    data_root: Path,
    replay_dir: Path,
    output_dir: Path,
    experiment_id: str,
) -> dict[str, Any]:
    """Create and validate one append-only v4 training-data release."""

    protocol_path = protocol_path.resolve()
    parent_dir = parent_dir.resolve()
    proof_supply_dir = proof_supply_dir.resolve()
    protected_ledger_dir = protected_ledger_dir.resolve()
    data_root = data_root.resolve()
    replay_dir = replay_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite versioned v4 data: {output_dir}")
    if not str(experiment_id).strip():
        raise ValueError("a nonempty Experiment ID is required")

    ledger_path, ledger_report_path, ledger_manifest_path, _ledger_report = (
        validate_protected_ledger_release(protected_ledger_dir)
    )
    protected_ledger_binding = _release_binding(
        {
            "ledger": ledger_path,
            "report": ledger_report_path,
            "manifest": ledger_manifest_path,
        }
    )
    protected_rows = _read_jsonl(ledger_path)

    protocol, frozen_paths, protocol_manifest_path = _load_protocol(protocol_path)
    _validate_protocol_protected_ledger(
        protocol, live_binding=protected_ledger_binding
    )
    population = _read_jsonl(frozen_paths["population"])
    _validate_population(population)
    population_protected_overlap = identity_overlap_counts(population, protected_rows)
    if any(population_protected_overlap.values()):
        raise ValueError(
            "v4 rollout population overlaps complete protected ledger: "
            f"{population_protected_overlap}"
        )
    weights = _read_jsonl(frozen_paths["sampling_weights"])
    groups = _read_jsonl(frozen_paths["prompt_groups"])
    schedule = _read_jsonl(frozen_paths["fixed_rollout_schedule"])
    schedule_gates = validate_schedule_assets(population, weights, groups, schedule)
    if not all(schedule_gates.values()):
        raise ValueError(f"frozen v4 schedule/weight gates failed: {schedule_gates}")

    parent_files, parent_rows = _validate_parent_release(
        parent_dir, final_protocol=protocol
    )
    parent_silver = _index(
        parent_rows, label="parent silver"
    )

    ordinary_contexts, ordinary_release_files, ordinary_source_counts = (
        _load_ordinary_context_release(
            protocol,
            population=population,
            parent_silver_path=parent_files["silver_train.jsonl"],
        )
    )
    (
        hm_requirements,
        hm_reused_contexts,
        hm_new_requests,
        hm_reconciliation_files,
    ) = _load_hm_reconciliation_contract(protocol)

    expansion_root = expansion_retrieval.resolve()
    expansion_path = _resolve_release_file(expansion_root, "retrieval_contexts.jsonl")
    expansion_metadata = _validate_expansion_release(
        expansion_root,
        expansion_path,
        reconciliation_protocol_path=hm_reconciliation_files["protocol"],
    )
    expansion_rows = _read_jsonl(expansion_path)
    expansion = _index(expansion_rows, label="H/M expansion retrieval")
    retrieval_request_rows = _read_jsonl(frozen_paths["retrieval_requests"])
    retrieval_requests, expected_counts = _validate_expansion_requirement_join(
        population, retrieval_request_rows, expansion
    )
    if set(retrieval_requests) != set(hm_requirements):
        raise ValueError(
            "final protocol retrieval requirements differ from H/M reconciliation"
        )
    for key, request in retrieval_requests.items():
        _identity_fields_match(
            request, hm_requirements[key], label="H/M reconciliation requirement"
        )
    _validate_hm_reconciled_contexts(
        expansion,
        requirements=hm_requirements,
        reused=hm_reused_contexts,
        new=hm_new_requests,
    )
    if any(
        row.get("gold_access") is not False
        or not _ten_safe_passages(row.get("passages"))
        or FORBIDDEN_PROMPT_FIELDS & set(row)
        or row.get("retrieval_source") != EXPECTED_RETRIEVAL_STACK
        for row in expansion_rows
    ):
        raise ValueError(
            "expansion retrieval violates answer-free ten-passage/canonical-backend contract"
        )

    proof_files = {
        name: _resolve_release_file(proof_supply_dir, name)
        for name in EXPECTED_PROOF_SUPPLY_OUTPUTS.values()
    }
    proof_files = {
        output_name: proof_files[filename]
        for output_name, filename in EXPECTED_PROOF_SUPPLY_OUTPUTS.items()
    }
    proof_release_metadata = _validate_proof_supply_release(
        proof_supply_dir,
        proof_files=proof_files,
        protected_ledger_binding=protected_ledger_binding,
    )
    proof_silver = _index(_read_jsonl(proof_files["silver_train"]), label="Proof silver")
    proof_records = _index(
        _read_jsonl(proof_files["question_kg_records"]), label="Proof records"
    )
    selected_proof_keys = {
        str(row["question_key"])
        for row in population
        if row.get("process_reward_eligible")
    }
    if not selected_proof_keys.issubset(proof_silver) or not selected_proof_keys.issubset(
        proof_records
    ):
        raise ValueError("unified Proof supply misses a frozen selected identity")

    replay_files, replay_rows = _validate_replay_release(
        replay_dir, protected_ledger_binding=protected_ledger_binding
    )
    replay_overlap = identity_overlap_counts(population, replay_rows)
    if any(replay_overlap.values()):
        raise ValueError(f"rollout/replay identity overlap: {replay_overlap}")
    replay_protected_overlap = identity_overlap_counts(replay_rows, protected_rows)
    if any(replay_protected_overlap.values()):
        raise ValueError(
            "Strong-SFT replay overlaps complete protected ledger: "
            f"{replay_protected_overlap}"
        )

    raw_paths = {dataset: data_root / dataset / "train.jsonl" for dataset in DATASETS}
    for path in raw_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    raw_by_key = load_selected_raw(raw_paths, population)
    cutoff = str((protocol.get("inputs") or {}).get("historical_cutoff") or "").strip()
    if not cutoff:
        raise ValueError("v4 protocol does not bind inputs.historical_cutoff")
    silver_rows, kg_rows, gate_rows, passage_sources = assemble_materialized_rows(
        population=population,
        raw_by_key=raw_by_key,
        parent_silver_by_key=parent_silver,
        ordinary_context_by_key=ordinary_contexts,
        expansion_retrieval_by_key=expansion,
        proof_silver_by_key=proof_silver,
        proof_record_by_key=proof_records,
        cutoff=cutoff,
    )

    keys = {question_key(row["dataset"], row["qid"]) for row in silver_rows}
    kg_index = load_question_kg_index(kg_rows)
    gate_index = {str(row["question_key"]): row for row in gate_rows}
    graph_count = sum(int(row["m_graph"]) for row in gate_rows)
    ordinary_count = len(gate_rows) - graph_count
    scheduled_graph = sum(
        gate_index[question_key(row["dataset"], row["qid"])]["m_graph"]
        for row in schedule
    )
    gates = {
        "population_3000_unique": len(silver_rows) == len(keys) == 3000,
        "dataset_1000_each": Counter(row["dataset"] for row in silver_rows)
        == Counter({dataset: 1000 for dataset in DATASETS}),
        "question_kg_identity_join_1": keys == set(kg_index) == set(gate_index),
        "proof800_m_graph_exact": graph_count == 800,
        "ordinary2200_m_graph_zero": ordinary_count == 2200,
        "all_steps_empty": all(row.get("steps") == [] for row in silver_rows),
        "all_exactly_ten_safe_passages": all(
            _ten_safe_passages(row.get("retrieved_passages")) for row in silver_rows
        ),
        "ordinary200_source_provenance_join_1": len(ordinary_contexts) == 200
        and sum(ordinary_source_counts.values()) == 200,
        "hm_retrieval_requirements_h417_m406_exact": len(retrieval_requests) == 823
        and expected_counts == Counter({"hotpotqa": 417, "musique": 406})
        and set(expansion) == set(retrieval_requests),
        "hm_reconciliation_812_reuse_11_new_exact": len(hm_reused_contexts) == 812
        and len(hm_new_requests) == 11
        and set(hm_reused_contexts) | set(hm_new_requests)
        == set(retrieval_requests),
        "gold_outcome_labels_nonempty": all(
            str((row.get("metadata") or {}).get("gold_answer") or "").strip()
            and (row.get("metadata") or {}).get("gold_use")
            == "outcome_reward_label_only"
            for row in silver_rows
        ),
        "source_steps_or_failed_edges_zero": all(
            row.get("steps") == []
            and not row.get("passage_evidence")
            and row.get("evidence_mode") is None
            and (row.get("metadata") or {}).get("failed_qpeg_or_saeg_p_edges_included")
            is False
            for row in silver_rows
        ),
        "source_gate_all_checks_for_eligible": all(
            row["m_graph"] == 0 or all(row["eligibility_checks"].values())
            for row in gate_rows
        ),
        "schedule_12000_k4": len(schedule) == 12000 and len(groups) == 3000,
        "scheduled_graph_trajectories_3200": scheduled_graph == 3200,
        "replay_n2000_bound": len(replay_rows) == 2000,
        "replay_qid_question_hash_family_overlap_zero": not any(
            replay_overlap.values()
        ),
        "population_protected_qid_question_hash_family_overlap_zero": not any(
            population_protected_overlap.values()
        ),
        "replay_protected_qid_question_hash_family_overlap_zero": not any(
            replay_protected_overlap.values()
        ),
        **schedule_gates,
    }
    if not all(gates.values()):
        raise RuntimeError(f"v4 materialization gates failed: {gates}")

    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "silver_train": output_dir / "silver_train.jsonl",
        "question_kg_records": output_dir / "question_kg_records.jsonl",
        "source_gate_records": output_dir / "source_gate_records.jsonl",
        "sampling_weights": output_dir / "sampling_weights.jsonl",
        "prompt_groups": output_dir / "prompt_groups.jsonl",
        "fixed_rollout_schedule": output_dir / "fixed_rollout_schedule.jsonl",
    }
    _write_jsonl(outputs["silver_train"], silver_rows)
    _write_jsonl(outputs["question_kg_records"], kg_rows)
    _write_jsonl(outputs["source_gate_records"], gate_rows)
    for name in ("sampling_weights", "prompt_groups", "fixed_rollout_schedule"):
        shutil.copyfile(frozen_paths[name], outputs[name])
        if _sha256(outputs[name]) != _sha256(frozen_paths[name]):
            raise RuntimeError(f"byte identity failed copying frozen {name}")

    # Exercise the production loader and applicator, not only a local set join.
    reader = SilverDatasetReader(outputs["silver_train"])
    records = read_question_kg_records(outputs["question_kg_records"])
    join = apply_training_question_kg(
        reader.accepted(), records, min_coverage=1.0, require_nonempty=False
    ).to_dict()
    if join["coverage_rate"] != 1.0 or join["covered"] != 3000:
        failed = output_dir / "FAILED_MATERIALIZATION.json"
        failed.write_text(json.dumps({"join": join}, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"production question-KG join failed: {join}")

    optional_inputs: dict[str, Any] = {
        "expansion_retrieval": {
            name: _identity(path) for name, path in expansion_metadata.items()
        },
        "proof_supply": {
            name: _identity(path) for name, path in proof_release_metadata.items()
        },
        "ordinary200_source_release": {
            name: _identity(path) for name, path in ordinary_release_files.items()
        },
        "hm_reconciliation": {
            name: _identity(path) for name, path in hm_reconciliation_files.items()
        },
        "protected_ledger": protected_ledger_binding,
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "experiment_id": str(experiment_id).strip(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "counts": {
            "unique_population": len(silver_rows),
            "unique_by_dataset": dict(
                sorted(Counter(row["dataset"] for row in silver_rows).items())
            ),
            "graph_eligible": graph_count,
            "graph_ineligible": ordinary_count,
            "graph_eligible_by_dataset": dict(
                sorted(
                    Counter(
                        row["dataset"] for row in gate_rows if row["m_graph"] == 1
                    ).items()
                )
            ),
            "passage_sources": passage_sources,
            "ordinary_source_origins": ordinary_source_counts,
            "retrieval_requirements": len(retrieval_requests),
            "prompt_groups": len(groups),
            "trajectories": len(schedule),
            "scheduled_graph_eligible_trajectories": scheduled_graph,
            "replay_rows": len(replay_rows),
        },
        "gates": gates,
        "question_kg_join_stats": join,
        "replay_identity_overlap": replay_overlap,
        "protected_identity_overlap": {
            "population": population_protected_overlap,
            "replay": replay_protected_overlap,
        },
        "gate_semantics": {
            "dataset_name_used_as_gate_feature": False,
            "gold_answer_used_as_gate_feature": False,
            "fail_closed": True,
            "historical_cutoff": cutoff,
        },
        "scientific_boundary": {
            "train_only_gold_use": "outcome reward label only",
            "gold_inserted_into_prompt_passages_or_kg": False,
            "source_gold_steps_copied": 0,
            "failed_qpeg_or_saeg_p_edges_consumed": False,
            "replay_regenerated": False,
            "evaluation_protocol_modified": False,
            "training_started": False,
            "old_assets_overwritten": False,
        },
        "inputs": {
            "protocol": _identity(protocol_path),
            "protocol_manifest": _identity(protocol_manifest_path),
            "parent": {name: _identity(path) for name, path in parent_files.items()},
            "ordinary200_source_release": {
                name: _identity(path) for name, path in ordinary_release_files.items()
            },
            "hm_reconciliation": {
                name: _identity(path) for name, path in hm_reconciliation_files.items()
            },
            "expansion_retrieval_contexts": _identity(expansion_path),
            "proof_supply": {name: _identity(path) for name, path in proof_files.items()},
            "replay": {name: _identity(path) for name, path in replay_files.items()},
            "raw_train": {dataset: _identity(path) for dataset, path in raw_paths.items()},
            "release_metadata": optional_inputs,
        },
        "outputs": {},
        "training_started": False,
    }
    report["outputs"] = {name: _identity(path) for name, path in outputs.items()}
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest_path = dump_manifest(
        output_dir,
        status=STATUS,
        extra={
            "phase": "mixed_ppo_v4_proof800_data_materialization",
            "experiment_id": report["experiment_id"],
            "protocol": report["inputs"]["protocol"],
            "protocol_manifest": report["inputs"]["protocol_manifest"],
            "report": _identity(report_path),
            "outputs": report["outputs"],
            # Bind both official unified-v3 data payloads and its release
            # report/manifest explicitly.  The report hash above already
            # closes this chain, but retaining the direct identities makes a
            # downstream preflight able to reject source-contract drift
            # without interpreting an older v2 layout.
            "proof_supply": {
                "schema_version": EXPECTED_PROOF_SUPPLY_SCHEMA,
                "status": EXPECTED_PROOF_SUPPLY_STATUS,
                "payloads": report["inputs"]["proof_supply"],
                "release_metadata": report["inputs"]["release_metadata"][
                    "proof_supply"
                ],
            },
            "replay": report["inputs"]["replay"],
            "training_started": False,
        },
    )
    report["outputs"]["report"] = _identity(report_path)
    report["outputs"]["manifest"] = _identity(manifest_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT)
    parser.add_argument(
        "--expansion-retrieval", type=Path, default=DEFAULT_EXPANSION_RETRIEVAL
    )
    parser.add_argument("--proof-supply-dir", type=Path, default=DEFAULT_PROOF_SUPPLY)
    parser.add_argument(
        "--protected-ledger-dir",
        type=Path,
        default=DEFAULT_PROTECTED_LEDGER_DIR,
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    args = parser.parse_args()
    report = materialize(
        protocol_path=args.protocol,
        parent_dir=args.parent_dir,
        expansion_retrieval=args.expansion_retrieval,
        proof_supply_dir=args.proof_supply_dir,
        protected_ledger_dir=args.protected_ledger_dir,
        data_root=args.data_root,
        replay_dir=args.replay_dir,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
    )
    print(
        json.dumps(
            {"status": report["status"], "counts": report["counts"], "gates": report["gates"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
