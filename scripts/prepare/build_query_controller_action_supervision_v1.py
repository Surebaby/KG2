#!/usr/bin/env python
"""Build Controller-v1 q1/q2_dynamic action supervision.

The release is deliberately narrow: it accepts only clean, linear, two-step
2WikiMultiHopQA relation plans and MuSiQue subquery plans.  The first query is
rendered from the existing answer-free planner target.  The second query is
rendered only after binding the *train-side intermediate answer* to an
annotated supporting sentence.  Final answers and future-hop tails are never
serialized into controller state or targets.

This is a data materializer, not a trainer.  Output directories are append-only
and an existing directory is never overwritten.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.eval.query_controller_v1 import (
    ActionValidationError,
    OBSERVATION_ANNOTATION_PATHS,
    OBSERVATION_BINDING_METHODS,
    OBSERVATION_FIELDS,
    OBSERVATION_PROVENANCE_FIELDS,
    PREVIOUS_ACTION_FIELDS,
    SCHEMA_VERSION,
    STATE_VERSION,
    validate_action_record,
)
from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.retrieval.dependent import (
    dependency_refs,
    instantiate_dependent_queries,
    normalize_dependency_ref,
    render_root_query,
)
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.build_query_planner_supervision import _is_conservative_alias
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("2wikimultihopqa", "musique")
DEFAULT_SPLIT_ROOT = Path(
    "data/silver_data/query_planner_supervision_split_v1_seed20260829"
)
DEFAULT_OUTPUT_DIR = Path("data/silver_data/query_controller_actions_v1_seed42_v4_4")
DEFAULT_CANDIDATE_OUTPUT_DIR = Path(
    "data/silver_data/query_controller_action_candidates_v1_seed42_v4_1"
)
DEFAULT_PROTOCOL_DIR = Path(
    "outputs/audits/query_controller_v1_exact_text_pilot_seed42_protocol_v4_4"
)
DEFAULT_EXPERIMENT_ID = "QUERY-CONTROLLER-ACTION-V1-SEED42-V4-4"
DEFAULT_CONSUMED_COHORTS = (
    Path(
        "outputs/audits/subquestion_decomposition_v8_cohort_freeze_"
        "dev30_prospective300_seed20260904_v1/development.identity_only.jsonl"
    ),
    Path(
        "outputs/audits/subquestion_decomposition_v8_consumed_smoke4x3_"
        "seed20260904_v1/smoke.identity_only.jsonl"
    ),
    Path(
        "outputs/audits/subquestion_decomposition_v9_canonical_subqa_"
        "pilot30x3_seed20260904_v1/pilot.identity_only.jsonl"
    ),
)

RELEASE_SCHEMA_VERSION = "query-controller-action-release-v4.4-pair-locked-same-identities"
STATUS = "COMPLETE_V4_4_PAIR_LOCKED_SAME_IDENTITIES_NOT_TRAINED"
CANDIDATE_SCHEMA_VERSION = "query-controller-action-candidate-pool-v2-pair-hashed"
CANDIDATE_STATUS = "COMPLETE_V4_HARDENED_ELIGIBLE_CANDIDATES_NOT_COHORT_NOT_TRAINED"
FORMAL_PROTOCOL_SCHEMA_VERSION = "query-controller-v1-pilot-protocol-4.4"
FORMAL_PROTOCOL_REPORT_SCHEMA_VERSION = "query-controller-v1-pilot-freeze-report-4.4"
FORMAL_PROTOCOL_MANIFEST_SCHEMA_VERSION = "query-controller-v1-pilot-manifest-4.4"
FORMAL_PROTOCOL_EXPERIMENT_ID = (
    "QUERY-CONTROLLER-V1-EXACT-TEXT-PILOT-SEED42-PROTOCOL-V4_4"
)
FORMAL_PROTOCOL_STATUS = (
    "FROZEN_V4_4_PEFT_DEFAULT_FP32_CLEAN_RELOAD_"
    "SAME_IDENTITIES_2DATASET_ACTIONS_NOT_TRAINED"
)
FORMAL_CANDIDATE_HASHES = {
    "train": "67cdb276c02a42ecd76febae947a1d0ded64c182b30ca21ad1d02e8ce83dc281",
    "dev": "a74a24e92120537562b63f277f0c220291c4355e22e90ea156873b7b2aa3fa83",
    "pair_hashes": "3669a4e788ef306ee86f3d80fe40c70bc8fbc72d8aa6ec74eeac3447484d5b00",
}
FORMAL_V4_3_IDENTITY_HASHES = {
    "train": "2a31e10a1d37e2090e9909fe05975fba3179c1f4301b8fe0a3e1c94192da2da3",
    "dev": "5c78577ccf5bea18e401f1863580871b09d09d1a0483adcdf1a9816172dc9a07",
    "confirmation": "b88bbae2c0758e5f7ffc77ff4aefff4dae6b822c967ba77108229d38b5e4b46b",
}
FORMAL_V4_3_ACTION_HASHES = {
    "train": "2ded8b3a8bd9fa42eba657307e8cdf816297d58542f2abba0c8014b9b93789ca",
    "dev": "86b405599d845b2b35a5e1e3e9557fce9bd19a85530db7ebc67d96050e59ad19",
    "confirmation": "87336a5ff0efe5ac90b51ea616a61cd1734dc22dfc8eb93a3a3d3fd06dcb796c",
}
FORMAL_V4_3_FAILURE_HASHES = {
    "protocol": "8ba09daae131c3a2a2860bbc7f28dfb4e71fb46bc03585921a47e0c4aa92a998",
    "protocol_supersession_addendum": (
        "11349678124371a42b32f8542b5a0cc5a672ff31398d6693e9ab29114734b73a"
    ),
    "probe_manifest": "65f9d7593116cb1bed13b336118355432458b3cd32dd014874d3dc47c9002eb1",
    "probe_failure_addendum": (
        "8ee7b3dfdb5de3b051fbd26e80ff416f3b7d5b4ce47b628c74b8d7d8d4cb35c1"
    ),
    "training_history": "1206cbc7af5ca9d54a701cd16b142320f7a81d4c65d482adcbf88a6d6bd8d113",
}
FORMAL_SEALED_PARENT_HASHES = {
    "report": "233b931716d96e0a6e40e0cb2c0e961a5c79c04884d6cac584c301e9ce9fe4b7",
    "manifest": "cda6525e1562697c31e17cb457280fe272de039ebebee23a2ddcabaa942730e6",
    "prospective_declared": (
        "36b680cabef059dae7370bb131b1bafc0f120baf372f4e7666aa0e2d13b13c99"
    ),
}
FORMAL_IMPLEMENTATION_PATHS = {
    "central_action_validator": Path("kgproweight/eval/query_controller_v1.py"),
    "protocol_freezer": Path("scripts/prepare/freeze_query_controller_v1_protocol.py"),
    "action_builder": Path(
        "scripts/prepare/build_query_controller_action_supervision_v1.py"
    ),
    "controller_trainer": Path("kgproweight/training/query_controller.py"),
    "controller_train_cli": Path("scripts/train/query_controller.py"),
    "controller_greedy_runner": Path("kgproweight/eval/query_controller_runner.py"),
    "controller_generate_cli": Path("scripts/eval/generate_query_controller_actions.py"),
    "controller_mechanism_scorer": Path(
        "scripts/eval/evaluate_query_controller_actions.py"
    ),
}
FORMAL_RUNTIME_CONFIG = {
    "phase": "probe",
    "experiment_id": "QUERY-CONTROLLER-ACTION-V1-PROBE20-SEED42-V4-4",
    "model": {
        "base_model": "llama3-8B-instruct",
        "method": "qlora",
        "initialization": "base_instruct",
        "init_adapter_path": None,
        "load_in_4bit": True,
        "dtype": "bf16",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "data": {
        "allowed_source_actions": ["text"],
        "train_quotas": {
            dataset: {"q1": 32, "q2_dynamic": 32} for dataset in DATASETS
        },
        "dev_quotas": {
            dataset: {"q1": 16, "q2_dynamic": 16} for dataset in DATASETS
        },
    },
    "training": {
        "seed": 42,
        "max_seq_length": 1024,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 0.0001,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "num_train_epochs": 1,
        "max_steps": 20,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "logging_steps": 1,
        "eval_strategy": "no",
        "eval_steps": 20,
        "save_strategy": "no",
        "save_steps": 20,
        "save_total_limit": 1,
        "verify_saved_adapter_reload": True,
    },
}
FORMAL_TRAINING_PROBE_GATES = {
    "cuda_runtime_available": True,
    "finite_loss_rate": 1.0,
    "nonzero_supervised_token_rate": 1.0,
    "nonzero_trainable_gradient_observed": True,
    "adapter_save_reload_exact": True,
    "adapter_save_fidelity_exact": True,
    "adapter_clean_reload_single_adapter": True,
    "adapter_clean_reload_tensor_exact": True,
    "adapter_dtype_inventories_recorded": True,
    "adapter_saved_live_clean_reload_dtype_inventories_equal": True,
    "oom_count": 0,
    "nan_or_inf_count": 0,
    "dev_only_validation": True,
    "confirmation_open_count": 0,
    "prospective_open_or_hash_count": 0,
}
FORMAL_PROBE_EVALUATION_CONTRACT = {
    "authorization": "post_probe_dev_teacher_forced_only",
    "cohort_role": "dev",
    "input_split": "dev",
    "datasets": list(DATASETS),
    "slots": ["q1", "q2_dynamic"],
    "confirmation_access": False,
    "prospective_access": False,
    "exact_qids_per_enabled_dataset": 60,
    "exact_action_rows_per_enabled_dataset": 120,
    "exact_actions": 240,
    "state_source": "annotation_derived_but_passage_bound",
    "runtime_reader_predicted": False,
    "required_probe_artifacts": {
        "experiment_id": "QUERY-CONTROLLER-ACTION-V1-PROBE20-SEED42-V4-4",
        "manifest_status": "COMPLETE",
        "completed_optimizer_steps": 20,
        "training_manifest_sha256": "required_external_eval_lock",
        "adapter_sha256": "required_external_eval_lock",
        "asset_lock_lineage_match": True,
    },
    "decoding": {
        "strategy": "greedy",
        "do_sample": False,
        "temperature": 0.0,
        "seed": 42,
        "batch_size": 4,
        "max_input_tokens": 1024,
        "max_new_tokens": 192,
        "base_model": "llama3-8B-instruct",
        "dtype": "bf16",
        "load_in_4bit": True,
    },
    "mechanism_gates": {
        "identity_join_rate": 1.0,
        "query_nonrepeat_rate": 1.0,
        "placeholder_free_rate": 1.0,
        "dependency_closed_rate_min_each_dataset_slot": {
            "q1": 0.97,
            "q2_dynamic": 0.95,
        },
        "state_use_valid_rate_min_each_dataset_slot": {
            "q1": 0.97,
            "q2_dynamic": 0.95,
        },
        "schema_valid_rate_min_each_dataset_slot": {
            "q1": 0.97,
            "q2_dynamic": 0.95,
        },
    },
    "outcome_metrics_authorized": {"em": False, "f1": False, "ihr": False},
    "status_semantics": {
        "generation_complete_status": "COMPLETE_GENERATION_NOT_MECHANISM_PASS",
        "mechanism_pass_requires_separate_scorer_gate": True,
        "generation_complete_implies_mechanism_pass": False,
    },
    "scientific_boundary": (
        "Teacher-forced action mechanics on the frozen release dev split only; "
        "not Reader-predicted runtime and not retrieval or QA utility."
    ),
}
FORMAL_FUTURE_GOLD_FREE_MECHANISM_GATE = {
    "authorization": (
        "not granted by this protocol; requires frozen adapter and a new append-only "
        "runtime protocol"
    ),
    "cohort": "train-side confirmation30 per enabled dataset",
    "runtime_intermediate_source": "Reader-predicted and passage-bound",
    "annotation_derived_intermediate_visible_at_runtime": False,
    "gold_access_count": 0,
    "runtime_error_count": 0,
    "active_adapter_hash_match_rate": 1.0,
    "cache_and_call_accounting_rate": 1.0,
    "q1_schema_valid_rate_min_each_dataset": 0.97,
    "q2_dynamic_schema_valid_rate_min_each_dataset": 0.95,
    "query_nonrepeat_rate": 1.0,
    "placeholder_free_rate": 1.0,
    "a1_admissible_rate_min_each_dataset": 0.40,
    "dynamic_state_binding_integrity_rate": 1.0,
    "dynamic_transition_rate_min_each_dataset": 0.50,
    "final_passage_budget_and_unique_rate": 1.0,
    "fallback_byte_identity_rate": 1.0,
    "reader_one_shot_regression_identity_rate": 1.0,
}
FORMAL_OUTCOME_UNLOCK_RULE = {
    "em_f1_authorized_now": False,
    "required_sequence": [
        "pass every data-release gate",
        "pass the 20-step training probe gates",
        "train and freeze a separately authorized Controller adapter",
        "freeze a new Gold-free runtime protocol",
        "pass every train-side confirmation Gold-free mechanism gate",
        "freeze prediction bytes and hashes",
        "obtain separate authorization for independent outcome scoring",
    ],
    "sealed_prospective900": (
        "remains unopened and unhashed; this protocol implements no unlock"
    ),
    "ihr": "not authorized before frozen prospective EM/F1",
}
_SPACE_RE = re.compile(r"\s+")
_DUPLICATE_ARTICLE_RE = re.compile(r"\b(a|an|the)\s+\1\b", re.IGNORECASE)


@dataclass(frozen=True)
class PlannerCandidate:
    dataset: str
    qid: str
    question: str
    family: str
    plan: Mapping[str, Any]
    planner_path: str
    planner_sha256: str


class BuildReject(ValueError):
    """A source row cannot safely yield a Controller-v1 action pair."""

    def __init__(
        self, code: str, message: str = "", *, details: Mapping[str, Any] | None = None
    ):
        self.code = code
        self.details = dict(details or {})
        super().__init__(message or code)


def _clean(value: object) -> str:
    return _SPACE_RE.sub(" ", str(value or "").strip())


def _surface(value: object) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", str(value or "").casefold()).split())


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_adjacent_duplicate_articles(value: str) -> tuple[str, int]:
    """Collapse repeated adjacent English articles without guessing semantics.

    The casing of the second occurrence is retained: ``the The WB`` becomes
    ``The WB``.  Repeated runs are collapsed to one article deterministically.
    """

    text = _clean(value)
    total = 0
    while True:
        match = _DUPLICATE_ARTICLE_RE.search(text)
        if match is None:
            return text, total
        second = match.group(0).split()[-1]
        text = f"{text[:match.start()]}{second}{text[match.end():]}"
        text = _clean(text)
        total += 1


def canonical_action_pair_sha256(
    pair: Sequence[Mapping[str, Any]], *, split_override: str | None = None
) -> str:
    """Hash the exact canonical q1/q2 action pair for a release role.

    Candidate confirmation rows originate in the old planner train split.
    ``split_override='confirmation'`` lets the freezer lock the exact bytes the
    final materializer must emit, rather than hashing the source-role record.
    """

    if len(pair) != 2:
        raise ValueError("action pair must contain exactly two records")
    canonical: list[dict[str, Any]] = []
    for source in pair:
        row = json.loads(json.dumps(dict(source), ensure_ascii=False))
        if split_override is not None:
            row["split"] = split_override
        validate_action_record(row, expected_split=row["split"])
        canonical.append(row)
    canonical.sort(key=lambda row: int(row["turn_index"]))
    if [row["slot"] for row in canonical] != ["q1", "q2_dynamic"]:
        raise ValueError("action pair must contain q1 then q2_dynamic")
    identities = {(row["dataset"], row["qid"]) for row in canonical}
    if len(identities) != 1:
        raise ValueError("action pair identity mismatch")
    payload = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def action_pair_hash_rows(
    records: Sequence[Mapping[str, Any]], *, source_split: str
) -> list[dict[str, str]]:
    """Return a deterministic pair-hash index for a candidate action file."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("split") != source_split:
            raise ValueError("candidate action split does not match pair-index source split")
        grouped[(str(record.get("dataset")), str(record.get("qid")))].append(record)
    result: list[dict[str, str]] = []
    for (dataset, qid), pair in sorted(grouped.items()):
        families = {str(row.get("family_sha256")) for row in pair}
        if len(families) != 1:
            raise ValueError(f"pair family mismatch: {dataset}::{qid}")
        result.append(
            {
                "dataset": dataset,
                "qid": qid,
                "source_split": source_split,
                "family_sha256": next(iter(families)),
                "action_pair_canonical_sha256": canonical_action_pair_sha256(pair),
            }
        )
    return result


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"row is not an object at {path}:{line_number}")
            yield row


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def _jsonl_rows_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (
                json.dumps(
                    dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _canonical_dependency(value: object) -> str | None:
    normalized = normalize_dependency_ref(value)
    if normalized is not None:
        return normalized
    text = _clean(value)
    match = re.fullmatch(r"(?:slot|q)_([1-9]\d*)", text, flags=re.IGNORECASE)
    return f"slot_{int(match.group(1))}" if match else None


def is_clean_linear_two_step(row: Mapping[str, Any]) -> bool:
    """Return whether an old planner record is a strict executable chain."""

    dataset = _clean(row.get("dataset")).lower()
    target_type = _clean(row.get("target_type"))
    target = row.get("target")
    steps = target.get("steps") if isinstance(target, Mapping) else None
    if dataset not in DATASETS or not isinstance(steps, list) or len(steps) != 2:
        return False
    if not all(isinstance(step, Mapping) for step in steps):
        return False
    first, second = steps
    first_dependencies = list(first.get("dependencies") or [])
    second_dependencies = list(second.get("dependencies") or [])
    if first_dependencies or len(second_dependencies) != 1:
        return False
    if _canonical_dependency(second_dependencies[0]) != "slot_1":
        return False
    if dataset == "2wikimultihopqa":
        return bool(
            target_type == "relation_graph"
            and _clean(first.get("subject"))
            and not dependency_refs(first.get("subject"))
            and dependency_refs(second.get("subject")) == ["slot_1"]
            and _clean(first.get("relation_label") or first.get("relation"))
            and _clean(second.get("relation_label") or second.get("relation"))
        )
    return bool(
        target_type == "subquery_graph"
        and _clean(first.get("subquery_template"))
        and not dependency_refs(first.get("subquery_template"))
        and dependency_refs(second.get("subquery_template")) == ["slot_1"]
    )


def _support_sentences_2wiki(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = row.get("metadata") or {}
    context = metadata.get("context") or {}
    titles = list(context.get("title") or [])
    contents = list(context.get("content") or [])
    support = metadata.get("supporting_facts") or {}
    support_titles = list(support.get("title") or [])
    support_indices = list(support.get("sent_id") or [])
    title_to_positions: dict[str, list[int]] = defaultdict(list)
    for position, title in enumerate(titles):
        title_to_positions[_surface(title)].append(position)
    result: list[dict[str, Any]] = []
    for title, sentence_index in zip(support_titles, support_indices):
        if type(sentence_index) is not int or sentence_index < 0:
            continue
        positions = title_to_positions.get(_surface(title), [])
        if len(positions) != 1:
            continue
        position = positions[0]
        sentences = contents[position] if position < len(contents) else []
        if not isinstance(sentences, list) or sentence_index >= len(sentences):
            continue
        excerpt = _clean(sentences[sentence_index])
        if excerpt:
            result.append(
                {
                    "document_id": f"2wikimultihopqa::{row.get('id')}::context::{position}",
                    "document_title": _clean(title),
                    "sentence_index": sentence_index,
                    "evidence_excerpt": excerpt,
                }
            )
    return result


def _contains_surface(container: object, needle: object) -> bool:
    haystack = _surface(container)
    item = _surface(needle)
    return bool(item and re.search(rf"(?<!\w){re.escape(item)}(?!\w)", haystack))


def _bind_2wiki_intermediate(row: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    metadata = row.get("metadata") or {}
    evidences = metadata.get("evidences") or {}
    facts = list(evidences.get("fact") or [])
    tails = list(evidences.get("entity") or [])
    if len(facts) != 2 or len(tails) != 2:
        raise BuildReject("raw_not_linear_two_step")
    intermediate = _clean(tails[0])
    if not intermediate:
        raise BuildReject("missing_intermediate_annotation")
    candidates = [
        item for item in _support_sentences_2wiki(row)
        if _contains_surface(item["evidence_excerpt"], intermediate)
    ]
    title_matches = [
        item for item in candidates
        if _surface(item["document_title"]) == _surface(facts[0])
        or _is_conservative_alias(item["document_title"], facts[0])
    ]
    if len(title_matches) == 1:
        chosen, method = title_matches[0], "fact_title_and_answer_surface"
    else:
        # Never bind merely because the intermediate surface occurs in one
        # support sentence: that sentence may be the future hop.  The first-hop
        # evidence title must also agree with the annotated first fact.
        raise BuildReject("ambiguous_or_missing_intermediate_support")
    chosen = dict(chosen)
    chosen["binding_method"] = method
    chosen["annotation_path"] = "metadata.evidences.entity[0]"
    return intermediate, chosen


def _sentences(text: object) -> list[str]:
    clean = _clean(text)
    if not clean:
        return []
    return [_clean(item) for item in re.split(r"(?<=[.!?])\s+", clean) if _clean(item)]


def _bind_musique_intermediate(row: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    decomposition = (
        ((row.get("metadata") or {}).get("metadata") or {}).get("question_decomposition")
        or []
    )
    if len(decomposition) != 2 or not isinstance(decomposition[0], Mapping):
        raise BuildReject("raw_not_linear_two_step")
    first = decomposition[0]
    intermediate = _clean(first.get("answer"))
    support = first.get("support_paragraph") or {}
    if not intermediate or not isinstance(support, Mapping):
        raise BuildReject("missing_intermediate_annotation")
    matches = [
        (index, sentence)
        for index, sentence in enumerate(_sentences(support.get("paragraph_text")))
        if _contains_surface(sentence, intermediate)
    ]
    if len(matches) != 1:
        raise BuildReject("ambiguous_or_missing_intermediate_support")
    sentence_index, excerpt = matches[0]
    paragraph_id = support.get("idx", first.get("paragraph_support_idx"))
    binding = {
        "document_id": f"musique::{row.get('id')}::paragraph::{paragraph_id}",
        "document_title": _clean(support.get("title")) or f"paragraph-{paragraph_id}",
        "sentence_index": sentence_index,
        "evidence_excerpt": excerpt,
        "binding_method": "decomposition_step_support_answer_surface",
        "annotation_path": "metadata.metadata.question_decomposition[0].answer",
    }
    return intermediate, binding


def _future_secrets(dataset: str, row: Mapping[str, Any]) -> list[str]:
    secrets = [_clean(value) for value in (row.get("golden_answers") or [])]
    if dataset == "2wikimultihopqa":
        tails = list((((row.get("metadata") or {}).get("evidences") or {}).get("entity") or []))
        if len(tails) >= 2:
            secrets.append(_clean(tails[1]))
    else:
        decomposition = (
            ((row.get("metadata") or {}).get("metadata") or {}).get("question_decomposition")
            or []
        )
        if len(decomposition) >= 2 and isinstance(decomposition[1], Mapping):
            secrets.append(_clean(decomposition[1].get("answer")))
    return list(dict.fromkeys(value for value in secrets if value))


def _short_secret_policy_audit(
    raw_by_dataset: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    single_alpha_qids: set[tuple[str, str]] = set()
    single_digit_qids: set[tuple[str, str]] = set()
    canonical_final_qids: set[tuple[str, str]] = set()
    single_alpha_alias_occurrences = 0
    single_digit_alias_occurrences = 0
    for dataset, rows in raw_by_dataset.items():
        for qid, row in rows.items():
            answers = [_clean(value) for value in (row.get("golden_answers") or []) if _clean(value)]
            if answers:
                canonical_final_qids.add((dataset, qid))
            for secret in _future_secrets(dataset, row):
                compact = _surface(secret).replace(" ", "")
                if len(compact) == 1 and compact.isalpha():
                    single_alpha_alias_occurrences += 1
                    single_alpha_qids.add((dataset, qid))
                elif len(compact) == 1 and compact.isdigit():
                    single_digit_alias_occurrences += 1
                    single_digit_qids.add((dataset, qid))
    return {
        "scope": "all strict planner candidates after identity/family exclusions, before support binding",
        "canonical_final_qids_checked": len(canonical_final_qids),
        "single_alphabetic_alias_occurrences": single_alpha_alias_occurrences,
        "single_alphabetic_alias_qids": len(single_alpha_qids),
        "single_alphabetic_policy": (
            "exact normalized intermediate/anchor equality is always rejected; broad "
            "one-letter containment is ambiguous and is not used; longer canonical/full "
            "final surfaces remain checked"
        ),
        "single_digit_alias_occurrences": single_digit_alias_occurrences,
        "single_digit_alias_qids": len(single_digit_qids),
        "single_digit_policy": "exact equality and whole-token semantic-text containment checked",
        "hash_identifier_fields_scanned": False,
    }


def _relation_intent(step: Mapping[str, Any], target_type: str) -> str:
    if target_type == "relation_graph":
        return _clean(step.get("relation_label") or step.get("relation"))
    template = _clean(step.get("subquery_template"))
    if ">>" in template:
        relation = _clean(template.rpartition(">>")[2])
        if relation:
            return relation
    return _clean(re.sub(r"(?<!\w)#1(?!\w)", "prior result", template))


def _root_anchor(step: Mapping[str, Any], target_type: str) -> str | None:
    if target_type == "relation_graph":
        return _clean(step.get("subject")) or None
    template = _clean(step.get("subquery_template"))
    if ">>" in template:
        return _clean(template.partition(">>")[0]) or None
    return None


def _pid(step: Mapping[str, Any]) -> str | None:
    value = step.get("pid")
    return _clean(value) or None


def _observation(intermediate: str, binding: Mapping[str, Any]) -> dict[str, Any]:
    excerpt = str(binding["evidence_excerpt"])
    return {
        "answer": intermediate,
        "answer_sha256": _sha256_text(intermediate),
        "evidence_excerpt": excerpt,
        "evidence_excerpt_sha256": _sha256_text(excerpt),
        "document_id": str(binding["document_id"]),
        "document_title": str(binding["document_title"]),
        "sentence_index": int(binding["sentence_index"]),
        "provenance": {
            "source": "train_annotation_support",
            "annotation_path": str(binding["annotation_path"]),
            "binding_method": str(binding["binding_method"]),
        },
    }


def _semantic_visible_text_leaves(value: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Project only semantic model-visible text, never IDs, hashes or metadata."""

    result: list[tuple[str, str]] = []
    state = value.get("state") if isinstance(value.get("state"), Mapping) else {}
    target = value.get("target") if isinstance(value.get("target"), Mapping) else {}
    for index, action in enumerate(state.get("previous_actions") or []):
        if isinstance(action, Mapping) and isinstance(action.get("query"), str):
            result.append((f"state.previous_actions[{index}].query", action["query"]))
    for index, observation in enumerate(state.get("verified_observations") or []):
        if not isinstance(observation, Mapping):
            continue
        for field in ("answer", "evidence_excerpt", "document_title"):
            if isinstance(observation.get(field), str):
                result.append(
                    (f"state.verified_observations[{index}].{field}", observation[field])
                )
    for field in ("query", "anchor", "relation_intent"):
        if isinstance(target.get(field), str):
            result.append((f"target.{field}", target[field]))
    return result


def _leaks_future(
    value: Mapping[str, Any], *, secrets: Sequence[str], question: str
) -> dict[str, str] | None:
    """Detect future aliases in semantic controller text without hash false positives.

    Intermediate answers and target anchors are compared to every final alias
    by exact normalized equality, including one-character aliases.  A
    single-digit alias is also checked by whole-token containment over all
    semantic leaves.  Broad containment for a single alphabetic character is
    intentionally not used because it is ambiguous; that population is
    counted and disclosed separately in the candidate-pool report.  Longer
    canonical answers and aliases receive the normal whole-token scan.
    """

    leaves = _semantic_visible_text_leaves(value)
    exact_fields = {
        path: text
        for path, text in leaves
        if path.endswith(".answer") or path == "target.anchor"
    }
    for secret in secrets:
        normalized = _surface(secret)
        compact = normalized.replace(" ", "")
        if not compact:
            continue
        for path, text in exact_fields.items():
            if _surface(text) == normalized:
                return {
                    "secret": secret,
                    "path": path,
                    "match_kind": "intermediate_or_anchor_exact",
                }
        # A literal inherited from the original question cannot be removed by
        # action construction.  Exact intermediate/anchor equality above is
        # still rejected even in that case.
        if _contains_surface(question, secret):
            continue
        is_single_alpha = len(compact) == 1 and compact.isalpha()
        if is_single_alpha:
            continue
        for path, text in leaves:
            if _contains_surface(text, secret):
                return {
                    "secret": secret,
                    "path": path,
                    "match_kind": (
                        "single_digit_semantic_containment"
                        if len(compact) == 1 and compact.isdigit()
                        else "semantic_containment"
                    ),
                }
    return None


def build_action_pair(
    candidate: PlannerCandidate,
    raw_row: Mapping[str, Any],
    *,
    split: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build and strictly validate q1 and q2_dynamic for one train row."""

    if split not in {"train", "dev", "confirmation"}:
        raise BuildReject("invalid_split")
    raw_qid = _clean(raw_row.get("id") or raw_row.get("qid"))
    raw_question = _clean(raw_row.get("question"))
    if raw_qid != candidate.qid or question_sha256(raw_question) != question_sha256(candidate.question):
        raise BuildReject("raw_identity_mismatch")
    if not is_clean_linear_two_step(
        {"dataset": candidate.dataset, **dict(candidate.plan)}
    ):
        raise BuildReject("planner_not_clean_linear_two_step")

    target_type = str(candidate.plan["target_type"])
    steps = candidate.plan["target"]["steps"]
    if candidate.dataset == "2wikimultihopqa":
        intermediate, binding = _bind_2wiki_intermediate(raw_row)
        slot_values = {"hop_1": intermediate}
    elif candidate.dataset == "musique":
        intermediate, binding = _bind_musique_intermediate(raw_row)
        slot_values = {"step_1": intermediate}
    else:  # defensive; PlannerCandidate is public to tests/callers
        raise BuildReject("unsupported_dataset")

    try:
        raw_q1_query = render_root_query(steps[0], target_type)
        q2_queries = instantiate_dependent_queries(
            steps[1], target_type, slot_values, max_variants=1
        )
    except ValueError as exc:
        raise BuildReject("query_render_failed", str(exc)) from exc
    q1_query, q1_article_normalizations = normalize_adjacent_duplicate_articles(
        raw_q1_query
    )
    q2_query, q2_article_normalizations = normalize_adjacent_duplicate_articles(
        q2_queries[0]
    )
    raw_q1_anchor = _root_anchor(steps[0], target_type)
    if raw_q1_anchor is None:
        q1_anchor, q1_anchor_normalizations = None, 0
    else:
        q1_anchor, q1_anchor_normalizations = normalize_adjacent_duplicate_articles(
            raw_q1_anchor
        )
    q1_relation_intent, q1_relation_normalizations = normalize_adjacent_duplicate_articles(
        _relation_intent(steps[0], target_type)
    )
    q2_relation_intent, q2_relation_normalizations = normalize_adjacent_duplicate_articles(
        _relation_intent(steps[1], target_type)
    )
    q1_normalized_fields = [
        field
        for field, count in (
            ("query", q1_article_normalizations),
            ("anchor", q1_anchor_normalizations),
            ("relation_intent", q1_relation_normalizations),
        )
        if count
    ]
    q2_normalized_fields = [
        field
        for field, count in (
            ("query", q2_article_normalizations),
            ("relation_intent", q2_relation_normalizations),
        )
        if count
    ]
    q1_normalization_count = (
        q1_article_normalizations + q1_anchor_normalizations + q1_relation_normalizations
    )
    q2_normalization_count = q2_article_normalizations + q2_relation_normalizations
    observation = _observation(intermediate, binding)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "dataset": candidate.dataset,
        "qid": candidate.qid,
        "question_key": question_key(candidate.dataset, candidate.qid),
        "question_sha256": question_sha256(candidate.question),
        "family_sha256": candidate.family,
        "split": split,
    }
    planner_provenance = {
        "planner_record_path": candidate.planner_path,
        "planner_record_file_sha256": candidate.planner_sha256,
        "planner_target_type": target_type,
        "source_action_policy": "text_only_v1",
    }
    q1 = {
        **identity,
        "example_id": f"{candidate.dataset}::{candidate.qid}::q1",
        "slot": "q1",
        "turn_index": 1,
        "state": {
            "state_version": STATE_VERSION,
            "original_question": candidate.question,
            "previous_actions": [],
            "verified_observations": [],
        },
        "target": {
            "action": "retrieve",
            "query": q1_query,
            "anchor": q1_anchor,
            "relation_intent": q1_relation_intent,
            "pid": _pid(steps[0]),
            "dependencies": [],
            "output_slot": "q1",
            "source_action": "text",
        },
        "source_provenance": {
            **planner_provenance,
            "train_intermediate_annotation_used": False,
            "adjacent_duplicate_article_normalized": bool(q1_normalization_count),
            "adjacent_duplicate_article_normalization_count": q1_normalization_count,
            "adjacent_duplicate_article_normalized_fields": q1_normalized_fields,
        },
        "gold_boundary": {
            "train_intermediate_annotation_used": False,
            "gold_final_answer_visible": False,
            "evaluation_gold_access": False,
        },
    }
    q2 = {
        **identity,
        "example_id": f"{candidate.dataset}::{candidate.qid}::q2_dynamic",
        "slot": "q2_dynamic",
        "turn_index": 2,
        "state": {
            "state_version": STATE_VERSION,
            "original_question": candidate.question,
            "previous_actions": [
                {
                    "slot": "q1",
                    "action": "retrieve",
                    "query": q1_query,
                    "output_slot": "q1",
                }
            ],
            "verified_observations": [observation],
        },
        "target": {
            "action": "retrieve",
            "query": q2_query,
            "anchor": intermediate,
            "relation_intent": q2_relation_intent,
            "pid": _pid(steps[1]),
            "dependencies": ["q1"],
            "output_slot": "q2",
            "source_action": "text",
        },
        "source_provenance": {
            **planner_provenance,
            "raw_annotation_path": str(binding["annotation_path"]),
            "support_document_id": str(binding["document_id"]),
            "support_document_title_sha256": _sha256_text(str(binding["document_title"])),
            "support_sentence_index": int(binding["sentence_index"]),
            "support_excerpt_sha256": _sha256_text(str(binding["evidence_excerpt"])),
            "binding_method": str(binding["binding_method"]),
            "train_intermediate_annotation_used": True,
            "adjacent_duplicate_article_normalized": bool(q2_normalization_count),
            "adjacent_duplicate_article_normalization_count": q2_normalization_count,
            "adjacent_duplicate_article_normalized_fields": q2_normalized_fields,
        },
        "gold_boundary": {
            "train_intermediate_annotation_used": True,
            "gold_final_answer_visible": False,
            "evaluation_gold_access": False,
        },
    }

    secrets = _future_secrets(candidate.dataset, raw_row)
    # source_provenance contains hashes only; the model-visible controller state
    # and target are checked directly for any future-hop/final-answer literal.
    for record in (q1, q2):
        visible = {"state": record["state"], "target": record["target"]}
        leaked = _leaks_future(visible, secrets=secrets, question=candidate.question)
        if leaked is not None:
            raise BuildReject(
                "final_or_future_tail_leak",
                details={
                    "match_kind": leaked["match_kind"],
                    "field": leaked["path"],
                    "secret_compact_length": len(_surface(leaked["secret"]).replace(" ", "")),
                },
            )
        validate_action_record(record, expected_split=split)
    return q1, q2


def _candidate_from_row(
    row: Mapping[str, Any], *, planner_path: Path, planner_sha256: str
) -> PlannerCandidate | None:
    if not is_clean_linear_two_step(row):
        return None
    dataset = _clean(row.get("dataset")).lower()
    qid = _clean(row.get("qid"))
    question = _clean(row.get("question"))
    if not qid or not question:
        return None
    if row.get("question_key") != question_key(dataset, qid):
        return None
    if row.get("question_sha256") != question_sha256(question):
        return None
    return PlannerCandidate(
        dataset=dataset,
        qid=qid,
        question=question,
        family=family_sha256(question),
        plan={
            "target_type": row["target_type"],
            "target": row["target"],
        },
        planner_path=str(planner_path),
        planner_sha256=planner_sha256,
    )


def _load_exclusions(paths: Sequence[Path]) -> tuple[set[tuple[str, str]], set[tuple[str, str]], list[dict[str, Any]]]:
    qids: set[tuple[str, str]] = set()
    families: set[tuple[str, str]] = set()
    inventory: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"required exclusion asset missing: {path}")
        rows = 0
        for row in _read_jsonl(path):
            dataset = _clean(row.get("dataset")).lower()
            qid = _clean(row.get("qid"))
            question = _clean(row.get("question"))
            if dataset not in DATASETS:
                continue
            if not qid or not question:
                raise ValueError(f"invalid exclusion identity in {path}")
            qids.add((dataset, qid))
            families.add((dataset, family_sha256(question)))
            rows += 1
        inventory.append({"path": str(path), "sha256": _sha256_file(path), "rows_used": rows})
    return qids, families, inventory


def _load_candidates(
    path: Path,
    *,
    excluded_qids: set[tuple[str, str]],
    excluded_families: set[tuple[str, str]],
) -> tuple[dict[str, dict[str, list[PlannerCandidate]]], Counter[str]]:
    digest = _sha256_file(path)
    groups: dict[str, dict[str, list[PlannerCandidate]]] = {
        dataset: defaultdict(list) for dataset in DATASETS
    }
    counts: Counter[str] = Counter()
    seen_qids: set[tuple[str, str]] = set()
    for row in _read_jsonl(path):
        counts["rows"] += 1
        candidate = _candidate_from_row(row, planner_path=path, planner_sha256=digest)
        if candidate is None:
            counts["not_clean_linear_two_step"] += 1
            continue
        key = (candidate.dataset, candidate.qid)
        if key in seen_qids:
            raise ValueError(f"duplicate planner qid in {path}: {key}")
        seen_qids.add(key)
        if key in excluded_qids:
            counts["excluded_qid"] += 1
            continue
        if (candidate.dataset, candidate.family) in excluded_families:
            counts["excluded_family"] += 1
            continue
        groups[candidate.dataset][candidate.family].append(candidate)
        counts[f"eligible_{candidate.dataset}"] += 1
    return groups, counts


def _load_json_object(path: Path, *, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"formal {role} missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"formal {role} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"formal {role} must be a JSON object: {path}")
    return value


def _canonical_object_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_formal_protocol(
    protocol_dir: Path,
    *,
    split_sizes: Mapping[str, int],
    expected_protocol_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fail closed on protocol identity, lineage, implementation, and seals."""

    paths = {
        "protocol": protocol_dir / "protocol.json",
        "protocol_report": protocol_dir / "report.json",
        "protocol_manifest": protocol_dir / "manifest.json",
    }
    documents = {
        role: _load_json_object(path, role=role) for role, path in paths.items()
    }
    digests = {role: _sha256_file(path) for role, path in paths.items()}
    if re.fullmatch(r"[0-9a-f]{64}", expected_protocol_sha256) is None:
        raise ValueError("formal expected protocol SHA256 is missing or malformed")
    if digests["protocol"] != expected_protocol_sha256:
        raise ValueError("formal protocol external SHA256 lock mismatch")
    protocol = documents["protocol"]
    report = documents["protocol_report"]
    manifest = documents["protocol_manifest"]

    for role, document, schema in (
        ("protocol", protocol, FORMAL_PROTOCOL_SCHEMA_VERSION),
        ("protocol report", report, FORMAL_PROTOCOL_REPORT_SCHEMA_VERSION),
        ("protocol manifest", manifest, FORMAL_PROTOCOL_MANIFEST_SCHEMA_VERSION),
    ):
        if document.get("schema_version") != schema:
            raise ValueError(f"formal {role} schema mismatch")
        if document.get("experiment_id") != FORMAL_PROTOCOL_EXPERIMENT_ID:
            raise ValueError(f"formal {role} Experiment ID mismatch")
        if document.get("status") != FORMAL_PROTOCOL_STATUS:
            raise ValueError(f"formal {role} status mismatch")

    body = dict(protocol)
    declared_body_hash = body.pop("protocol_body_canonical_sha256", None)
    if declared_body_hash != _canonical_object_sha256(body):
        raise ValueError("formal protocol canonical body hash mismatch")
    if protocol.get("enabled_training_datasets") != list(DATASETS):
        raise ValueError("formal protocol enabled dataset set/order mismatch")
    readiness = protocol.get("dataset_readiness") or {}
    if set(readiness) != {*DATASETS, "hotpotqa"}:
        raise ValueError("formal protocol dataset readiness schema mismatch")
    if any(
        (readiness.get(dataset) or {}).get("status")
        != "AUTHORIZED_EXACT_TEXT_SUPERVISION_ENGINEERING_RELEASE"
        for dataset in DATASETS
    ):
        raise ValueError("formal protocol enabled dataset readiness mismatch")
    if (readiness.get("hotpotqa") or {}).get("status") != "NOT_INCLUDED_LABEL_COVERAGE_UNKNOWN":
        raise ValueError("formal protocol must preserve HotpotQA coverage as UNKNOWN")

    cohort = protocol.get("cohort") or {}
    if cohort.get("source_split_for_role") != {
        "train": "train", "dev": "dev", "confirmation": "train"
    }:
        raise ValueError("formal protocol source split mapping mismatch")
    if cohort.get("qid_overlap_between_splits") != 0 or cohort.get(
        "family_overlap_between_splits"
    ) != 0:
        raise ValueError("formal protocol does not declare split isolation")
    if cohort.get("identity_fields") != [
        "dataset", "qid", "question_key", "question", "question_sha256",
        "family_sha256", "split", "action_pair_sha256",
    ]:
        raise ValueError("formal protocol identity/action-pair lock schema mismatch")
    declared_sizes = cohort.get("materialized_split_sizes_per_enabled_dataset")
    if declared_sizes != dict(split_sizes) or cohort.get(
        "split_sizes_per_enabled_dataset"
    ) != dict(split_sizes):
        raise ValueError("formal protocol split quota declaration mismatch")
    cohort_locks = cohort.get("cohort_locks") or {}
    if set(cohort_locks) != {"train", "dev", "confirmation"}:
        raise ValueError("formal protocol cohort-lock roles mismatch")
    for split, per_dataset in split_sizes.items():
        lock = cohort_locks.get(split) or {}
        expected_source = "train" if split == "confirmation" else split
        if (
            lock.get("rows") != per_dataset * len(DATASETS)
            or lock.get("per_dataset") != {dataset: per_dataset for dataset in DATASETS}
            or lock.get("source_planner_split") != expected_source
            or lock.get("gold_fields_emitted") is not False
            or re.fullmatch(r"[0-9a-f]{64}", str(lock.get("sha256") or "")) is None
            or lock.get("sha256") != FORMAL_V4_3_IDENTITY_HASHES[split]
        ):
            raise ValueError(f"formal protocol {split} cohort lock mismatch")
    identity_continuity = cohort.get("v4_3_identity_continuity") or {}
    if identity_continuity != {
        "selection_salt": (
            "QUERY-CONTROLLER-V1-EXACT-TEXT-PILOT-SEED42-PROTOCOL-V4_2"
        ),
        "predecessor_identity_sha256": FORMAL_V4_3_IDENTITY_HASHES,
        "successor_identity_sha256": FORMAL_V4_3_IDENTITY_HASHES,
        "byte_identical_to_v4_3": True,
        "comparison_used_identity_hashes_only": True,
        "confirmation_action_records_opened": False,
    }:
        raise ValueError("formal protocol v4.3 identity-continuity gate mismatch")

    contract = protocol.get("action_contract") or {}
    expected_contract = {
        "schema_version": SCHEMA_VERSION,
        "allowed_release_splits": ["train", "dev", "confirmation"],
        "trainer_allowed_splits": ["train", "dev"],
        "source_action": "text",
        "dual_source_routing": False,
        "anchor_nullable": True,
        "pid_nullable": True,
        "q1_previous_actions": 0,
        "q1_verified_observations": 0,
        "q2_previous_actions": 1,
        "q2_verified_observations": 1,
        "target_format": "exact JSON object; retrieve action only",
        "identity_lock_binds_canonical_q1_q2_action_pair_sha256": True,
        "nested_model_visible_state_fields_are_exact": True,
        "previous_action_fields": sorted(PREVIOUS_ACTION_FIELDS),
        "observation_fields": sorted(OBSERVATION_FIELDS),
        "observation_provenance_fields": sorted(OBSERVATION_PROVENANCE_FIELDS),
        "observation_provenance_source": "train_annotation_support",
        "observation_annotation_paths": sorted(OBSERVATION_ANNOTATION_PATHS),
        "observation_binding_methods": sorted(OBSERVATION_BINDING_METHODS),
    }
    if contract != expected_contract:
        raise ValueError("formal protocol action contract mismatch")
    training_contract = protocol.get("training_contract") or {}
    if training_contract.get("runtime_config") != FORMAL_RUNTIME_CONFIG:
        raise ValueError("formal protocol probe runtime config mismatch")
    if (
        training_contract.get("authorization")
        != "20-step probe only after every data-release gate passes"
        or training_contract.get("initialization")
        != "base Llama-3-8B instruct; independent Controller LoRA"
        or training_contract.get("forbidden_initialization")
        != ["Strong-SFT Reader adapter", "historical query-planner adapter"]
        or training_contract.get("trainer_input_splits") != ["train", "dev"]
        or training_contract.get("objective")
        != "assistant target JSON tokens only; all state/observation tokens masked"
        or training_contract.get("target_truncation") != "forbidden"
        or training_contract.get("checkpoint_selection")
        != "fixed final probe step; no confirmation access"
        or training_contract.get("confirmation_read_by_trainer") is not False
        or training_contract.get("probe_optimizer_steps") != 20
    ):
        raise ValueError("formal protocol training objective/boundary mismatch")
    if training_contract.get("probe_gates") != FORMAL_TRAINING_PROBE_GATES:
        raise ValueError("formal protocol training probe gates mismatch")
    if protocol.get("probe_evaluation_contract") != FORMAL_PROBE_EVALUATION_CONTRACT:
        raise ValueError("formal protocol probe evaluation contract mismatch")
    if (
        protocol.get("future_gold_free_mechanism_gate")
        != FORMAL_FUTURE_GOLD_FREE_MECHANISM_GATE
    ):
        raise ValueError("formal protocol future Gold-free mechanism gate mismatch")
    if protocol.get("outcome_unlock_rule") != FORMAL_OUTCOME_UNLOCK_RULE:
        raise ValueError("formal protocol outcome unlock rule mismatch")

    inputs = protocol.get("inputs") or {}
    failure_lineage = inputs.get("v4_3_failure_lineage") or {}
    failure_artifacts = failure_lineage.get("artifacts") or {}
    if (
        failure_lineage.get("predecessor_protocol_version") != "v4_3"
        or failure_lineage.get("predecessor_terminal_status") != "FAIL_STOP"
        or failure_lineage.get("failure_class")
        != "clean_reload_adapter_dtype_coercion_fp32_to_bf16"
        or failure_lineage.get("failure_preserved_not_upgraded") is not True
        or failure_lineage.get("same_identity_selection_required") is not True
        or failure_lineage.get("same_action_record_bytes_required") is not True
        or failure_lineage.get("successor_single_implementation_variable")
        != (
            "PEFT default FP32 autocast for clean single-adapter reload with "
            "torch.equal zero tolerance and saved/live/reloaded dtype telemetry"
        )
        or failure_lineage.get("confirmation_action_records_opened") is not False
        or failure_lineage.get("prospective_content_opened_or_hashed") is not False
        or set(failure_artifacts) != set(FORMAL_V4_3_FAILURE_HASHES)
        or any(
            (failure_artifacts.get(role) or {}).get("sha256") != digest
            for role, digest in FORMAL_V4_3_FAILURE_HASHES.items()
        )
    ):
        raise ValueError("formal protocol v4.3 FAIL_STOP lineage mismatch")
    candidate_rows = inputs.get("strict_valid_action_candidate_pools") or []
    source_candidates = {
        str(item.get("source_planner_split")): item
        for item in candidate_rows
        if isinstance(item, Mapping) and item.get("source_planner_split") in {"train", "dev"}
    }
    pair_candidates = [
        item for item in candidate_rows
        if isinstance(item, Mapping)
        and item.get("role") == "candidate_action_pair_hash_index"
    ]
    if set(source_candidates) != {"train", "dev"} or len(pair_candidates) != 1:
        raise ValueError("formal protocol candidate inventory schema mismatch")
    expected_candidate_counts = {"train": (20798, 10399), "dev": (488, 244)}
    for split, (actions, pairs) in expected_candidate_counts.items():
        item = source_candidates[split]
        if (
            item.get("sha256") != FORMAL_CANDIDATE_HASHES[split]
            or item.get("action_rows") != actions
            or item.get("strict_valid_qid_pairs") != pairs
        ):
            raise ValueError(f"formal protocol {split} candidate lineage mismatch")
    pair_item = pair_candidates[0]
    if (
        pair_item.get("sha256") != FORMAL_CANDIDATE_HASHES["pair_hashes"]
        or pair_item.get("rows") != 10643
        or pair_item.get("per_source_split") != {"dev": 244, "train": 10399}
    ):
        raise ValueError("formal protocol candidate pair-hash lineage mismatch")
    hardening = inputs.get("candidate_v4_1_hardening_evidence") or {}
    hardening_audit = hardening.get("hardening_audit") or {}
    append_validation = hardening.get("append_only_validation") or {}
    if (
        hardening_audit.get("gate") != "PASS"
        or hardening_audit.get("selected_final_or_future_secret_leaks") != 0
        or hardening_audit.get("pair_hash_recompute_mismatches") != 0
        or append_validation.get("gate") != "PASS"
    ):
        raise ValueError("formal protocol candidate hardening evidence mismatch")

    sealed = inputs.get("sealed_prospective_parent_metadata_only") or {}
    if (
        sealed.get("parent_report_sha256") != FORMAL_SEALED_PARENT_HASHES["report"]
        or sealed.get("parent_manifest_sha256") != FORMAL_SEALED_PARENT_HASHES["manifest"]
        or sealed.get("prospective_declared_sha256")
        != FORMAL_SEALED_PARENT_HASHES["prospective_declared"]
        or sealed.get("prospective_content_opened") is not False
        or sealed.get("prospective_content_hashed") is not False
        or sealed.get("prospective_unlocked") is not False
    ):
        raise ValueError("formal protocol prospective seal mismatch")
    if manifest.get("prospective_opened_or_hashed") is not False:
        raise ValueError("formal protocol manifest prospective seal mismatch")

    implementation = protocol.get("implementation_locks") or {}
    if set(implementation) != set(FORMAL_IMPLEMENTATION_PATHS):
        raise ValueError("formal protocol implementation lock roles mismatch")
    for role, relative in FORMAL_IMPLEMENTATION_PATHS.items():
        lock = implementation.get(role) or {}
        if lock.get("path") != relative.as_posix():
            raise ValueError(f"formal protocol implementation path mismatch: {role}")
        if lock.get("sha256") != _sha256_file(PROJECT_ROOT / relative):
            raise ValueError(f"formal protocol implementation hash drift: {role}")

    gates = protocol.get("data_release_gates") or {}
    expected_data_gates = {
        "exact_train_qids_per_enabled_dataset": 600,
        "exact_dev_qids_per_enabled_dataset": 60,
        "exact_confirmation_qids_per_enabled_dataset": 30,
        "exact_actions_per_qid": 2,
        "eligibility_applied_before_identity_selection": True,
        "action_schema_valid_rate": 1.0,
        "question_identity_join_rate": 1.0,
        "query_nonrepeat_rate": 1.0,
        "placeholder_free_rate": 1.0,
        "dependency_closed_rate": 1.0,
        "source_action_text_rate": 1.0,
        "state_use_valid_rate": 1.0,
        "gold_boundary_valid_rate": 1.0,
        "duplicate_example_id_count": 0,
        "lock_consumption_policy": "exact join; construction failure aborts; no replacement",
        "action_pair_hash_match_rate": 1.0,
        "v4_3_identity_lock_byte_match_rate": 1.0,
        "v4_3_action_record_bytes_required": True,
        "v4_3_fail_stop_preserved": True,
        "hotpot_policy": (
            "report UNKNOWN label coverage; absence does not fail this two-dataset release"
        ),
    }
    if gates != expected_data_gates:
        raise ValueError("formal protocol data-release gates mismatch")
    gold = protocol.get("gold_boundary") or {}
    expected_gold_boundary = {
        "source_train_files_may_contain_gold": True,
        "freezer_direct_gold_final_answer_accessed": False,
        "upstream_candidate_builder_gold_final_answer_use": "leakage_exclusion_only",
        "candidate_eligibility_is_gold_screened": True,
        "construction_train_intermediate_annotation_used": True,
        "q1_train_intermediate_annotation_used": False,
        "q2_train_intermediate_annotation_used": True,
        "training_q2_model_visible_intermediate": (
            "annotation-derived_but_passage-bound; includes exact supporting excerpt "
            "and provenance; this is not annotation-free supervision"
        ),
        "training_model_visible_forbidden": [
            "gold final answer",
            "future-hop answer/tail",
            "unbound intermediate annotation",
            "raw decomposition/evidence objects",
            "evaluation Gold",
        ],
        "runtime_q2_model_visible_intermediate": (
            "Reader-predicted and retrieved-passage-bound; train annotation forbidden"
        ),
        "gold_final_answer_visible_rate": 0.0,
        "evaluation_gold_access_rate": 0.0,
        "confirmation_selection_reads_mechanical_label_eligibility": True,
        "confirmation_identity_output_contains_targets": False,
        "answer_scoring_authorized": False,
    }
    if gold != expected_gold_boundary:
        raise ValueError("formal protocol Gold boundary mismatch")

    report_checks = report.get("checks") or {}
    if (
        report.get("cohorts") != cohort_locks
        or report_checks.get("qid_overlap_between_splits") != 0
        or report_checks.get("family_overlap_between_splits") != 0
        or report_checks.get("sealed_prospective_content_opened") is not False
        or report_checks.get("sealed_prospective_content_hashed") is not False
        or report_checks.get("em_f1_ihr_authorized") is not False
        or report_checks.get("v4_3_failure_preserved_not_upgraded") is not True
        or report_checks.get("identity_bytes_equal_v4_3") is not True
    ):
        raise ValueError("formal protocol report gate mismatch")
    manifest_outputs = manifest.get("outputs") or {}
    if (
        manifest_outputs.get("protocol.json") != digests["protocol"]
        or manifest_outputs.get("report.json") != digests["protocol_report"]
        or manifest.get("action_data_built") is not False
        or manifest.get("training_started") is not False
        or manifest.get("answer_scoring_performed") is not False
    ):
        raise ValueError("formal protocol manifest gate mismatch")
    for split in ("train", "dev", "confirmation"):
        if manifest_outputs.get(f"{split}.identity_only.jsonl") != cohort_locks[split].get(
            "sha256"
        ):
            raise ValueError(f"formal protocol manifest {split} lock mismatch")

    inventory = [
        {"role": role, "path": str(paths[role]), "sha256": digests[role]}
        for role in ("protocol", "protocol_report", "protocol_manifest")
    ]
    return protocol, inventory


def _load_identity_locks(
    protocol_dir: Path,
    *,
    split_sizes: Mapping[str, int],
    expected_protocol_sha256: str,
) -> tuple[
    dict[str, list[dict[str, str]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Validate the formal protocol, then load its exact action-pair locks."""

    protocol, inventory = _validate_formal_protocol(
        protocol_dir,
        split_sizes=split_sizes,
        expected_protocol_sha256=expected_protocol_sha256,
    )
    cohort_locks = ((protocol.get("cohort") or {}).get("cohort_locks") or {})
    result: dict[str, list[dict[str, str]]] = {}
    required_fields = {
        "dataset", "qid", "question_key", "question", "question_sha256",
        "family_sha256", "split", "action_pair_sha256",
    }
    seen_qids: set[tuple[str, str]] = set()
    seen_families: set[tuple[str, str]] = set()
    for split in ("train", "dev", "confirmation"):
        declared = cohort_locks.get(split)
        if not isinstance(declared, Mapping):
            raise ValueError(f"protocol does not declare {split} cohort lock")
        relative = declared.get("path") or f"{split}.identity_only.jsonl"
        path = protocol_dir / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"identity lock missing: {path}")
        digest = _sha256_file(path)
        if digest != declared.get("sha256"):
            raise ValueError(f"identity lock hash mismatch: {path}")
        rows: list[dict[str, str]] = []
        for row in _read_jsonl(path):
            if set(row) != required_fields:
                raise ValueError(f"identity lock schema mismatch: {path}")
            dataset = _clean(row.get("dataset")).lower()
            qid = _clean(row.get("qid"))
            question = _clean(row.get("question"))
            family = _clean(row.get("family_sha256"))
            if dataset not in DATASETS or row.get("split") != split:
                raise ValueError(f"identity lock dataset/split mismatch: {path}")
            if row.get("question_key") != question_key(dataset, qid):
                raise ValueError(f"identity lock question_key mismatch: {dataset}::{qid}")
            if row.get("question_sha256") != question_sha256(question):
                raise ValueError(f"identity lock question hash mismatch: {dataset}::{qid}")
            if family != family_sha256(question):
                raise ValueError(f"identity lock family hash mismatch: {dataset}::{qid}")
            qid_key, family_key = (dataset, qid), (dataset, family)
            if qid_key in seen_qids or family_key in seen_families:
                raise ValueError("identity locks are not qid/family disjoint")
            seen_qids.add(qid_key)
            seen_families.add(family_key)
            rows.append({key: str(row[key]) for key in required_fields})
        if len(rows) != declared.get("rows"):
            raise ValueError(f"identity lock row count mismatch: {path}")
        result[split] = rows
        inventory.append(
            {"role": f"identity_lock_{split}", "path": str(path), "sha256": digest, "rows": len(rows)}
        )
    return result, inventory, protocol


def _stable_key(seed: int, *values: str) -> str:
    return _sha256_text("|".join((str(seed), *values)))


def _ordered_candidates(
    groups: Mapping[str, Sequence[PlannerCandidate]], *, dataset: str, split: str, seed: int
) -> list[PlannerCandidate]:
    ordered: list[PlannerCandidate] = []
    families = sorted(groups, key=lambda family: _stable_key(seed, split, dataset, family))
    for family in families:
        rows = sorted(
            groups[family],
            key=lambda row: _stable_key(seed, split, dataset, family, row.qid),
        )
        ordered.extend(rows)
    return ordered


def _raw_rows(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        qid = _clean(row.get("id") or row.get("qid"))
        if qid not in wanted:
            continue
        if qid in result:
            raise ValueError(f"duplicate raw qid in {path}: {qid}")
        result[qid] = row
    return result


def _select_pairs(
    *,
    groups: Mapping[str, Sequence[PlannerCandidate]],
    raw: Mapping[str, Mapping[str, Any]],
    dataset: str,
    split: str,
    quota: int,
    seed: int,
    additionally_excluded_families: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str], Counter[str]]:
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    selected_families: set[str] = set()
    counts: Counter[str] = Counter()
    by_family: dict[str, list[PlannerCandidate]] = defaultdict(list)
    for candidate in _ordered_candidates(groups, dataset=dataset, split=split, seed=seed):
        by_family[candidate.family].append(candidate)
    family_order = sorted(by_family, key=lambda family: _stable_key(seed, split, dataset, family))
    for family in family_order:
        if family in additionally_excluded_families:
            counts["cross_split_family_excluded"] += 1
            continue
        accepted = False
        for candidate in by_family[family]:
            raw_row = raw.get(candidate.qid)
            if raw_row is None:
                reason = "raw_qid_missing"
            else:
                try:
                    pair = build_action_pair(candidate, raw_row, split=split)
                except (BuildReject, ActionValidationError) as exc:
                    reason = getattr(exc, "code", "action_pair_validation_failed")
                else:
                    selected.extend(pair)
                    selected_families.add(family)
                    accepted = True
                    break
            counts[reason] += 1
            rejected.append(
                {
                    "dataset": dataset,
                    "qid": candidate.qid,
                    "question_sha256": question_sha256(candidate.question),
                    "family_sha256": family,
                    "split": split,
                    "reason": reason,
                }
            )
        if accepted and len(selected_families) >= quota:
            break
    if len(selected_families) < quota:
        raise ValueError(
            f"{dataset}/{split}: only {len(selected_families)} valid families; quota={quota}"
        )
    return selected, rejected, selected_families, counts


def _materialize_locked_pairs(
    *,
    locks: Sequence[Mapping[str, str]],
    groups_by_dataset: Mapping[str, Mapping[str, Sequence[PlannerCandidate]]],
    raw_by_dataset: Mapping[str, Mapping[str, Mapping[str, Any]]],
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    """Exact-join frozen identities; never substitute another candidate."""

    indexes: dict[str, dict[str, PlannerCandidate]] = {}
    for dataset in DATASETS:
        index: dict[str, PlannerCandidate] = {}
        for rows in groups_by_dataset[dataset].values():
            for candidate in rows:
                if candidate.qid in index:
                    raise ValueError(f"duplicate eligible planner qid: {dataset}::{candidate.qid}")
                index[candidate.qid] = candidate
        indexes[dataset] = index
    actions: list[dict[str, Any]] = []
    families: dict[str, set[str]] = {dataset: set() for dataset in DATASETS}
    for lock in locks:
        dataset, qid = lock["dataset"], lock["qid"]
        candidate = indexes[dataset].get(qid)
        if candidate is None:
            raise ValueError(f"identity lock is not strict-action eligible: {dataset}::{qid}")
        if (
            candidate.question != lock["question"]
            or question_sha256(candidate.question) != lock["question_sha256"]
            or candidate.family != lock["family_sha256"]
        ):
            raise ValueError(f"identity lock/planner hash join mismatch: {dataset}::{qid}")
        raw_row = raw_by_dataset[dataset].get(qid)
        if raw_row is None:
            raise ValueError(f"identity lock/raw join miss: {dataset}::{qid}")
        try:
            pair = build_action_pair(candidate, raw_row, split=split)
        except (BuildReject, ActionValidationError) as exc:
            code = getattr(exc, "code", type(exc).__name__)
            raise ValueError(
                f"frozen identity failed action construction: {dataset}::{qid}:{code}"
            ) from exc
        actual_pair_sha256 = canonical_action_pair_sha256(pair)
        if actual_pair_sha256 != lock["action_pair_sha256"]:
            raise ValueError(
                "frozen identity/action pair hash mismatch: "
                f"{dataset}::{qid}:{actual_pair_sha256} != {lock['action_pair_sha256']}"
            )
        actions.extend(pair)
        families[dataset].add(candidate.family)
    return actions, families


def build_candidate_pool(
    *,
    project_root: Path,
    data_root: Path,
    split_root: Path,
    output_dir: Path,
    experiment_id: str = f"{DEFAULT_EXPERIMENT_ID}-ELIGIBLE-CANDIDATES",
    seed: int = 42,
    consumed_cohorts: Sequence[Path] = DEFAULT_CONSUMED_COHORTS,
) -> dict[str, Any]:
    """Emit one strictly valid action pair per source-split family.

    This pool is not a train/dev release and freezes no cohort.  The protocol
    freezer consumes it, selects identity locks, and the formal release pass
    then exact-joins those locks back to planner/raw sources.
    """

    project_root = Path(project_root).resolve()
    data_root = Path(data_root).resolve()
    split_root = Path(split_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing candidate pool: {output_dir}")
    split_paths = {split: split_root / f"{split}.jsonl" for split in ("train", "dev")}
    isolation_paths = [
        split_root / "confirmation.jsonl",
        split_root / "seen_diagnostics.jsonl",
        *(Path(path).resolve() for path in consumed_cohorts),
    ]
    excluded_qids, excluded_families, exclusion_inventory = _load_exclusions(isolation_paths)
    groups_by_split: dict[str, dict[str, dict[str, list[PlannerCandidate]]]] = {}
    filtering: dict[str, dict[str, int]] = {}
    wanted: dict[str, set[str]] = {dataset: set() for dataset in DATASETS}
    for split, path in split_paths.items():
        groups, counts = _load_candidates(
            path, excluded_qids=excluded_qids, excluded_families=excluded_families
        )
        groups_by_split[split] = groups
        filtering[split] = dict(counts)
        for dataset in DATASETS:
            wanted[dataset].update(
                candidate.qid
                for family_rows in groups[dataset].values()
                for candidate in family_rows
            )
    raw_paths = {dataset: data_root / dataset / "train.jsonl" for dataset in DATASETS}
    raw = {dataset: _raw_rows(raw_paths[dataset], wanted[dataset]) for dataset in DATASETS}
    pools: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    rejected: list[dict[str, Any]] = []
    reject_counts: Counter[str] = Counter()
    leak_match_counts: Counter[str] = Counter()
    for split in ("train", "dev"):
        for dataset in DATASETS:
            groups = groups_by_split[split][dataset]
            for family in sorted(groups, key=lambda value: _stable_key(seed, split, dataset, value)):
                candidates = sorted(
                    groups[family],
                    key=lambda item: _stable_key(seed, split, dataset, family, item.qid),
                )
                for candidate in candidates:
                    try:
                        pair = build_action_pair(
                            candidate, raw[dataset][candidate.qid], split=split
                        )
                    except (BuildReject, ActionValidationError) as exc:
                        reason = getattr(exc, "code", "action_pair_validation_failed")
                        reject_counts[reason] += 1
                        if isinstance(exc, BuildReject) and exc.details:
                            match_kind = str(exc.details.get("match_kind") or "")
                            field = str(exc.details.get("field") or "")
                            compact_length = exc.details.get("secret_compact_length")
                            if match_kind:
                                leak_match_counts[f"match_kind::{match_kind}"] += 1
                            if field:
                                leak_match_counts[f"field::{field}"] += 1
                            if type(compact_length) is int:
                                leak_match_counts[f"secret_compact_length::{compact_length}"] += 1
                        rejected.append(
                            {
                                "dataset": dataset,
                                "qid": candidate.qid,
                                "question_sha256": question_sha256(candidate.question),
                                "family_sha256": family,
                                "split": split,
                                "reason": reason,
                            }
                        )
                    else:
                        pools[split].extend(pair)
                        break
    for split, rows in pools.items():
        rows.sort(key=lambda row: (row["dataset"], row["family_sha256"], row["qid"], row["turn_index"]))
        for row in rows:
            validate_action_record(row, expected_split=split)
    rejected.sort(key=lambda row: (row["split"], row["dataset"], row["qid"], row["reason"]))
    capacity = {
        split: {
            dataset: {
                "valid_qids": len({row["qid"] for row in pools[split] if row["dataset"] == dataset}),
                "valid_families": len({row["family_sha256"] for row in pools[split] if row["dataset"] == dataset}),
                "action_rows": sum(row["dataset"] == dataset for row in pools[split]),
            }
            for dataset in DATASETS
        }
        for split in ("train", "dev")
    }
    pair_hash_index = [
        *action_pair_hash_rows(pools["train"], source_split="train"),
        *action_pair_hash_rows(pools["dev"], source_split="dev"),
    ]
    pair_hash_index.sort(key=lambda row: (row["source_split"], row["dataset"], row["qid"]))
    article_normalization = {
        "affected_action_rows": sum(
            bool(row["source_provenance"].get("adjacent_duplicate_article_normalized"))
            for rows in pools.values() for row in rows
        ),
        "replacement_count": sum(
            int(row["source_provenance"].get("adjacent_duplicate_article_normalization_count", 0))
            for rows in pools.values() for row in rows
        ),
        "affected_fields": dict(
            sorted(
                Counter(
                    field
                    for rows in pools.values()
                    for row in rows
                    for field in row["source_provenance"].get(
                        "adjacent_duplicate_article_normalized_fields", []
                    )
                ).items()
            )
        ),
        "rule": "case-insensitive identical adjacent a/an/the collapsed to the second surface casing",
    }
    report: dict[str, Any] = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "status": CANDIDATE_STATUS,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "selection_authority": "NONE_PROTOCOL_FREEZER_MUST_SELECT_IDENTITIES",
        "candidate_policy": "one_valid_qid_pair_per_dataset_scoped_family",
        "source_action": "text",
        "capacity": capacity,
        "filtering": filtering,
        "rejection_reasons": dict(sorted(reject_counts.items())),
        "future_leak_diagnostics": dict(sorted(leak_match_counts.items())),
        "short_secret_policy_audit": _short_secret_policy_audit(raw),
        "adjacent_duplicate_article_normalization": article_normalization,
        "exception_policy": {
            "expected_rejections": ["BuildReject", "ActionValidationError"],
            "unknown_exception": "FAIL_STOP_NOT_CONVERTED_TO_REJECTION",
        },
        "pair_hash_contract": {
            "algorithm": "sha256(canonical_json([q1,q2]))",
            "canonical_json": "utf8_sort_keys_compact_separators_turn_index_order",
            "confirmation_rule": "freezer recomputes with split_override=confirmation",
            "rows": len(pair_hash_index),
        },
        "exclusions": exclusion_inventory,
        "inputs": [
            *(
                {"role": f"planner_{split}", "path": str(path), "sha256": _sha256_file(path)}
                for split, path in split_paths.items()
            ),
            *(
                {"role": f"raw_{dataset}_train", "path": str(path), "sha256": _sha256_file(path)}
                for dataset, path in raw_paths.items()
            ),
        ],
        "outputs": {},
        "gold_boundary": {
            "train_intermediate_annotation_used_for_q2_only": True,
            "final_answer_or_future_tail_serialized": False,
            "evaluation_gold_access": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for split, rows in pools.items():
        path = output_dir / f"{split}.jsonl"
        _write_jsonl(path, rows)
        report["outputs"][path.name] = {"rows": len(rows), "sha256": _sha256_file(path)}
    pair_hash_path = output_dir / "pair_hashes.jsonl"
    _write_jsonl(pair_hash_path, pair_hash_index)
    report["outputs"][pair_hash_path.name] = {
        "rows": len(pair_hash_index), "sha256": _sha256_file(pair_hash_path)
    }
    rejected_path = output_dir / "rejected.jsonl"
    _write_jsonl(rejected_path, rejected)
    report["outputs"][rejected_path.name] = {
        "rows": len(rejected), "sha256": _sha256_file(rejected_path)
    }
    _write_json(output_dir / "report.json", report)
    dump_manifest(
        output_dir,
        status=CANDIDATE_STATUS,
        extra=report,
    )
    return report


def build_release(
    *,
    project_root: Path,
    data_root: Path,
    split_root: Path,
    output_dir: Path,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    train_per_dataset: int = 600,
    dev_per_dataset: int = 60,
    confirmation_per_dataset: int = 30,
    seed: int = 42,
    consumed_cohorts: Sequence[Path] = DEFAULT_CONSUMED_COHORTS,
    protocol_dir: Path | None = None,
    expected_protocol_sha256: str | None = None,
    allow_unfrozen_selection_for_tests: bool = False,
) -> dict[str, Any]:
    """Materialize an append-only Controller-v1 action release."""

    project_root = Path(project_root).resolve()
    data_root = Path(data_root).resolve()
    split_root = Path(split_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing release: {output_dir}")
    if train_per_dataset <= 0 or dev_per_dataset <= 0 or confirmation_per_dataset <= 0:
        raise ValueError("train/dev/confirmation quotas must be positive")
    if not _clean(experiment_id):
        raise ValueError("experiment_id must be non-empty")
    if protocol_dir is None and not allow_unfrozen_selection_for_tests:
        raise ValueError("formal release requires protocol_dir identity locks")
    if protocol_dir is not None and not expected_protocol_sha256:
        raise ValueError("formal release requires an external expected_protocol_sha256 lock")
    if protocol_dir is not None and not allow_unfrozen_selection_for_tests:
        if experiment_id != DEFAULT_EXPERIMENT_ID:
            raise ValueError("formal release Experiment ID mismatch")
        if {
            "train": train_per_dataset,
            "dev": dev_per_dataset,
            "confirmation": confirmation_per_dataset,
        } != {"train": 600, "dev": 60, "confirmation": 30}:
            raise ValueError("formal release cohort quotas must be exactly 600/60/30")
        if seed != 42:
            raise ValueError("formal release selection seed must be 42")

    split_paths = {split: split_root / f"{split}.jsonl" for split in ("train", "dev")}
    isolation_paths = [
        split_root / "confirmation.jsonl",
        split_root / "seen_diagnostics.jsonl",
        *(Path(path).resolve() for path in consumed_cohorts),
    ]
    excluded_qids, excluded_families, exclusion_inventory = _load_exclusions(isolation_paths)
    candidate_groups: dict[str, dict[str, dict[str, list[PlannerCandidate]]]] = {}
    filtering: dict[str, dict[str, int]] = {}
    all_wanted: dict[str, set[str]] = {dataset: set() for dataset in DATASETS}
    for split, path in split_paths.items():
        groups, counts = _load_candidates(
            path,
            excluded_qids=excluded_qids,
            excluded_families=excluded_families,
        )
        candidate_groups[split] = groups
        filtering[split] = dict(counts)
        for dataset in DATASETS:
            all_wanted[dataset].update(
                candidate.qid
                for family_rows in groups[dataset].values()
                for candidate in family_rows
            )

    raw_paths = {dataset: data_root / dataset / "train.jsonl" for dataset in DATASETS}
    raw = {
        dataset: _raw_rows(raw_paths[dataset], all_wanted[dataset])
        for dataset in DATASETS
    }

    release: dict[str, list[dict[str, Any]]] = {
        "train": [], "dev": [], "confirmation": []
    }
    rejected: list[dict[str, Any]] = []
    selected_families: dict[str, dict[str, set[str]]] = {
        split: {dataset: set() for dataset in DATASETS}
        for split in ("train", "dev", "confirmation")
    }
    rejection_counts: Counter[str] = Counter()
    identity_inventory: list[dict[str, Any]] = []
    formal_protocol: dict[str, Any] | None = None
    if protocol_dir is not None:
        locks, identity_inventory, formal_protocol = _load_identity_locks(
            Path(protocol_dir).resolve(),
            split_sizes={
                "train": train_per_dataset,
                "dev": dev_per_dataset,
                "confirmation": confirmation_per_dataset,
            },
            expected_protocol_sha256=str(expected_protocol_sha256),
        )
        expected = {
            "train": train_per_dataset,
            "dev": dev_per_dataset,
            "confirmation": confirmation_per_dataset,
        }
        for split in ("train", "dev", "confirmation"):
            lock_counts = Counter(row["dataset"] for row in locks[split])
            if any(lock_counts[dataset] != expected[split] for dataset in DATASETS):
                raise ValueError(
                    f"{split} identity lock quota mismatch: {dict(lock_counts)} != {expected[split]} each"
                )
            rows, families = _materialize_locked_pairs(
                locks=locks[split],
                groups_by_dataset=candidate_groups[
                    "train" if split == "confirmation" else split
                ],
                raw_by_dataset=raw,
                split=split,
            )
            release[split].extend(rows)
            selected_families[split] = families
    else:
        # Test-only escape hatch.  Formal CLI always supplies protocol locks.
        for dataset in DATASETS:
            rows, rejects, families, counts = _select_pairs(
                groups=candidate_groups["dev"][dataset], raw=raw[dataset], dataset=dataset,
                split="dev", quota=dev_per_dataset, seed=seed,
                additionally_excluded_families=set(),
            )
            release["dev"].extend(rows)
            rejected.extend(rejects)
            selected_families["dev"][dataset] = families
            rejection_counts.update(counts)
            rows, rejects, families, counts = _select_pairs(
                groups=candidate_groups["train"][dataset], raw=raw[dataset], dataset=dataset,
                split="train", quota=train_per_dataset, seed=seed,
                additionally_excluded_families=selected_families["dev"][dataset],
            )
            release["train"].extend(rows)
            rejected.extend(rejects)
            selected_families["train"][dataset] = families
            rejection_counts.update(counts)

    materialized_splits = (
        ("train", "dev", "confirmation") if protocol_dir is not None else ("train", "dev")
    )
    for split in materialized_splits:
        release[split].sort(key=lambda row: (row["dataset"], row["qid"], row["turn_index"]))
        for row in release[split]:
            validate_action_record(row, expected_split=split)
    release_action_hashes = {
        split: _jsonl_rows_sha256(release[split]) for split in materialized_splits
    }
    canonical_successor = protocol_dir is not None and not allow_unfrozen_selection_for_tests
    if canonical_successor and release_action_hashes != FORMAL_V4_3_ACTION_HASHES:
        raise ValueError(
            "v4.4 action records differ from the frozen v4.3 release: "
            f"actual={release_action_hashes}"
        )
    rejected.sort(key=lambda row: (row["split"], row["dataset"], row["qid"], row["reason"]))

    split_qids = {
        split: {(row["dataset"], row["qid"]) for row in release[split]}
        for split in materialized_splits
    }
    split_families = {
        split: {(row["dataset"], row["family_sha256"]) for row in release[split]}
        for split in materialized_splits
    }
    all_qids = set().union(*split_qids.values())
    all_families = set().union(*split_families.values())
    qid_overlap = sum(
        len(split_qids[left] & split_qids[right])
        for index, left in enumerate(materialized_splits)
        for right in materialized_splits[index + 1 :]
    )
    family_overlap = sum(
        len(split_families[left] & split_families[right])
        for index, left in enumerate(materialized_splits)
        for right in materialized_splits[index + 1 :]
    )
    checks = {
        "all_records_schema_valid": True,
        "two_actions_per_qid": all(
            Counter((row["dataset"], row["qid"]) for row in release[split]).most_common()
            and set(
                Counter((row["dataset"], row["qid"]) for row in release[split]).values()
            ) == {2}
            for split in materialized_splits
        ),
        "cross_split_qid_overlap": qid_overlap,
        "cross_split_family_overlap": family_overlap,
        "excluded_qid_overlap": len(all_qids & excluded_qids),
        "excluded_family_overlap": len(all_families & excluded_families),
        "source_action_values": sorted(
            {row["target"]["source_action"] for split in release.values() for row in split}
        ),
        "gold_final_answer_visible_count": sum(
            bool(row["gold_boundary"]["gold_final_answer_visible"])
            for split in release.values() for row in split
        ),
        "evaluation_gold_access_count": sum(
            bool(row["gold_boundary"]["evaluation_gold_access"])
            for split in release.values() for row in split
        ),
        "action_pair_hash_join_rate": 1.0 if protocol_dir is not None else None,
        "v4_3_action_record_byte_match_rate": (
            1.0 if canonical_successor else None
        ),
    }
    gates_pass = bool(
        checks["two_actions_per_qid"]
        and checks["cross_split_qid_overlap"] == 0
        and checks["cross_split_family_overlap"] == 0
        and checks["excluded_qid_overlap"] == 0
        and checks["excluded_family_overlap"] == 0
        and checks["source_action_values"] == ["text"]
        and checks["gold_final_answer_visible_count"] == 0
        and checks["evaluation_gold_access_count"] == 0
        and (protocol_dir is None or checks["action_pair_hash_join_rate"] == 1.0)
        and (
            not canonical_successor
            or checks["v4_3_action_record_byte_match_rate"] == 1.0
        )
    )
    if not gates_pass:
        raise ValueError(f"release gates failed: {checks}")

    input_inventory = [
        {"role": f"planner_{split}", "path": str(path), "sha256": _sha256_file(path)}
        for split, path in split_paths.items()
    ] + [
        {"role": f"raw_{dataset}_train", "path": str(path), "sha256": _sha256_file(path)}
        for dataset, path in raw_paths.items()
    ]
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    per_split_dataset: dict[str, dict[str, Any]] = {}
    for split in materialized_splits:
        per_split_dataset[split] = {}
        for dataset in DATASETS:
            rows = [row for row in release[split] if row["dataset"] == dataset]
            per_split_dataset[split][dataset] = {
                "qids": len({row["qid"] for row in rows}),
                "actions": len(rows),
                "q1": sum(row["slot"] == "q1" for row in rows),
                "q2_dynamic": sum(row["slot"] == "q2_dynamic" for row in rows),
            }
    report: dict[str, Any] = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "status": STATUS,
        "generated_at_utc": generated_at,
        "selection": {
            "seed": seed,
            "one_qid_per_dataset_scoped_family": True,
            "train_per_dataset": train_per_dataset,
            "dev_per_dataset": dev_per_dataset,
            "confirmation_per_dataset": confirmation_per_dataset,
            "source_action": "text",
            "scope": "strict_linear_two_step_2wiki_and_musique",
            "identity_authority": (
                "frozen_protocol_exact_join" if protocol_dir is not None
                else "TEST_ONLY_UNFROZEN_INTERNAL_SELECTION"
            ),
        },
        "gold_boundary": {
            "train_intermediate_annotation_used_for_q2_only": True,
            "final_answer_or_future_tail_serialized": False,
            "evaluation_gold_access": False,
            "hotpotqa": "UNKNOWN_NO_MECHANICALLY_VERIFIED_QUERY_ACTION_TARGET",
        },
        "counts": per_split_dataset,
        "filtering": filtering,
        "rejection_reasons": dict(sorted(rejection_counts.items())),
        "checks": checks,
        "all_release_gates_pass": gates_pass,
        "inputs": input_inventory,
        "identity_locks": identity_inventory,
        "protocol_lineage": (
            {
                "path": str((Path(protocol_dir).resolve() / "protocol.json")),
                "sha256": str(expected_protocol_sha256),
                "schema_version": formal_protocol["schema_version"],
                "experiment_id": formal_protocol["experiment_id"],
                "status": formal_protocol["status"],
                "protocol_body_canonical_sha256": formal_protocol[
                    "protocol_body_canonical_sha256"
                ],
                "cohort_locks": formal_protocol["cohort"]["cohort_locks"],
                "candidate_hashes": dict(FORMAL_CANDIDATE_HASHES),
                "implementation_locks": formal_protocol["implementation_locks"],
                "probe_evaluation_contract": formal_protocol[
                    "probe_evaluation_contract"
                ],
                "predecessor_v4_3_action_sha256": dict(FORMAL_V4_3_ACTION_HASHES),
                "successor_action_sha256": dict(release_action_hashes),
                "action_records_byte_identical_to_v4_3": canonical_successor,
            }
            if formal_protocol is not None
            else None
        ),
        "exclusions": exclusion_inventory,
        "outputs": {},
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    for split in materialized_splits:
        path = output_dir / f"{split}.jsonl"
        _write_jsonl(path, release[split])
        report["outputs"][path.name] = {
            "rows": len(release[split]), "sha256": _sha256_file(path)
        }
    rejected_path = output_dir / "rejected.jsonl"
    _write_jsonl(rejected_path, rejected)
    report["outputs"][rejected_path.name] = {
        "rows": len(rejected), "sha256": _sha256_file(rejected_path)
    }
    report_path = output_dir / "report.json"
    _write_json(report_path, report)
    dump_manifest(output_dir, status=STATUS, extra=report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", type=Path, default=Path("."))
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--split_root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--candidate_pool_only",
        action="store_true",
        help="Emit strict eligible action pairs for protocol selection; do not freeze a release.",
    )
    parser.add_argument("--protocol_dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    parser.add_argument(
        "--expected_protocol_sha256",
        help="Required external SHA256 lock for a formal protocol.json",
    )
    parser.add_argument("--experiment_id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--train_per_dataset", type=int, default=600)
    parser.add_argument("--dev_per_dataset", type=int, default=60)
    parser.add_argument("--confirmation_per_dataset", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude_cohort", type=Path, action="append", default=[],
        help="Additional identity-only consumed cohort (defaults remain active).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    consumed = [
        *(project_root / path for path in DEFAULT_CONSUMED_COHORTS),
        *(path if path.is_absolute() else project_root / path for path in args.exclude_cohort),
    ]
    common = dict(
        project_root=project_root,
        data_root=args.data_root if args.data_root.is_absolute() else project_root / args.data_root,
        split_root=args.split_root if args.split_root.is_absolute() else project_root / args.split_root,
        experiment_id=args.experiment_id,
        seed=args.seed,
        consumed_cohorts=consumed,
    )
    if args.candidate_pool_only:
        requested_output = args.output_dir
        # Keep a safe dedicated default so a candidate pass cannot occupy the
        # canonical final-release path.
        if requested_output == DEFAULT_OUTPUT_DIR:
            requested_output = DEFAULT_CANDIDATE_OUTPUT_DIR
        report = build_candidate_pool(
            **common,
            output_dir=(
                requested_output if requested_output.is_absolute()
                else project_root / requested_output
            ),
        )
    else:
        report = build_release(
            **common,
            output_dir=(
                args.output_dir if args.output_dir.is_absolute()
                else project_root / args.output_dir
            ),
            train_per_dataset=args.train_per_dataset,
            dev_per_dataset=args.dev_per_dataset,
            confirmation_per_dataset=args.confirmation_per_dataset,
            protocol_dir=(
                args.protocol_dir if args.protocol_dir.is_absolute()
                else project_root / args.protocol_dir
            ),
            expected_protocol_sha256=args.expected_protocol_sha256,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
