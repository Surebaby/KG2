#!/usr/bin/env python3
"""Materialise the v4-safe Strong-SFT full-trajectory replay release.

The historical v1c replay was frozen before the complete v4 protected ledger
and the final HotpotQA/MuSiQue rollout population existed.  This CPU-only
materialiser returns to the same versioned Strong-SFT silver source, keeps only
accepted train-fold trajectories whose rendered target has 3--5 complete
steps, and deterministically chooses a fresh n=2,000 pool after excluding:

* every identity in the complete protected-ledger v2; and
* every identity in the frozen H1000/M1000 v4 interim population.

All identity tests are dataset-scoped and use qid, exact-question SHA256, and
the *currently recomputed* lexical-family SHA256.  Stored family hashes are
never treated as authority.  Selected source JSON rows are copied byte for
byte (apart from normalising the line ending to ``\n``), so teacher steps, KG,
and retrieved passages cannot drift during the refreeze.

No model weights, network resources, Gold labels, evaluation outputs, or raw
datasets are changed.  A local-tokenizer dry-run mirrors PPO replay packing and
must retain a nonempty supervised assistant suffix for all 2,000 rows.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.data.prompts import build_sft_messages
from kgproweight.data.silver_dataset import SilverTrajectory
from kgproweight.data.silver_split import SplitSpec, assign_split
from kgproweight.kg.question_kg import question_sha256
from kgproweight.training.phase3_sft import _render_assistant_trace
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256
from scripts.prepare.materialize_source_gated_mixed3_v3 import _valid_replay_trace


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "sft-replay-rendered-3to5-v2-clean-isolated"
SELECTION_SCHEMA_VERSION = "sft-replay-selection-v2-clean-isolated"
STATUS = "COMPLETE_DATA_NOT_TRAINED"
EXPERIMENT_ID = "SFT-REPLAY-STRONG-LEGACY-TRAIN-3TO5-N2000-SEED42-V2"

DEFAULT_SOURCE = Path(
    "checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42_"
    "no_text_head/silver_with_logprobs.jsonl"
)
DEFAULT_SOURCE_MANIFEST = Path(
    "checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42_"
    "no_text_head/manifest.json"
)
DEFAULT_LEGACY_REPLAY = Path(
    "data/silver_data/sft_replay_legacy_strong_train_rendered3to5_"
    "n2000_seed42_v1c"
)
DEFAULT_PROTECTED_LEDGER = Path(
    "outputs/audits/mixed_ppo_v4_protected_identity_ledger_v2"
)
DEFAULT_HM_FREEZE = Path(
    "outputs/audits/mixed_ppo_three_dataset_v4_hm_expansion_"
    "h1000_m1000_seed42_preregistration"
)
DEFAULT_TOKENIZER = Path(
    "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final"
)
DEFAULT_OUTPUT = Path(
    "data/silver_data/sft_replay_legacy_strong_train_rendered3to5_"
    "n2000_seed42_v2"
)

PROTECTED_SCHEMA = "mixed-ppo-v4-protected-identity-ledger-v2"
PROTECTED_ROW_SCHEMA = "mixed-ppo-v4-protected-question-identity-v2"
PROTECTED_STATUS = "COMPLETE_FROZEN_IDENTITY_ONLY_NOT_TRAINING_DATA"
HM_SCHEMA = "mixed-ppo-three-dataset-v4-hm-expansion-preregistration-v1"
HM_STATUS = (
    "FROZEN_HM_IDENTITIES_RETRIEVAL_NOT_MATERIALIZED_"
    "2WIKI_UNRESOLVED_NOT_TRAINED"
)
VALID_DATASETS = {"hotpotqa", "2wikimultihopqa", "musique"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _resolve_bound_path(raw: Any) -> Path:
    path = Path(str(raw or ""))
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _identity_from_row(
    row: Mapping[str, Any],
    *,
    label: str,
    require_stored_hashes: bool,
) -> dict[str, str]:
    dataset = str(row.get("dataset") or row.get("dataset_name") or "").strip().lower()
    qid = str(row.get("qid") or row.get("id") or "").strip()
    question = str(row.get("question") or "").strip()
    if dataset not in VALID_DATASETS or not qid or not question:
        raise ValueError(
            f"incomplete {label} identity: dataset={dataset!r}, qid={qid!r}"
        )
    qhash = question_sha256(question)
    family = family_sha256(question)
    if require_stored_hashes and (
        str(row.get("question_sha256") or "") != qhash
        or str(row.get("family_sha256") or "") != family
        or str(row.get("family_version") or "") != FAMILY_VERSION
    ):
        raise ValueError(f"stale/malformed {label} identity: {dataset}::{qid}")
    return {
        "dataset": dataset,
        "qid": qid,
        "question": question,
        "question_sha256": qhash,
        "family_sha256": family,
    }


@dataclass
class IdentityIndex:
    qids: set[tuple[str, str]]
    question_hashes: set[tuple[str, str]]
    families: set[tuple[str, str]]

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, Any]],
        *,
        label: str,
        require_stored_hashes: bool = True,
    ) -> "IdentityIndex":
        index = cls(set(), set(), set())
        for row in rows:
            identity = _identity_from_row(
                row,
                label=label,
                require_stored_hashes=require_stored_hashes,
            )
            dataset = identity["dataset"]
            index.qids.add((dataset, identity["qid"]))
            index.question_hashes.add((dataset, identity["question_sha256"]))
            index.families.add((dataset, identity["family_sha256"]))
        return index

    def matches(self, identity: Mapping[str, str]) -> dict[str, bool]:
        dataset = identity["dataset"]
        return {
            "qid": (dataset, identity["qid"]) in self.qids,
            "question_sha256": (
                dataset,
                identity["question_sha256"],
            )
            in self.question_hashes,
            "family_sha256": (dataset, identity["family_sha256"])
            in self.families,
        }


def _validate_protected_ledger(
    directory: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    paths = {
        name: directory.resolve() / name
        for name in (
            "protected_identities.question_only.jsonl",
            "report.json",
            "manifest.json",
        )
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(paths["report.json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    boundary = report.get("scientific_boundary") or {}
    if not (
        report.get("schema_version") == PROTECTED_SCHEMA
        and report.get("status") == PROTECTED_STATUS
        and manifest.get("status") == PROTECTED_STATUS
        and report.get("complete") is True
        and report.get("identity_scope") == "dataset-scoped"
        and report.get("current_family_recomputed") is True
        and boundary.get("gold_or_outcome_values_used_for_identity_selection")
        is False
        and boundary.get("gold_fields_emitted") is False
        and boundary.get("data_raw_modified") is False
        and boundary.get("training_started") is False
    ):
        raise ValueError("protected ledger schema/status/scientific boundary failed")
    ledger_sha = sha256_file(paths["protected_identities.question_only.jsonl"])
    report_sha = sha256_file(paths["report.json"])
    run = manifest.get("run") or {}
    if (
        str((report.get("output") or {}).get("sha256") or "") != ledger_sha
        or str(run.get("protected_identities_sha256") or "") != ledger_sha
        or str(run.get("report_sha256") or "") != report_sha
    ):
        raise ValueError("protected ledger report/manifest hash binding failed")
    rows = read_jsonl(paths["protected_identities.question_only.jsonl"])
    if len(rows) != int((report.get("unique") or {}).get("dataset_qids", -1)):
        raise ValueError("protected ledger row count differs from report")
    for row in rows:
        if (
            row.get("schema_version") != PROTECTED_ROW_SCHEMA
            or row.get("gold_access") is not False
        ):
            raise ValueError("malformed protected-ledger identity row")
    # This also proves every stored family is the current implementation.
    IdentityIndex.from_rows(rows, label="protected-ledger")
    return rows, {name: file_identity(path) for name, path in paths.items()}


def _validate_hm_population(
    directory: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    protocol_path = directory.resolve() / "protocol.json"
    manifest_path = directory.resolve() / "manifest.json"
    if not protocol_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(directory)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version") != HM_SCHEMA
        or protocol.get("status") != HM_STATUS
        or manifest.get("status") != HM_STATUS
        or protocol.get("scope") != "hotpotqa_musique_expansion_only"
        or not all(bool(value) for value in (protocol.get("gates") or {}).values())
    ):
        raise ValueError("H/M interim freeze schema/status/gates failed")
    binding = (protocol.get("outputs") or {}).get("hm_population") or {}
    population_path = _resolve_bound_path(binding.get("path"))
    if (
        not population_path.is_file()
        or str(binding.get("sha256") or "") != sha256_file(population_path)
        or int(binding.get("size_bytes", -1)) != population_path.stat().st_size
    ):
        raise ValueError("H/M interim population hash binding failed")
    rows = read_jsonl(population_path)
    counts = Counter(str(row.get("dataset") or "").strip().lower() for row in rows)
    if len(rows) != 2000 or counts != Counter({"hotpotqa": 1000, "musique": 1000}):
        raise ValueError(f"H/M interim population count mismatch: {counts}")
    for row in rows:
        if row.get("gold_access") is not False:
            raise ValueError("H/M population contains a non-answer-free row")
    IdentityIndex.from_rows(rows, label="H/M v4 population")
    paths = {
        "protocol.json": protocol_path,
        "manifest.json": manifest_path,
        "hm_population.question_only.jsonl": population_path,
    }
    return rows, {name: file_identity(path) for name, path in paths.items()}


def _validate_source_provenance(
    *, source: Path, source_manifest: Path, legacy_replay: Path
) -> dict[str, Any]:
    source = source.resolve()
    source_manifest = source_manifest.resolve()
    legacy_report_path = legacy_replay.resolve() / "report.json"
    for path in (source, source_manifest, legacy_report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_sha = sha256_file(source)
    source_md5 = md5_file(source)
    old_report = json.loads(legacy_report_path.read_text(encoding="utf-8"))
    if (
        old_report.get("schema_version") != "sft-replay-rendered-3to5-v1"
        or old_report.get("status") != STATUS
        or str((old_report.get("source") or {}).get("sha256") or "")
        != source_sha
        or int((old_report.get("source") or {}).get("size_bytes", -1))
        != source.stat().st_size
    ):
        raise ValueError("legacy replay does not bind the supplied Strong-SFT source")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    run = manifest.get("run") or {}
    enriched = (run.get("output_artifacts") or {}).get("enriched_silver") or {}
    configured_source = _resolve_bound_path(enriched.get("path"))
    if not (
        manifest.get("status") == "COMPLETE"
        and run.get("phase") == "phase2_prm"
        and configured_source == source
        and int(enriched.get("size_bytes", -1)) == source.stat().st_size
        and str(enriched.get("md5") or "") == source_md5
        and str((run.get("config") or {}).get("split") or "") == "train"
    ):
        raise ValueError("Strong-SFT source manifest binding failed")
    return {
        "silver_with_logprobs": file_identity(source),
        "silver_with_logprobs_md5": source_md5,
        "phase2_manifest": file_identity(source_manifest),
        "legacy_replay_report": file_identity(legacy_report_path),
        "legacy_replay_experiment_id": old_report.get("experiment_id"),
        "upstream_original_silver": str(
            (run.get("config") or {}).get("silver_path") or "UNKNOWN"
        ),
        "source_fold": {
            "split": "train",
            "spec": SplitSpec().__dict__,
        },
    }


def _selection_rank(identity: Mapping[str, str], seed: int) -> str:
    payload = "\0".join(
        (
            str(seed),
            identity["dataset"],
            identity["qid"],
            identity["question_sha256"],
            identity["family_sha256"],
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class Candidate:
    trajectory: SilverTrajectory
    identity: dict[str, str]
    source_line_number: int
    source_row_bytes: bytes
    source_row_sha256: str
    rendered_steps: int
    rendered_trace_sha256: str
    selection_rank_sha256: str


def collect_candidates(
    *,
    source: Path,
    protected: IdentityIndex,
    hm_population: IdentityIndex,
    seed: int,
) -> tuple[list[Candidate], dict[str, Any]]:
    counters: Counter[str] = Counter()
    overlaps: dict[str, Counter[str]] = {
        "protected": Counter(),
        "hm_population": Counter(),
    }
    candidates: list[Candidate] = []
    split_spec = SplitSpec()
    with source.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            payload = raw_line.rstrip(b"\r\n")
            if not payload.strip():
                continue
            counters["source_rows"] += 1
            try:
                raw = json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError(f"invalid source JSONL {source}:{line_number}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"expected source object at {source}:{line_number}")
            trajectory = SilverTrajectory.from_dict(raw)
            if not trajectory.accepted:
                counters["rejected_not_accepted"] += 1
                continue
            if assign_split(trajectory, split_spec) != "train":
                counters["rejected_nontrain_fold"] += 1
                continue
            counters["accepted_train"] += 1
            identity = _identity_from_row(
                raw, label="Strong-SFT source", require_stored_hashes=False
            )
            protected_hits = protected.matches(identity)
            hm_hits = hm_population.matches(identity)
            for source_name, hits in (
                ("protected", protected_hits),
                ("hm_population", hm_hits),
            ):
                for field, hit in hits.items():
                    overlaps[source_name][field] += int(hit)
                overlaps[source_name]["any"] += int(any(hits.values()))
            if any(protected_hits.values()):
                counters["rejected_protected_ledger"] += 1
                continue
            if any(hm_hits.values()):
                counters["rejected_hm_population"] += 1
                continue
            valid, n_steps, reason = _valid_replay_trace(trajectory)
            if not valid:
                counters[f"rejected_{reason}"] += 1
                continue
            rendered = _render_assistant_trace(trajectory)
            candidates.append(
                Candidate(
                    trajectory=trajectory,
                    identity=identity,
                    source_line_number=line_number,
                    source_row_bytes=payload,
                    source_row_sha256=hashlib.sha256(payload).hexdigest(),
                    rendered_steps=n_steps,
                    rendered_trace_sha256=hashlib.sha256(
                        rendered.encode("utf-8")
                    ).hexdigest(),
                    selection_rank_sha256=_selection_rank(identity, seed),
                )
            )
    candidates.sort(
        key=lambda row: (
            row.selection_rank_sha256,
            row.identity["dataset"],
            row.identity["qid"],
        )
    )
    return candidates, {
        "source_flow": dict(sorted(counters.items())),
        "overlap_all_accepted_train": {
            name: dict(sorted(counts.items())) for name, counts in overlaps.items()
        },
        "valid_safe_before_internal_dedup": len(candidates),
    }


def _tokenizer_dry_run(
    candidate: Candidate,
    tokenizer: Any,
    *,
    max_input_length: int,
    max_new_tokens: int,
    max_passages: int,
    max_kg_triples: int,
) -> tuple[bool, dict[str, int | str]]:
    trajectory = candidate.trajectory
    answer_trace = _render_assistant_trace(trajectory)
    max_total = max_input_length + max_new_tokens
    n_passages = min(max_passages, len(trajectory.retrieved_passages))
    while True:
        prompt_messages = build_sft_messages(
            question=trajectory.question,
            retrieved_passages=list(trajectory.retrieved_passages)[:n_passages],
            kg_triples=trajectory.kg_subgraph,
            top_k=n_passages,
            max_kg_triples=max_kg_triples,
        )
        full_messages = build_sft_messages(
            question=trajectory.question,
            retrieved_passages=list(trajectory.retrieved_passages)[:n_passages],
            kg_triples=trajectory.kg_subgraph,
            answer_trace=answer_trace,
            top_k=n_passages,
            max_kg_triples=max_kg_triples,
        )
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        prompt_ids = tokenizer(
            prompt_text, truncation=False, add_special_tokens=False
        )["input_ids"]
        full_ids = tokenizer(
            full_text, truncation=False, add_special_tokens=False
        )["input_ids"]
        prefix_ok = full_ids[: len(prompt_ids)] == prompt_ids
        fits = len(prompt_ids) <= max_input_length and len(full_ids) <= max_total
        if (fits and prefix_ok and len(full_ids) > len(prompt_ids)) or n_passages == 0:
            break
        n_passages -= 1
    if not prefix_ok:
        return False, {"reason": "prompt_not_exact_prefix"}
    if not fits:
        return False, {
            "reason": "target_does_not_fit_even_without_passages",
            "prompt_tokens": len(prompt_ids),
            "full_tokens": len(full_ids),
        }
    if len(full_ids) <= len(prompt_ids):
        return False, {"reason": "no_supervised_assistant_tokens"}
    return True, {
        "reason": "accepted",
        "prompt_tokens": len(prompt_ids),
        "full_tokens": len(full_ids),
        "assistant_tokens": len(full_ids) - len(prompt_ids),
        "passages_retained": n_passages,
        "passages_dropped": min(max_passages, len(trajectory.retrieved_passages))
        - n_passages,
    }


def select_candidates(
    candidates: Sequence[Candidate],
    *,
    n_samples: int,
    tokenizer: Any | None,
    max_input_length: int = 6144,
    max_new_tokens: int = 384,
    max_passages: int = 15,
    max_kg_triples: int = 12,
) -> tuple[list[Candidate], list[dict[str, Any]], dict[str, Any]]:
    selected: list[Candidate] = []
    selection_rows: list[dict[str, Any]] = []
    seen = IdentityIndex(set(), set(), set())
    rejected: Counter[str] = Counter()
    step_counts: Counter[int] = Counter()
    token_stats: Counter[str] = Counter()
    token_extrema: dict[str, int] = {
        "max_prompt_tokens": 0,
        "max_full_tokens": 0,
        "min_assistant_tokens": 10**9,
        "max_passages_dropped": 0,
    }
    for candidate in candidates:
        duplicate = seen.matches(candidate.identity)
        if any(duplicate.values()):
            for name, hit in duplicate.items():
                rejected[f"internal_duplicate_{name}"] += int(hit)
            rejected["internal_duplicate_any"] += 1
            continue
        token_detail: dict[str, Any] = {"reason": "not_run"}
        if tokenizer is not None:
            ok, token_detail = _tokenizer_dry_run(
                candidate,
                tokenizer,
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
                max_passages=max_passages,
                max_kg_triples=max_kg_triples,
            )
            if not ok:
                rejected[f"tokenizer_{token_detail['reason']}"] += 1
                continue
            token_stats["rows_checked"] += 1
            token_stats["passages_dropped_total"] += int(
                token_detail["passages_dropped"]
            )
            token_extrema["max_prompt_tokens"] = max(
                token_extrema["max_prompt_tokens"], int(token_detail["prompt_tokens"])
            )
            token_extrema["max_full_tokens"] = max(
                token_extrema["max_full_tokens"], int(token_detail["full_tokens"])
            )
            token_extrema["min_assistant_tokens"] = min(
                token_extrema["min_assistant_tokens"],
                int(token_detail["assistant_tokens"]),
            )
            token_extrema["max_passages_dropped"] = max(
                token_extrema["max_passages_dropped"],
                int(token_detail["passages_dropped"]),
            )
        selected.append(candidate)
        dataset = candidate.identity["dataset"]
        seen.qids.add((dataset, candidate.identity["qid"]))
        seen.question_hashes.add((dataset, candidate.identity["question_sha256"]))
        seen.families.add((dataset, candidate.identity["family_sha256"]))
        step_counts[candidate.rendered_steps] += 1
        selection_rows.append(
            {
                "schema_version": SELECTION_SCHEMA_VERSION,
                **candidate.identity,
                "family_version": FAMILY_VERSION,
                "source_line_number": candidate.source_line_number,
                "source_row_sha256": candidate.source_row_sha256,
                "rendered_steps": candidate.rendered_steps,
                "rendered_trace_sha256": candidate.rendered_trace_sha256,
                "selection_rank_sha256": candidate.selection_rank_sha256,
                "tokenizer_dry_run": token_detail,
            }
        )
        if len(selected) == n_samples:
            break
    if len(selected) != n_samples:
        raise ValueError(
            f"only {len(selected)} valid isolated replay rows; require {n_samples}"
        )
    if token_extrema["min_assistant_tokens"] == 10**9:
        token_extrema["min_assistant_tokens"] = 0
    return selected, selection_rows, {
        "rejected_after_ranking": dict(sorted(rejected.items())),
        "rendered_step_counts": {
            str(key): value for key, value in sorted(step_counts.items())
        },
        "tokenizer": {
            **dict(sorted(token_stats.items())),
            **token_extrema,
        },
    }


def _write_outputs(
    *,
    output_dir: Path,
    selected: Sequence[Candidate],
    selection_rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    silver_path = output_dir / "silver_train.jsonl"
    with silver_path.open("xb") as handle:
        for candidate in selected:
            handle.write(candidate.source_row_bytes + b"\n")
    selection_path = output_dir / "selection_records.jsonl"
    with selection_path.open("x", encoding="utf-8") as handle:
        for row in selection_rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")

    # Prove the output rows are exactly the selected source rows and in the
    # frozen order.  This catches accidental dataclass reserialisation drift.
    output_hashes = []
    with silver_path.open("rb") as handle:
        for line in handle:
            if line.strip():
                output_hashes.append(hashlib.sha256(line.rstrip(b"\r\n")).hexdigest())
    expected = [candidate.source_row_sha256 for candidate in selected]
    if output_hashes != expected:
        raise RuntimeError("source/output row byte-preservation gate failed")
    return silver_path, selection_path


def materialize_replay(
    *,
    source: Path,
    source_manifest: Path,
    legacy_replay: Path,
    protected_ledger: Path,
    hm_freeze: Path,
    tokenizer_path: Path | None,
    output_dir: Path,
    seed: int,
    n_samples: int,
    experiment_id: str,
    max_input_length: int = 6144,
    max_new_tokens: int = 384,
    max_passages: int = 15,
    max_kg_triples: int = 12,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite replay release: {output_dir}")
    if n_samples != 2000:
        raise ValueError("formal v4 Strong-SFT replay must contain exactly 2,000 rows")
    if not str(experiment_id).strip():
        raise ValueError("a nonempty Experiment ID is required")

    provenance = _validate_source_provenance(
        source=source,
        source_manifest=source_manifest,
        legacy_replay=legacy_replay,
    )
    protected_rows, protected_bindings = _validate_protected_ledger(protected_ledger)
    hm_rows, hm_bindings = _validate_hm_population(hm_freeze)
    protected_index = IdentityIndex.from_rows(
        protected_rows, label="protected-ledger"
    )
    hm_index = IdentityIndex.from_rows(hm_rows, label="H/M v4 population")
    candidates, pool_report = collect_candidates(
        source=source.resolve(),
        protected=protected_index,
        hm_population=hm_index,
        seed=seed,
    )
    if len(candidates) < n_samples:
        raise ValueError(
            f"only {len(candidates)} valid safe candidates before tokenization; "
            f"require {n_samples}"
        )

    tokenizer = None
    tokenizer_binding: dict[str, Any]
    if tokenizer_path is not None:
        from transformers import AutoTokenizer

        tokenizer_path = tokenizer_path.resolve()
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(tokenizer_path)
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True
        )
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Strong-SFT tokenizer has no chat template")
        tokenizer_binding = {
            "path": str(tokenizer_path),
            "class": tokenizer.__class__.__name__,
            "local_files_only": True,
            "tokenizer_json": file_identity(tokenizer_path / "tokenizer.json"),
            "tokenizer_config": file_identity(
                tokenizer_path / "tokenizer_config.json"
            ),
            "special_tokens_map": file_identity(
                tokenizer_path / "special_tokens_map.json"
            ),
        }
    else:
        tokenizer_binding = {"status": "NOT_RUN"}

    selected, selection_rows, selection_report = select_candidates(
        candidates,
        n_samples=n_samples,
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
        max_passages=max_passages,
        max_kg_triples=max_kg_triples,
    )
    silver_path, selection_path = _write_outputs(
        output_dir=output_dir,
        selected=selected,
        selection_rows=selection_rows,
    )

    selected_identities = [candidate.identity for candidate in selected]
    selected_index = IdentityIndex.from_rows(
        selected_identities,
        label="selected replay",
        require_stored_hashes=False,
    )
    protected_overlap = Counter()
    hm_overlap = Counter()
    for identity in selected_identities:
        for name, hit in protected_index.matches(identity).items():
            protected_overlap[name] += int(hit)
        for name, hit in hm_index.matches(identity).items():
            hm_overlap[name] += int(hit)
    selected_datasets = Counter(row["dataset"] for row in selected_identities)
    all_token_checked = tokenizer is not None and int(
        selection_report["tokenizer"].get("rows_checked", 0)
    ) == n_samples
    gates = {
        "exactly_2000_rows": len(selected) == n_samples == 2000,
        "all_accepted_train_source": True,
        "all_rendered_3to5": set(selection_report["rendered_step_counts"])
        <= {"3", "4", "5"},
        "dataset_scoped_qid_unique": len(selected_index.qids) == n_samples,
        "dataset_scoped_question_hash_unique": len(selected_index.question_hashes)
        == n_samples,
        "dataset_scoped_current_family_unique": len(selected_index.families)
        == n_samples,
        "protected_qid_hash_current_family_overlap_zero": not any(
            protected_overlap.values()
        ),
        "hm_v4_qid_hash_current_family_overlap_zero": not any(
            hm_overlap.values()
        ),
        "source_rows_preserved_bytewise": True,
        "tokenizer_dry_run_all_2000": all_token_checked,
        "training_not_started": True,
        "data_raw_not_modified": True,
        "legacy_v1c_not_overwritten": output_dir.resolve()
        != legacy_replay.resolve(),
    }
    if not all(gates.values()):
        raise RuntimeError(f"Strong-SFT replay v2 gates failed: {gates}")

    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(experiment_id).strip(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "source": provenance,
        "protected_ledger": protected_bindings,
        "hm_v4_interim_freeze": hm_bindings,
        "selection": {
            "seed": seed,
            "rank": (
                "sha256(seed\\0dataset\\0qid\\0question_sha256\\0"
                "current_family_sha256)"
            ),
            "n_samples": len(selected),
            "dataset_counts": dict(sorted(selected_datasets.items())),
            "candidate_pool": pool_report,
            **selection_report,
            "internal_identity_policy": (
                "one row per dataset-scoped qid, exact question hash, and "
                "current lexical family"
            ),
        },
        "tokenizer_dry_run_contract": {
            "tokenizer": tokenizer_binding,
            "max_input_length": max_input_length,
            "max_new_tokens": max_new_tokens,
            "max_total_tokens": max_input_length + max_new_tokens,
            "max_passages": max_passages,
            "max_kg_triples": max_kg_triples,
            "packing_logic": "mirrors phase3_ppo._prepare_sft_anchor_data",
            "lowest_ranked_passages_may_be_dropped_to_preserve_full_target": True,
        },
        "selected_overlap": {
            "protected_ledger": dict(sorted(protected_overlap.items())),
            "hm_v4_interim_population": dict(sorted(hm_overlap.items())),
        },
        "outputs": {
            "silver_train": file_identity(silver_path),
            "selection_records": file_identity(selection_path),
        },
        "gates": gates,
        "scientific_boundary": {
            "source_rows_contain_gold_answers_for_supervised_replay": True,
            "gold_or_outcome_values_used_for_selection": False,
            "selection_fields": [
                "accepted",
                "deterministic train fold",
                "rendered step/schema validity",
                "dataset/qid/question identity",
                "token packing validity",
            ],
            "evaluation_outputs_read": False,
            "data_raw_read": False,
            "data_raw_modified": False,
            "network_used": False,
            "gpu_used": False,
            "model_weights_loaded": False,
            "training_started": False,
            "legacy_v1c_preserved": True,
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_path = dump_manifest(
        output_dir,
        status=STATUS,
        extra={
            "phase": "strong_sft_replay_v4_clean_refreeze",
            "experiment_id": report["experiment_id"],
            "report_sha256": sha256_file(report_path),
            "silver_train_sha256": sha256_file(silver_path),
            "selection_records_sha256": sha256_file(selection_path),
            "protected_ledger_sha256": protected_bindings[
                "protected_identities.question_only.jsonl"
            ]["sha256"],
            "hm_population_sha256": hm_bindings[
                "hm_population.question_only.jsonl"
            ]["sha256"],
            "training_started": False,
        },
    )
    report["outputs"]["report"] = file_identity(report_path)
    report["outputs"]["manifest"] = file_identity(manifest_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST
    )
    parser.add_argument("--legacy-replay", type=Path, default=DEFAULT_LEGACY_REPLAY)
    parser.add_argument(
        "--protected-ledger", type=Path, default=DEFAULT_PROTECTED_LEDGER
    )
    parser.add_argument("--hm-freeze", type=Path, default=DEFAULT_HM_FREEZE)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--max-input-length", type=int, default=6144)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-passages", type=int, default=15)
    parser.add_argument("--max-kg-triples", type=int, default=12)
    args = parser.parse_args()
    report = materialize_replay(
        source=args.source,
        source_manifest=args.source_manifest,
        legacy_replay=args.legacy_replay,
        protected_ledger=args.protected_ledger,
        hm_freeze=args.hm_freeze,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        seed=args.seed,
        n_samples=args.n_samples,
        experiment_id=args.experiment_id,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        max_passages=args.max_passages,
        max_kg_triples=args.max_kg_triples,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output_dir),
                "selection": report["selection"],
                "selected_overlap": report["selected_overlap"],
                "gates": report["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
