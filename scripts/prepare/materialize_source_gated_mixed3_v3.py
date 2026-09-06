#!/usr/bin/env python3
"""Materialise the append-only three-dataset source-gated PPO data release.

This command does not generate trajectories and does not train a model.  It
copies the five already-frozen mixed3 assets byte-for-byte, computes a strict
dataset-agnostic Graph mask sidecar, and derives a separate 3--5 step SFT replay
pool from the frozen Strong-SFT silver source.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
from typing import Any, Dict, Iterable

from kgproweight.data.parsers import extract_final_answer, parse_steps
from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.data.silver_split import SplitSpec
from kgproweight.kg.question_kg import load_question_kg_index, question_key
from kgproweight.reward.trajectory_source_gate import make_source_gate_record
from kgproweight.training.phase3_sft import _render_assistant_trace


BASE_FILES = (
    "silver_train.jsonl",
    "question_kg_records.jsonl",
    "sampling_weights.jsonl",
    "prompt_groups.jsonl",
    "fixed_rollout_schedule.jsonl",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows = []
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


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _git_state(root: Path) -> Dict[str, Any]:
    def run_text(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, check=False
        )
        return (
            result.stdout.decode("utf-8", errors="replace").strip()
            if result.returncode == 0
            else "UNKNOWN"
        )

    commit = run_text("rev-parse", "HEAD")
    status = run_text("status", "--short")
    diff_result = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    diff = diff_result.stdout if diff_result.returncode == 0 else b"UNKNOWN"
    dirty_payload = status.encode("utf-8") + b"\n" + diff
    return {
        "commit": commit or "UNKNOWN",
        "dirty": bool(status and status != "UNKNOWN"),
        "dirty_state_sha256": hashlib.sha256(dirty_payload).hexdigest(),
    }


def _has_text_evidence(row: Dict[str, Any]) -> bool:
    passages = row.get("retrieved_passages") or []
    return len(passages) == 10 and all(
        isinstance(passage, dict)
        and bool(str(passage.get("id") or "").strip())
        and bool(str(passage.get("source") or "").strip())
        and bool(str(passage.get("contents") or "").strip())
        for passage in passages
    )


def _valid_replay_trace(trajectory: Any) -> tuple[bool, int, str]:
    rendered = _render_assistant_trace(trajectory)
    steps = parse_steps(rendered, known_kg=trajectory.kg_subgraph)
    n_steps = len(steps)
    if not 3 <= n_steps <= 5:
        return False, n_steps, "rendered_step_count_outside_3_5"
    if [step.index for step in steps] != list(range(1, n_steps + 1)):
        return False, n_steps, "nonsequential_steps"
    if not extract_final_answer(rendered):
        return False, n_steps, "missing_final_answer"
    required_fields = ("reasoning:", "knowledge used:", "conclusion:")
    for step in steps:
        body = str(step.raw_text or "").casefold()
        if any(field not in body for field in required_fields):
            return False, n_steps, "missing_step_field"
        if step.unknown_citation_surfaces:
            return False, n_steps, "citation_outside_stored_kg"
    return True, n_steps, "accepted"


def _materialize_replay(
    *, source: Path, output_dir: Path, seed: int, n_samples: int, experiment_id: str
) -> Dict[str, Any]:
    reader = SilverDatasetReader(source, split="train", split_spec=SplitSpec())
    candidates = sorted(
        reader.accepted(), key=lambda item: (str(item.dataset), str(item.qid))
    )
    random.Random(seed).shuffle(candidates)

    selected = []
    selection_rows = []
    rejected = Counter()
    step_counts = Counter()
    for trajectory in candidates:
        valid, n_steps, reason = _valid_replay_trace(trajectory)
        if not valid:
            rejected[reason] += 1
            continue
        selected.append(trajectory)
        step_counts[n_steps] += 1
        selection_rows.append(
            {
                "dataset": trajectory.dataset,
                "qid": trajectory.qid,
                "rendered_steps": n_steps,
                "rendered_trace_sha256": hashlib.sha256(
                    _render_assistant_trace(trajectory).encode("utf-8")
                ).hexdigest(),
            }
        )
        if len(selected) == n_samples:
            break
    if len(selected) != n_samples:
        raise ValueError(
            f"only {len(selected)} valid replay trajectories; need {n_samples}"
        )

    output_dir.mkdir(parents=False, exist_ok=False)
    silver_path = output_dir / "silver_train.jsonl"
    SilverDatasetReader.write_jsonl(silver_path, selected)
    selection_path = output_dir / "selection_records.jsonl"
    _write_jsonl(selection_path, selection_rows)
    report = {
        "schema_version": "sft-replay-rendered-3to5-v1",
        "experiment_id": experiment_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_DATA_NOT_TRAINED",
        "source": _identity(source),
        "split": {"name": "train", "spec": SplitSpec().__dict__},
        "selection": {
            "seed": seed,
            "sort_key": ["dataset", "qid"],
            "n_samples": len(selected),
            "rendered_step_counts": dict(sorted(step_counts.items())),
            "rejected_before_pool_complete": dict(sorted(rejected.items())),
            "requires_fields": ["Reasoning", "Knowledge Used", "Conclusion"],
            "requires_known_kg_citation_schema": True,
        },
        "training_started": False,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "dir": str(output_dir),
        "silver_train": _identity(silver_path),
        "selection_records": _identity(selection_path),
        "report": _identity(report_path),
        "selected_keys": {
            question_key(item.dataset, item.qid) for item in selected
        },
        "step_counts": dict(sorted(step_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-source", type=Path, required=True)
    parser.add_argument("--replay-output-dir", type=Path, required=True)
    parser.add_argument("--stage3-report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay-n", type=int, default=2000)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--replay-experiment-id", required=True)
    args = parser.parse_args()

    root = Path.cwd().resolve()
    parent = args.parent_dir.resolve()
    output = args.output_dir.resolve()
    replay_output = args.replay_output_dir.resolve()
    if output.exists() or replay_output.exists():
        raise FileExistsError(
            f"append-only outputs already exist: {output} or {replay_output}"
        )
    for filename in BASE_FILES:
        if not (parent / filename).is_file():
            raise FileNotFoundError(parent / filename)

    stage3 = json.loads(args.stage3_report.read_text(encoding="utf-8"))
    cutoff = str((stage3.get("cache_policy") or {}).get("historical_cutoff") or "")
    if not cutoff:
        raise ValueError("stage3 report does not bind cache_policy.historical_cutoff")

    silver_rows = _read_jsonl(parent / "silver_train.jsonl")
    kg_rows = _read_jsonl(parent / "question_kg_records.jsonl")
    kg_index = load_question_kg_index(kg_rows)
    if len(silver_rows) != 1799 or len(kg_index) != 1799:
        raise ValueError("parent mixed3 population is not the frozen n=1799 release")

    output.mkdir(parents=False, exist_ok=False)
    parent_identities = {}
    output_identities = {}
    for filename in BASE_FILES:
        source = parent / filename
        destination = output / filename
        shutil.copyfile(source, destination)
        parent_identities[filename] = _identity(source)
        output_identities[filename] = _identity(destination)
        if parent_identities[filename]["sha256"] != output_identities[filename]["sha256"]:
            raise RuntimeError(f"byte identity failed while copying {filename}")

    gate_rows = []
    dataset_counts = Counter()
    graph_counts = Counter()
    for row in silver_rows:
        dataset = str(row.get("dataset") or "").strip().lower()
        qid = str(row.get("qid") or "").strip()
        question = str(row.get("question") or "").strip()
        key = question_key(dataset, qid)
        if key not in kg_index:
            raise ValueError(f"missing question-KG record for {key}")
        text_available = _has_text_evidence(row)
        if not text_available:
            raise ValueError(f"missing canonical ten-passage evidence for {key}")
        gate = make_source_gate_record(
            kg_index[key],
            dataset=dataset,
            qid=qid,
            question=question,
            text_evidence_available=text_available,
            historical_cutoff=cutoff,
        )
        gate_rows.append(gate)
        dataset_counts[dataset] += 1
        graph_counts[(dataset, gate["m_graph"])] += 1

    gate_path = output / "source_gate_records.jsonl"
    _write_jsonl(gate_path, gate_rows)

    schedules = _read_jsonl(output / "fixed_rollout_schedule.jsonl")
    groups = _read_jsonl(output / "prompt_groups.jsonl")
    gate_index = {row["question_key"]: row for row in gate_rows}
    scheduled_graph = 0
    for schedule in schedules:
        key = question_key(schedule["dataset"], schedule["qid"])
        if key not in gate_index:
            raise ValueError(f"schedule identity missing gate row: {key}")
        scheduled_graph += gate_index[key]["m_graph"]

    replay = _materialize_replay(
        source=args.replay_source.resolve(),
        output_dir=replay_output,
        seed=args.seed,
        n_samples=args.replay_n,
        experiment_id=args.replay_experiment_id,
    )
    rollout_keys = {
        question_key(str(row.get("dataset") or ""), str(row.get("qid") or ""))
        for row in silver_rows
    }
    overlap = sorted(rollout_keys & replay.pop("selected_keys"))
    if overlap:
        raise ValueError(
            f"replay overlaps mixed3 rollout population: {len(overlap)} keys"
        )

    gates = {
        "population_1799_unique": len(gate_rows) == len(gate_index) == 1799,
        "dataset_counts_exact": dict(dataset_counts)
        == {"hotpotqa": 600, "2wikimultihopqa": 600, "musique": 599},
        "all_text_evidence_available": all(
            row["text_evidence_available"] for row in gate_rows
        ),
        "graph_eligible_exact_400": sum(row["m_graph"] for row in gate_rows) == 400,
        "graph_ineligible_exact_1399": sum(1 - row["m_graph"] for row in gate_rows) == 1399,
        "schedule_7200_k4": len(schedules) == 7200
        and len(groups) == 1800
        and all(
            len({r["qid"] for r in schedules[start : start + 4]}) == 1
            for start in range(0, len(schedules), 4)
        ),
        "scheduled_graph_eligible_exact_1600": scheduled_graph == 1600,
        "replay_exact_2000": args.replay_n == 2000,
        "replay_all_rendered_3to5": set(replay["step_counts"]).issubset({3, 4, 5}),
        "rollout_replay_identity_overlap_zero": not overlap,
    }
    if not all(gates.values()):
        raise RuntimeError(f"source-gated data release failed gates: {gates}")

    report = {
        "schema_version": "mixed-ppo-three-dataset-source-gated-v3",
        "experiment_id": args.experiment_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_DATA_NOT_TRAINED",
        "counts": {
            "unique_population": len(gate_rows),
            "unique_by_dataset": dict(dataset_counts),
            "graph_mask_by_dataset": {
                dataset: {
                    "m_graph_1": graph_counts[(dataset, 1)],
                    "m_graph_0": graph_counts[(dataset, 0)],
                }
                for dataset in sorted(dataset_counts)
            },
            "scheduled_prompt_groups": len(groups),
            "scheduled_trajectories": len(schedules),
            "scheduled_graph_eligible_trajectories": scheduled_graph,
        },
        "gate_semantics": {
            "dataset_name_used_as_feature": False,
            "gold_answer_used_as_feature": False,
            "fail_closed": True,
            "historical_cutoff": cutoff,
            "known_distribution_boundary": (
                "All currently eligible Graph rows happen to be 2Wiki; this is an "
                "observed upstream supply distribution, not a dataset branch."
            ),
        },
        "gates": gates,
        "training_started": False,
        "replay_release": replay,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": "mixed-ppo-three-dataset-source-gated-v3-manifest",
        "experiment_id": report["experiment_id"],
        "status": report["status"],
        "parent": {
            "directory": str(parent),
            "report": _identity(parent / "report.json"),
            "manifest": _identity(parent / "manifest.json"),
            "base_files": parent_identities,
        },
        "upstream_cutoff_source": _identity(args.stage3_report.resolve()),
        "replay_source": _identity(args.replay_source.resolve()),
        "code": {
            "gate": _identity(root / "kgproweight/reward/trajectory_source_gate.py"),
            "materializer": _identity(root / "scripts/prepare/materialize_source_gated_mixed3_v3.py"),
            "renderer": _identity(root / "kgproweight/training/phase3_sft.py"),
            "parser": _identity(root / "kgproweight/data/parsers.py"),
        },
        "outputs": {
            **output_identities,
            "source_gate_records.jsonl": _identity(gate_path),
            "report.json": _identity(report_path),
            "replay": replay,
        },
        "git": _git_state(root),
        "training_started": False,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
