#!/usr/bin/env python3
"""Fail-fast CPU preflight for the source-gated mixed3 v3 data release."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from kgproweight.data.parsers import extract_final_answer, parse_steps
from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.data.silver_split import SplitSpec
from kgproweight.kg.question_kg import question_key
from kgproweight.training.phase3_ppo import Phase3PPOConfig, _prepare_sft_anchor_data
from kgproweight.training.phase3_sft import _render_assistant_trace


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _valid_rendered_replay(trajectory: Any) -> bool:
    rendered = _render_assistant_trace(trajectory)
    steps = parse_steps(rendered, known_kg=trajectory.kg_subgraph)
    return bool(
        3 <= len(steps) <= 5
        and [step.index for step in steps] == list(range(1, len(steps) + 1))
        and extract_final_answer(rendered)
        and all(
            not step.unknown_citation_surfaces
            and all(
                field in str(step.raw_text or "").casefold()
                for field in ("reasoning:", "knowledge used:", "conclusion:")
            )
            for step in steps
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--experiment-id", default="SOURCE-GATED-MIXED3-V3-PREFLIGHT-ADHOC")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    replay_dir = args.replay_dir.resolve()

    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((data_dir / "report.json").read_text(encoding="utf-8"))
    checks: Dict[str, bool] = {
        "status_complete_not_trained": report.get("status") == "COMPLETE_DATA_NOT_TRAINED"
        and manifest.get("status") == "COMPLETE_DATA_NOT_TRAINED"
        and report.get("training_started") is False
        and manifest.get("training_started") is False,
        "materialization_gates_all_pass": all((report.get("gates") or {}).values()),
    }
    for name, identity in (manifest.get("outputs") or {}).items():
        if name == "replay":
            continue
        path = Path(identity["path"])
        checks[f"hash_{name}"] = path.is_file() and _sha256(path) == identity["sha256"]
    for name, identity in (manifest.get("parent") or {}).get("base_files", {}).items():
        checks[f"parent_copy_identity_{name}"] = (
            identity["sha256"]
            == (manifest.get("outputs") or {}).get(name, {}).get("sha256")
        )

    silver = _read_jsonl(data_dir / "silver_train.jsonl")
    gates = _read_jsonl(data_dir / "source_gate_records.jsonl")
    question_kg = _read_jsonl(data_dir / "question_kg_records.jsonl")
    schedule = _read_jsonl(data_dir / "fixed_rollout_schedule.jsonl")
    gate_by_key = {row["question_key"]: row for row in gates}
    kg_by_key = {row["question_key"]: row for row in question_kg}
    dataset_counts = Counter(str(row["dataset"]) for row in silver)
    checks.update(
        {
            "population_and_join_1799": len(silver) == len(gates) == len(question_kg) == len(gate_by_key) == len(kg_by_key) == 1799,
            "dataset_counts_exact": dict(dataset_counts)
            == {"hotpotqa": 600, "2wikimultihopqa": 600, "musique": 599},
            "all_text_available": all(row.get("text_evidence_available") is True for row in gates),
            "graph_mask_exact_400": sum(int(row.get("m_graph", -1)) for row in gates) == 400,
            "ineligible_graphs_are_empty": all(
                kg_by_key[row["question_key"]].get("kg_subgraph") == []
                for row in gates
                if row.get("m_graph") == 0
            ),
            "eligible_checks_all_true": all(
                all((row.get("eligibility_checks") or {}).values())
                for row in gates
                if row.get("m_graph") == 1
            ),
            "schedule_7200": len(schedule) == 7200,
            "schedule_k4_grouped": all(
                len(
                    {
                        question_key(item["dataset"], item["qid"])
                        for item in schedule[start : start + 4]
                    }
                )
                == 1
                for start in range(0, len(schedule), 4)
            ),
            "scheduled_graph_exact_1600": sum(
                gate_by_key[question_key(row["dataset"], row["qid"])]["m_graph"]
                for row in schedule
            )
            == 1600,
        }
    )

    replay_reader = SilverDatasetReader(
        replay_dir / "silver_train.jsonl", split="train", split_spec=SplitSpec()
    )
    replay = replay_reader.accepted()
    rollout_keys = {question_key(row["dataset"], row["qid"]) for row in silver}
    replay_keys = {question_key(row.dataset, row.qid) for row in replay}
    checks.update(
        {
            "replay_exact_2000": len(replay) == 2000,
            "replay_all_rendered_3to5_valid": all(_valid_rendered_replay(row) for row in replay),
            "replay_rollout_overlap_zero": not (replay_keys & rollout_keys),
        }
    )

    tokenizer_stats = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(args.tokenizer), local_files_only=True, use_fast=True
        )
        cfg = Phase3PPOConfig(
            silver_path=str(data_dir / "silver_train.jsonl"),
            output_dir="/tmp/source-gated-preflight",
            seed=42,
            split=None,
            split_allow_none=True,
            max_input_length=6144,
            max_new_tokens=384,
            ppo_max_passages=15,
            ppo_max_kg_triples=12,
        )
        prepared = _prepare_sft_anchor_data(
            silver_path=str(replay_dir / "silver_train.jsonl"),
            tokenizer=tokenizer,
            cfg=cfg,
            max_samples=2000,
            apply_rollout_question_kg=False,
            replay_split="train",
        )
        tokenizer_stats = {
            "prepared": len(prepared),
            "max_total_tokens": max(len(row["input_ids"]) for row in prepared),
            "min_passages_retained": min(row["num_passages"] for row in prepared),
        }
        checks["tokenizer_prepared_exact_2000"] = len(prepared) == 2000
        checks["tokenizer_retained_all_15_passages"] = tokenizer_stats["min_passages_retained"] == 15

    result = {
        "schema_version": "source-gated-mixed3-v3-preflight-v1",
        "experiment_id": args.experiment_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "tokenizer": tokenizer_stats,
        "training_started": False,
    }
    if args.output_dir:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
        report_path = output / "report.json"
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": "source-gated-mixed3-v3-preflight-manifest-v1",
            "experiment_id": result["experiment_id"],
            "status": result["status"],
            "inputs": {
                "data_manifest": {
                    "path": str(data_dir / "manifest.json"),
                    "sha256": _sha256(data_dir / "manifest.json"),
                },
                "replay_report": {
                    "path": str(replay_dir / "report.json"),
                    "sha256": _sha256(replay_dir / "report.json"),
                },
                "tokenizer": str(args.tokenizer.resolve()) if args.tokenizer else None,
            },
            "code": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "outputs": {
                "report": {"path": str(report_path), "sha256": _sha256(report_path)}
            },
            "training_started": False,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
