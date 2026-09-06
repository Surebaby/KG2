#!/usr/bin/env python
"""CPU preflight for the frozen QPEG-v4 schema-adaptation SFT run."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from peft import PeftConfig

from kgproweight.config import ProjectConfig, load_config
from kgproweight.data.parsers import parse_steps
from kgproweight.utils.logging import dump_manifest


EXPERIMENT_ID = "QPEG-V4-SCHEMA-ADAPT-SFT-N2400-SEED42"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/training/phase3_sft_qpeg_v4_schema_adaptation_n2400_seed42.yaml"),
    )
    parser.add_argument(
        "--protocol", type=Path,
        default=Path("outputs/audits/qpeg_v4_schema_adaptation_protocol_v1/protocol.json"),
    )
    parser.add_argument(
        "--data_report", type=Path,
        default=Path("data/silver_data/qpeg_v4_schema_adaptation_n2400_seed42_v2/report.json"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("outputs/audits/qpeg_v4_schema_adaptation_sft_preflight_v1"),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite preflight: {args.out}")
    args.out.mkdir(parents=True)
    cfg = load_config(args.config, validate=ProjectConfig).training
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    data_report = json.loads(args.data_report.read_text(encoding="utf-8"))
    silver_path = Path(cfg.silver_path)
    adapter_path = Path(cfg.sft_init_adapter_path)
    output_dir = Path(cfg.output_dir)
    rows = _read_jsonl(silver_path)

    counts = Counter((str(row["dataset"]), str((row.get("metadata") or {}).get("curriculum_variant"))) for row in rows)
    parse_valid = 0
    citation_exact = 0
    for row in rows:
        trace = "\n\n".join(f"[Step {step['index']}]\n{step['text']}" for step in row["steps"])
        parsed = parse_steps(trace, known_kg=row["kg_subgraph"])
        parse_valid += len(parsed) == len(row["steps"]) and all(step.knowledge_used_valid for step in parsed)
        observed = {triple for step in parsed for triple in step.cited_triples}
        declared = {tuple(triple) for step in row["steps"] for triple in step.get("cited_triples", [])}
        citation_exact += observed == declared

    peft_cfg = PeftConfig.from_pretrained(adapter_path)
    expected_updates = math.ceil(len(rows) / (int(cfg.sft_batch_size) * int(cfg.sft_grad_accum)))
    checks = {
        "protocol_approval_matches": protocol.get("researcher_approval_id") == "USER-APPROVED-2026-09-03-QPEG-V4-SCHEMA-ADAPTATION",
        "data_status_complete_not_trained": data_report.get("status") == "COMPLETE_NOT_TRAINED",
        "silver_hash_matches_report": _sha256(silver_path) == data_report["outputs"]["silver"]["sha256"],
        "rows_2400": len(rows) == 2400,
        "counts_600_graph_200_replay_each": all(
            counts[(dataset, "qpeg")] == 600 and counts[(dataset, "no_graph_replay")] == 200
            for dataset in ("hotpotqa", "2wikimultihopqa", "musique")
        ),
        "unique_row_ids": len({str(row["qid"]) for row in rows}) == 2400,
        "all_parser_valid": parse_valid == 2400,
        "all_citations_exact": citation_exact == 2400,
        "output_dir_absent": not output_dir.exists(),
        "start_adapter_exists": adapter_path.is_dir(),
        "lora_r_32": int(peft_cfg.r) == 32,
        "lora_alpha_64": int(peft_cfg.lora_alpha) == 64,
        "lora_targets_exact": set(peft_cfg.target_modules or []) == {"q_proj", "k_proj", "v_proj", "o_proj"},
        "lr_2e_6": float(cfg.sft_lr) == 2e-6,
        "effective_batch_32": int(cfg.sft_batch_size) * int(cfg.sft_grad_accum) == 32,
        "max_length_6144": int(cfg.sft_max_length) == 6144,
        "one_epoch": int(cfg.sft_epochs) == 1,
        "expected_updates_75": expected_updates == 75,
        "save_every_25": cfg.sft_save_strategy == "steps" and int(cfg.sft_save_steps) == 25,
        "derived_whole_file_explicit": cfg.split is None and bool(cfg.split_allow_none),
    }
    report = {
        "schema_version": "qpeg-v4-schema-adaptation-sft-preflight-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_NOT_TRAINED" if all(checks.values()) else "FAIL_STOP",
        "counts": {f"{dataset}::{variant}": value for (dataset, variant), value in counts.items()},
        "expected_updates": expected_updates,
        "checks": checks,
        "inputs": {
            "config": {"path": str(args.config), "sha256": _sha256(args.config)},
            "protocol": {"path": str(args.protocol), "sha256": _sha256(args.protocol)},
            "data_report": {"path": str(args.data_report), "sha256": _sha256(args.data_report)},
            "silver": {"path": str(silver_path), "sha256": _sha256(silver_path)},
            "adapter_model": {
                "path": str(adapter_path / "adapter_model.safetensors"),
                "sha256": _sha256(adapter_path / "adapter_model.safetensors"),
            },
        },
        "output_dir": str(output_dir),
        "scientific_boundary": "CPU/config/data readiness only. This report does not contain a trained model or downstream utility result.",
    }
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, extra={"phase": "qpeg_v4_sft_preflight", **report}, status=report["status"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
