"""SFT curriculum controls must reach the runtime dataclass."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

import scripts.train.phase3_sft as phase3_cli


def test_continued_sft_and_question_kg_controls_are_forwarded(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "sft.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "training": {
                    "phase": "phase3_sft",
                    "silver_path": "data/curriculum.jsonl",
                    "output_dir": "outputs/sft",
                    "split": None,
                    "split_allow_none": True,
                    "sft_init_adapter_path": "checkpoints/old/final",
                    "question_kg_records_path": "data/proofkg.jsonl",
                    "min_question_kg_record_coverage": 0.97,
                    "require_nonempty_question_kg_records": True,
                    "sft_save_strategy": "steps",
                    "sft_save_steps": 40,
                    "sft_save_total_limit": 4,
                    "sft_save_only_model": True,
                    "sft_log_with": "tensorboard",
                    "sft_logging_dir": "/root/tf-logs/test-sft",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.setattr(
        phase3_cli,
        "run_phase3_sft",
        lambda cfg: captured.setdefault("cfg", cfg),
    )
    monkeypatch.setattr(sys, "argv", ["phase3_sft.py", "--config", str(config_path)])

    phase3_cli.main()
    cfg = captured["cfg"]

    assert cfg.init_adapter_path == "checkpoints/old/final"
    assert cfg.question_kg_records_path == "data/proofkg.jsonl"
    assert cfg.min_question_kg_record_coverage == 0.97
    assert cfg.require_nonempty_question_kg_records is True
    assert cfg.save_strategy == "steps"
    assert cfg.save_steps == 40
    assert cfg.save_total_limit == 4
    assert cfg.save_only_model is True
    assert cfg.log_with == "tensorboard"
    assert cfg.logging_dir == "/root/tf-logs/test-sft"
    assert cfg.split is None
    assert cfg.split_allow_none is True
