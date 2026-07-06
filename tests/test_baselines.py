"""Baseline registry and metric export tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kgproweight.eval.baselines import BASELINES, baseline_config
from kgproweight.eval.runner import export_metric_json, parse_metric_score_txt
from kgproweight.retrieval.hybrid import (
    EVAL_GENERATOR_MAX_INPUT_LEN,
    apply_retrieval_overrides,
    build_flashrag_config,
)


def test_baseline_flashrag_module_paths():
    modules = {b.name: b.pipeline_module for b in BASELINES}
    assert modules["zero_shot"] == "flashrag.pipeline.pipeline"
    assert modules["naive_rag"] == "flashrag.pipeline.pipeline"
    assert modules["self_rag"] == "flashrag.pipeline.active_pipeline"
    assert modules["rearag"] == "flashrag.pipeline.reasoning_pipeline"
    assert modules["r1_searcher"] == "flashrag.pipeline.reasoning_pipeline"


def test_zero_shot_uses_naive_run():
    zero = next(b for b in BASELINES if b.name == "zero_shot")
    assert zero.run_mode == "naive"
    assert zero.system_prompt is not None
    assert zero.user_prompt is not None


def test_reasoning_baselines_flag():
    rearag = next(b for b in BASELINES if b.name == "rearag")
    r1 = next(b for b in BASELINES if b.name == "r1_searcher")
    naive = next(b for b in BASELINES if b.name == "naive_rag")
    assert rearag.is_reasoning is True
    assert r1.is_reasoning is True
    assert naive.is_reasoning is False


def test_baseline_config_sets_is_reasoning():
    spec = next(b for b in BASELINES if b.name == "rearag")
    cfg = baseline_config(spec, "hotpotqa", "/tmp/out", seed=42)
    assert cfg["is_reasoning"] is True


def test_apply_retrieval_e5_only_override():
    cfg = build_flashrag_config("hotpotqa", "t", "/tmp/out")
    cfg = apply_retrieval_overrides(cfg, {"use_multi_retriever": False, "retrieval_topk": 50})
    assert cfg["use_multi_retriever"] is False
    assert cfg["retrieval_topk"] == 50


def test_kg_proweight_config_includes_lora_path(tmp_path):
    ckpt = str(tmp_path / "final")
    cfg = build_flashrag_config(
        "hotpotqa",
        save_note="kg_proweight",
        save_dir=str(tmp_path / "eval"),
        generator_lora_path=ckpt,
    )
    assert cfg["generator_lora_path"] == ckpt
    assert cfg["generator_max_input_len"] == EVAL_GENERATOR_MAX_INPUT_LEN


def test_parse_metric_score_txt(tmp_path):
    txt = tmp_path / "metric_score.txt"
    txt.write_text("em: 0.53\nf1: 0.42\n", encoding="utf-8")
    metrics = parse_metric_score_txt(txt)
    assert metrics["em"] == pytest.approx(0.53)
    assert metrics["f1"] == pytest.approx(0.42)


def test_export_metric_json(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metric_score.txt").write_text("em: 0.5\nf1: 0.6\n", encoding="utf-8")
    out = export_metric_json(run_dir)
    assert out is not None
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["em"] == pytest.approx(0.5)
    assert data["f1"] == pytest.approx(0.6)
