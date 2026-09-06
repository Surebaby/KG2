"""Teacher and student must render the SAME KG budget (retraining_plan §12.3).

Phase 1 used to show the Teacher a top-50 KG block while PPO/inference render
only 12, so 44.5% of teacher citations pointed at triples the student can never
see. Two things made that easy to miss and easy to reintroduce:

* ``SilverDataConfig`` inherits ``extra="allow"``, so a YAML key that nothing
  reads is accepted silently (this is how ``ppo_max_kg_triples`` stayed at its
  default while the YAML appeared to set it);
* ``scripts/train/phase1_generate_silver.py`` hardcoded ``max_kg_triples=50``.

These tests fail if either returns.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from kgproweight.config import ProjectConfig, load_config
from kgproweight.training.phase1_distill import Phase1Config, _needs_format_retry
from kgproweight.training.phase3_grpo import Phase3GRPOConfig
from kgproweight.training.phase3_ppo import Phase3PPOConfig

ROOT = Path(__file__).resolve().parents[1]
BUDGET, MIN_KEEP = 12, 5


def _default(cls, name):
    return next(f.default for f in dataclasses.fields(cls) if f.name == name)


def _ints(text: str, pattern: str) -> set[int]:
    return {int(m) for m in re.findall(pattern, text)}


def _silver_cfg():
    cfg = load_config(str(ROOT / "configs/training/phase1_silver.yaml"), validate=ProjectConfig)
    return cfg.training.silver_data


def test_every_stage_uses_the_same_budget():
    prm = (ROOT / "kgproweight/training/phase2_prm.py").read_text()
    pipe = (ROOT / "kgproweight/pipeline/kg_proweight_pipeline.py").read_text()
    observed = {
        "yaml": {_silver_cfg().max_kg_triples},
        "phase1": {_default(Phase1Config, "max_kg_triples")},
        "phase2_prm": _ints(prm, r"max_keep=(\d+)"),
        "phase3_ppo": {_default(Phase3PPOConfig, "ppo_max_kg_triples")},
        "phase3_grpo": {_default(Phase3GRPOConfig, "max_kg_triples")},
        "inference": _ints(pipe, r"max_kg_triples: int = (\d+)"),
    }
    bad = {k: sorted(v) for k, v in observed.items() if v != {BUDGET}}
    assert not bad, f"KG budget must be {BUDGET} everywhere; disagreeing stages: {bad}"


def test_every_stage_uses_the_same_min_keep():
    prm = (ROOT / "kgproweight/training/phase2_prm.py").read_text()
    assert _default(Phase1Config, "min_kg_keep") == MIN_KEEP
    assert _silver_cfg().min_kg_keep == MIN_KEEP
    assert _ints(prm, r"min_keep=(\d+)") == {MIN_KEEP}
    assert _default(Phase3PPOConfig, "ppo_min_kg_triples") == MIN_KEEP


def test_ppo_yaml_budget_is_declared_and_effective():
    cfg = load_config(str(ROOT / "configs/training/phase3_ppo.yaml"), validate=ProjectConfig)
    ppo = cfg.training.ppo
    assert ppo.ppo_min_kg_triples == MIN_KEEP
    assert ppo.ppo_max_kg_triples == BUDGET
    assert not ({"ppo_min_kg_triples", "ppo_max_kg_triples"} & set(ppo.model_extra or {}))

    cli = (ROOT / "scripts/train/phase3_ppo.py").read_text()
    assert "ppo_min_kg_triples=ppo_cfg.ppo_min_kg_triples" in cli
    assert "ppo_max_kg_triples=ppo_cfg.ppo_max_kg_triples" in cli


def test_phase1_passes_min_keep_to_the_filter():
    """Phase 1 used to omit min_keep (=0), a stricter filter than inference."""
    src = (ROOT / "kgproweight/training/phase1_distill.py").read_text()
    call = re.search(r"teacher_kg = filter_and_rank_triples\((.*?)\)", src, re.S)
    assert call, "filter_and_rank_triples call not found in _process_one"
    assert "min_keep=cfg.min_kg_keep" in call.group(1)
    assert "max_keep=cfg.max_kg_triples" in call.group(1)


def test_cli_forwards_the_config_value():
    src = (ROOT / "scripts/train/phase1_generate_silver.py").read_text()
    assert "max_kg_triples=max_kg_triples" in src
    assert "max_kg_triples=50" not in src, "hardcoded 50 is back; the YAML would be ignored"


def test_yaml_keys_are_declared_fields_not_extras():
    """extra='allow' would swallow a typo'd or undeclared key without a word."""
    cfg = _silver_cfg()
    extra = cfg.model_extra or {}
    for key in ("max_kg_triples", "min_kg_keep"):
        assert key in type(cfg).model_fields, f"{key} must be a declared field"
        assert key not in extra, f"{key} landed in model_extra -- nothing reads it"


def test_question_kg_index_default_matches_the_budget():
    src = (ROOT / "scripts/prepare/06_build_question_kg_index.py").read_text()
    m = re.search(r'"--max_keep",\s*type=int,\s*default=(\d+)', src)
    assert m and int(m.group(1)) == BUDGET, "index default must equal the student budget"


class _Step:
    def __init__(self, cited):
        self.cited_triples = cited


KG = [("a", "b", "c")]


def test_retry_allows_empty_citations_when_nonempty_kg_is_irrelevant():
    raw = """[Step 1]\nReasoning: A.\nKnowledge Used: []\nConclusion: A.
[Step 2]\nReasoning: B.\nKnowledge Used: []\nConclusion: B.
[Step 3]\nReasoning: C.\nKnowledge Used: []\nConclusion: C.\n[Final Answer] C"""
    steps = [_Step([]), _Step([]), _Step([])]
    assert not _needs_format_retry(steps, KG, min_steps=3, raw_output=raw)


def test_retry_rejects_out_of_visible_kg_citation():
    raw = """[Step 1]\nReasoning: A.\nKnowledge Used: [(x, y, z)]\nConclusion: A.
[Step 2]\nReasoning: B.\nKnowledge Used: []\nConclusion: B.
[Step 3]\nReasoning: C.\nKnowledge Used: []\nConclusion: C.\n[Final Answer] C"""
    steps = [_Step([]), _Step([]), _Step([])]
    assert _needs_format_retry(steps, KG, min_steps=3, raw_output=raw)


def test_retry_rejects_too_few_steps():
    assert _needs_format_retry([_Step(KG), _Step(KG)], KG, min_steps=3)


def test_annotation_uses_the_filtered_subgraph():
    """PRM labels must be computed against what is stored, not the raw fetch."""
    src = (ROOT / "kgproweight/training/phase1_distill.py").read_text()
    body = src[src.index("def _process_one"):]
    body = body[: body.index("return _Candidate")]
    assert "_annotate_steps(raw_output, teacher_kg, annotator)" in body
    assert "_annotate_steps(raw_output, triples, annotator)" not in body
    assert '"kg_empty": len(teacher_kg) == 0' in body
