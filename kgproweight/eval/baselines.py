"""Baseline registry (paper §5.4).

Every baseline shares the same hybrid RRF top-50 retrieval; only the
pipeline class and generator differ. This module produces FlashRAG-ready
config dicts; the runner in :mod:`scripts.eval.run_baselines` instantiates
the pipeline and runs evaluation.

YAML files under ``configs/eval/baseline_*.yaml`` document the same
settings but are **not** loaded at runtime — edit this registry instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from kgproweight.retrieval.hybrid import build_flashrag_config

RunMode = Literal["standard", "naive"]


@dataclass
class BaselineSpec:
    name: str
    pipeline_class: str
    pipeline_module: str
    generator_model: str
    framework: str = "hf"
    is_reasoning: bool = False
    run_mode: RunMode = "standard"
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    use_kg_retrieval: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)


BASELINES: List[BaselineSpec] = [
    BaselineSpec(
        name="zero_shot",
        pipeline_class="SequentialPipeline",
        pipeline_module="flashrag.pipeline.pipeline",
        generator_model="llama3-8B-instruct",
        run_mode="naive",
        system_prompt="Answer the question based on your own knowledge. Only give the answer.",
        user_prompt="Question: {question}",
    ),
    BaselineSpec(
        name="naive_rag",
        pipeline_class="SequentialPipeline",
        pipeline_module="flashrag.pipeline.pipeline",
        generator_model="llama3-8B-instruct",
        system_prompt="Answer the question based on the retrieved passages. Only give the answer.",
        user_prompt="Reference passages:\n{reference}\n\nQuestion: {question}\nAnswer:",
    ),
    BaselineSpec(
        name="self_rag",
        pipeline_class="SelfRAGPipeline",
        pipeline_module="flashrag.pipeline.active_pipeline",
        generator_model="selfrag",
    ),
    BaselineSpec(
        name="trace",
        pipeline_class="SequentialPipeline",
        pipeline_module="flashrag.pipeline.pipeline",
        generator_model="llama3-8B-instruct",
        system_prompt="Answer the question based on the retrieved passages.",
        user_prompt="Reference passages:\n{reference}\n\nQuestion: {question}\nAnswer:",
        extras={"refiner_name": "kg_trace_refiner"},
    ),
    BaselineSpec(
        name="r1_searcher",
        pipeline_class="ReasoningPipeline",
        pipeline_module="flashrag.pipeline.reasoning_pipeline",
        generator_model="r1-searcher",
        is_reasoning=True,
    ),
    BaselineSpec(
        name="rearag",
        pipeline_class="ReaRAGPipeline",
        pipeline_module="flashrag.pipeline.reasoning_pipeline",
        generator_model="rearag",
        is_reasoning=True,
        # ReaRAG pipeline overrides temperature=0 in per-step params; keep decoding
        # deterministic globally to avoid sampling-time temperature validation errors.
        extras={"generation_params": {"do_sample": False}},
    ),
]


def baseline_config(
    spec: BaselineSpec,
    dataset_name: str,
    save_dir: str,
    *,
    split: str = "dev",
    test_sample_num: Optional[int] = None,
    seed: int = 42,
    gpu_id: str = "0",
) -> Dict[str, Any]:
    """Build a FlashRAG config dict for one baseline run."""
    cfg = build_flashrag_config(
        dataset_name=dataset_name,
        save_note=spec.name,
        save_dir=save_dir,
        method_name=spec.name,
        pipeline_class=spec.pipeline_class,
        generator_model=spec.generator_model,
        framework=spec.framework,
        split=split,
        test_sample_num=test_sample_num,
        seed=seed,
        gpu_id=gpu_id,
        is_reasoning=spec.is_reasoning,
    )
    cfg.update(spec.extras)
    return cfg
