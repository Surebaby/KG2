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

from kgproweight.retrieval.hybrid import DEFAULT_TOPK, build_flashrag_config
from flashrag.utils.pred_parse import ircot_pred_parse

RunMode = Literal["standard", "naive"]


def corag_extract_answer(dataset):
    """CoRAG pred processing: extract first non-empty content line."""
    for item in dataset:
        pred = getattr(item, "pred", "") or ""
        lines = [l.strip() for l in pred.split("\n") if l.strip()]
        for line in lines:
            if not line.startswith("Question:") and not line.startswith("Answer:"):
                if len(line) > 3:
                    item.pred = line
                    break
        else:
            item.pred = lines[0] if lines else ""
    return dataset


def r1_extract_answer(dataset):
    """R1-Searcher pred processing: extract content inside <answer> tags.

    The raw generation is a reasoning chain (prompt ends in ``<think>``) followed
    by ``<answer>…</answer>``. ``Item.__setattr__`` routes ``item.pred`` into
    ``output["pred"]``, so overwriting ``pred`` destroys the chain. Preserve the
    full chain in ``output["raw_pred"]`` so the IHR judge can score the reasoning
    steps even though EM/F1 score only the extracted answer.
    """
    import re
    for item in dataset:
        pred = getattr(item, "pred", "") or ""
        item.output["raw_pred"] = pred
        # Try <answer>...</answer> first
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", pred, re.DOTALL | re.IGNORECASE)
        if m:
            item.pred = m.group(1).strip()
            continue
        # Fallback: try to find "answer:" or "Answer:" line
        lines = [l.strip() for l in pred.split("\n") if l.strip()]
        for line in lines:
            if re.match(r"^(answer|final answer)\s*[:：]\s*", line, re.IGNORECASE):
                item.pred = re.sub(r"^(answer|final answer)\s*[:：]\s*", "", line, flags=re.IGNORECASE).strip()
                break
        else:
            # Last resort: last non-empty line
            item.pred = lines[-1] if lines else ""
    return dataset


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
    pipeline_kwargs: Dict[str, Any] = field(default_factory=dict)


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
        pipeline_class="IRCOTPipeline",
        pipeline_module="flashrag.pipeline.active_pipeline",
        generator_model="llama3-8B-instruct",
        is_reasoning=True,
        # IRCOT generates one sentence per iteration (stop=['.', '\n']) and needs
        # enough rounds to reach the "So the answer is:" marker on multi-hop
        # questions. The stock default (max_iter=2) yields 2 single sentences and
        # never reaches the answer, so we raise it.
        pipeline_kwargs={"max_iter": 6},
        # The runner passes pred_process_fun=None (overriding IRCOT's default),
        # which would score EM on the full reasoning chain. Force the parser so
        # only the text after "So the answer is[:]" is scored.
        extras={"pred_process_fun": ircot_pred_parse},
    ),
    BaselineSpec(
        name="r1_searcher",
        pipeline_class="SequentialPipeline",
        pipeline_module="flashrag.pipeline.pipeline",
        generator_model="r1-searcher",
        is_reasoning=False,
        system_prompt=(
            "You are a helpful assistant. Answer the question based on the provided reference passages. "
            "Put your final answer within <answer> </answer> tags."
        ),
        user_prompt="Reference passages:\n{reference}\n\nQuestion: {question}\n\n<think>",
        extras={
            "pred_process_fun": r1_extract_answer,
            "generation_params": {"max_tokens": 1024, "temperature": 0.0, "do_sample": False},
        },
    ),
    BaselineSpec(
        name="corag",
        pipeline_class="SequentialPipeline",
        pipeline_module="flashrag.pipeline.pipeline",
        generator_model="corag",
        is_reasoning=False,
        system_prompt="Answer the question based on the retrieved passages. Give only the answer.",
        user_prompt="Reference passages:\n{reference}\n\nQuestion: {question}\nAnswer:",
        extras={"pred_process_fun": corag_extract_answer},
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
    topk: int = DEFAULT_TOPK,
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
        topk=topk,
        split=split,
        test_sample_num=test_sample_num,
        seed=seed,
        gpu_id=gpu_id,
        is_reasoning=spec.is_reasoning,
    )
    # Merge extras, but deep-merge ``generation_params`` so a baseline that only
    # sets ``do_sample`` (e.g. rearag) does not clobber the config's ``max_tokens``
    # / ``temperature`` defaults. A missing ``max_tokens`` makes HF `generate()`
    # fall back to `max_length=20`, truncating the ~300-token ReaRAG prompt to
    # nothing and yielding "No valid answer found".
    for key, val in spec.extras.items():
        if key == "generation_params" and isinstance(val, dict):
            cfg.setdefault("generation_params", {}).update(val)
        else:
            cfg[key] = val
    return cfg
