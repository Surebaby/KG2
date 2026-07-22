"""Generic FlashRAG evaluation runner.

The scripts in ``scripts/eval/*.py`` are thin CLI wrappers around
:func:`run_evaluation`, which handles config validation, FlashRAG
bootstrap, dataset loading, pipeline instantiation, and the
``manifest.json`` write-out.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from kgproweight.data.flashrag_loader import flashrag_config, get_dataset
from kgproweight.utils.flashrag_bootstrap import setup_flashrag
from kgproweight.utils.logging import dump_manifest, get_logger
from kgproweight.utils.seed import set_seed

logger = get_logger(__name__)

RunMode = Literal["standard", "naive"]


def parse_metric_score_txt(path: Path) -> Dict[str, Any]:
    """Parse FlashRAG ``metric_score.txt`` into a JSON-serialisable dict."""
    metrics: Dict[str, Any] = {}
    if not path.exists():
        return metrics
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        try:
            metrics[key] = float(value)
        except ValueError:
            metrics[key] = value
    return metrics


def export_metric_json(run_dir: Path) -> Optional[Path]:
    """Write ``metric_score.json`` next to FlashRAG's ``metric_score.txt``."""
    txt_path = run_dir / "metric_score.txt"
    if not txt_path.exists():
        return None
    metrics = parse_metric_score_txt(txt_path)
    if not metrics:
        return None
    json_path = run_dir / "metric_score.json"
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path


def _import_class(module_path: str, class_name: str):
    setup_flashrag()
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _build_pipeline(
    PipelineCls,
    cfg,
    pipeline_kwargs: Optional[Dict[str, Any]],
    *,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
):
    kwargs = dict(pipeline_kwargs or {})
    if system_prompt and user_prompt:
        from flashrag.prompt import PromptTemplate

        tmpl = PromptTemplate(cfg, system_prompt=system_prompt, user_prompt=user_prompt)
        kwargs.setdefault("prompt_template", tmpl)
    return PipelineCls(cfg, **kwargs)


def run_evaluation(
    flashrag_cfg: Dict[str, Any],
    pipeline_module: str,
    pipeline_class: str,
    pipeline_kwargs: Optional[Dict[str, Any]] = None,
    dropout_dataset: Optional[List[Dict[str, Any]]] = None,
    seed: int = 42,
    run_mode: RunMode = "standard",
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    pred_process_fun: Optional[Callable] = None,
    after_run: Optional[Callable[[Any], None]] = None,
) -> Dict[str, Any]:
    """Run one evaluation pass.

    Parameters
    ----------
    flashrag_cfg:
        Resolved FlashRAG config dict (see :mod:`kgproweight.retrieval.hybrid`).
    pipeline_module, pipeline_class:
        Where to import the pipeline class.
    pipeline_kwargs:
        Keyword arguments passed to the pipeline constructor.
    dropout_dataset:
        Optional list of dicts (e.g., from :class:`DropoutDataset.to_flashrag_dataset`).
        When provided the FlashRAG dataset is wholesale replaced by this list.
    run_mode:
        ``"standard"`` calls ``pipeline.run()``; ``"naive"`` skips retrieval.
    system_prompt, user_prompt:
        Optional FlashRAG prompt overrides (used by zero-shot / naive RAG baselines).
    after_run:
        Optional callback invoked with the FlashRAG pipeline instance
        after ``run()`` returns (e.g., to dump α distributions).
    """
    set_seed(seed)
    cfg = flashrag_config(flashrag_cfg)
    PipelineCls = _import_class(pipeline_module, pipeline_class)

    split = flashrag_cfg.get("split", ["dev"])[0] if isinstance(flashrag_cfg.get("split"), list) else flashrag_cfg.get("split", "dev")

    if dropout_dataset is not None:
        from flashrag.dataset.dataset import Dataset as FlashRAGDataset

        ds = FlashRAGDataset(cfg, sample_num=len(dropout_dataset))
        ds.data = list(dropout_dataset)
    else:
        ds = get_dataset(cfg, split=split)

    pipeline = _build_pipeline(
        PipelineCls,
        cfg,
        pipeline_kwargs,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    if run_mode == "naive":
        result = pipeline.naive_run(ds, do_eval=True, pred_process_fun=pred_process_fun)
    else:
        result = pipeline.run(ds, do_eval=True, pred_process_fun=pred_process_fun)

    if after_run is not None:
        try:
            after_run(pipeline)
        except Exception as exc:
            logger.warning("after_run callback failed: %s", exc)

    final_dir: Optional[Path] = None
    save_dir = cfg["save_dir"] if "save_dir" in cfg else None
    if save_dir:
        final_dir = Path(save_dir)
        export_metric_json(final_dir)

    if final_dir is not None and final_dir.exists():
        dump_manifest(
            final_dir,
            extra={
                "phase": "eval",
                "pipeline_class": pipeline_class,
                "dataset": flashrag_cfg.get("dataset_name"),
                "split": flashrag_cfg.get("split"),
                "seed": seed,
                "run_mode": run_mode,
            },
        )

    return {
        "result": result,
        "save_dir": str(final_dir) if final_dir else None,
    }
