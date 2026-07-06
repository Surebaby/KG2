#!/usr/bin/env python
"""Evaluate KG-ProWeight on D_std and D_dropout.

Records per-step α distributions and the heuristic IHR signal. The pipeline
itself honours ``metadata.dropout.modified_kg`` (bug-fix #5).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from kgproweight.config import load_config
from kgproweight.data.d_dropout_loader import load_dropout_dataset
from kgproweight.eval.pred_processing import kg_proweight_pred_process
from kgproweight.eval.runner import run_evaluation
from kgproweight.retrieval.hybrid import apply_retrieval_overrides, build_flashrag_config
from kgproweight.utils.logging import configure_logging, get_logger
from kgproweight.utils.paths import checkpoint_dir, data_dir, output_dir

configure_logging("INFO")
logger = get_logger(__name__)

DEFAULT_PIPELINE = (
    "kgproweight.pipeline.kg_proweight_pipeline",
    "KGProWeightPipeline",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="Optional YAML (e.g. configs/ablation/no_kg.yaml).")
    p.add_argument("--checkpoint", default=None, help="LoRA / PPO checkpoint dir.")
    p.add_argument("--alpha_gate_path", default=None)
    p.add_argument("--entity_cache_path", default=None)
    p.add_argument("--kg_cache_dir", default=None)
    p.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimultihopqa", "musique", "d_dropout"])
    p.add_argument("--split", default="dev")
    p.add_argument("--test_sample_num", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 2024])
    p.add_argument("--gpu_id", default="0")
    p.add_argument("--save_root", default=None)
    return p.parse_args()


def _resolve_paths(args) -> dict:
    # --checkpoint none|base|"" → evaluate the BARE base model (no LoRA adapter).
    # Lets the same script produce the base-vs-SFT-vs-PPO comparison.
    if args.checkpoint is not None and args.checkpoint.lower() in ("none", "base", ""):
        ckpt = None
    else:
        ckpt = args.checkpoint or str(Path(checkpoint_dir()) / "kg_proweight_final" / "final")
    alpha = args.alpha_gate_path or str(Path(checkpoint_dir()) / "prm_alpha_gate" / "alpha_gate.pt")
    return {"checkpoint": ckpt, "alpha_gate_path": alpha}


def _load_yaml_overrides(config_path: Optional[str]) -> Tuple[Dict[str, Any], str, str, bool]:
    """Return (retrieval_dict, pipeline_module, pipeline_class, record_alpha)."""
    if not config_path:
        module, cls = DEFAULT_PIPELINE
        return {}, module, cls, True

    doc = load_config(config_path)
    pipeline_cfg = doc.get("pipeline") or {}
    eval_cfg = doc.get("eval") or {}
    module = pipeline_cfg.get("module", DEFAULT_PIPELINE[0])
    cls = pipeline_cfg.get("class", DEFAULT_PIPELINE[1])
    record_alpha = bool(eval_cfg.get("use_real_alpha", True))
    return doc.get("retrieval") or {}, module, cls, record_alpha


def main():
    args = parse_args()
    save_root = Path(args.save_root) if args.save_root else Path(output_dir()) / "kg_proweight"
    paths = _resolve_paths(args)
    retrieval_overrides, pipeline_module, pipeline_class, record_alpha = _load_yaml_overrides(args.config)

    for ds in args.datasets:
        for seed in args.seeds:
            save_dir = str(save_root / ds / f"seed_{seed}")
            cfg = build_flashrag_config(
                dataset_name=ds,
                save_note="kg_proweight",
                save_dir=save_dir,
                method_name="kg_proweight",
                pipeline_class=pipeline_class,
                split=args.split,
                test_sample_num=args.test_sample_num,
                seed=seed,
                gpu_id=args.gpu_id,
                generator_lora_path=paths["checkpoint"],
            )
            cfg = apply_retrieval_overrides(cfg, retrieval_overrides)

            dropout_dataset = None
            if ds == "d_dropout":
                drop_path = Path(data_dir()) / "d_dropout" / f"{args.split}.jsonl"
                if not drop_path.exists():
                    logger.error("D_dropout file missing at %s — run scripts/prepare/05_build_d_dropout.py", drop_path)
                    continue
                dropout_dataset = load_dropout_dataset(drop_path).to_flashrag_dataset()

            pipeline_kwargs = {
                "alpha_gate_path": paths["alpha_gate_path"],
                "entity_cache_path": args.entity_cache_path,
                "kg_cache_dir": args.kg_cache_dir,
                "record_alpha": record_alpha,
            }

            def _after_run(pipe, save_dir=save_dir):  # noqa: ARG001
                if not record_alpha:
                    return
                try:
                    pipe.save_alpha_distribution(str(Path(save_dir) / "alpha_distribution.jsonl"))
                    pipe.print_alpha_summary()
                except Exception as exc:
                    logger.warning("Failed to dump alpha distribution: %s", exc)

            logger.info(
                "Running KG-ProWeight on %s (seed=%d, pipeline=%s, lora=%s) → %s",
                ds,
                seed,
                pipeline_class,
                paths["checkpoint"],
                save_dir,
            )
            try:
                run_evaluation(
                    flashrag_cfg=cfg,
                    pipeline_module=pipeline_module,
                    pipeline_class=pipeline_class,
                    pipeline_kwargs=pipeline_kwargs,
                    dropout_dataset=dropout_dataset,
                    seed=seed,
                    pred_process_fun=kg_proweight_pred_process,
                    after_run=_after_run,
                )
            except Exception as exc:
                logger.error("KG-ProWeight eval failed (%s seed=%d): %s", ds, seed, exc)


if __name__ == "__main__":
    main()
