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

# Loaded when --config is omitted so the documented command actually runs the
# paper's retrieval setting instead of the bare DEFAULT_TOPK=15 fallback.
DEFAULT_EVAL_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "eval" / "kg_proweight.yaml"


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
    p.add_argument("--no_kg", action="store_true", help="Disable KG injection (alpha=0 ablation)")
    p.add_argument("--rerank", type=int, default=None,
                   help="Rerank top-K after RRF. Default: follow config's rerank_topk "
                        "(hybrid_rrf_top50.yaml -> 10). 0=disabled, N=override.")
    p.add_argument("--alpha_bias_correction", type=float, default=None,
                   help="AlphaGate inference bias additive correction. "
                        "Default (None) = 0.0, i.e. NO correction (changed "
                        "2026-08-23; +0.78 was reverse-engineered to make b_eff a "
                        "round -1.0 and the mechanism it claims allows at most "
                        "+0.363). Pass 0.78 to reproduce pre-2026-08-23 runs.")
    p.add_argument("--kg_supply_mode", default=argparse.SUPPRESS,
                   choices=["legacy", "qpeg_v1", "proofkg_v1", "legacy_plus_proofkg", "legacy_plus_complete_proofkg"],
                   help="KG supply for inference. Default follows the eval config's "
                        "eval.kg_supply_mode (legacy). 'proofkg_v1' reads a versioned "
                        "per-question ProofKG record via --question_kg_records_path; "
                        "'qpeg_v1' reads versioned passage-derived QPEG records; "
                        "'legacy_plus_proofkg' / 'legacy_plus_complete_proofkg' augment "
                        "legacy with selective trusted ProofKG edges (fail-closed).")
    p.add_argument("--question_kg_records_path", default=argparse.SUPPRESS,
                   help="Path to question_kg_records.jsonl (required when "
                        "--kg_supply_mode=proofkg_v1).")
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
    """Return (retrieval_dict, pipeline_module, pipeline_class, record_alpha).

    With no ``--config`` this used to return ``{}``, so the run silently used
    ``DEFAULT_TOPK=15`` — including the README's own command — while
    ``configs/retrieval/hybrid_rrf_top50.yaml`` sat unused. We now load the
    canonical eval config by default so train and eval share one retrieval
    setting; pass ``--config`` explicitly to override (e.g. an ablation).
    """
    if not config_path:
        config_path = str(DEFAULT_EVAL_CONFIG)
        if not Path(config_path).exists():
            logger.warning("Default eval config %s missing — falling back to built-in defaults.",
                           config_path)
            module, cls = DEFAULT_PIPELINE
            return {}, module, cls, True
        logger.info("No --config given; using default eval config %s", config_path)

    doc = load_config(config_path)
    pipeline_cfg = doc.get("pipeline") or {}
    eval_cfg = doc.get("eval") or {}
    module = pipeline_cfg.get("module", DEFAULT_PIPELINE[0])
    cls = pipeline_cfg.get("class", DEFAULT_PIPELINE[1])
    record_alpha = bool(eval_cfg.get("use_real_alpha", True))
    kg_supply_mode = eval_cfg.get("kg_supply_mode", "legacy")
    question_kg_records_path = eval_cfg.get("question_kg_records_path")
    return doc.get("retrieval") or {}, module, cls, record_alpha, kg_supply_mode, question_kg_records_path


def main():
    args = parse_args()
    save_root = Path(args.save_root) if args.save_root else Path(output_dir()) / "kg_proweight"
    paths = _resolve_paths(args)
    retrieval_overrides, pipeline_module, pipeline_class, record_alpha, yaml_kg_supply_mode, yaml_qkg_records_path = _load_yaml_overrides(args.config)
    # CLI explicitly-passes override the YAML eval config; otherwise fall back to
    # the eval config value (so a materialised ProofKG-v1 template config runs
    # without needing --kg_supply_mode on the command line).
    kg_supply_mode = getattr(args, "kg_supply_mode", yaml_kg_supply_mode)
    question_kg_records_path = getattr(args, "question_kg_records_path", yaml_qkg_records_path)

    # R9 v6 fix: two-stage reranking was silently disabled — the config's
    # ``rerank_topk`` (10) never reached the pipeline because ``--rerank``
    # defaulted to 0, and without rerank the 50-passage prompt overran 6144
    # tokens and right-truncation dropped the KG block + assistant anchor.
    # Now: no flag → follow the config; ``--rerank N`` overrides; ``--rerank 0`` disables.
    rerank = args.rerank if args.rerank is not None else int(retrieval_overrides.get("rerank_topk", 0) or 0)

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
                "alpha_bias_correction": args.alpha_bias_correction,
                "kg_supply_mode": kg_supply_mode,
                "question_kg_records_path": question_kg_records_path,
            }
            if args.no_kg:
                pipeline_kwargs["inject_kg"] = False
            if rerank > 0:
                # Report the ACTUAL configured pool, not a hardcoded "50": with
                # no --config the retriever returned 15 docs while this line
                # claimed RRF@50, so logs disagreed with what ran.
                pool = int(cfg.get("retrieval_topk") or 0)
                per_retriever = [
                    r.get("retrieval_topk")
                    for r in cfg.get("multi_retriever_setting", {}).get("retriever_list", [])
                ]
                if pool < rerank:
                    logger.warning(
                        "rerank_topk=%d >= candidate pool=%d — reranking is a no-op. "
                        "Raise retrieval.retrieval_topk in the config.",
                        rerank, pool,
                    )
                pipeline_kwargs["rerank_topk"] = rerank
                pipeline_kwargs["retrieval_topk"] = pool
                pipeline_kwargs["rerank_method"] = retrieval_overrides.get(
                    "rerank_method", "cross-encoder")
                pipeline_kwargs["cross_encoder_model"] = retrieval_overrides.get(
                    "cross_encoder_model", "models/bge-reranker-v2-m3")
                logger.info(
                    "Two-stage retrieval: per-retriever@%s → RRF@%d → %s@%d → prompt<=%dtok",
                    per_retriever, pool,
                    retrieval_overrides.get("rerank_method", "cross-encoder"),
                    rerank, 3860,
                )

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
