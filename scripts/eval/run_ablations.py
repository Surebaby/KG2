#!/usr/bin/env python
"""Paper §7 ablations.

Fixes bug #11: instead of monkey-patching α at inference time, the
α=0 / α=0.5 / α=1 / binary_labels variants each launch a short PPO
re-training run with the override. The single_retriever variant simply
swaps the retrieval config but reuses the existing checkpoint.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from kgproweight.utils.logging import configure_logging, get_logger
from kgproweight.utils.paths import checkpoint_dir, output_dir

configure_logging("INFO")
logger = get_logger(__name__)


# The variants that require *retraining*; map each to PPO override args.
_RETRAIN_VARIANTS: Dict[str, Dict[str, object]] = {
    "alpha_zero": {"alpha_override": 0.0, "binary_labels_only": False, "phase2": False},
    "alpha_one": {"alpha_override": 1.0, "binary_labels_only": False, "phase2": False},
    "alpha_half": {"alpha_override": 0.5, "binary_labels_only": False, "phase2": False},
    "binary_labels": {"alpha_override": None, "binary_labels_only": False, "phase2": True},
}

# Inference-only ablations (no retraining required).
_INFERENCE_VARIANTS: List[str] = ["single_retriever", "no_kg", "e5_only"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", nargs="+", default=list(_RETRAIN_VARIANTS.keys()) + _INFERENCE_VARIANTS)
    p.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimultihopqa", "musique", "d_dropout"])
    p.add_argument("--split", default="dev")
    p.add_argument("--total_steps", type=int, default=1000, help="Mini-PPO budget for retrain variants.")
    p.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 2024])
    p.add_argument("--gpu_id", default="0")
    p.add_argument("--save_root", default=None)
    p.add_argument("--checkpoint", default=None, help="Main PPO checkpoint for inference-only ablations.")
    p.add_argument("--python", default=sys.executable, help="Python binary used for PPO sub-process.")
    return p.parse_args()


def _run_eval(args, variant: str, checkpoint: Path, alpha_gate: Path | None = None) -> None:
    eval_cmd = [
        args.python,
        "scripts/eval/run_kg_proweight.py",
        "--checkpoint",
        str(checkpoint),
        "--datasets",
        *args.datasets,
        "--split",
        args.split,
        "--save_root",
        str(Path(args.save_root or output_dir() / "ablations") / variant / "eval"),
        "--gpu_id",
        args.gpu_id,
        "--seeds",
        *[str(s) for s in args.seeds],
    ]
    if alpha_gate is not None:
        eval_cmd += ["--alpha_gate_path", str(alpha_gate)]
    logger.info("Eval ablation %s: %s", variant, eval_cmd)
    subprocess.run(eval_cmd, check=True)


def main():
    args = parse_args()
    save_root = Path(args.save_root) if args.save_root else Path(output_dir()) / "ablations"
    save_root.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    main_ckpt = Path(
        args.checkpoint or str(Path(checkpoint_dir()) / "kg_proweight_final" / "final")
    )

    for variant in args.variants:
        if variant in _RETRAIN_VARIANTS:
            ovr = _RETRAIN_VARIANTS[variant]
            variant_root = save_root / variant
            ppo_out = variant_root / "checkpoint"
            alpha_gate_path = Path(checkpoint_dir()) / "prm_alpha_gate" / "alpha_gate.pt"
            silver_path = Path(checkpoint_dir()) / "prm_alpha_gate" / "silver_with_logprobs.jsonl"

            if ovr.get("phase2"):
                phase2_out = variant_root / "phase2"
                phase2_cmd = [
                    args.python,
                    "scripts/train/phase2_train_prm.py",
                    "--output_dir",
                    str(phase2_out),
                    "--binary_labels_only",
                ]
                logger.info("Phase 2 retrain for %s: %s", variant, phase2_cmd)
                try:
                    subprocess.run(phase2_cmd, check=True)
                except subprocess.CalledProcessError as exc:
                    logger.error("Phase 2 retrain of %s failed: %s", variant, exc)
                    summary[variant] = {"error": str(exc)}
                    continue
                alpha_gate_path = phase2_out / "alpha_gate.pt"
                silver_path = phase2_out / "silver_with_logprobs.jsonl"

            cmd = [
                args.python,
                "scripts/train/phase3_ppo.py",
                "--output_dir",
                str(ppo_out),
                "--total_steps",
                str(args.total_steps),
                "--silver_data",
                str(silver_path),
                "--alpha_gate_path",
                str(alpha_gate_path),
            ]
            if ovr["alpha_override"] is not None:
                cmd += ["--alpha_override", str(ovr["alpha_override"])]
            logger.info("Retrain ablation %s: %s", variant, cmd)
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                logger.error("Retrain of %s failed: %s", variant, exc)
                summary[variant] = {"error": str(exc)}
                continue

            try:
                _run_eval(args, variant, ppo_out / "final", alpha_gate=alpha_gate_path)
                summary[variant] = {"save_dir": str(variant_root)}
            except subprocess.CalledProcessError as exc:
                logger.error("Eval of %s failed: %s", variant, exc)
                summary[variant] = {"error": str(exc)}

        elif variant in _INFERENCE_VARIANTS:
            config_path = f"configs/ablation/{variant}.yaml"
            eval_cmd = [
                args.python,
                "scripts/eval/run_kg_proweight.py",
                "--config",
                config_path,
                "--checkpoint",
                str(main_ckpt),
                "--datasets",
                *args.datasets,
                "--split",
                args.split,
                "--save_root",
                str(save_root / variant),
                "--gpu_id",
                args.gpu_id,
                "--seeds",
                *[str(s) for s in args.seeds],
            ]
            logger.info("Inference ablation %s: %s", variant, eval_cmd)
            try:
                subprocess.run(eval_cmd, check=True)
                summary[variant] = {"save_dir": str(save_root / variant)}
            except subprocess.CalledProcessError as exc:
                logger.error("Eval of %s failed: %s", variant, exc)
                summary[variant] = {"error": str(exc)}
        else:
            logger.warning("Unknown variant %s", variant)

    summary_path = save_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Ablation summary → %s", summary_path)


if __name__ == "__main__":
    main()
