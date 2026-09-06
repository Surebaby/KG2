"""Evaluate hard/soft alpha gates on the same deterministic held-out fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.data.silver_split import SplitSpec
from kgproweight.kg.coverage import graph_density
from kgproweight.kg.entity_linker import EntityLinker
from kgproweight.reward.alpha_gate import AlphaGate, entropy_from_logprobs
from kgproweight.retrieval.bootstrap import resolve_entity_cache_path
from kgproweight.training.phase2_prm import (
    _alpha_calibration_target,
    _build_samples_accepted_only,
)


def _metrics(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    prob = torch.sigmoid(logits)
    residual = prob - target
    target_var = target.var(unbiased=False)
    r2 = float("nan")
    if target_var.item() > 0:
        r2 = float((1.0 - residual.square().mean() / target_var).item())
    return {
        "bce": float(F.binary_cross_entropy_with_logits(logits, target).item()),
        "brier": float(residual.square().mean().item()),
        "mae": float(residual.abs().mean().item()),
        "r2_vs_constant": r2,
        "alpha_mean": float(prob.mean().item()),
        "alpha_std": float(prob.std(unbiased=False).item()),
        "target_mean": float(target.mean().item()),
    }


def _calibration_bins(logits: torch.Tensor, target: torch.Tensor) -> List[Dict[str, float]]:
    prob = torch.sigmoid(logits)
    bins = []
    for i in range(10):
        lo, hi = i / 10.0, (i + 1) / 10.0
        mask = (prob >= lo) & (prob < hi if i < 9 else prob <= hi)
        if mask.any():
            bins.append({
                "lo": lo,
                "hi": hi,
                "n": int(mask.sum().item()),
                "alpha_mean": float(prob[mask].mean().item()),
                "target_mean": float(target[mask].mean().item()),
            })
    return bins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", required=True, help="Enriched silver with token_logprobs.")
    parser.add_argument(
        "--gate", action="append", required=True,
        help="Named gate in NAME=PATH form; pass once for each ablation arm.",
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    spec = SplitSpec(
        val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.split_seed
    )
    reader = SilverDatasetReader(args.silver, split=args.split, split_spec=spec)
    accepted = reader.accepted()
    linker = EntityLinker(cache_path=resolve_entity_cache_path(), offline=True)
    prov = _build_samples_accepted_only(reader, entity_linker=linker, accepted=accepted)
    if not prov:
        raise SystemExit(f"No accepted step samples in split={args.split!r}")

    samples = [p.sample for p in prov]
    for p in prov:
        stored = accepted[p.traj_idx].steps[p.step_idx].token_logprobs
        p.sample.semantic_entropy = entropy_from_logprobs(stored)

    density = torch.tensor([graph_density(s.kg_subgraph) for s in samples])
    confidence = torch.tensor([s.coverage for s in samples])
    entropy = torch.tensor([s.semantic_entropy for s in samples])
    cite_any = torch.tensor([s.cite_any for s in samples])
    cite_match = torch.tensor([s.cite_match for s in samples])
    labels_rkg = torch.tensor([s.label for s in samples])
    labels_class = torch.tensor([s.label_class for s in samples])
    targets = {
        name: _alpha_calibration_target(labels_class, labels_rkg, name)
        for name in ("hard_verdict", "soft_abs_rkg")
    }

    report = {
        "silver": args.silver,
        "split": args.split,
        "split_spec": {
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.split_seed,
        },
        "n_trajectories": len(accepted),
        "n_steps": len(samples),
        "gates": {},
    }
    for item in args.gate:
        if "=" not in item:
            raise SystemExit(f"--gate must be NAME=PATH, got {item!r}")
        name, raw_path = item.split("=", 1)
        gate = AlphaGate()
        gate.load_state_dict(torch.load(raw_path, map_location="cpu"))
        gate.eval()
        with torch.no_grad():
            logits = gate.forward_logits(
                density, confidence, entropy, cite_any, cite_match
            )
        report["gates"][name] = {
            "path": raw_path,
            "weights": gate.W.detach().tolist(),
            "bias": float(gate.b.detach().item()),
            "tau": float(gate.tau.detach().item()),
            "against_targets": {
                target_name: {
                    **_metrics(logits, target),
                    "calibration_bins": _calibration_bins(logits, target),
                }
                for target_name, target in targets.items()
            },
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
