"""One frozen, longest-input ReaRAG scoring probe on local CUDA/BF16."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.relative_to(ROOT / "outputs")
    output.mkdir(parents=True, exist_ok=False)
    input_path = ROOT / "outputs/audits/sourcegate_rearag_local_token_preflight_20260905_v1/longest_step.json"
    bank_path = ROOT / "outputs/audits/source_quality_candidate_bank_v1_inputs_seed42_tensorboard_v1/manifest.json"
    report = {
        "schema_version": "sourcegate-local-rearag-gpu-probe-v1",
        "experiment_id": "SOURCEGATE-REARAG-LOCAL-GPU-PROBE-20260905-V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "research_policy_updates": 0,
        "input_path": str(input_path.relative_to(ROOT)),
        "input_sha256": sha256(input_path),
        "bank_manifest_path": str(bank_path.relative_to(ROOT)),
        "bank_manifest_sha256": sha256(bank_path),
        "probe_code_sha256": sha256(Path(__file__)),
        "backend_code_sha256": sha256(ROOT / "kgproweight/reward/text_reward_model.py"),
        "code_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scoring_contract": "rearag-passage-only-raw-tanh-nll-v1",
        "device": "cuda:0",
        "dtype": "bf16",
        "precision_or_scoring_changes": False,
        "seed": 42,
        "environment": {key: os.environ.get(key) for key in ("HF_MODULES_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "package_versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "accelerate")},
    }

    def save() -> None:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    scorer = None
    torch = None
    started = time.perf_counter()
    try:
        from scripts.prepare.source_quality_candidate_bank_v1 import validate_model
        from kgproweight.reward.text_reward_model import RearagPromptScorer
        import math
        import torch

        torch.manual_seed(42)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Probe requires actual local CUDA with BF16 support")
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
        bank = json.loads(bank_path.read_text())
        sample = json.loads(input_path.read_text())
        model_path = ROOT / bank["rearag_model"]["path"]
        report["model_path"] = str(model_path.relative_to(ROOT))
        t0 = time.perf_counter()
        validate_model(model_path, bank["rearag_model"])
        report["model_validation_seconds"] = time.perf_counter() - t0
        report["verified_model_and_tokenizer_files"] = len(bank["rearag_model"]["files"])
        report["model_and_tokenizer_hashes_match"] = True
        save()
        print("Frozen model and tokenizer hashes verified; loading original ReaRAG in BF16", flush=True)
        torch.cuda.reset_peak_memory_stats(0)
        t0 = time.perf_counter()
        scorer = RearagPromptScorer.from_pretrained(str(model_path), device="cuda:0", dtype="bf16")
        torch.cuda.synchronize(0)
        report["load_seconds"] = time.perf_counter() - t0
        report["model_parameter_dtypes"] = sorted({str(parameter.dtype) for parameter in scorer.model.parameters()})
        report["model_training"] = scorer.model.training
        report["parameters_require_grad"] = any(parameter.requires_grad for parameter in scorer.model.parameters())
        report["max_length"] = scorer.max_length
        report["candidate_id"] = sample["candidate_id"]
        report["step"] = sample["step"]
        report["prompt_tokens"] = len(scorer.tokenizer(sample["prompt"], add_special_tokens=False)["input_ids"])
        report["step_tokens"] = len(scorer.tokenizer(sample["text"], add_special_tokens=False)["input_ids"])
        report["total_tokens"] = report["prompt_tokens"] + report["step_tokens"]
        report["truncated_tokens"] = max(0, report["total_tokens"] - scorer.max_length)
        if report["total_tokens"] != sample["total_tokens"] or report["truncated_tokens"]:
            raise ValueError("Probe input/tokenizer mismatch or unintended truncation")
        save()
        print(f"Loaded; scoring longest actual step ({report['total_tokens']} tokens)", flush=True)
        t0 = time.perf_counter()
        report["score"] = scorer.score_step(sample["prompt"], sample["text"])
        torch.cuda.synchronize(0)
        report["score_seconds"] = time.perf_counter() - t0
        if not math.isfinite(report["score"]) or not -1 <= report["score"] <= 1:
            raise ValueError("Non-finite or out-of-range ReaRAG score")
        report["score_finite_and_in_range"] = True
        report["status"] = "PASSED"
    except Exception as exc:
        report["status"] = "FAILED"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    finally:
        if torch is not None and torch.cuda.is_available():
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
            report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
            report["allocated_before_cleanup_bytes"] = torch.cuda.memory_allocated(0)
        del scorer
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            report["allocated_after_cleanup_bytes"] = torch.cuda.memory_allocated(0)
        report["elapsed_seconds"] = time.perf_counter() - started
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
