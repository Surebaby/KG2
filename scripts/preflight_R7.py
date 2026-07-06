#!/usr/bin/env python
"""R7 pre-flight check — run this before starting training.

Usage: python scripts/preflight_R7.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKS: list[tuple[str, str, str]] = [
    # (name, relative_path, check_type)
    ("Elite SFT adapter", "checkpoints/sft_student_elite/final/adapter_config.json", "file"),
    ("Alpha-gate weights", "checkpoints/prm_alpha_gate/alpha_gate.pt", "file"),
    ("PPO silver data", "checkpoints/prm_alpha_gate/silver_with_logprobs.jsonl", "file"),
    ("Silver trajectories", "data/silver_data/silver_trajectories.jsonl", "file"),
    ("PPO config YAML", "configs/training/phase3_ppo.yaml", "file"),
]

IMPORT_CHECKS = [
    "torch",
    "transformers",
    "trl",
    "peft",
    "tensorboard",
]

FAIL = 0


def check_file(path: Path) -> bool:
    if not path.exists():
        print(f"  ❌ MISSING: {path}")
        return False
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  ✅ {path} ({size_mb:.1f} MB)")
    return True


def check_import(name: str) -> bool:
    try:
        __import__(name)
        print(f"  ✅ {name}")
        return True
    except ImportError:
        print(f"  ❌ NOT INSTALLED: {name}")
        return False


def check_no_step_format_bonus(path: Path) -> bool:
    """Ensure no executable code references the removed step_format_bonus."""
    if not path.exists():
        return True
    content = path.read_text(encoding="utf-8")
    lines = content.split("\n")
    issues = []
    for i, line in enumerate(lines, 1):
        if "step_format_bonus" in line and not line.strip().startswith("#"):
            # Allow comment-only references
            stripped = line.split("#")[0] if "#" in line else line
            if "step_format_bonus" in stripped:
                issues.append(i)
    if issues:
        print(f"  ⚠️  step_format_bonus referenced at lines: {issues}")
        return False
    print(f"  ✅ No active step_format_bonus references")
    return True


def check_config_yaml(path: Path) -> bool:
    """Quick sanity check on the R7 config."""
    if not path.exists():
        print(f"  ❌ Config not found: {path}")
        return False
    content = path.read_text(encoding="utf-8")
    ok = True
    if "step_format_bonus" in content:
        print("  ⚠️  Config still contains step_format_bonus!")
        ok = False
    if "min_valid_steps" not in content:
        print("  ⚠️  Config missing min_valid_steps (R7 field)!")
        ok = False
    if "sft_anchor_weight" not in content:
        print("  ⚠️  Config missing sft_anchor_weight (R7 field)!")
        ok = False
    if ok:
        print("  ✅ Config has R7 fields, no step_format_bonus")
    return ok


def main():
    global FAIL
    print("=" * 60)
    print("R7 Pre-flight Check")
    print("=" * 60)

    # 1. File checks
    print("\n📁 Required files:")
    for name, rel_path, _ in CHECKS:
        print(f"  {name}:")
        if not check_file(PROJECT_ROOT / rel_path):
            FAIL += 1

    # 2. Import checks
    print("\n📦 Python packages:")
    for name in IMPORT_CHECKS:
        if not check_import(name):
            FAIL += 1

    # 3. Code checks
    print("\n🔍 R7 code integrity:")
    print("  reward_function.py:")
    check_no_step_format_bonus(
        PROJECT_ROOT / "kgproweight" / "training" / "reward_function.py"
    )
    print("  phase3_ppo.py:")
    check_no_step_format_bonus(
        PROJECT_ROOT / "kgproweight" / "training" / "phase3_ppo.py"
    )
    print("  composite_reward.py:")
    check_no_step_format_bonus(
        PROJECT_ROOT / "kgproweight" / "reward" / "composite_reward.py"
    )
    print("  schemas.py:")
    check_no_step_format_bonus(
        PROJECT_ROOT / "kgproweight" / "config" / "schemas.py"
    )
    print("  scripts/train/phase3_ppo.py:")
    check_no_step_format_bonus(
        PROJECT_ROOT / "scripts" / "train" / "phase3_ppo.py"
    )
    print("  configs/training/phase3_ppo.yaml:")
    check_config_yaml(PROJECT_ROOT / "configs" / "training" / "phase3_ppo.yaml")

    # 4. GPU
    print("\n🖥️  GPU:")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"  ✅ {gpu_name} ({gpu_mem:.0f} GB)")
        else:
            print("  ❌ CUDA not available!")
            FAIL += 1
    except Exception:
        print("  ⚠️  Could not check GPU (torch not installed?)")

    # 5. Summary
    print("\n" + "=" * 60)
    if FAIL == 0:
        print("✅ ALL CHECKS PASSED — ready for R7 training!")
        print()
        print("Launch command:")
        print("  cd ~/kgpaper")
        print("  nohup python scripts/train/phase3_ppo.py \\")
        print("    --config configs/training/phase3_ppo.yaml \\")
        print("    --sft_checkpoint checkpoints/sft_student_elite/final \\")
        print("    --alpha_gate_path checkpoints/prm_alpha_gate/alpha_gate.pt \\")
        print("    --silver_data checkpoints/prm_alpha_gate/silver_with_logprobs.jsonl \\")
        print("    --output_dir checkpoints/kg_proweight_R7A \\")
        print("    --seed 42 \\")
        print("    > logs/R7A_train.log 2>&1 &")
    else:
        print(f"❌ {FAIL} CHECKS FAILED — fix before training!")
    print("=" * 60)
    return FAIL


if __name__ == "__main__":
    sys.exit(main())
