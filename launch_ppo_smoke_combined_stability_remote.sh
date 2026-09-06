#!/usr/bin/env bash
# One paid combination smoke: truncation + KL calibration + critic repair.
# This is intentionally multi-variable; do not report it as an ablation.
set -euo pipefail
KGPW_ROOT=/root/autodl-tmp/kgpaper
cd "$KGPW_ROOT"

export PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src"
export KGPW_TB_DIR=/root/tf-logs
export CUDA_VISIBLE_DEVICES=0
KGPW_PYTHON=/root/autodl-tmp/kgpw_env/bin/python

RUN_ID=ppo_legacy_repaired_v2_quota70_hard_seed42_no_text_head_smoke600_combined_stability_v1
LOG_PATH=logs/training/ppo_quota70_hard_seed42_no_text_head_smoke600_combined_stability_v1.log

mkdir -p logs/training
test ! -e "outputs/$RUN_ID"
test ! -e "$LOG_PATH"

# Abort before loading any 8B/9B model if deployment missed a forwarding,
# reference, critic, replay, or manifest regression.
"$KGPW_PYTHON" -m pytest -q \
  tests/test_ppo_diagnostics.py \
  tests/test_phase3_ppo_config_forwarding.py \
  tests/test_ppo_explicit_reference.py \
  tests/test_run_preflight_manifest.py \
  tests/test_ppo_sft_replay.py \
  tests/test_advantage_telemetry.py

exec "$KGPW_PYTHON" scripts/train/phase3_ppo.py \
  --config configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600_combined_stability_v1.yaml \
  2>&1 | tee "$LOG_PATH"
