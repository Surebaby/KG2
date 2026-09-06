#!/usr/bin/env bash
# PPO smoke (600 trajectories / 150 updates / 10% replay) on the remote 96GB box.
# TensorBoard events go to /root/tf-logs/ (AutoDL built-in TB reads that path).
#
#   bash launch_ppo_smoke_quota70_remote.sh
set -euo pipefail
KGPW_ROOT=/root/autodl-tmp/kgpaper
cd "$KGPW_ROOT"

export PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src"
export KGPW_TB_DIR=/root/tf-logs
export CUDA_VISIBLE_DEVICES=0

mkdir -p logs/training
test ! -e outputs/ppo_legacy_repaired_v2_quota70_hard_seed42_no_text_head_smoke600_replay10

exec /root/autodl-tmp/kgpw_env/bin/python scripts/train/phase3_ppo.py \
  --config configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600.yaml \
  2>&1 | tee logs/training/ppo_quota70_hard_seed42_no_text_head_smoke600.log
