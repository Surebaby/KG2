#!/usr/bin/env bash
# PPO correctness smoke: force the explicit frozen SFT reference under PEFT.
# Research knobs remain at the ce010 baseline: kl_coef=0.15, replay=10%, CE=0.10.
set -euo pipefail
KGPW_ROOT=/root/autodl-tmp/kgpaper
cd "$KGPW_ROOT"

export PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src"
export KGPW_TB_DIR=/root/tf-logs
export CUDA_VISIBLE_DEVICES=0

mkdir -p logs/training
test ! -e outputs/ppo_legacy_repaired_v2_quota70_hard_seed42_no_text_head_smoke600_explicit_sft_ref
test ! -e logs/training/ppo_quota70_hard_seed42_no_text_head_smoke600_explicit_sft_ref.log

exec /root/autodl-tmp/kgpw_env/bin/python scripts/train/phase3_ppo.py \
  --config configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600_explicit_sft_ref.yaml \
  2>&1 | tee logs/training/ppo_quota70_hard_seed42_no_text_head_smoke600_explicit_sft_ref.log
