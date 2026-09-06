#!/usr/bin/env bash
# Phase 3a SFT over the quota70 legacy-repaired silver (branch A output of Phase 2).
#
#   bash launch_sft_quota70_no_text_head.sh
set -euo pipefail
KGPW_ROOT=/home/zjulab/kgpaper
KGPW_PY=/home/zjulab/anaconda3/envs/kgpaper/bin/python
SFT_LOG="$KGPW_ROOT/logs/training/sft_quota70_hard_seed42_no_text_head.log"
cd "$KGPW_ROOT"
mkdir -p logs/training
test ! -e "$SFT_LOG"
test ! -e checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head

export PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src"
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

exec "$KGPW_PY" scripts/train/phase3_sft.py \
  --config configs/training/phase3_sft_legacy_repaired_v2_quota70.yaml \
  2>&1 | tee "$SFT_LOG"
