#!/usr/bin/env bash
# Phase 2 (PRM head + 5-feature alpha gate) over the quota70 legacy-repaired
# silver, branch A: --no_text_head (PPO uses ReaRAG, the fallback text head is
# not loaded, so we don't train it). This is the D16=A decision.
#
#   bash launch_phase2_quota70_no_text_head.sh
set -euo pipefail
KGPW_ROOT=/home/zjulab/kgpaper
KGPW_PY=/home/zjulab/anaconda3/envs/kgpaper/bin/python
PHASE2_LOG="$KGPW_ROOT/logs/training/phase2_quota70_hard_seed42_no_text_head.log"
cd "$KGPW_ROOT"
mkdir -p logs/training
test ! -e "$PHASE2_LOG"
test ! -e checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42_no_text_head

export PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src"
export CUDA_VISIBLE_DEVICES="0"

exec "$KGPW_PY" scripts/train/phase2_train_prm.py \
  --config configs/training/phase2_prm_legacy_repaired_v2_quota70.yaml \
  --no_text_head \
  2>&1 | tee "$PHASE2_LOG"
