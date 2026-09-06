#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/kgpaper
export PYTHONPATH=/root/autodl-tmp/kgpaper:/root/autodl-tmp/kgpaper/flashrag_src
export CUDA_VISIBLE_DEVICES=0
export KGPW_TB_DIR=/root/tf-logs/PROOFKG-2WIKI-HARD-V1-PPO-O-1200-SEED42

CONFIG=configs/training/phase3_ppo_2wiki_hard_curriculum_v1_outcome1200_seed42.yaml
LOCK=${CONFIG}.lock.v2.json
OUT=outputs/ppo_2wiki_hard_curriculum_v1_outcome1200_seed42
LOG=logs/training/ppo_2wiki_hard_curriculum_v1_outcome1200_seed42.log
PREFLIGHT=logs/training/ppo_2wiki_hard_curriculum_v1_outcome1200_seed42.preflight.json

test -s "$CONFIG"
test -s "$LOCK"
test -s data/silver_data/automatic_proofkg_2wiki_hard_contrastive_v1/manifest.json
test ! -e "$OUT"
test ! -e "$KGPW_TB_DIR"
test ! -e "$LOG"
test ! -e "$PREFLIGHT"
mkdir -p logs/training "$KGPW_TB_DIR"

/root/autodl-tmp/kgpw_env/bin/python scripts/prepare/preflight_2wiki_hard_curriculum_ppo.py \
  --config "$CONFIG" --lock "$LOCK" --run_tests --report_path "$PREFLIGHT"

nohup /root/autodl-tmp/kgpw_env/bin/python scripts/train/phase3_ppo.py \
  --config "$CONFIG" >"$LOG" 2>&1 &
echo $!
echo "log=$LOG"
echo "tensorboard=$KGPW_TB_DIR"
