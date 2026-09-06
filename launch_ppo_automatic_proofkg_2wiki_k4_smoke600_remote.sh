#!/usr/bin/env bash
# Combination smoke: automatic ProofKG reward + dynamic validity + same-q K=4.
set -euo pipefail
KGPW_ROOT=/root/autodl-tmp/kgpaper
cd "$KGPW_ROOT"

export PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src"
export KGPW_TB_DIR=/root/tf-logs
export CUDA_VISIBLE_DEVICES=0
KGPW_PYTHON=/root/autodl-tmp/kgpw_env/bin/python
CONFIG=configs/training/phase3_ppo_automatic_proofkg_2wiki_k4_smoke600_seed42.yaml
CONFIG_LOCK=${CONFIG}.lock.json
LOG_PATH=logs/training/ppo_automatic_proofkg_2wiki_k4_smoke600_seed42.log
PREFLIGHT_PATH=logs/training/ppo_automatic_proofkg_2wiki_k4_smoke600_seed42.preflight.json
RUN_DIR=outputs/ppo_automatic_proofkg_2wiki_k4_smoke600_seed42

mkdir -p logs/training
test -s "$CONFIG"
test -s "$CONFIG_LOCK"
test -s data/silver_data/automatic_proofkg_2wiki_train_k4_v1/silver_train.jsonl
test -s data/silver_data/automatic_proofkg_2wiki_train_k4_v1/question_kg_records.jsonl
test -d checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final
test -s checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42_no_text_head/alpha_gate.pt
test ! -e "$RUN_DIR"
test ! -e "$LOG_PATH"
test ! -e "$PREFLIGHT_PATH"

GPU_FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | head -1 | tr -d ' ')
[[ "$GPU_FREE_MIB" =~ ^[0-9]+$ ]] && (( GPU_FREE_MIB >= 80000 ))
DISK_FREE_KIB=$(df -Pk "$KGPW_ROOT" | awk 'NR == 2 {print $4}')
[[ "$DISK_FREE_KIB" =~ ^[0-9]+$ ]] && (( DISK_FREE_KIB >= 20971520 ))

"$KGPW_PYTHON" scripts/prepare/preflight_automatic_proofkg_ppo.py \
  --config "$CONFIG" \
  --lock "$CONFIG_LOCK" \
  --run_tests \
  --report_path "$PREFLIGHT_PATH"

exec "$KGPW_PYTHON" scripts/train/phase3_ppo.py --config "$CONFIG" \
  2>&1 | tee "$LOG_PATH"
