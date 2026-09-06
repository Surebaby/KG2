#!/usr/bin/env bash
# Checklist §6: base-only vs SFT held-out val validation (n=200, seed=42, greedy).
# Replays the PPO-time RL prompt from stored passages — no retrieval needed.
#
#   bash launch_validate_sft_quota70.sh
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="/home/zjulab/kgpaper:/home/zjulab/kgpaper/flashrag_src"
export CUDA_VISIBLE_DEVICES="0"

VAL_DIR="outputs/validation/sft_quota70_hard_seed42_no_text_head_val_n200"
SILVER="checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42_no_text_head/silver_with_logprobs.jsonl"
ADAPTER="checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final"

mkdir -p "$VAL_DIR"
test ! -e "$VAL_DIR/base_seed42_n200.jsonl"
test ! -e "$VAL_DIR/sft_seed42_n200.jsonl"

echo "===== BASE-ONLY (val n=200) ====="
python scripts/eval/validate_sft.py \
  --base_only --silver "$SILVER" --split val --n 200 --seed 42 \
  --out "$VAL_DIR/base_seed42_n200.jsonl"

echo
echo "===== SFT (val n=200) ====="
python scripts/eval/validate_sft.py \
  --adapter "$ADAPTER" --silver "$SILVER" --split val --n 200 --seed 42 \
  --out "$VAL_DIR/sft_seed42_n200.jsonl"

echo
echo "ALL VAL DONE"
