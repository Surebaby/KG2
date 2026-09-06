#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="configs/training/query_controller_action_v1_probe20_seed42_v4_3.yaml"
OUTPUT="outputs/probes/query_controller_action_v1_probe20_seed42_v4_3"
LOG="logs/training/query_controller_action_v1_probe20_seed42_v4_3.log"

cd "$PROJECT_ROOT"
test ! -e "$OUTPUT"
test ! -e "$LOG"
test -s "$CONFIG"
test -s data/silver_data/query_controller_actions_v1_seed42_v4_3/train.jsonl
test -s data/silver_data/query_controller_actions_v1_seed42_v4_3/dev.jsonl
mkdir -p logs/training

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/flashrag_src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" scripts/train/query_controller.py \
  --config "$CONFIG" \
  --dry_run

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$PYTHON_BIN" scripts/train/query_controller.py \
    --config "$CONFIG" \
    --probe \
    2>&1 | tee "$LOG"
