#!/usr/bin/env bash
# Run the approved hard-vs-soft alpha target ablation. This is GPU training and
# must be launched only after the regenerated silver pilot/full quality gate.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

SILVER=${1:?usage: run_phase2_alpha_ablation.sh NEW_SILVER.jsonl EXPERIMENT_ID}
EXP_ID=${2:?usage: run_phase2_alpha_ablation.sh NEW_SILVER.jsonl EXPERIMENT_ID}
PYTHON_BIN=${PYTHON_BIN:-/home/zjulab/anaconda3/envs/kgpaper/bin/python}
OUT_ROOT="$ROOT/checkpoints/experiments/$EXP_ID"
HARD_OUT="$OUT_ROOT/phase2_alpha_hard"
SOFT_OUT="$OUT_ROOT/phase2_alpha_soft_abs_rkg"

[ -f "$SILVER" ] || { echo "FATAL: missing silver $SILVER" >&2; exit 2; }
[ ! -e "$OUT_ROOT" ] || {
  echo "FATAL: experiment output already exists: $OUT_ROOT (refusing overwrite)" >&2
  exit 2
}
mkdir -p "$OUT_ROOT"

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"'

"$PYTHON_BIN" scripts/train/phase2_train_prm.py \
  --config configs/training/phase2_prm.yaml \
  --silver_data "$SILVER" \
  --output_dir "$HARD_OUT"

"$PYTHON_BIN" scripts/train/phase2_train_prm.py \
  --config configs/training/phase2_prm_soft_alpha.yaml \
  --silver_data "$SILVER" \
  --output_dir "$SOFT_OUT"

"$PYTHON_BIN" scripts/pilot/score_alpha_gate.py \
  --silver "$HARD_OUT/silver_with_logprobs.jsonl" \
  --gate "hard=$HARD_OUT/alpha_gate.pt" \
  --gate "soft_abs_rkg=$SOFT_OUT/alpha_gate.pt" \
  --split val --val_ratio 0.10 --test_ratio 0.10 --split_seed 42 \
  --output "$OUT_ROOT/alpha_gate_val_report.json"

echo "PHASE2_ABLATION_COMPLETE experiment_id=$EXP_ID report=$OUT_ROOT/alpha_gate_val_report.json"
