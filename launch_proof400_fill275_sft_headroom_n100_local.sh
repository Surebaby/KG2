#!/usr/bin/env bash
set -euo pipefail

# Zero-update inference audit only: 100 greedy + 400 sampled generations.
# It never calls the PPO trainer and never constructs a reward model or critic.
KGPW_RUN_ROOT=${KGPW_PROJECT_ROOT:-/home/zjulab/kgpaper}
KGPW_RUN_PYTHON=${KGPW_PYTHON:-/home/zjulab/anaconda3/envs/kgpaper/bin/python}
cd "$KGPW_RUN_ROOT"
export PYTHONPATH="$KGPW_RUN_ROOT:$KGPW_RUN_ROOT/flashrag_src"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PROTOCOL=outputs/audits/proof400_fill275_strong_sft_headroom_n100_k4_seed42_v3_preregistration/protocol.json
OUTPUT=outputs/validation/proof400_fill275_strong_sft_headroom_n100_k4_seed42_v3
LOG=logs/training/proof400_fill275_strong_sft_headroom_n100_k4_seed42_v3.log
EXPERIMENT_ID=PROOF400-FILL275-STRONG-SFT-HEADROOM-N100-K4-SEED42-V3

test -x "$KGPW_RUN_PYTHON"
test -s "$PROTOCOL"
test ! -e "$OUTPUT"
test ! -e "$LOG"
nvidia-smi
mkdir -p logs/training
"$KGPW_RUN_PYTHON" scripts/pilot/audit_proof400_fill275_sft_headroom.py \
  --protocol "$PROTOCOL" \
  --output_dir "$OUTPUT" \
  --experiment_id "$EXPERIMENT_ID" \
  --batch_size 4 2>&1 | tee "$LOG"
