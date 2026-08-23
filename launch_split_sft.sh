#!/bin/bash
# ===========================================================================
# Phase 3a SFT — first run on the real train fold  (baseline for the PPO run)
#
#   bash launch_split_sft.sh
#
# What makes this run different from checkpoints/sft_student (2026-06-22):
#   --split train      holds back val/test, so Phase 3b's numbers can be
#                      reported as held-out. The old run used the whole file.
#   sft_max_length     6144, not 4096. Measured on the train fold (llama3-8b
#                      tokenizer, topk=15): median 4072, p90 4594, max 5901.
#                      At 4096, 48% of rows overflowed and _build_dataset
#                      dropped their lowest-ranked passages — so SFT trained on
#                      ~10-14 passages while PPO rolls out with 15. At 6144
#                      nothing overflows and the two phases match.
#
# Expected shape (computed, not guessed):
#   7,913 accepted trajectories in the train fold (9,839 in the whole file)
#   bs 4 x accum 8 = 32 traj/step -> 247 optimizer steps, 1 epoch
#   ~12 loss lines (logging_steps=20), one save at the end
#   peak GPU: ~46 GB active. nvidia-smi will show MORE than that — the caching
#   allocator does not return freed blocks, so watch the
#   "SFT peak GPU memory: allocated ... | reserved ..." line instead.
# ===========================================================================
set -euo pipefail

REMOTE_ROOT=/root/autodl-tmp/kgpaper
cd "$REMOTE_ROOT"

export PYTHONPATH="$REMOTE_ROOT:$REMOTE_ROOT/flashrag_src"
export KGPW_FLASHRAG_ROOT="$REMOTE_ROOT/flashrag_src"
export KGPW_PROJECT_ROOT="$REMOTE_ROOT"
export KGPW_DATA_DIR="$REMOTE_ROOT/data"
export KGPW_INDEX_DIR="$REMOTE_ROOT/indexes"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset HTTP_PROXY HTTPS_PROXY

PY=/root/autodl-tmp/kgpw_env/bin/python
SILVER="$REMOTE_ROOT/data/silver_data/silver_v1_reannotated.jsonl"
OUT="$REMOTE_ROOT/checkpoints/sft_student_split"

# Fail before loading 16 GB of weights rather than after.
[ -f "$SILVER" ] || { echo "MISSING silver: $SILVER"; exit 1; }

mkdir -p "$OUT"

nohup "$PY" scripts/train/phase3_sft.py \
  --config configs/training/phase3_sft.yaml \
  --silver_data "$SILVER" \
  --output_dir "$OUT" \
  --split train \
  > "$OUT/train.log" 2>&1 &

echo "Launched Phase 3a SFT (pid $!)"
echo "  log:    tail -f $OUT/train.log"
echo "  split:  grep 'Phase 3a split' $OUT/train.log     # must say fold=train"
echo "  loss:   grep -o \"'loss': [0-9.]*\" $OUT/train.log | tail"
echo "  memory: grep 'peak GPU memory' $OUT/train.log     # after it finishes"
echo "  expect: 247 steps, ~12 loss lines, final at $OUT/final"
