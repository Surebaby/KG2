#!/usr/bin/env bash
# 96GB remote: light (20% ProofKG) continued-SFT repair with checkpoints.
set -euo pipefail
KGPW_ROOT=${KGPW_ROOT:-/root/autodl-tmp/kgpaper}
KGPW_PYTHON=${KGPW_PYTHON:-/root/autodl-tmp/kgpw_env/bin/python}
cd "$KGPW_ROOT"

export PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_DIR=data/silver_data/proofkg_curriculum_light20_v2_n5000_seed42_20260831
OUT=checkpoints/sft_proofkg_curriculum_light20_v2_n5000_seed42
LOG=logs/training/sft_proofkg_curriculum_light20_v2_n5000_seed42.log
mkdir -p logs/training
test ! -e "$OUT"
test ! -e "$LOG"
test -s "$DATA_DIR/silver_curriculum.jsonl"
test -s "$DATA_DIR/question_kg_records.jsonl"
test -d checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final
test "$(sha256sum "$DATA_DIR/silver_curriculum.jsonl" | cut -d' ' -f1)" = \
  1cfd5f989f65a476390151e74ae4511e6199129d4394d884276fea92487a067b
test "$(sha256sum "$DATA_DIR/question_kg_records.jsonl" | cut -d' ' -f1)" = \
  8434ed430a644d6e5b175a2eba1799ba2ac33075731b2af7456b60786fd32521

"$KGPW_PYTHON" -m pytest -q \
  tests/test_phase3_sft_config_forwarding.py \
  tests/test_training_question_kg.py \
  tests/test_silver_split.py \
  tests/test_prm_annotator.py \
  tests/test_kg_index_guard.py

exec "$KGPW_PYTHON" scripts/train/phase3_sft.py \
  --config configs/training/phase3_sft_proofkg_curriculum_light20_v2_n5000_seed42.yaml \
  2>&1 | tee "$LOG"
