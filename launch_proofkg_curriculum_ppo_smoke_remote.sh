#!/usr/bin/env bash
# 96GB remote: 600-trajectory PPO smoke after the curriculum SFT completes.
set -euo pipefail
KGPW_ROOT=${KGPW_ROOT:-/root/autodl-tmp/kgpaper}
KGPW_PYTHON=${KGPW_PYTHON:-/root/autodl-tmp/kgpw_env/bin/python}
cd "$KGPW_ROOT"

export PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KGPW_TB_DIR=${KGPW_TB_DIR:-/root/tf-logs}

DATA_DIR=data/silver_data/proofkg_curriculum_mix_v1_n8000_seed42_20260829
SFT=checkpoints/sft_proofkg_curriculum_mix_v1_n8000_seed42
OUT=outputs/ppo_proofkg_curriculum_mix_v1_smoke600_seed42
LOG=logs/training/ppo_proofkg_curriculum_mix_v1_smoke600_seed42.log
mkdir -p logs/training
test ! -e "$OUT"
test ! -e "$LOG"
test "$("$KGPW_PYTHON" -c "import json; print(json.load(open('$SFT/manifest.json'))['status'])")" = COMPLETE
test -s "$SFT/final/adapter_model.safetensors"
test -s "$DATA_DIR/silver_curriculum.jsonl"
test -s "$DATA_DIR/question_kg_records.jsonl"
test "$(sha256sum "$DATA_DIR/silver_curriculum.jsonl" | cut -d' ' -f1)" = \
  c8d5de2d76db56f767cfedb31e6182ef80050a6047cb4fca2c7bda16f18a519d
test "$(sha256sum "$DATA_DIR/question_kg_records.jsonl" | cut -d' ' -f1)" = \
  56aab2b6502adfb1bfe51c0933cc821573039b3ae4f7f52aa1db771803873053

"$KGPW_PYTHON" -m pytest -q \
  tests/test_training_question_kg.py \
  tests/test_prm_annotator.py \
  tests/test_kg_index_guard.py \
  tests/test_ppo_sft_replay.py \
  tests/test_phase3_ppo_config_forwarding.py \
  tests/test_ppo_explicit_reference.py \
  tests/test_ppo_diagnostics.py \
  tests/test_run_preflight_manifest.py

exec "$KGPW_PYTHON" scripts/train/phase3_ppo.py \
  --config configs/training/phase3_ppo_proofkg_curriculum_mix_v1_smoke600_seed42.yaml \
  2>&1 | tee "$LOG"
