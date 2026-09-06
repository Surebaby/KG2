#!/usr/bin/env bash
# 96GB remote: guarded 600-trajectory PPO smoke after light20 SFT PASS.
set -euo pipefail
KGPW_ROOT=${KGPW_ROOT:-/root/autodl-tmp/kgpaper}
KGPW_PYTHON=${KGPW_PYTHON:-/root/autodl-tmp/kgpw_env/bin/python}
cd "$KGPW_ROOT"

export PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KGPW_TB_DIR=${KGPW_TB_DIR:-/root/tf-logs}

CONFIG=configs/training/phase3_ppo_proofkg_curriculum_light20_v2_smoke600_seed42.resolved.yaml
TMP_VALUES=$(mktemp)
trap 'rm -f "$TMP_VALUES"' EXIT
test -s "$CONFIG"

"$KGPW_PYTHON" -c \
  "from kgproweight.config import load_config,ProjectConfig; c=load_config('$CONFIG',validate=ProjectConfig).training; print(c.sft_checkpoint); print(c.sft_selection_report_path); print(c.output_dir)" \
  > "$TMP_VALUES"
mapfile -t VALUES < "$TMP_VALUES"
SFT=${VALUES[0]}
SELECTION=${VALUES[1]}
OUT=${VALUES[2]}
LOG="logs/training/$(basename "$OUT").log"

test -s "$SFT/adapter_model.safetensors"
test -s "$SELECTION"
test ! -e "$OUT"
test ! -e "$LOG"
test "$(sha256sum data/silver_data/proofkg_curriculum_light20_v2_n5000_seed42_20260831/silver_curriculum.jsonl | cut -d' ' -f1)" = \
  1cfd5f989f65a476390151e74ae4511e6199129d4394d884276fea92487a067b
test "$(sha256sum data/silver_data/proofkg_curriculum_light20_v2_n5000_seed42_20260831/question_kg_records.jsonl | cut -d' ' -f1)" = \
  8434ed430a644d6e5b175a2eba1799ba2ac33075731b2af7456b60786fd32521
"$KGPW_PYTHON" -c \
  "import json; from pathlib import Path; from scripts.prepare.materialize_light20_ppo_config import resolve_selected_checkpoint; r=json.load(open('$SELECTION')); _,p=resolve_selected_checkpoint(r,Path('checkpoints/sft_proofkg_curriculum_light20_v2_n5000_seed42')); assert p.resolve()==Path('$SFT').resolve()"
test "$($KGPW_PYTHON -c "import json; print(json.load(open('checkpoints/sft_proofkg_curriculum_light20_v2_n5000_seed42/manifest.json'))['status'])")" = COMPLETE
mkdir -p logs/training

"$KGPW_PYTHON" -m pytest -q \
  tests/test_materialize_light20_ppo_config.py \
  tests/test_phase3_ppo_config_forwarding.py \
  tests/test_training_question_kg.py \
  tests/test_prm_annotator.py \
  tests/test_kg_index_guard.py \
  tests/test_ppo_sft_replay.py \
  tests/test_ppo_explicit_reference.py \
  tests/test_ppo_diagnostics.py \
  tests/test_run_preflight_manifest.py

exec "$KGPW_PYTHON" scripts/train/phase3_ppo.py --config "$CONFIG" 2>&1 | tee "$LOG"
