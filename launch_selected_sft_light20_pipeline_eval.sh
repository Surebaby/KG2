#!/usr/bin/env bash
# Three-dataset standard pipeline evaluation for the gate-selected light20 SFT.
set -euo pipefail
cd "$(dirname "$0")"

KGPW_PYTHON=${KGPW_PYTHON:-/home/zjulab/anaconda3/envs/kgpaper/bin/python}
CONFIG=configs/training/phase3_ppo_proofkg_curriculum_light20_v2_smoke600_seed42.resolved.yaml
SELECTION=outputs/validation/sft_proofkg_curriculum_light20_v2_n5000_seed42_val_n200/checkpoint_selection.json
SAVE_ROOT=outputs/sft_proofkg_curriculum_light20_v2_selected_pipeline_eval
TMP_SELECTED=$(mktemp)
trap 'rm -f "$TMP_SELECTED"' EXIT

test -s "$CONFIG"
test -s "$SELECTION"
test ! -e "$SAVE_ROOT"
test -s indexes_wiki18/corpus_flashrag.jsonl
test -s indexes_wiki18/e5_fp16.dat
test -d indexes_wiki18/bm25

export PYTHONPATH="$PWD/flashrag_src:$PWD"
export KGPW_CORPUS_PATH="$PWD/indexes_wiki18/corpus_flashrag.jsonl"
export KGPW_DENSE_INDEX_PATH="$PWD/indexes_wiki18/e5_fp16.dat"
export KGPW_BM25_INDEX_PATH="$PWD/indexes_wiki18/bm25"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$KGPW_PYTHON" -c \
  "from kgproweight.config import load_config,ProjectConfig; print(load_config('$CONFIG', validate=ProjectConfig).training.sft_checkpoint)" \
  > "$TMP_SELECTED"
IFS= read -r SELECTED_ADAPTER < "$TMP_SELECTED"
test -s "$SELECTED_ADAPTER/adapter_model.safetensors"

exec "$KGPW_PYTHON" scripts/eval/run_kg_proweight.py \
  --checkpoint "$SELECTED_ADAPTER" \
  --alpha_gate_path checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42_no_text_head/alpha_gate.pt \
  --datasets hotpotqa 2wikimultihopqa musique \
  --split dev \
  --seeds 42 \
  --test_sample_num 300 \
  --rerank 10 \
  --gpu_id 0 \
  --save_root "$SAVE_ROOT"
