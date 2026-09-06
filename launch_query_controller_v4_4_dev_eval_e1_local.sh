#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/zjulab/kgpaper"
PYTHON_BIN="/home/zjulab/anaconda3/envs/kgpaper/bin/python"
INPUT="data/silver_data/query_controller_actions_v1_seed42_v4_4/dev.jsonl"
ADAPTER="outputs/probes/query_controller_action_v1_probe20_seed42_v4_4/final"
PARENT_PROTOCOL="outputs/audits/query_controller_v1_exact_text_pilot_seed42_protocol_v4_4/protocol.json"
EVAL_PROTOCOL="outputs/audits/query_controller_v4_4_dev_eval_e1_protocol/protocol.json"
TRAINING_MANIFEST="outputs/probes/query_controller_action_v1_probe20_seed42_v4_4/manifest.json"
OUTPUT="outputs/validation/query_controller_action_v1_probe20_seed42_v4_4_dev_eval_e1"
LOG="logs/eval/query_controller_action_v1_probe20_seed42_v4_4_dev_eval_e1.log"

cd "$PROJECT_ROOT"
test -s "$INPUT"
test -s "$ADAPTER/adapter_model.safetensors"
test -s "$PARENT_PROTOCOL"
test -s "$EVAL_PROTOCOL"
test -s "$TRAINING_MANIFEST"
test ! -e "$OUTPUT"
test ! -e "$LOG"
mkdir -p "$(dirname "$LOG")"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/flashrag_src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$PYTHON_BIN" scripts/eval/generate_query_controller_actions.py \
  --input "$INPUT" \
  --adapter "$ADAPTER" \
  --protocol "$PARENT_PROTOCOL" \
  --eval_protocol "$EVAL_PROTOCOL" \
  --training_manifest "$TRAINING_MANIFEST" \
  --expected_protocol_sha256 be9eb2cf1fc00b6ca61fb0f4af4edc6075ef3f3e0aab916555eae9c9e263d55b \
  --expected_eval_protocol_sha256 dcbc1280b9f9a70cb9ad030a5c59f2efd5e4d5d617f35472fa74757e94b544c8 \
  --expected_training_manifest_sha256 23d8d2df11e923d24e3f9326374a475d53da9c5e2363e663c1aae8d8848b1dad \
  --expected_adapter_sha256 b3bae36afd770eba4e2d144ed6d07e6ae07c3bc73d09e526294686dcd669f88e \
  --cohort_role dev \
  --output_dir "$OUTPUT" \
  --experiment_id QUERY-CONTROLLER-ACTION-V1-PROBE20-SEED42-V4-4-DEV-EVAL-E1 \
  --base_model llama3-8B-instruct \
  --batch_size 4 \
  --max_input_tokens 1024 \
  --max_new_tokens 192 \
  --seed 42 \
  --dtype bf16 \
  --load_in_4bit \
  2>&1 | tee "$LOG"
