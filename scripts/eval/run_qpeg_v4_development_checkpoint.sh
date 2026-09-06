#!/usr/bin/env bash
# Register, evaluate, and score one QPEG-v4 adapted checkpoint on development.
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^(25|50|75)$ ]]; then
  echo "usage: $0 {25|50|75}" >&2
  exit 2
fi

STEP=$1
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PYTHON=${KGPW_PYTHON:-/home/zjulab/anaconda3/envs/kgpaper/bin/python}
BASE_PROTOCOL=outputs/audits/qpeg_v4_schema_adaptation_development_inputs_v1/protocol.json
EXPERIMENT_PROTOCOL=outputs/audits/qpeg_v4_schema_adaptation_protocol_v1/protocol.json
ADAPTER=checkpoints/sft_qpeg_v4_schema_adaptation_n2400_seed42/checkpoint-$STEP
REGISTRY=outputs/audits/qpeg_v4_schema_adaptation_development_checkpoint_registry_v1/step${STEP}.protocol.json
RUN_DIR=outputs/validation/qpeg_v4_development_adapted_step${STEP}_ab_v1
LOG_DIR=outputs/validation/qpeg_v4_development_adapted_step${STEP}_ab_v1_logs
SCORE=outputs/audits/qpeg_v4_schema_adaptation_development_scores_v1/step${STEP}.json
STRONG=outputs/validation/qpeg_v4_development_strong_sft_ab_v1/predictions.jsonl

test -s "$STRONG"
test -s "$ADAPTER/adapter_model.safetensors"
test ! -e "$REGISTRY"
test ! -e "$RUN_DIR"
test ! -e "$SCORE"
mkdir -p "$LOG_DIR"

env PYTHONPATH="$ROOT:$ROOT/flashrag_src" "$PYTHON" \
  scripts/prepare/register_qpeg_v4_development_checkpoint.py \
  --parent_protocol "$BASE_PROTOCOL" \
  --adapter "$ADAPTER" \
  --checkpoint_step "$STEP" \
  --output "$REGISTRY"

env PYTHONPATH="$ROOT:$ROOT/flashrag_src" CUDA_VISIBLE_DEVICES=0 "$PYTHON" \
  scripts/eval/evaluate_a1_fixed_context_kg.py \
  --protocol "$REGISTRY" \
  --legacy_input outputs/audits/qpeg_v4_schema_adaptation_development_inputs_v1/arm_no_graph.jsonl \
  --proof_input outputs/audits/qpeg_v4_schema_adaptation_development_inputs_v1/arm_qpeg.jsonl \
  --adapter "$ADAPTER" \
  --model_label "adapted_step$STEP" \
  --base_model models/llama3-8b \
  --run_dir "$RUN_DIR" \
  --experiment_id "QPEG-V4-DEVELOPMENT-ADAPTED-STEP${STEP}-AB-SEED42" \
  --max_new_tokens 512 --seed 42 2>&1 | tee "$LOG_DIR/run.log"

env PYTHONPATH="$ROOT:$ROOT/flashrag_src" "$PYTHON" \
  scripts/eval/score_qpeg_v4_development_interaction.py \
  --experiment_protocol "$EXPERIMENT_PROTOCOL" \
  --strong_predictions "$STRONG" \
  --adapted_predictions "$RUN_DIR/predictions.jsonl" \
  --adapted_label "adapted_step$STEP" \
  --checkpoint_step "$STEP" \
  --output "$SCORE"
