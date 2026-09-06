#!/usr/bin/env bash
# Wait for the approved QPEG-v4 SFT to complete, then evaluate step 25 on dev.
set -euo pipefail
ROOT=${KGPW_PROJECT_ROOT:-/root/autodl-tmp/kgpaper}
PYTHON=${KGPW_PYTHON:-/root/autodl-tmp/kgpw_env/bin/python}
cd "$ROOT"

TRAIN_DIR=checkpoints/sft_qpeg_v4_schema_adaptation_n2400_seed42
PARENT=outputs/audits/qpeg_v4_schema_adaptation_development_inputs_v1/protocol.json
REGISTRY=outputs/audits/qpeg_v4_schema_adaptation_development_checkpoint_registry_v1/step25.protocol.json
RUN_DIR=outputs/validation/qpeg_v4_development_adapted_step25_ab_v1
LOG=logs/training/qpeg_v4_development_adapted_step25_ab_v1.log

while true; do
  status=$($PYTHON -c "import json; print(json.load(open('$TRAIN_DIR/manifest.json')).get('status',''))")
  if [[ "$status" == "COMPLETE" ]]; then
    break
  fi
  if ! pgrep -f 'scripts/train/phase3_sft.py --config configs/training/phase3_sft_qpeg_v4_schema_adaptation_n2400_seed42.yaml' >/dev/null; then
    echo "training process ended without COMPLETE manifest (status=$status)" >&2
    exit 1
  fi
  sleep 20
done

test -s "$TRAIN_DIR/checkpoint-25/adapter_model.safetensors"
test -s outputs/audits/qpeg_v4_schema_adaptation_development_inputs_v1/arm_no_graph.jsonl
test -s outputs/audits/qpeg_v4_schema_adaptation_development_inputs_v1/arm_qpeg.jsonl
test ! -e "$REGISTRY"
test ! -e "$RUN_DIR"
mkdir -p "$(dirname "$REGISTRY")" "$(dirname "$LOG")"

env PYTHONPATH="$ROOT:$ROOT/flashrag_src" "$PYTHON" \
  scripts/prepare/register_qpeg_v4_development_checkpoint.py \
  --parent_protocol "$PARENT" \
  --adapter "$TRAIN_DIR/checkpoint-25" \
  --checkpoint_step 25 \
  --output "$REGISTRY"

env PYTHONPATH="$ROOT:$ROOT/flashrag_src" CUDA_VISIBLE_DEVICES=0 "$PYTHON" \
  scripts/eval/evaluate_a1_fixed_context_kg.py \
  --protocol "$REGISTRY" \
  --legacy_input outputs/audits/qpeg_v4_schema_adaptation_development_inputs_v1/arm_no_graph.jsonl \
  --proof_input outputs/audits/qpeg_v4_schema_adaptation_development_inputs_v1/arm_qpeg.jsonl \
  --adapter "$TRAIN_DIR/checkpoint-25" \
  --model_label adapted_step25 \
  --base_model models/llama3-8b \
  --run_dir "$RUN_DIR" \
  --experiment_id QPEG-V4-DEVELOPMENT-ADAPTED-STEP25-AB-SEED42 \
  --max_new_tokens 512 --seed 42 2>&1 | tee "$LOG"
