#!/usr/bin/env bash
# Evaluate all light20 SFT candidates on the already-frozen replay val n=200.
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="${PYTHONPATH:-$PWD:$PWD/flashrag_src}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

ROOT=checkpoints/sft_proofkg_curriculum_light20_v2_n5000_seed42
SILVER=checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42_no_text_head/silver_with_logprobs.jsonl
OUT=outputs/validation/sft_proofkg_curriculum_light20_v2_n5000_seed42_val_n200
LABELS=(step40 step80 step120 final)
ADAPTERS=("$ROOT/checkpoint-40" "$ROOT/checkpoint-80" "$ROOT/checkpoint-120" "$ROOT/final")

test -s "$SILVER"
for adapter in "${ADAPTERS[@]}"; do test -s "$adapter/adapter_model.safetensors"; done
mkdir -p "$OUT"
test ! -e "$OUT/checkpoint_selection.json"

for index in "${!LABELS[@]}"; do
  label=${LABELS[$index]}
  adapter=${ADAPTERS[$index]}
  if [[ -s "$OUT/$label.jsonl" ]]; then
    echo "REUSE_COMPLETE_CANDIDATE=$label"
    continue
  fi
  test ! -e "$OUT/$label.log"
  python scripts/eval/validate_sft.py \
    --adapter "$adapter" --silver "$SILVER" --split val --n 200 --seed 42 \
    --out "$OUT/$label.jsonl" \
    2>&1 | tee "$OUT/$label.log"
done

python scripts/eval/select_sft_checkpoint.py \
  --candidate "step40=$OUT/step40.jsonl" \
  --candidate "step80=$OUT/step80.jsonl" \
  --candidate "step120=$OUT/step120.jsonl" \
  --candidate "final=$OUT/final.jsonl" \
  --output "$OUT/checkpoint_selection.json"

# A runnable PPO config is materialized only after the selector exits PASS.
# This prepares provenance; it does not start PPO.
python scripts/prepare/materialize_light20_ppo_config.py \
  --selection_report "$OUT/checkpoint_selection.json" \
  --checkpoint_root "$ROOT" \
  --output_config \
    configs/training/phase3_ppo_proofkg_curriculum_light20_v2_smoke600_seed42.resolved.yaml

echo "SFT_GATE_PASS; PPO config prepared but training NOT started."
