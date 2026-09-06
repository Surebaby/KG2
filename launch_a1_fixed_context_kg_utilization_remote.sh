#!/usr/bin/env bash
set -euo pipefail

KGPW_ROOT="${KGPW_ROOT:-/root/autodl-tmp/kgpaper}"
KGPW_PYTHON="${KGPW_PYTHON:-/root/anaconda3/envs/kgpaper/bin/python}"
KGPW_LLAMA3_PATH="${KGPW_LLAMA3_PATH:-/root/autodl-tmp/models/llama3-8b}"
KGPW_GPU_ID="${KGPW_GPU_ID:-0}"

cd "$KGPW_ROOT"
export PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src"
export CUDA_VISIBLE_DEVICES="$KGPW_GPU_ID"

PROTOCOL="outputs/audits/a1_fixed_context_kg_utilization_preregistration_20260829/protocol.json"
LEGACY_INPUT="outputs/audits/a1_fixed_context_kg_paired_inputs_n30_20260829/arm_legacy.jsonl"
PROOF_INPUT="outputs/audits/a1_fixed_context_kg_paired_inputs_n30_20260829/arm_proof.jsonl"
SFT_ADAPTER="checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final"
PPO_ADAPTER="outputs/ppo_legacy_repaired_v2_quota70_hard_seed42_no_text_head_smoke600_hybrid_old10_bridge5_v3/final"
SFT_RUN="outputs/validation/a1_fixed_context_kg_utilization_sft_n30_20260829"
PPO_RUN="outputs/validation/a1_fixed_context_kg_utilization_hybrid_ppo_n30_20260829"
PAIRED_REPORT="outputs/validation/a1_fixed_context_kg_utilization_paired_n30_20260829/report.json"

mkdir -p logs/evaluation

"$KGPW_PYTHON" scripts/eval/evaluate_a1_fixed_context_kg.py \
  --protocol "$PROTOCOL" \
  --legacy_input "$LEGACY_INPUT" \
  --proof_input "$PROOF_INPUT" \
  --adapter "$SFT_ADAPTER" \
  --model_label sft \
  --base_model "$KGPW_LLAMA3_PATH" \
  --run_dir "$SFT_RUN" \
  --experiment_id a1_fixed_context_kg_utilization_sft_n30_20260829 \
  --max_new_tokens 512 \
  --seed 42 \
  2>&1 | tee logs/evaluation/a1_fixed_context_kg_utilization_sft_n30_20260829.log

"$KGPW_PYTHON" scripts/eval/evaluate_a1_fixed_context_kg.py \
  --protocol "$PROTOCOL" \
  --legacy_input "$LEGACY_INPUT" \
  --proof_input "$PROOF_INPUT" \
  --adapter "$PPO_ADAPTER" \
  --model_label hybrid_ppo \
  --base_model "$KGPW_LLAMA3_PATH" \
  --run_dir "$PPO_RUN" \
  --experiment_id a1_fixed_context_kg_utilization_hybrid_ppo_n30_20260829 \
  --max_new_tokens 512 \
  --seed 42 \
  2>&1 | tee logs/evaluation/a1_fixed_context_kg_utilization_hybrid_ppo_n30_20260829.log

"$KGPW_PYTHON" scripts/pilot/score_a1_fixed_context_kg.py \
  --protocol "$PROTOCOL" \
  --sft_predictions "$SFT_RUN/predictions.jsonl" \
  --hybrid_predictions "$PPO_RUN/predictions.jsonl" \
  --output "$PAIRED_REPORT" \
  2>&1 | tee logs/evaluation/a1_fixed_context_kg_utilization_paired_n30_20260829.log
