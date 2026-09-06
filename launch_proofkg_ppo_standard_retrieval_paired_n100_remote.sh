#!/usr/bin/env bash
set -euo pipefail

KGPW_ROOT=${KGPW_ROOT:-/root/autodl-tmp/kgpaper}
KGPW_PYTHON=${KGPW_PYTHON:-/root/autodl-tmp/kgpw_env/bin/python}
KGPW_LLAMA3_PATH=${KGPW_LLAMA3_PATH:-/root/autodl-tmp/models/llama3-8b}
KGPW_GPU_ID=${KGPW_GPU_ID:-0}
KGPW_MIN_FREE_MIB=${KGPW_MIN_FREE_MIB:-18000}
cd "$KGPW_ROOT"
export PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src"
export CUDA_VISIBLE_DEVICES="$KGPW_GPU_ID"

PROTOCOL=outputs/audits/historical_hybrid_v2_independent_n100_seed20260901_standard_retrieval_proofkg_ppo_registration/protocol.json
ANALYSIS=outputs/audits/historical_hybrid_v2_independent_n100_seed20260901_standard_retrieval_proofkg_ppo_registration/analysis_protocol.json
LEGACY=outputs/audits/historical_hybrid_v2_independent_n100_seed20260901_standard_retrieval_inputs/arm_legacy.jsonl
PROOF=outputs/audits/historical_hybrid_v2_independent_n100_seed20260901_standard_retrieval_inputs/arm_proof.jsonl
BASELINE=outputs/validation/historical_hybrid_v2_independent_n100_seed20260901_standard_retrieval_sft/predictions.jsonl
ADAPTER=outputs/ppo_automatic_proofkg_2wiki_k4_smoke600_seed42/final
RUN_DIR=outputs/validation/proofkg_ppo_standard_retrieval_paired_n100_seed42
DECISION=outputs/validation/proofkg_ppo_standard_retrieval_paired_n100_seed42_decision/report.json
INFER_LOG=logs/evaluation/proofkg_ppo_standard_retrieval_paired_n100_seed42.log
SCORE_LOG=logs/evaluation/proofkg_ppo_standard_retrieval_paired_n100_seed42_score.log

for path in "$PROTOCOL" "$ANALYSIS" "$LEGACY" "$PROOF" "$BASELINE" \
  "$ADAPTER/adapter_config.json" "$ADAPTER/adapter_model.safetensors" \
  "$KGPW_LLAMA3_PATH/config.json" "$KGPW_LLAMA3_PATH/model.safetensors.index.json"; do
  test -s "$path"
done
test ! -e "$RUN_DIR"
test ! -e "$(dirname "$DECISION")"
test ! -e "$INFER_LOG"
test ! -e "$SCORE_LOG"

GPU_FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$KGPW_GPU_ID" | head -1 | tr -d ' ')
[[ "$GPU_FREE_MIB" =~ ^[0-9]+$ ]] && (( GPU_FREE_MIB >= KGPW_MIN_FREE_MIB ))
mkdir -p logs/evaluation

"$KGPW_PYTHON" scripts/eval/evaluate_a1_fixed_context_kg.py \
  --protocol "$PROTOCOL" \
  --legacy_input "$LEGACY" \
  --proof_input "$PROOF" \
  --adapter "$ADAPTER" \
  --model_label proofkg_ppo \
  --base_model "$KGPW_LLAMA3_PATH" \
  --run_dir "$RUN_DIR" \
  --experiment_id PROOFKG-PPO-STANDARD-RETRIEVAL-PAIRED-N100-SEED42 \
  --max_new_tokens 512 \
  --seed 42 \
  2>&1 | tee "$INFER_LOG"

"$KGPW_PYTHON" scripts/pilot/score_paired_kg_model_comparison.py \
  --analysis_protocol "$ANALYSIS" \
  --baseline_predictions "$BASELINE" \
  --candidate_predictions "$RUN_DIR/predictions.jsonl" \
  --output "$DECISION" \
  2>&1 | tee "$SCORE_LOG"
