#!/usr/bin/env bash
# Frozen, zero-training old-KG vs passage-aware-KG-v2 paired evaluation.
# hidden33 is val; hard25 is an explicitly labelled train diagnostic.
set -euo pipefail

KGPW_ROOT=${KGPW_ROOT:-/root/autodl-tmp/kgpaper}
KGPW_PYTHON=${KGPW_PYTHON:-/root/autodl-tmp/kgpw_env/bin/python}
cd "$KGPW_ROOT"

export PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPERIMENT_ID=kg_v2_zero_train_paired_hidden33_hard25_seed20260828_v1
OUTPUT_DIR="outputs/validation/$EXPERIMENT_ID"
LOG_PATH="logs/validation/$EXPERIMENT_ID.log"
SILVER=checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42_no_text_head/silver_with_logprobs.jsonl
SFT=checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final
PPO=outputs/ppo_legacy_repaired_v2_quota70_hard_seed42_no_text_head_smoke600_hybrid_old10_bridge5_v3/final
HIDDEN_COHORT=outputs/audits/hidden_retrieval_audit.jsonl
HARD_COHORT=outputs/audits/rankability_hard25_retrieval_attribution_seed20260828_v1/hard25_attribution.jsonl
HIDDEN_OLD=data/silver_data/pilots/retrieval_bridge_v3_hidden33_seed42_20260827_gpu/passages_top30.jsonl
HIDDEN_V2=data/silver_data/pilots/passage_aware_kg_v2_local_roots_hidden33_seed42_20260828/passages_top15_with_kg_v2.jsonl
HARD_OLD=data/silver_data/pilots/ppo_smoke_hybrid_train_seed42_20260828/old10_bridge5_v3.jsonl
HARD_V2=data/silver_data/pilots/passage_aware_kg_v2_local_roots_hard25_seed20260828/passages_top15_with_kg_v2.jsonl
HARD_STRUCTURAL=outputs/audits/passage_aware_kg_v2_hard25_structural_gate_seed20260828/report.json

mkdir -p logs/validation outputs/validation
test ! -e "$OUTPUT_DIR"
test ! -e "$LOG_PATH"
for path in "$SILVER" "$SFT" "$PPO" "$HIDDEN_COHORT" "$HARD_COHORT" \
  "$HIDDEN_OLD" "$HIDDEN_V2" "$HARD_OLD" "$HARD_V2" "$HARD_STRUCTURAL"; do
  test -e "$path"
done
test "$(sha256sum "$HIDDEN_COHORT" | cut -d' ' -f1)" = \
  202c1d3e6a3c07056ba084bb1bedb49b0730ea374f0d4eceb42241d224aa4569
test "$(sha256sum "$HARD_COHORT" | cut -d' ' -f1)" = \
  a87efa8e42a6a27686c0bba177a09fe5f778af1f47ef0901143f1a645aad63f7
test "$(sha256sum "$HIDDEN_OLD" | cut -d' ' -f1)" = \
  bae5ea05ff04266d34857d5f7f8a26389973cace437ef5507b27d6c38588b6c4
test "$(sha256sum "$HIDDEN_V2" | cut -d' ' -f1)" = \
  47d39df0d8af746fa53dbaca9e6ae82b6c5667d74b95f2ef1251518e43a0a454
test "$(sha256sum "$HARD_OLD" | cut -d' ' -f1)" = \
  7c199a0e272323a8739d232c21ee4f084ba0fc071f1de67fb97a7cb593eb1a1f
test "$(sha256sum "$HARD_V2" | cut -d' ' -f1)" = \
  ade16ae58b5a4a5e150b2b987af407a608d102251a5a4b966aa13c07cea12b96

"$KGPW_PYTHON" -m pytest -q \
  tests/test_validate_sft_sampling.py \
  tests/test_passage_aware_kg_v2.py

if [[ "${1:-}" == "--preflight-only" ]]; then
  echo "preflight complete: hashes, checkpoints, cohorts, and tests passed"
  exit 0
fi

mkdir "$OUTPUT_DIR"
exec > >(tee "$LOG_PATH") 2>&1

run_hidden() {
  local name=$1
  local adapter=$2
  local overrides=$3
  "$KGPW_PYTHON" scripts/eval/validate_sft.py \
    --adapter "$adapter" \
    --silver "$SILVER" \
    --split val \
    --selection_jsonl "$HIDDEN_COHORT" \
    --input_overrides "$overrides" \
    --max_new_tokens 512 \
    --seed 42 \
    --out "$OUTPUT_DIR/$name.jsonl"
}

run_hard() {
  local name=$1
  local adapter=$2
  local overrides=$3
  "$KGPW_PYTHON" scripts/eval/validate_sft.py \
    --adapter "$adapter" \
    --silver "$SILVER" \
    --split train \
    --allow_train_diagnostic \
    --selection_jsonl "$HARD_COHORT" \
    --input_overrides "$overrides" \
    --max_new_tokens 512 \
    --seed 42 \
    --out "$OUTPUT_DIR/$name.jsonl"
}

run_hidden hidden_sft_old "$SFT" "$HIDDEN_OLD"
run_hidden hidden_sft_v2 "$SFT" "$HIDDEN_V2"
run_hidden hidden_ppo_old "$PPO" "$HIDDEN_OLD"
run_hidden hidden_ppo_v2 "$PPO" "$HIDDEN_V2"
run_hard hard_sft_old "$SFT" "$HARD_OLD"
run_hard hard_sft_v2 "$SFT" "$HARD_V2"
run_hard hard_ppo_old "$PPO" "$HARD_OLD"
run_hard hard_ppo_v2 "$PPO" "$HARD_V2"

"$KGPW_PYTHON" scripts/pilot/score_kg_v2_zero_train_eval.py \
  --hidden_sft_old "$OUTPUT_DIR/hidden_sft_old.jsonl" \
  --hidden_sft_v2 "$OUTPUT_DIR/hidden_sft_v2.jsonl" \
  --hidden_ppo_old "$OUTPUT_DIR/hidden_ppo_old.jsonl" \
  --hidden_ppo_v2 "$OUTPUT_DIR/hidden_ppo_v2.jsonl" \
  --hard_sft_old "$OUTPUT_DIR/hard_sft_old.jsonl" \
  --hard_sft_v2 "$OUTPUT_DIR/hard_sft_v2.jsonl" \
  --hard_ppo_old "$OUTPUT_DIR/hard_ppo_old.jsonl" \
  --hard_ppo_v2 "$OUTPUT_DIR/hard_ppo_v2.jsonl" \
  --hidden_cohort "$HIDDEN_COHORT" \
  --hard_cohort "$HARD_COHORT" \
  --hard_structural_report "$HARD_STRUCTURAL" \
  --output "$OUTPUT_DIR/report.json" \
  --run_dir "$OUTPUT_DIR/report_run" \
  --experiment_id "$EXPERIMENT_ID"
