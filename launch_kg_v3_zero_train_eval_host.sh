#!/usr/bin/env bash
# Frozen zero-training precision-first KG-v3 paired evaluation.
# Reuses immutable old arms from the KG-v2 run and computes four new v3 arms.
set -euo pipefail

KGPW_ROOT=${KGPW_ROOT:-/root/autodl-tmp/kgpaper}
KGPW_PYTHON=${KGPW_PYTHON:-/root/autodl-tmp/kgpw_env/bin/python}
cd "$KGPW_ROOT"

export PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPERIMENT_ID=kg_v3_precision_zero_train_paired_hidden33_hard25_seed20260828_v1
OUTPUT_DIR="outputs/validation/$EXPERIMENT_ID"
LOG_PATH="logs/validation/$EXPERIMENT_ID.log"
OLD_RUN=outputs/validation/kg_v2_zero_train_paired_hidden33_hard25_seed20260828_v1
SILVER=checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42_no_text_head/silver_with_logprobs.jsonl
SFT=checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final
PPO=outputs/ppo_legacy_repaired_v2_quota70_hard_seed42_no_text_head_smoke600_hybrid_old10_bridge5_v3/final
HIDDEN_COHORT=outputs/audits/hidden_retrieval_audit.jsonl
HARD_COHORT=outputs/audits/rankability_hard25_retrieval_attribution_seed20260828_v1/hard25_attribution.jsonl
HIDDEN_V3=data/silver_data/pilots/passage_aware_kg_v3_explicit_question_hidden33_seed42_20260828/passages_top15_with_kg_v3.jsonl
HARD_V3=data/silver_data/pilots/passage_aware_kg_v3_explicit_question_hard25_seed20260828/passages_top15_with_kg_v3.jsonl
HARD_STRUCTURAL=outputs/audits/passage_aware_kg_v3_explicit_question_hard25_structural_gate_seed20260828/report.json

mkdir -p logs/validation outputs/validation
test ! -e "$OUTPUT_DIR"
test ! -e "$LOG_PATH"
for path in "$SILVER" "$SFT" "$PPO" "$HIDDEN_COHORT" "$HARD_COHORT" \
  "$HIDDEN_V3" "$HARD_V3" "$HARD_STRUCTURAL" \
  "$OLD_RUN/hidden_sft_old.jsonl" "$OLD_RUN/hidden_ppo_old.jsonl" \
  "$OLD_RUN/hard_sft_old.jsonl" "$OLD_RUN/hard_ppo_old.jsonl"; do
  test -e "$path"
done
test "$(sha256sum "$HIDDEN_COHORT" | cut -d' ' -f1)" = \
  202c1d3e6a3c07056ba084bb1bedb49b0730ea374f0d4eceb42241d224aa4569
test "$(sha256sum "$HARD_COHORT" | cut -d' ' -f1)" = \
  a87efa8e42a6a27686c0bba177a09fe5f778af1f47ef0901143f1a645aad63f7
test "$(sha256sum "$HIDDEN_V3" | cut -d' ' -f1)" = \
  9efa72720ec2d31add7745d4c02c6e3a382725c37cbdcc8e400773c0bc86b190
test "$(sha256sum "$HARD_V3" | cut -d' ' -f1)" = \
  b47c2624f98cc0345fd7ebd6c751ddf4ec1969ff491b893f7f448b554ac855c5

"$KGPW_PYTHON" -m pytest -q \
  tests/test_validate_sft_sampling.py \
  tests/test_passage_aware_kg_v2.py

if [[ "${1:-}" == "--preflight-only" ]]; then
  echo "preflight complete: hashes, checkpoints, cohorts, reused arms, and tests passed"
  exit 0
fi

mkdir "$OUTPUT_DIR"
exec > >(tee "$LOG_PATH") 2>&1

run_hidden() {
  local name=$1
  local adapter=$2
  "$KGPW_PYTHON" scripts/eval/validate_sft.py \
    --adapter "$adapter" \
    --silver "$SILVER" \
    --split val \
    --selection_jsonl "$HIDDEN_COHORT" \
    --input_overrides "$HIDDEN_V3" \
    --max_new_tokens 512 \
    --seed 42 \
    --out "$OUTPUT_DIR/$name.jsonl"
}

run_hard() {
  local name=$1
  local adapter=$2
  "$KGPW_PYTHON" scripts/eval/validate_sft.py \
    --adapter "$adapter" \
    --silver "$SILVER" \
    --split train \
    --allow_train_diagnostic \
    --selection_jsonl "$HARD_COHORT" \
    --input_overrides "$HARD_V3" \
    --max_new_tokens 512 \
    --seed 42 \
    --out "$OUTPUT_DIR/$name.jsonl"
}

run_hidden hidden_sft_v3 "$SFT"
run_hidden hidden_ppo_v3 "$PPO"
run_hard hard_sft_v3 "$SFT"
run_hard hard_ppo_v3 "$PPO"

"$KGPW_PYTHON" scripts/pilot/score_kg_v3_zero_train_eval.py \
  --hidden_sft_old "$OLD_RUN/hidden_sft_old.jsonl" \
  --hidden_sft_v3 "$OUTPUT_DIR/hidden_sft_v3.jsonl" \
  --hidden_ppo_old "$OLD_RUN/hidden_ppo_old.jsonl" \
  --hidden_ppo_v3 "$OUTPUT_DIR/hidden_ppo_v3.jsonl" \
  --hard_sft_old "$OLD_RUN/hard_sft_old.jsonl" \
  --hard_sft_v3 "$OUTPUT_DIR/hard_sft_v3.jsonl" \
  --hard_ppo_old "$OLD_RUN/hard_ppo_old.jsonl" \
  --hard_ppo_v3 "$OUTPUT_DIR/hard_ppo_v3.jsonl" \
  --hidden_cohort "$HIDDEN_COHORT" \
  --hard_cohort "$HARD_COHORT" \
  --hard_structural_report "$HARD_STRUCTURAL" \
  --output "$OUTPUT_DIR/report.json" \
  --run_dir "$OUTPUT_DIR/report_run" \
  --experiment_id "$EXPERIMENT_ID"
