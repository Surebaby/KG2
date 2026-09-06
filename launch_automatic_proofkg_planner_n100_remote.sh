#!/usr/bin/env bash
# Gold-free planner generation for the frozen automatic Proof-KG n=100 cohort.
set -euo pipefail

ROOT=${KGPW_ROOT:-/root/autodl-tmp/kgpaper}
PYTHON=${KGPW_PYTHON:-/root/autodl-tmp/kgpw_env/bin/python}
COHORT=outputs/audits/automatic_proofkg_2wiki_unseen_n100_seed20260830_preregistration/cohort.question_only.jsonl
PROTOCOL=outputs/audits/automatic_proofkg_2wiki_unseen_n100_seed20260830_preregistration/protocol.json
ADAPTER=checkpoints/query_planner_learned_scale_v1_1_seed42/final
OUTPUT=outputs/validation/automatic_proofkg_2wiki_unseen_n100_seed20260830_plans
LOG=logs/validation/automatic_proofkg_2wiki_unseen_n100_seed20260830_plans.log

cd "$ROOT"
test -x "$PYTHON"
test "$(sha256sum "$COHORT" | cut -d' ' -f1)" = \
  ddbd751f332a99430a4c58559fb2e9083614f0474419e133e25961f1824da35a
test "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" = \
  afe7d3ed95e730bb475409a8621d8ada490c9d3cc1c16b3a974ca8f9abde4f7c
test "$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)" = \
  0bd41d01140b00413c7d8a908d7d4482c4d955dd772072f2f3c74c8fe1c2c776
test ! -e "$OUTPUT"
test ! -e "$LOG"
mkdir -p logs/validation

export PYTHONPATH="$ROOT:$ROOT/flashrag_src"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
"$PYTHON" scripts/eval/generate_query_plans_unseen.py \
  --cohort "$COHORT" \
  --adapter "$ADAPTER" \
  --config configs/training/query_planner_learned_scale_v1_1_seed42.yaml \
  --protocol "$PROTOCOL" \
  --output_dir "$OUTPUT" \
  --experiment_id AUTOMATIC-PROOFKG-2WIKI-UNSEEN-N100-PLANS-SEED20260830 \
  --batch_size 4 \
  --max_new_tokens 512 \
  2>&1 | tee "$LOG"
