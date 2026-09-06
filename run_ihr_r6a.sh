#!/bin/bash
set -euo pipefail
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in the environment; never store it in this script}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.deepseek.com}"
cd /root/autodl-tmp/kgpaper
/root/autodl-tmp/kgpw_env/bin/python -u scripts/eval/run_ihr_judge.py \
  --predictions outputs/eval_r6a/hotpotqa_smoke/seed_42/hotpotqa_smoke_2026_06_30_12_58_kg_proweight/intermediate_data.json \
  --sample 50 --seed 42 --judge_model deepseek-v4-pro \
  --output outputs/eval_r6a/hotpotqa_smoke/seed_42/ihr_result.json
