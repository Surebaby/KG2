#!/usr/bin/env bash
# Re-run IHR (LLM-as-judge) for the two reasoning baselines that emit extractable
# steps: trace (IRCOT) and r1_searcher, across 3 datasets, on the rerank-10 outputs.
#
# Judge = deepseek-v4-pro (user-chosen 2026-08-20), endpoint api.deepseek.com, key
# OPENAI_API_KEY in .env. Prior IHR results used deepseek-chat and are NOT comparable.
#
#   bash launch_ihr_rerank.sh
set -euo pipefail
cd "$(dirname "$0")"

# Load OPENAI_API_KEY / OPENAI_BASE_URL from the gitignored .env (never printed).
set -a
source .env
set +a

export PYTHONPATH="/home/zjulab/kgpaper/flashrag_src:/home/zjulab/kgpaper"

declare -a JOBS=(
  "ircot       outputs/baselines_rerank/trace/hotpotqa/seed_42/hotpotqa_2026_08_21_00_50_trace/intermediate_data.json"
  "ircot       outputs/baselines_rerank/trace/2wikimultihopqa/seed_42/2wikimultihopqa_2026_08_21_05_17_trace/intermediate_data.json"
  "ircot       outputs/baselines_rerank/trace/musique/seed_42/musique_2026_08_21_05_55_trace/intermediate_data.json"
  "r1_searcher outputs/baselines_rerank/r1_searcher/hotpotqa/seed_42/hotpotqa_2026_08_21_01_23_r1_searcher/intermediate_data.json"
  "r1_searcher outputs/baselines_rerank/r1_searcher/2wikimultihopqa/seed_42/2wikimultihopqa_2026_08_21_06_32_r1_searcher/intermediate_data.json"
  "r1_searcher outputs/baselines_rerank/r1_searcher/musique/seed_42/musique_2026_08_21_08_16_r1_searcher/intermediate_data.json"
)

for job in "${JOBS[@]}"; do
  read -r method pred <<< "$job"
  echo "=== IHR $method -> $pred ==="
  python scripts/eval/run_baseline_ihr.py \
    --predictions "$pred" \
    --method "$method" \
    --sample 50 \
    --seed 42 \
    --judge_model deepseek-v4-pro
done

echo "ALL IHR DONE"
