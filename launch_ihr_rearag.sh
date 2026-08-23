#!/usr/bin/env bash
# ReaRAG IHR (LLM-as-judge, deepseek-v4-pro) across all 3 datasets, on the
# rerank-10 outputs. ReaRAG steps = "Thought N" assistant turns (extract_rearag_steps).
# The --predictions paths below are globs resolved to the latest timestamp dir for
# each dataset; run only after each dataset's EM/F1 rerun has written intermediate_data.json.
#
#   bash launch_ihr_rearag.sh
set -euo pipefail
cd "$(dirname "$0")"

# Load OPENAI_API_KEY / OPENAI_BASE_URL from the gitignored .env (never printed).
set -a
source .env
set +a

export PYTHONPATH="/home/zjulab/kgpaper/flashrag_src:/home/zjulab/kgpaper"

for ds in hotpotqa 2wikimultihopqa musique; do
  pred=$(ls -1t outputs/baselines_rerank/rearag/$ds/seed_42/*/intermediate_data.json 2>/dev/null | head -1)
  if [ -z "$pred" ]; then
    echo "SKIP $ds: no intermediate_data.json yet"
    continue
  fi
  echo "=== IHR rearag -> $pred ==="
  python scripts/eval/run_baseline_ihr.py \
    --predictions "$pred" \
    --method rearag \
    --sample 50 \
    --seed 42 \
    --judge_model deepseek-v4-pro
done

echo "ALL REARAG IHR DONE"
