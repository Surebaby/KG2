#!/usr/bin/env bash
# IHR (LLM-as-judge, deepseek-v4-pro) for the ppo_r10_split eval, matching the
# baseline IHR protocol (n=50, seed=42, judge=deepseek-v4-pro).
#
#   bash launch_ihr_ppo_r10_split.sh
set -euo pipefail
cd "$(dirname "$0")"

# Load OPENAI_API_KEY / OPENAI_BASE_URL from the gitignored .env (never printed).
set -a
source .env
set +a

export PYTHONPATH="/home/zjulab/kgpaper/flashrag_src:/home/zjulab/kgpaper"

out_root="/home/zjulab/kgpaper/outputs/ppo_r10_split_eval_ihr"
mkdir -p "$out_root"

declare -A PREDS=(
  [hotpotqa]="/home/zjulab/kgpaper/outputs/ppo_r10_split_eval/hotpotqa/seed_42/hotpotqa_2026_08_25_13_50_kg_proweight/intermediate_data.json"
  [2wikimultihopqa]="/home/zjulab/kgpaper/outputs/ppo_r10_split_eval/2wikimultihopqa/seed_42/2wikimultihopqa_2026_08_25_14_14_kg_proweight/intermediate_data.json"
  [musique]="/home/zjulab/kgpaper/outputs/ppo_r10_split_eval/musique/seed_42/musique_2026_08_25_14_38_kg_proweight/intermediate_data.json"
)

for ds in hotpotqa 2wikimultihopqa musique; do
  pred="${PREDS[$ds]}"
  if [ ! -f "$pred" ]; then
    echo "SKIP $ds: missing $pred"
    continue
  fi
  echo "=== IHR ppo_r10_split -> $ds ==="
  python scripts/eval/run_ihr_judge.py \
    --predictions "$pred" \
    --judge_model deepseek-v4-pro \
    --sample 50 \
    --seed 42 \
    --output "$out_root/${ds}_ihr.json"
done

echo "ALL PPO_R10_SPLIT IHR DONE"
