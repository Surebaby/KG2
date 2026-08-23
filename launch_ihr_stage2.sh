#!/bin/bash
set -a
source /home/zjulab/kgpaper/.env
set +a
export PYTHONPATH="/home/zjulab/kgpaper/flashrag_src:/home/zjulab/kgpaper"

python scripts/eval/run_baseline_ihr.py \
  --predictions /home/zjulab/kgpaper/outputs/baselines_stage2/trace/2wikimultihopqa/seed_42/2wikimultihopqa_2026_08_20_19_32_trace/intermediate_data.json \
  --method ircot --sample 50 --seed 42 --judge_model deepseek-v4-pro \
  --output /home/zjulab/kgpaper/outputs/baselines_stage2/trace/2wikimultihopqa/seed_42/2wikimultihopqa_2026_08_20_19_32_trace/ihr_result_ircot.json

python scripts/eval/run_baseline_ihr.py \
  --predictions /home/zjulab/kgpaper/outputs/baselines_stage2/trace/musique/seed_42/musique_2026_08_20_20_14_trace/intermediate_data.json \
  --method ircot --sample 50 --seed 42 --judge_model deepseek-v4-pro \
  --output /home/zjulab/kgpaper/outputs/baselines_stage2/trace/musique/seed_42/musique_2026_08_20_20_14_trace/ihr_result_ircot.json

echo "IHR_STAGE2_DONE $(date +%H:%M:%S)"
