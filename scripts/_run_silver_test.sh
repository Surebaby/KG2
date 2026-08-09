#!/bin/bash
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:?set it in .env, never hardcode: git history scrubbed 2026-08-23}"
source /home/zjulab/anaconda3/bin/activate kgpaper
# PIPELINE SMOKE TEST ONLY — --allow_eval_split means the output is generated
# from the eval split and MUST NOT be used for SFT/PPO training. For real silver
# data use --split train (run scripts/prepare/03_download_datasets.py first;
# train.jsonl is not present locally yet).
KGPW_FLASHRAG_ROOT=/home/zjulab/kgpaper/flashrag_src python scripts/train/phase1_generate_silver.py \
  --config configs/training/phase1_silver.yaml \
  --dataset hotpotqa \
  --split dev \
  --allow_eval_split \
  --max_queries 50 \
  --rerank 10 \
  --offline on \
  --output /home/zjulab/kgpaper/data/silver_data/_smoketest_DO_NOT_TRAIN.jsonl \
  --seed 42 \
  2>&1 | tail -20
