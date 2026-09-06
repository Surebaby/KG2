#!/usr/bin/env bash
# Full pipeline EM/F1 for the hybrid PPO smoke checkpoint, under the SAME
# conditions as the baseline table: two-stage retrieval (dense E5@100 + sparse
# BM25@100 -> RRF k=60 -> top-50 -> bge-reranker -> top-10), seed=42, n=300, temp=0.
#
#   bash launch_hybrid_pipeline_eval.sh
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="/home/zjulab/kgpaper/flashrag_src:/home/zjulab/kgpaper"
export KGPW_CORPUS_PATH="/home/zjulab/kgpaper/indexes_wiki18/corpus_flashrag.jsonl"
export KGPW_DENSE_INDEX_PATH="/home/zjulab/kgpaper/indexes_wiki18/e5_fp16.dat"
export KGPW_BM25_INDEX_PATH="/home/zjulab/kgpaper/indexes_wiki18/bm25"
export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

exec python scripts/eval/run_kg_proweight.py \
  --checkpoint /home/zjulab/kgpaper/outputs/ppo_legacy_repaired_v2_quota70_hard_seed42_no_text_head_smoke600_hybrid_old10_bridge5_v3/final \
  --alpha_gate_path /home/zjulab/kgpaper/checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42_no_text_head/alpha_gate.pt \
  --datasets hotpotqa 2wikimultihopqa musique \
  --split dev \
  --seeds 42 \
  --test_sample_num 300 \
  --rerank 10 \
  --gpu_id 0 \
  --save_root /home/zjulab/kgpaper/outputs/hybrid_pipeline_eval
