#!/usr/bin/env bash
# ReaRAG EM/F1 rerun for the two remaining datasets (hotpotqa already done),
# with the SAME two-stage retrieval (top-50 -> bge-reranker -> top-10) and the
# SAME OOM guard (expandable_segments) that let hotpotqa clear its earlier
# failure point. Run only AFTER hotpotqa has finished and freed the GPU.
#
#   bash launch_rearag_2wiki_musique.sh
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="/home/zjulab/kgpaper/flashrag_src:/home/zjulab/kgpaper"
export KGPW_CORPUS_PATH="/home/zjulab/kgpaper/indexes_wiki18/corpus_flashrag.jsonl"
export KGPW_DENSE_INDEX_PATH="/home/zjulab/kgpaper/indexes_wiki18/e5_fp16.dat"
export KGPW_BM25_INDEX_PATH="/home/zjulab/kgpaper/indexes_wiki18/bm25"
export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

exec python scripts/eval/run_baselines.py \
  --methods rearag \
  --datasets 2wikimultihopqa musique \
  --seeds 42 \
  --test_sample_num 300 \
  --retrieval_topk 50 \
  --rerank 10 \
  --save_root outputs/baselines_rerank
