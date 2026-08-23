#!/usr/bin/env bash
# Baseline EM/F1 rerun with two-stage retrieval (top-50 → bge-reranker → top-10).
#
# The Stage-2 numbers used the legacy single-stage top-15 (no reranker), while the
# main method uses top-50 → cross-encoder → 10 — an unfair comparison. This reruns
# the 5 retrieval baselines (naive_rag / self_rag / trace / r1_searcher / corag) ×
# 3 datasets with the SAME two-stage retrieval as the main method. zero_shot has no
# retrieval and is unaffected; rearag is tracked separately (hotpotqa only).
#
#   bash launch_baselines_rerank.sh                         # all 3 datasets, 15 combos
#   bash launch_baselines_rerank.sh hotpotqa                # stage one dataset first
#
# Indices MUST point at the real 21M wiki18 corpus (indexes_wiki18/), NOT the
# 989-vector smoke placeholder under indexes/.
set -euo pipefail

cd "$(dirname "$0")"

DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=(hotpotqa 2wikimultihopqa musique)
fi

export PYTHONPATH="/home/zjulab/kgpaper/flashrag_src:/home/zjulab/kgpaper"
export KGPW_CORPUS_PATH="/home/zjulab/kgpaper/indexes_wiki18/corpus_flashrag.jsonl"
export KGPW_DENSE_INDEX_PATH="/home/zjulab/kgpaper/indexes_wiki18/e5_fp16.dat"
export KGPW_BM25_INDEX_PATH="/home/zjulab/kgpaper/indexes_wiki18/bm25"
export CUDA_VISIBLE_DEVICES="0"

exec python scripts/eval/run_baselines.py \
  --methods naive_rag self_rag trace r1_searcher corag \
  --datasets "${DATASETS[@]}" \
  --seeds 42 \
  --test_sample_num 300 \
  --retrieval_topk 50 \
  --rerank 10 \
  --save_root outputs/baselines_rerank
