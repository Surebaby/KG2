#!/usr/bin/env bash
# Baseline EM/F1 eval — 7 methods × 1 seed (42) × n=300.
#
# Datasets are positional args (default: all three).  Stage the eval:
#   bash launch_baselines.sh hotpotqa                       # 1 dataset first
#   bash launch_baselines.sh 2wikimultihopqa musique        # then the rest
#
# Retrieval indices MUST point at the real 21M wiki18 corpus (indexes_wiki18/),
# NOT the 989-vector smoke placeholder under indexes/.  The dense index is the
# fp16 memmap (e5_fp16.dat, auto-detected via .dat → MemmapSearch), BM25 was
# built by scripts/prepare/02_build_bm25_index.sh.
#
# History: the first full run (outputs/baselines_final_run.log) silently used
# the smoke index and CPU-offloaded the generator on the 2nd+ dataset (~320s/it).
# Both fixed: KGPW_* env below + empty_cache in kgproweight/eval/runner.py.
set -euo pipefail

cd "$(dirname "$0")"

DATASETS=("${@:-hotpotqa 2wikimultihopqa musique}")

export PYTHONPATH="/home/zjulab/kgpaper/flashrag_src:/home/zjulab/kgpaper"
export KGPW_CORPUS_PATH="/home/zjulab/kgpaper/indexes_wiki18/corpus_flashrag.jsonl"
export KGPW_DENSE_INDEX_PATH="/home/zjulab/kgpaper/indexes_wiki18/e5_fp16.dat"
export KGPW_BM25_INDEX_PATH="/home/zjulab/kgpaper/indexes_wiki18/bm25"
export CUDA_VISIBLE_DEVICES="0"

exec python scripts/eval/run_baselines.py \
  --methods zero_shot naive_rag self_rag trace r1_searcher corag rearag \
  --datasets "${DATASETS[@]}" \
  --seeds 42 \
  --test_sample_num 300
