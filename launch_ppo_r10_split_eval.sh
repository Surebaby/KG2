#!/usr/bin/env bash
# Evaluate the PPO checkpoint `ppo_r10_split/final` under the SAME conditions
# as the baseline table (baseline_results.md): two-stage retrieval
# (dense E5@100 + sparse BM25@100 -> RRF k=60 -> top-50 -> bge-reranker -> top-10),
# seed=42, n=300, temperature=0. Outputs land under outputs/ppo_r10_split_eval.
#
# Alpha gate matches what ppo_r10_split was trained with
# (prm_alpha_gate_v1reann_negfix), NOT the (dangling) default prm_alpha_gate.
# The full wiki18 corpus/dense/bm25 are in indexes_wiki18/ — the indexes/ dir
# symlinks point at the SMOKE corpus, so we override them explicitly (same as
# the baseline launchers do).
#
#   bash launch_ppo_r10_split_eval.sh
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
  --checkpoint /home/zjulab/kgpaper/checkpoints/ppo_r10_split/final \
  --alpha_gate_path /home/zjulab/kgpaper/checkpoints/prm_alpha_gate_v1reann_negfix/alpha_gate.pt \
  --datasets hotpotqa 2wikimultihopqa musique \
  --split dev \
  --seeds 42 \
  --test_sample_num 300 \
  --rerank 10 \
  --gpu_id 0 \
  --save_root /home/zjulab/kgpaper/outputs/ppo_r10_split_eval
