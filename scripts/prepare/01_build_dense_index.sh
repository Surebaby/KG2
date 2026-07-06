#!/usr/bin/env bash
# =============================================================================
# Step 1 — Build dense retrieval index (e5-base-v2 + FAISS Flat).
#
# Defaults are tuned for the RTX PRO 6000 Blackwell (96 GB): batch_size 1024,
# bf16/fp16 encoding. Override via env vars KGPW_E5_PATH, KGPW_INDEX_DIR,
# KGPW_BATCH_SIZE.
#
# Runtime estimate: ~6h for 15M documents on a single Pro 6000.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CORPUS_PATH="${KGPW_CORPUS_PATH:-${KGPW_INDEX_DIR:-${PROJECT_ROOT}/indexes}/corpus_flashrag.jsonl}"
SAVE_DIR="${KGPW_INDEX_DIR:-${PROJECT_ROOT}/indexes}"
MODEL_PATH="${KGPW_E5_PATH:-intfloat/e5-base-v2}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE="${KGPW_BATCH_SIZE:-1024}"

if [[ ! -f "${CORPUS_PATH}" ]]; then
  echo "ERROR: corpus not found at ${CORPUS_PATH}"
  echo "       Run scripts/prepare/00_convert_corpus.py first."
  exit 1
fi

echo "=============================================="
echo "  KG-ProWeight :: dense index build (e5)"
echo "=============================================="
echo "  Corpus     : ${CORPUS_PATH}"
echo "  Save dir   : ${SAVE_DIR}"
echo "  Model      : ${MODEL_PATH}"
echo "  GPU        : ${GPU_ID}"
echo "  Batch size : ${BATCH_SIZE}"
echo ""

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
python -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path "${MODEL_PATH}" \
    --corpus_path "${CORPUS_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --use_fp16 \
    --max_length 512 \
    --batch_size "${BATCH_SIZE}" \
    --pooling_method mean \
    --faiss_type Flat

echo ""
echo "Index built: ${SAVE_DIR}/e5_Flat.index"
