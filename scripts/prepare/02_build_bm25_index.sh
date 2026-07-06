#!/usr/bin/env bash
# =============================================================================
# Step 2 — Build BM25s sparse index.
# Runtime ~1h on a modern CPU; ~64 GB RAM recommended.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CORPUS_PATH="${KGPW_CORPUS_PATH:-${KGPW_INDEX_DIR:-${PROJECT_ROOT}/indexes}/corpus_flashrag.jsonl}"
SAVE_DIR="${KGPW_INDEX_DIR:-${PROJECT_ROOT}/indexes}"

if [[ ! -f "${CORPUS_PATH}" ]]; then
  echo "ERROR: corpus not found at ${CORPUS_PATH}"
  exit 1
fi

echo "=============================================="
echo "  KG-ProWeight :: BM25s sparse index"
echo "=============================================="
echo "  Corpus   : ${CORPUS_PATH}"
echo "  Save dir : ${SAVE_DIR}"
echo ""

python -m flashrag.retriever.index_builder \
    --retrieval_method bm25 \
    --corpus_path "${CORPUS_PATH}" \
    --bm25_backend bm25s \
    --save_dir "${SAVE_DIR}"

echo ""
echo "BM25 index ready at ${SAVE_DIR}/bm25/"
