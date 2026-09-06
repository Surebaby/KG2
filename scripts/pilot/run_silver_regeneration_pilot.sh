#!/usr/bin/env bash
# Paired Phase-1 regeneration pilot: 100 deterministic train questions per
# dataset by default. This spends Teacher API credit but does not train a model.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-/home/zjulab/anaconda3/envs/kgpaper/bin/python}
N_PER_DATASET=${N_PER_DATASET:-100}
SEED=${SEED:-42}
OFFLINE_MODE=${OFFLINE_MODE:-on}
STAMP=$(date +%Y%m%d_%H%M%S)
EXP_ID="silver_regen_pilot_n${N_PER_DATASET}_seed${SEED}_${STAMP}"
OUTDIR="$ROOT/data/silver_data/pilots/$EXP_ID"
LOGDIR="$OUTDIR/logs"
mkdir -p "$LOGDIR"

if [ -z "${DEEPSEEK_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "FATAL: source the API environment first (DEEPSEEK_API_KEY or OPENAI_API_KEY)." >&2
  exit 2
fi
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  export DEEPSEEK_API_KEY="$OPENAI_API_KEY"
fi

export KGPW_FLASHRAG_ROOT=${KGPW_FLASHRAG_ROOT:-$ROOT/flashrag_src}
# Use the semantically identical BM25 corpus copy because it already has the
# 21,015,324-row `corpus.mmindex.json` needed for low-RSS random access.
export KGPW_CORPUS_PATH=${KGPW_CORPUS_PATH:-$ROOT/indexes_wiki18/bm25/corpus.jsonl}
export KGPW_DENSE_INDEX_PATH=${KGPW_DENSE_INDEX_PATH:-$ROOT/indexes_wiki18/e5_fp16.dat}
export KGPW_BM25_INDEX_PATH=${KGPW_BM25_INDEX_PATH:-$ROOT/indexes_wiki18/bm25}
export KGPW_CORPUS_MMAP=1
export KGPW_REQUIRE_BATCH_RETRIEVAL=1
for p in "$KGPW_CORPUS_PATH" "$KGPW_DENSE_INDEX_PATH" "$KGPW_BM25_INDEX_PATH"; do
  [ -e "$p" ] || { echo "FATAL: missing retrieval asset $p" >&2; exit 2; }
done
"$PYTHON_BIN" scripts/prepare/check_wiki18_assets.py \
  --corpus "$KGPW_CORPUS_PATH" \
  --dense "$KGPW_DENSE_INDEX_PATH" \
  --bm25 "$KGPW_BM25_INDEX_PATH" \
  --output "$OUTDIR/wiki18_asset_preflight.json"

run_one() {
  local dataset=$1
  local output=$2
  local log=$3
  "$PYTHON_BIN" scripts/train/phase1_generate_silver.py \
    --config configs/training/phase1_silver_pilot.yaml \
    --dataset "$dataset" \
    --split train \
    --max_queries "$N_PER_DATASET" \
    --sample_strategy random \
    --rerank 10 \
    --offline "$OFFLINE_MODE" \
    --output "$output" \
    --seed "$SEED" \
    > "$log" 2>&1 || {
      tail -80 "$log"
      exit 1
    }
  tail -5 "$log"
  local n
  n=$(wc -l < "$output")
  [ "$n" -eq "$N_PER_DATASET" ] || {
    echo "FATAL: $dataset wrote $n/$N_PER_DATASET records; failed/empty Teacher items make the sample biased." >&2
    exit 1
  }
}

run_one hotpotqa "$OUTDIR/hotpotqa.jsonl" "$LOGDIR/hotpotqa.log"
run_one 2wikimultihopqa "$OUTDIR/2wikimultihopqa.jsonl" "$LOGDIR/2wikimultihopqa.log"
run_one musique "$OUTDIR/musique.jsonl" "$LOGDIR/musique.log"

"$PYTHON_BIN" scripts/pilot/score_silver_regeneration.py \
  --pilot "$OUTDIR/hotpotqa.jsonl" \
  --pilot "$OUTDIR/2wikimultihopqa.jsonl" \
  --pilot "$OUTDIR/musique.jsonl" \
  --expected_per_dataset "$N_PER_DATASET" \
  --baseline data/silver_data/silver_v1_reannotated.jsonl \
  --output "$OUTDIR/quality_report.json"

echo "PILOT_COMPLETE experiment_id=$EXP_ID report=$OUTDIR/quality_report.json"
