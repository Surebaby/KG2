#!/usr/bin/env bash
# Paired control/additive-v3 Phase-1 Teacher pilot. This spends Teacher API
# credit but does not train or overwrite any existing silver dataset.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-/home/zjulab/anaconda3/envs/kgpaper/bin/python}
N_PER_DATASET=${N_PER_DATASET:-30}
SEED=${SEED:-45}
OFFLINE_MODE=${OFFLINE_MODE:-on}
STAMP=$(date +%Y%m%d_%H%M%S)
EXP_ID="silver_bridge_teacher_paired_n${N_PER_DATASET}_seed${SEED}_${STAMP}"
OUTDIR="$ROOT/data/silver_data/pilots/$EXP_ID"
LOGDIR="$OUTDIR/logs"
mkdir -p "$LOGDIR" "$OUTDIR/control" "$OUTDIR/additive_v3"

on_failure() {
  local exit_code=$?
  printf 'FAILED exit_code=%s timestamp=%s\n' "$exit_code" "$(date --iso-8601=seconds)" > "$OUTDIR/status.failed"
  printf 'PILOT_FAILED experiment_id=%s outputs_preserved=%s\n' "$EXP_ID" "$OUTDIR" >&2
  exit "$exit_code"
}
trap on_failure ERR

if [ -z "${DEEPSEEK_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "FATAL: source the API environment first (DEEPSEEK_API_KEY or OPENAI_API_KEY)." >&2
  exit 2
fi
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  export DEEPSEEK_API_KEY="$OPENAI_API_KEY"
fi

export KGPW_FLASHRAG_ROOT=${KGPW_FLASHRAG_ROOT:-$ROOT/flashrag_src}
export PYTHONPATH="$ROOT:$ROOT/flashrag_src${PYTHONPATH:+:$PYTHONPATH}"
export KGPW_CORPUS_PATH=${KGPW_CORPUS_PATH:-$ROOT/indexes_wiki18/bm25/corpus.jsonl}
export KGPW_DENSE_INDEX_PATH=${KGPW_DENSE_INDEX_PATH:-$ROOT/indexes_wiki18/e5_fp16.dat}
export KGPW_BM25_INDEX_PATH=${KGPW_BM25_INDEX_PATH:-$ROOT/indexes_wiki18/bm25}
export KGPW_CORPUS_MMAP=1
export KGPW_REQUIRE_BATCH_RETRIEVAL=1
for path in "$KGPW_CORPUS_PATH" "$KGPW_DENSE_INDEX_PATH" "$KGPW_BM25_INDEX_PATH"; do
  [ -e "$path" ] || { echo "FATAL: missing retrieval asset $path" >&2; exit 2; }
done

"$PYTHON_BIN" scripts/prepare/check_wiki18_assets.py \
  --corpus "$KGPW_CORPUS_PATH" \
  --dense "$KGPW_DENSE_INDEX_PATH" \
  --bm25 "$KGPW_BM25_INDEX_PATH" \
  --output "$OUTDIR/wiki18_asset_preflight.json"

run_one() {
  local arm=$1
  local bridge_mode=$2
  local dataset=$3
  local output="$OUTDIR/$arm/$dataset.jsonl"
  local log="$LOGDIR/${arm}_${dataset}.log"
  "$PYTHON_BIN" scripts/train/phase1_generate_silver.py \
    --config configs/training/phase1_silver_bridge_paired_pilot.yaml \
    --dataset "$dataset" \
    --split train \
    --max_queries "$N_PER_DATASET" \
    --sample_strategy random \
    --rerank 10 \
    --offline "$OFFLINE_MODE" \
    --output "$output" \
    --seed "$SEED" \
    --bridge_mode "$bridge_mode" \
    > "$log" 2>&1
  tail -5 "$log"
  local candidate="${output%.jsonl}.candidates.jsonl"
  local candidate_count output_count
  candidate_count=$(wc -l < "$candidate")
  output_count=$(wc -l < "$output")
  [ "$candidate_count" -eq "$N_PER_DATASET" ] || {
    echo "FATAL: $arm/$dataset wrote $candidate_count/$N_PER_DATASET candidates." >&2
    return 1
  }
  [ "$output_count" -eq "$N_PER_DATASET" ] || {
    echo "FATAL: $arm/$dataset finalized $output_count/$N_PER_DATASET rows." >&2
    return 1
  }
}

for dataset in hotpotqa 2wikimultihopqa musique; do
  run_one control off "$dataset"
  run_one additive_v3 additive_v3 "$dataset"
done

"$PYTHON_BIN" scripts/pilot/score_paired_bridge_teacher_pilot.py \
  --root "$OUTDIR" \
  --expected_per_dataset "$N_PER_DATASET" \
  --seed "$SEED" \
  --output "$OUTDIR/quality_report.json" \
  > "$LOGDIR/scoring.log" 2>&1

printf 'COMPLETE timestamp=%s\n' "$(date --iso-8601=seconds)" > "$OUTDIR/status.completed"
trap - ERR
echo "PILOT_COMPLETE experiment_id=$EXP_ID report=$OUTDIR/quality_report.json"
