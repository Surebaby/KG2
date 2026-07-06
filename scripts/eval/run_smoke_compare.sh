#!/usr/bin/env bash
# =============================================================================
# Smoke-test eval: base vs SFT vs PPO on the tiny HotpotQA distractor corpus.
# Run on the SERVER in GPU mode. Produces EM/F1 for the three models so we can
# see whether PPO improved over SFT (the flat-reward question).
#
# Usage (on server):
#   bash scripts/eval/run_smoke_compare.sh [N_SAMPLES] [SEED]
#   N_SAMPLES default 100, SEED default 42.
# =============================================================================
set -uo pipefail

ROOT=/root/autodl-tmp/kgpaper
PY=/root/autodl-tmp/kgpw_env/bin/python
N="${1:-100}"
SEED="${2:-42}"

# Offline everything: HF unreachable from CN; KG offline = cache-only (no SPARQL).
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export KGPW_LLAMA3_PATH=/root/autodl-tmp/models/llama3-8b
export KGPW_INDEX_DIR=$ROOT/indexes_smoke
export KGPW_E5_PATH=/root/autodl-tmp/models/e5-base-v2
export KGPW_KG_OFFLINE=1
export ENTITY_CACHE=$ROOT/indexes/entity_cache.jsonl
export KG_CACHE_DIR=$ROOT/indexes/kg_cache   # holds kg_subgraph_cache.jsonl (96% coverage)

cd "$ROOT"

run_one () {  # $1=label  $2=checkpoint-arg  $3=save_root
  echo "=============================================="
  echo "  EVAL: $1   (ckpt=$2)"
  echo "=============================================="
  $PY scripts/eval/run_kg_proweight.py \
    --checkpoint "$2" \
    --datasets hotpotqa_smoke --seeds "$SEED" --test_sample_num "$N" \
    --entity_cache_path "$ENTITY_CACHE" \
    --kg_cache_dir "$KG_CACHE_DIR" \
    --save_root "$3" \
    2>&1 | grep -vE "huggingface_hub|MaxRetryError|Retrying|thrown while|faiss.loader|swigfaiss"
}

run_one "BASE (llama3-8b, no LoRA)" "none"                                   "$ROOT/outputs/smoke_base"
run_one "SFT"  "$ROOT/checkpoints/sft_student/final"                         "$ROOT/outputs/smoke_sft"
run_one "PPO"  "$ROOT/checkpoints/kg_proweight_final/final"                  "$ROOT/outputs/smoke_ppo_full"

echo ""
echo "=============================================="
echo "  RESULTS (EM / F1)"
echo "=============================================="
for m in smoke_base smoke_sft smoke_ppo_full; do
  f=$(find "$ROOT/outputs/$m" -name metric_score.txt 2>/dev/null | head -1)
  echo "--- $m ---"
  [ -n "$f" ] && cat "$f" || echo "  (no metric_score.txt — run may have failed)"
done
