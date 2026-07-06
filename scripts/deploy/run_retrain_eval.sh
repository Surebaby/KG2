#!/usr/bin/env bash
# =============================================================================
# Retrain SFT + PPO with the 2026-06-22 fixes, then smoke-eval base/SFT/PPO.
# Phase 2 (prm_alpha_gate) is UNCHANGED and reused. Run on the SERVER in GPU mode.
#
#   bash scripts/deploy/run_retrain_eval.sh
#
# Fixes being validated:
#  - PPO: SFT adapter now loaded TRAINABLE + frozen-SFT KL reference (was a no-op)
#  - SFT: prompt-masked loss + single BOS + answer-preserving truncation
#  - top_k 50->15 everywhere; PPO ppo_max_passages 5->15, max_input_length 6144
# =============================================================================
set -uo pipefail
cd /root/autodl-tmp/kgpaper
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/kgpw_env
set -a; source .env; set +a
export KGPW_FLASHRAG_ROOT=/root/autodl-tmp/kgpaper
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/root/autodl-tmp/kgpaper
CKPT=$ROOT/checkpoints
TS=$(date +%Y%m%d_%H%M%S)
stamp() { echo "[$(date '+%H:%M:%S')] $*"; }
die()   { stamp "FAILED: $*"; echo "PIPELINE_FAILED"; exit 1; }
need()  { [ -e "$1" ] || die "missing artifact: $1"; }

stamp "Verify Phase 2 artifacts (kept, not retrained)"
need "$CKPT/prm_alpha_gate/alpha_gate.pt"
need "$CKPT/prm_alpha_gate/silver_with_logprobs.jsonl"
need "$CKPT/prm_alpha_gate/prm_head"

stamp "Archiving old (broken) SFT + PPO checkpoints -> *.broken_$TS"
[ -d "$CKPT/sft_student" ]       && mv "$CKPT/sft_student"       "$CKPT/sft_student.broken_$TS"
[ -d "$CKPT/kg_proweight_final" ] && mv "$CKPT/kg_proweight_final" "$CKPT/kg_proweight_final.broken_$TS"

stamp "===== PHASE 3a: SFT (prompt-masked) ====="
python scripts/train/phase3_sft.py --config configs/training/phase3_sft.yaml || die "phase3_sft crashed"
need "$CKPT/sft_student/final/adapter_model.safetensors"
stamp "phase3_sft OK"

stamp "===== PHASE 3b: PPO (trainable SFT adapter + SFT ref) ====="
python scripts/train/phase3_ppo.py --config configs/training/phase3_ppo.yaml \
  --text_reward_backend rearag \
  --text_reward_fallback_path "$CKPT/prm_alpha_gate/prm_head" \
  || die "phase3_ppo crashed"
need "$CKPT/kg_proweight_final/final/adapter_model.safetensors"
stamp "phase3_ppo OK"

# Sanity: PPO adapter MUST differ from SFT (the bug we fixed produced identical bytes)
SFT_MD5=$(md5sum "$CKPT/sft_student/final/adapter_model.safetensors" | awk '{print $1}')
PPO_MD5=$(md5sum "$CKPT/kg_proweight_final/final/adapter_model.safetensors" | awk '{print $1}')
if [ "$SFT_MD5" = "$PPO_MD5" ]; then
  die "PPO adapter is byte-identical to SFT — PPO STILL a no-op! (MD5 $PPO_MD5)"
fi
stamp "PPO adapter differs from SFT (good): SFT=$SFT_MD5 PPO=$PPO_MD5"

stamp "===== EVAL: base / SFT / PPO (top_k=15 smoke corpus) ====="
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export KGPW_INDEX_DIR=$ROOT/indexes_smoke
export KGPW_E5_PATH=/root/autodl-tmp/models/e5-base-v2
export KGPW_KG_OFFLINE=1
EC=$ROOT/indexes/entity_cache.jsonl
KGC=$ROOT/indexes/kg_cache
eval_one() {
  python scripts/eval/run_kg_proweight.py --checkpoint "$2" \
    --datasets hotpotqa_smoke --seeds 42 --test_sample_num 100 \
    --entity_cache_path "$EC" --kg_cache_dir "$KGC" --save_root "$3" \
    2>&1 | grep -vE "huggingface_hub|MaxRetryError|Retrying|thrown while|faiss.loader|swigfaiss"
  stamp "eval $1 done"
}
eval_one BASE "none"                        "$ROOT/outputs/re_base"
eval_one SFT  "$CKPT/sft_student/final"     "$ROOT/outputs/re_sft"
eval_one PPO  "$CKPT/kg_proweight_final/final" "$ROOT/outputs/re_ppo"

echo ""
echo "================ RESULTS (EM / F1) ================"
for m in re_base re_sft re_ppo; do
  f=$(find "$ROOT/outputs/$m" -name metric_score.txt 2>/dev/null | head -1)
  echo "--- $m ---"; [ -n "$f" ] && cat "$f" || echo "  (missing)"
done
stamp "===== PIPELINE_DONE ====="
