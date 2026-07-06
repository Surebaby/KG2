#!/usr/bin/env bash
# Resume from Phase 3a — Phase 2 artifacts already exist; skip the ~55min re-run.
set -uo pipefail
cd /root/autodl-tmp/kgpaper
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/kgpw_env
set -a; source .env; set +a
export KGPW_FLASHRAG_ROOT=/root/autodl-tmp/kgpaper
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT=/root/autodl-tmp/kgpaper/checkpoints
stamp() { echo "[$(date '+%H:%M:%S')] $*"; }
die()   { stamp "FAILED: $*"; echo "PIPELINE_FAILED"; exit 1; }
need()  { [ -e "$1" ] || die "missing artifact: $1"; }

stamp "Verifying Phase 2 artifacts (no re-run)"
need "$CKPT/prm_alpha_gate/alpha_gate.pt"
need "$CKPT/prm_alpha_gate/silver_with_logprobs.jsonl"
need "$CKPT/prm_alpha_gate/prm_head"
stamp "phase2 artifacts OK"

stamp "===== PHASE 3a: SFT student ====="
python scripts/train/phase3_sft.py --config configs/training/phase3_sft.yaml || die "phase3_sft crashed"
need "$CKPT/sft_student/final"
stamp "phase3_sft OK"

stamp "===== PHASE 3b: PPO (KG-ProWeight) ====="
python scripts/train/phase3_ppo.py --config configs/training/phase3_ppo.yaml \
  --text_reward_backend rearag \
  --text_reward_fallback_path "$CKPT/prm_alpha_gate/prm_head" \
  || die "phase3_ppo crashed"
need "$CKPT/kg_proweight_final"
stamp "phase3_ppo OK"

stamp "===== PIPELINE_DONE — SFT + PPO complete ====="
