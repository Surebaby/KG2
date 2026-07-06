#!/usr/bin/env bash
# Run ONLY Phase 3b PPO — Phase 2 + SFT artifacts already exist.
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

stamp "Verifying upstream artifacts"
need "$CKPT/prm_alpha_gate/alpha_gate.pt"
need "$CKPT/prm_alpha_gate/prm_head"
need "$CKPT/sft_student/final"
stamp "upstream OK"

stamp "===== PHASE 3b: PPO (KG-ProWeight) ====="
python scripts/train/phase3_ppo.py --config configs/training/phase3_ppo.yaml \
  --text_reward_backend rearag \
  --text_reward_fallback_path "$CKPT/prm_alpha_gate/prm_head" \
  || die "phase3_ppo crashed"
need "$CKPT/kg_proweight_final"
stamp "===== PIPELINE_DONE — PPO complete ====="
