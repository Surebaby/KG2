#!/usr/bin/env bash
# KG-ProWeight — full Phase 2/3 training pipeline, fail-fast with artifact checks.
# Launched via nohup so it survives SSH disconnects. Logs to train_all.log.
set -uo pipefail

cd /root/autodl-tmp/kgpaper
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/kgpw_env
set -a; source .env; set +a
export KGPW_FLASHRAG_ROOT=/root/autodl-tmp/kgpaper
# Fight CUDA fragmentation (Phase 2 OOM'd at ~step 1000 from fragmentation on
# variable-length batches; the allocator suggested this flag explicitly).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT=/root/autodl-tmp/kgpaper/checkpoints
LOG=/root/autodl-tmp/train_all.log

stamp() { echo "[$(date '+%H:%M:%S')] $*"; }
die()   { stamp "FAILED: $*"; echo "PIPELINE_FAILED" ; exit 1; }
need()  { [ -e "$1" ] || die "missing artifact: $1"; }

stamp "===== PHASE 2: PRM + alpha-gate + text-reward head ====="
python scripts/train/phase2_train_prm.py --config configs/training/phase2_prm.yaml || die "phase2 crashed"
need "$CKPT/prm_alpha_gate/alpha_gate.pt"
need "$CKPT/prm_alpha_gate/silver_with_logprobs.jsonl"
need "$CKPT/prm_alpha_gate/prm_head"
stamp "phase2 OK"

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

stamp "===== PIPELINE_DONE — all phases complete ====="
