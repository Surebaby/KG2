#!/usr/bin/env bash
# Route A connectivity smoke: run Phase 2 -> 3a -> 3b on the SAME 80-item silver
# set, end to end, to verify the full pipeline wires together (NOT for quality).
# All 8B work is 4-bit so it fits a 24GB card. Run only when the GPU is free.
set -euo pipefail

cd /home/ai/flashrag/kgpaper
source .env 2>/dev/null || true
export KGPW_FLASHRAG_ROOT=/home/ai/flashrag/flashrag/FlashRAG-main
export PYTHONPATH="/home/ai/flashrag/kgpaper:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/ai/anaconda3/envs/kgpw/bin/python

SILVER=scripts/train/try/outputs/silver_try_80.jsonl
RUN=scripts/train/try/outputs/pipeline_smoke
mkdir -p "$RUN"
rm -rf "$RUN"/p2 "$RUN"/sft "$RUN"/ppo

echo "================ PHASE 2 (PRM + alpha-gate, 4-bit) ================"
$PY scripts/train/try/phase2_prm/phase2_prm_try.py \
    --silver "$SILVER" \
    --output_dir "$RUN/p2" \
    --epochs 1 --batch_size 1 --grad_accum 2 --max_length 1024
echo "PHASE 2 DONE -> $RUN/p2/alpha_gate.pt"

echo "================ PHASE 3a (SFT, 4-bit + merge) ==================="
$PY scripts/train/try/phase3_sft/phase3_sft_try.py \
    --silver "$SILVER" \
    --output_dir "$RUN/sft" \
    --epochs 1 --batch_size 1 --grad_accum 4 --max_length 1024 \
    --merge_output
echo "PHASE 3a DONE -> $RUN/sft/merged"

echo "================ PHASE 3b (PPO, 4-bit, from SFT + new alpha) ====="
$PY scripts/train/try/phase3_ppo/phase3_ppo_try.py \
    --silver "$SILVER" \
    --sft_checkpoint "$RUN/sft/merged" \
    --alpha_gate_path "$RUN/p2/alpha_gate.pt" \
    --text_reward_backend dummy \
    --use_4bit \
    --output_dir "$RUN/ppo" \
    --total_steps 2 --batch_size 2 --mini_batch_size 1 \
    --max_new_tokens 64 --max_input_length 1024
echo "PHASE 3b DONE -> $RUN/ppo/final"

echo "================ PIPELINE SMOKE COMPLETE ========================="
echo "Phase 2 alpha_gate : $RUN/p2/alpha_gate.pt"
echo "Phase 2 enriched   : $RUN/p2/silver_with_logprobs.jsonl"
echo "Phase 3a merged    : $RUN/sft/merged"
echo "Phase 3b ckpt      : $RUN/ppo/final"
