#!/bin/bash
# ===========================================================================
# Phase 3b PPO — SMOKE TEST (R10 config validation, ~10 optimiser updates)
#
#   bash launch_split_ppo_smoke.sh          # 80 trajectories = 10 updates
#   SMOKE_TRAJ=160 bash launch_split_ppo_smoke.sh
#
# Purpose: confirm the R10 changes behave as intended BEFORE spending a full
# session on launch_split_ppo.sh (16000 trajectories / 2000 updates). This is
# not a training run — 10 updates at lr=1e-6 changes nothing measurable.
#
# What it is checking, and what each answer means:
#
#   1. NO "right-truncation will drop the trailing KG block" warning.
#      ppo_max_kg_triples went 50 -> 30 to restore the 6144 budget's margin.
#      If this still fires, the KG context is being cut off and every KG claim
#      the paper makes is unsupported by the run.
#
#   2. custom/kg_reward_share near 0.10 (was 0.009).
#      outcome_weight 10 -> 4 and step_reward_scale 0.3 -> 1.5. If it comes
#      back near 0.01 the reward assembly is not picking up the new weights.
#
#   3. objective/kl in the tens, kl_coef barely moving off 0.15.
#      KL is a per-sequence SUM (ppo_trainer.py:1301), hence target 8 -> 40.
#
#   4. Peak GPU well under 96 GB. The ~58 GB figure in launch_split_ppo.sh is
#      scaled from Phase 3a's measured 55.60 GB, never measured for PPO.
#
#   5. A checkpoint actually lands (save_every_steps forced to 40 here).
# ===========================================================================
set -euo pipefail

# 2026-08-22 (retraining_plan §13-5): 80 -> 600. At batch_size 4 (see
# phase3_ppo.yaml) 80 trajectories is only 20 optimiser updates, and the failure
# mode this smoke test now exists to catch -- the policy collapsing to
# min_valid_steps steps per trajectory -- unfolds over the first ~150 updates
# (measured: PPO(1) 2.84 -> 2.04 steps, PPO(2) 3.13 -> 2.01). 20 updates cannot
# distinguish "fixed" from "not yet collapsed". 600 traj = 150 updates, ~45 min.
SMOKE_TRAJ="${SMOKE_TRAJ:-600}"
SMOKE_SAVE="${SMOKE_SAVE:-40}"

REMOTE_ROOT=/root/autodl-tmp/kgpaper
cd "$REMOTE_ROOT"

export PYTHONPATH="$REMOTE_ROOT:$REMOTE_ROOT/flashrag_src"
export KGPW_FLASHRAG_ROOT="$REMOTE_ROOT/flashrag_src"
export KGPW_PROJECT_ROOT="$REMOTE_ROOT"
export KGPW_DATA_DIR="$REMOTE_ROOT/data"
export KGPW_INDEX_DIR="$REMOTE_ROOT/indexes"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset HTTP_PROXY HTTPS_PROXY
export KGPW_REARAG_PATH=/root/autodl-tmp/models/rearag-9b

PY=/root/autodl-tmp/kgpw_env/bin/python
SILVER="$REMOTE_ROOT/data/silver_data/silver_v1_reannotated.jsonl"
SFT="$REMOTE_ROOT/checkpoints/sft_student_split/final"
ALPHA="$REMOTE_ROOT/checkpoints/prm_alpha_gate_v1reann_negfix/alpha_gate.pt"
OUT="$REMOTE_ROOT/outputs/split_ppo_smoke"

for p in "$SILVER" "$ALPHA"; do
  [ -e "$p" ] || { echo "MISSING: $p"; exit 1; }
done
[ -d "$SFT" ] || { echo "MISSING SFT checkpoint: $SFT"; exit 1; }
[ -d "$KGPW_REARAG_PATH" ] || { echo "MISSING rearag: $KGPW_REARAG_PATH (would download 18 GB)"; exit 1; }

rm -rf "$OUT"
mkdir -p "$OUT"

# batch_size is read from the YAML so this line cannot drift from what runs
# again (it hardcoded 8 while the YAML said 4, understating updates by 2x).
SMOKE_BS=$("$PY" -c "import yaml,sys; print(yaml.safe_load(open('configs/training/phase3_ppo.yaml'))['training']['ppo']['batch_size'])")
echo "Smoke test: $SMOKE_TRAJ trajectories (= $((SMOKE_TRAJ / SMOKE_BS)) optimiser updates at batch_size $SMOKE_BS)"

nohup "$PY" scripts/train/phase3_ppo.py \
  --config configs/training/phase3_ppo.yaml \
  --silver_data "$SILVER" \
  --sft_checkpoint "$SFT" \
  --alpha_gate_path "$ALPHA" \
  --text_reward_backend rearag \
  --output_dir "$OUT" \
  --split train \
  --total_steps "$SMOKE_TRAJ" \
  --save_every_steps "$SMOKE_SAVE" \
  > "$OUT/train.log" 2>&1 &

PID=$!
echo "Launched (pid $PID); log: $OUT/train.log"
echo
echo "When it finishes, run:  bash check_ppo_smoke.sh"
