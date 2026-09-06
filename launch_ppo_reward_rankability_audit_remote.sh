#!/usr/bin/env bash
# Zero-update SFT exploration/reward-rankability audit. No PPO optimiser is built.
set -euo pipefail

KGPW_ROOT=/root/autodl-tmp/kgpaper
cd "$KGPW_ROOT"

export PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src"
export CUDA_VISIBLE_DEVICES=0
KGPW_PYTHON=/root/autodl-tmp/kgpw_env/bin/python

EXPERIMENT_ID=ppo_reward_rankability_sft_hybrid_train_n100_k4_seed20260828_v1
OUTPUT_DIR="outputs/audits/$EXPERIMENT_ID"
LOG_PATH="logs/audits/$EXPERIMENT_ID.log"
CONFIG=configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600_hybrid_old10_bridge5_v3.yaml
OVERRIDES=data/silver_data/pilots/ppo_smoke_hybrid_train_seed42_20260828/old10_bridge5_v3.jsonl

mkdir -p logs/audits outputs/audits
test ! -e "$OUTPUT_DIR"
test ! -e "$LOG_PATH"
test -s "$CONFIG"
test -s "$OVERRIDES"
test "$(sha256sum "$OVERRIDES" | cut -d' ' -f1)" = \
  7c199a0e272323a8739d232c21ee4f084ba0fc071f1de67fb97a7cb593eb1a1f

# Unit checks run before GPU model allocation and before reserving Experiment ID.
"$KGPW_PYTHON" -m pytest -q \
  tests/test_ppo_reward_rankability.py \
  tests/test_phase3_ppo_config_forwarding.py \
  tests/test_ppo_hybrid_rollout_inputs.py \
  tests/test_run_preflight_manifest.py

"$KGPW_PYTHON" scripts/pilot/audit_ppo_reward_rankability.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --experiment-id "$EXPERIMENT_ID" \
  --rollouts-per-qid 4 \
  --warmup-qids 20 \
  --cohort-seed 20260828 \
  --generation-seed 42 \
  --bootstrap-seed 42 \
  --preflight-only

"$KGPW_PYTHON" scripts/pilot/audit_ppo_reward_rankability.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --experiment-id "$EXPERIMENT_ID" \
  --rollouts-per-qid 4 \
  --warmup-qids 20 \
  --cohort-seed 20260828 \
  --generation-seed 42 \
  --bootstrap-seed 42 \
  2>&1 | tee "$LOG_PATH"
