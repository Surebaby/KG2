#!/usr/bin/env bash
# Paid retrieval-input smoke. Reward/loss/KG/replay/stability knobs are frozen.
set -euo pipefail
KGPW_ROOT=/root/autodl-tmp/kgpaper
cd "$KGPW_ROOT"

export PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src"
export KGPW_TB_DIR=/root/tf-logs
export CUDA_VISIBLE_DEVICES=0
KGPW_PYTHON=/root/autodl-tmp/kgpw_env/bin/python

RUN_ID=ppo_legacy_repaired_v2_quota70_hard_seed42_no_text_head_smoke600_hybrid_old10_bridge5_v3
LOG_PATH=logs/training/ppo_quota70_hard_seed42_no_text_head_smoke600_hybrid_old10_bridge5_v3.log
OVERRIDES=data/silver_data/pilots/ppo_smoke_hybrid_train_seed42_20260828/old10_bridge5_v3.jsonl
SCHEDULE=data/silver_data/pilots/ppo_smoke_hybrid_train_seed42_20260828/schedule600.jsonl

mkdir -p logs/training
test ! -e "outputs/$RUN_ID"
test ! -e "$LOG_PATH"
test -s "$OVERRIDES"
test -s "$SCHEDULE"
test "$(sha256sum "$OVERRIDES" | cut -d' ' -f1)" = \
  7c199a0e272323a8739d232c21ee4f084ba0fc071f1de67fb97a7cb593eb1a1f
test "$(sha256sum "$SCHEDULE" | cut -d' ' -f1)" = \
  6e2b4871d9035ec7c3b494212c382c67aeb4d113ed5ef8367a1b5510af19b771

# Fail before allocating 8B/9B models if config forwarding, frozen scheduling,
# explicit reference, replay or manifest invariants regressed during transfer.
"$KGPW_PYTHON" -m pytest -q \
  tests/test_phase3_ppo_config_forwarding.py \
  tests/test_ppo_hybrid_rollout_inputs.py \
  tests/test_ppo_rollout_schedule.py \
  tests/test_ppo_diagnostics.py \
  tests/test_ppo_explicit_reference.py \
  tests/test_run_preflight_manifest.py \
  tests/test_ppo_sft_replay.py \
  tests/test_advantage_telemetry.py

exec "$KGPW_PYTHON" scripts/train/phase3_ppo.py \
  --config configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600_hybrid_old10_bridge5_v3.yaml \
  2>&1 | tee "$LOG_PATH"
