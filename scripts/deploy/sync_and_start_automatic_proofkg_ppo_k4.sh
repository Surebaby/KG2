#!/usr/bin/env bash
set -euo pipefail
: "${KGPW_SSH_PASS:?set KGPW_SSH_PASS (never store it in the repo)}"
export SSHPASS="$KGPW_SSH_PASS"
HOST=${KGPW_SSH_HOST:-connect.bjb1.seetacloud.com}
PORT=${KGPW_SSH_PORT:-30481}
USER=${KGPW_SSH_USER:-root}
REMOTE_ROOT=${KGPW_REMOTE_ROOT:-/root/autodl-tmp/kgpaper}
RUN_LARGE_TRAINING=${RUN_LARGE_TRAINING:-0}
LOCAL_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$LOCAL_ROOT"

FILES=(
  configs/base.yaml
  configs/training/phase3_ppo.yaml
  configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600.yaml
  configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600_explicit_sft_ref.yaml
  configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600_combined_stability_v1.yaml
  configs/training/phase3_ppo_automatic_proofkg_2wiki_k4_smoke600_seed42.yaml
  configs/training/phase3_ppo_automatic_proofkg_2wiki_k4_smoke600_seed42.yaml.lock.json
  data/silver_data/automatic_proofkg_2wiki_train_k4_v1/silver_train.jsonl
  data/silver_data/automatic_proofkg_2wiki_train_k4_v1/question_kg_records.jsonl
  data/silver_data/automatic_proofkg_2wiki_train_k4_v1/report.json
  data/silver_data/automatic_proofkg_2wiki_train_k4_v1/manifest.json
  kgproweight/config/schemas.py
  kgproweight/kg/training_question_kg.py
  kgproweight/reward/proofkg_process.py
  kgproweight/training/reward_function.py
  kgproweight/training/phase3_ppo.py
  scripts/train/phase3_ppo.py
  scripts/prepare/preflight_automatic_proofkg_ppo.py
  tests/test_proofkg_production_reward.py
  tests/test_ppo_rollout_schedule.py
  tests/test_phase3_ppo_config_forwarding.py
  tests/test_training_question_kg.py
  tests/test_ppo_sft_replay.py
  tests/test_ppo_explicit_reference.py
  tests/test_run_preflight_manifest.py
  launch_ppo_automatic_proofkg_2wiki_k4_smoke600_remote.sh
)
for path in "${FILES[@]}"; do test -f "$path"; done
command -v sshpass >/dev/null

sshpass -e rsync -avzp --relative \
  -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
  "${FILES[@]}" "$USER@$HOST:$REMOTE_ROOT/"

if [[ "$RUN_LARGE_TRAINING" != "1" ]]; then
  echo "SYNC_COMPLETE; PPO not started (set RUN_LARGE_TRAINING=1 after approval)."
  exit 0
fi

sshpass -e ssh -p "$PORT" -o StrictHostKeyChecking=no "$USER@$HOST" bash -s -- "$REMOTE_ROOT" <<'REMOTE'
set -euo pipefail
cd "$1"
nohup bash launch_ppo_automatic_proofkg_2wiki_k4_smoke600_remote.sh \
  > logs/training/ppo_automatic_proofkg_2wiki_k4_smoke600_seed42.launcher.log 2>&1 &
echo "PPO_PID=$!"
REMOTE
