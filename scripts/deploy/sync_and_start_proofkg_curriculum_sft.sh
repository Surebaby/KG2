#!/usr/bin/env bash
# Sync only the frozen Proof-KG curriculum training package, verify hashes, and
# start continued SFT in the background on the 96GB server.
set -euo pipefail
: "${KGPW_SSH_PASS:?set KGPW_SSH_PASS (never store it in the repo)}"
export SSHPASS="$KGPW_SSH_PASS"
HOST=${KGPW_SSH_HOST:-connect.bjb1.seetacloud.com}
PORT=${KGPW_SSH_PORT:-41354}
USER=${KGPW_SSH_USER:-root}
REMOTE_ROOT=${KGPW_REMOTE_ROOT:-/root/autodl-tmp/kgpaper}
LOCAL_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$LOCAL_ROOT"

FILES=(
  configs/base.yaml
  configs/training/phase3_sft.yaml
  configs/training/phase3_ppo.yaml
  configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600.yaml
  configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600_explicit_sft_ref.yaml
  configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600_combined_stability_v1.yaml
  kgproweight/config/schemas.py
  kgproweight/data/parsers.py
  kgproweight/data/prompts.py
  kgproweight/data/silver_dataset.py
  kgproweight/data/silver_split.py
  kgproweight/kg/question_kg.py
  kgproweight/kg/training_question_kg.py
  kgproweight/reward/alpha_gate.py
  kgproweight/reward/composite_reward.py
  kgproweight/reward/prm_annotator.py
  kgproweight/training/phase3_sft.py
  kgproweight/training/phase3_ppo.py
  kgproweight/training/reward_function.py
  kgproweight/training/step_reward_ppo_trainer.py
  kgproweight/utils/logging.py
  scripts/train/_split_args.py
  scripts/train/phase3_sft.py
  scripts/train/phase3_ppo.py
  configs/training/phase3_sft_proofkg_curriculum_mix_v1_n8000_seed42.yaml
  configs/training/phase3_ppo_proofkg_curriculum_mix_v1_smoke600_seed42.yaml
  data/silver_data/proofkg_curriculum_mix_v1_n8000_seed42_20260829/silver_curriculum.jsonl
  data/silver_data/proofkg_curriculum_mix_v1_n8000_seed42_20260829/question_kg_records.jsonl
  tests/test_training_question_kg.py
  tests/test_phase3_sft_config_forwarding.py
  tests/test_phase3_ppo_config_forwarding.py
  tests/test_kg_index_guard.py
  tests/test_ppo_sft_replay.py
  tests/test_prm_annotator.py
  launch_proofkg_curriculum_sft_remote.sh
  launch_proofkg_curriculum_ppo_smoke_remote.sh
)
for path in "${FILES[@]}"; do test -f "$path"; done
command -v sshpass >/dev/null

sshpass -e rsync -avzp --relative \
  -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
  "${FILES[@]}" "$USER@$HOST:$REMOTE_ROOT/"

sshpass -e ssh -p "$PORT" -o StrictHostKeyChecking=no "$USER@$HOST" bash -s -- "$REMOTE_ROOT" <<'REMOTE'
set -euo pipefail
ROOT=$1
cd "$ROOT"
DATA=data/silver_data/proofkg_curriculum_mix_v1_n8000_seed42_20260829
test "$(sha256sum "$DATA/silver_curriculum.jsonl" | cut -d' ' -f1)" = \
  c8d5de2d76db56f767cfedb31e6182ef80050a6047cb4fca2c7bda16f18a519d
test "$(sha256sum "$DATA/question_kg_records.jsonl" | cut -d' ' -f1)" = \
  56aab2b6502adfb1bfe51c0933cc821573039b3ae4f7f52aa1db771803873053
test ! -e checkpoints/sft_proofkg_curriculum_mix_v1_n8000_seed42
test ! -e logs/training/sft_proofkg_curriculum_mix_v1_n8000_seed42.log
mkdir -p logs/training
nohup bash launch_proofkg_curriculum_sft_remote.sh \
  > logs/training/sft_proofkg_curriculum_mix_v1_n8000_seed42.launcher.log 2>&1 &
echo "SFT_PID=$!"
echo "LOG=$ROOT/logs/training/sft_proofkg_curriculum_mix_v1_n8000_seed42.log"
REMOTE
