#!/usr/bin/env bash
# Sync the versioned light20 SFT package.  Starting the GPU job is intentionally
# gated by RUN_LARGE_TRAINING=1 so a code sync cannot spend GPU budget by itself.
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
  configs/training/phase3_sft.yaml
  configs/training/phase3_sft_proofkg_curriculum_light20_v2_n5000_seed42.yaml
  kgproweight/config/schemas.py
  kgproweight/data/parsers.py
  kgproweight/data/prompts.py
  kgproweight/data/silver_dataset.py
  kgproweight/data/silver_split.py
  kgproweight/kg/question_kg.py
  kgproweight/kg/training_question_kg.py
  kgproweight/training/phase3_sft.py
  kgproweight/utils/logging.py
  scripts/train/_split_args.py
  scripts/train/phase3_sft.py
  data/silver_data/proofkg_curriculum_light20_v2_n5000_seed42_20260831/silver_curriculum.jsonl
  data/silver_data/proofkg_curriculum_light20_v2_n5000_seed42_20260831/question_kg_records.jsonl
  data/silver_data/proofkg_curriculum_light20_v2_n5000_seed42_20260831/report.json
  tests/test_phase3_sft_config_forwarding.py
  tests/test_training_question_kg.py
  tests/test_silver_split.py
  tests/test_prm_annotator.py
  tests/test_kg_index_guard.py
  launch_proofkg_curriculum_light20_sft_remote.sh
)
for path in "${FILES[@]}"; do test -f "$path"; done
command -v sshpass >/dev/null

sshpass -e rsync -avzp --relative \
  -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
  "${FILES[@]}" "$USER@$HOST:$REMOTE_ROOT/"

if [[ "$RUN_LARGE_TRAINING" != "1" ]]; then
  echo "SYNC_COMPLETE; large training not started (set RUN_LARGE_TRAINING=1 after approval)."
  exit 0
fi

sshpass -e ssh -p "$PORT" -o StrictHostKeyChecking=no "$USER@$HOST" bash -s -- "$REMOTE_ROOT" <<'REMOTE'
set -euo pipefail
ROOT=$1
cd "$ROOT"
DATA=data/silver_data/proofkg_curriculum_light20_v2_n5000_seed42_20260831
test "$(sha256sum "$DATA/silver_curriculum.jsonl" | cut -d' ' -f1)" = \
  1cfd5f989f65a476390151e74ae4511e6199129d4394d884276fea92487a067b
test "$(sha256sum "$DATA/question_kg_records.jsonl" | cut -d' ' -f1)" = \
  8434ed430a644d6e5b175a2eba1799ba2ac33075731b2af7456b60786fd32521
test ! -e checkpoints/sft_proofkg_curriculum_light20_v2_n5000_seed42
test ! -e logs/training/sft_proofkg_curriculum_light20_v2_n5000_seed42.log
mkdir -p logs/training
nohup bash launch_proofkg_curriculum_light20_sft_remote.sh \
  > logs/training/sft_proofkg_curriculum_light20_v2_n5000_seed42.launcher.log 2>&1 &
echo "SFT_PID=$!"
echo "LOG=$ROOT/logs/training/sft_proofkg_curriculum_light20_v2_n5000_seed42.log"
REMOTE
