#!/usr/bin/env bash
# Sync the approved QPEG-v4 SFT package; start only with RUN_LARGE_TRAINING=1.
set -euo pipefail
: "${KGPW_SSH_PASS:?set KGPW_SSH_PASS (never store it in the repository)}"
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
  configs/training/phase3_sft_qpeg_v4_schema_adaptation_n2400_seed42.yaml
  kgproweight/config/schemas.py
  kgproweight/data/parsers.py
  kgproweight/data/prompts.py
  kgproweight/data/silver_dataset.py
  kgproweight/data/silver_split.py
  kgproweight/kg/qpeg.py
  kgproweight/kg/question_kg.py
  kgproweight/training/phase3_sft.py
  kgproweight/utils/logging.py
  kgproweight/utils/paths.py
  kgproweight/utils/seed.py
  scripts/train/_split_args.py
  scripts/train/phase3_sft.py
  scripts/prepare/build_qpeg_v4_schema_adaptation_data.py
  data/silver_data/qpeg_v4_schema_adaptation_n2400_seed42_v2/silver_curriculum.jsonl
  data/silver_data/qpeg_v4_schema_adaptation_n2400_seed42_v2/report.json
  outputs/audits/qpeg_v4_schema_adaptation_protocol_v1/protocol.json
  outputs/audits/qpeg_v4_schema_adaptation_protocol_v1/report.json
  outputs/audits/qpeg_v4_schema_adaptation_sft_preflight_v1/report.json
  tests/test_qpeg_v4_schema_adaptation_data.py
  tests/test_phase3_sft_config_forwarding.py
  tests/test_silver_split.py
  launch_qpeg_v4_schema_adaptation_sft_remote.sh
)
for path in "${FILES[@]}"; do test -f "$path"; done
command -v sshpass >/dev/null

sshpass -e rsync -avzp --relative \
  -e "ssh -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  "${FILES[@]}" "$USER@$HOST:$REMOTE_ROOT/"

if [[ "$RUN_LARGE_TRAINING" != "1" ]]; then
  echo "SYNC_COMPLETE; approved training not started (set RUN_LARGE_TRAINING=1)."
  exit 0
fi

sshpass -e ssh -p "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "$USER@$HOST" bash -s -- "$REMOTE_ROOT" <<'REMOTE'
set -euo pipefail
ROOT=$1
cd "$ROOT"
CONFIG=configs/training/phase3_sft_qpeg_v4_schema_adaptation_n2400_seed42.yaml
DATA=data/silver_data/qpeg_v4_schema_adaptation_n2400_seed42_v2/silver_curriculum.jsonl
PROTOCOL=outputs/audits/qpeg_v4_schema_adaptation_protocol_v1/protocol.json
OUT=checkpoints/sft_qpeg_v4_schema_adaptation_n2400_seed42
LOG=logs/training/sft_qpeg_v4_schema_adaptation_n2400_seed42.log
test "$(sha256sum "$CONFIG" | cut -d' ' -f1)" = e050d3bc100a6f4e6fc0ac835bd2af07deab8d8e8f725838088d64c7648c529c
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = 0107dcd8847e316127b24f490db8328edd3698d7cdbd64194856be6381f2d3a6
test "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" = 1f53b0957f3e82bd2b5fdc43992b4f7b7a8012215b079731cbb1424ee5a07c2f
test -d models/llama3-8b
test -d checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final
test ! -e "$OUT"
test ! -e "$LOG"
mkdir -p logs/training
nohup bash launch_qpeg_v4_schema_adaptation_sft_remote.sh \
  > logs/training/sft_qpeg_v4_schema_adaptation_n2400_seed42.launcher.log 2>&1 &
echo "SFT_PID=$!"
echo "LOG=$ROOT/$LOG"
REMOTE
