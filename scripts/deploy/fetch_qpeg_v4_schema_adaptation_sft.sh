#!/usr/bin/env bash
# Fetch every QPEG-v4 adapter/checkpoint and every retained training log.
set -euo pipefail
: "${KGPW_SSH_PASS:?set KGPW_SSH_PASS (never store it in the repo)}"
export SSHPASS="$KGPW_SSH_PASS"
HOST=${KGPW_SSH_HOST:-connect.bjb1.seetacloud.com}
PORT=${KGPW_SSH_PORT:-30481}
USER=${KGPW_SSH_USER:-root}
REMOTE_ROOT=${KGPW_REMOTE_ROOT:-/root/autodl-tmp/kgpaper}
REL=checkpoints/sft_qpeg_v4_schema_adaptation_n2400_seed42
LOCAL_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$LOCAL_ROOT"

test ! -e "$REL/.download_complete_verified"
mkdir -p "$(dirname "$REL")" logs/training
sshpass -e rsync -avzp --partial \
  --exclude='optimizer.pt' \
  --exclude='scheduler.pt' \
  --exclude='rng_state*.pth' \
  -e "ssh -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  "$USER@$HOST:$REMOTE_ROOT/$REL/" "$REL/"

for name in \
  sft_qpeg_v4_schema_adaptation_n2400_seed42.log \
  sft_qpeg_v4_schema_adaptation_n2400_seed42.launcher.log \
  sft_qpeg_v4_schema_adaptation_n2400_seed42.failed_preflight_import_20260903.log \
  sft_qpeg_v4_schema_adaptation_n2400_seed42.failed_preflight_import2_20260903.log \
  qpeg_v4_development_step25_after_sft.launcher.log \
  qpeg_v4_development_adapted_step25_ab_v1.log
do
  sshpass -e rsync -avzp --partial \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
    "$USER@$HOST:$REMOTE_ROOT/logs/training/$name" "logs/training/$name"
done

for candidate in checkpoint-25 checkpoint-50 checkpoint-75 final; do
  test -s "$REL/$candidate/adapter_model.safetensors"
  test -s "$REL/$candidate/adapter_config.json"
  remote_sha=$(sshpass -e ssh -p "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$USER@$HOST" "sha256sum '$REMOTE_ROOT/$REL/$candidate/adapter_model.safetensors'" | awk '{print $1}')
  local_sha=$(sha256sum "$REL/$candidate/adapter_model.safetensors" | awk '{print $1}')
  test "$remote_sha" = "$local_sha"
  echo "$candidate adapter_model.safetensors sha256=$local_sha VERIFIED"
done
test -s "$REL/manifest.json"
test -s "$REL/sft_loss.jsonl"
touch "$REL/.download_complete_verified"
echo "FETCH_COMPLETE=$LOCAL_ROOT/$REL"
