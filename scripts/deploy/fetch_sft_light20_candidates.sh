#!/usr/bin/env bash
# Fetch inference-relevant light20 candidate files while leaving remote
# optimizer/scheduler states in place for reproducibility and resume support.
set -euo pipefail
: "${KGPW_SSH_PASS:?set KGPW_SSH_PASS (never store it in the repo)}"
export SSHPASS="$KGPW_SSH_PASS"
HOST=${KGPW_SSH_HOST:-connect.bjb1.seetacloud.com}
PORT=${KGPW_SSH_PORT:-30481}
USER=${KGPW_SSH_USER:-root}
REMOTE_ROOT=${KGPW_REMOTE_ROOT:-/root/autodl-tmp/kgpaper}
REL=checkpoints/sft_proofkg_curriculum_light20_v2_n5000_seed42
LOCAL_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$LOCAL_ROOT"

test ! -e "$REL"
mkdir -p "$(dirname "$REL")"
sshpass -e rsync -avzp \
  --exclude='optimizer.pt' \
  --exclude='scheduler.pt' \
  --exclude='rng_state*.pth' \
  -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
  "$USER@$HOST:$REMOTE_ROOT/$REL/" "$REL/"

for candidate in checkpoint-40 checkpoint-80 checkpoint-120 final; do
  test -s "$REL/$candidate/adapter_model.safetensors"
  test -s "$REL/$candidate/adapter_config.json"
done
test -s "$REL/manifest.json"
echo "FETCH_COMPLETE=$LOCAL_ROOT/$REL"
