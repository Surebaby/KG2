#!/usr/bin/env bash
set -euo pipefail
: "${KGPW_SSH_PASS:?set KGPW_SSH_PASS (never store it in the repo)}"
export SSHPASS="$KGPW_SSH_PASS"
HOST=${KGPW_SSH_HOST:-connect.bjb1.seetacloud.com}
PORT=${KGPW_SSH_PORT:-30481}
USER=${KGPW_SSH_USER:-root}
REMOTE_ROOT=${KGPW_REMOTE_ROOT:-/root/autodl-tmp/kgpaper}
RUN_GPU_EVAL=${RUN_GPU_EVAL:-0}
LOCAL_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$LOCAL_ROOT"

FILES=(
  scripts/eval/evaluate_a1_fixed_context_kg.py
  scripts/pilot/score_a1_fixed_context_kg.py
  scripts/pilot/score_paired_kg_model_comparison.py
  outputs/audits/historical_hybrid_v2_independent_n100_seed20260901_standard_retrieval_proofkg_ppo_registration/protocol.json
  outputs/audits/historical_hybrid_v2_independent_n100_seed20260901_standard_retrieval_proofkg_ppo_registration/analysis_protocol.json
  outputs/audits/historical_hybrid_v2_independent_n100_seed20260901_standard_retrieval_inputs/arm_legacy.jsonl
  outputs/audits/historical_hybrid_v2_independent_n100_seed20260901_standard_retrieval_inputs/arm_proof.jsonl
  outputs/validation/historical_hybrid_v2_independent_n100_seed20260901_standard_retrieval_sft/predictions.jsonl
  launch_proofkg_ppo_standard_retrieval_paired_n100_remote.sh
)
for path in "${FILES[@]}"; do test -s "$path"; done
command -v sshpass >/dev/null

sshpass -e rsync -avzp --relative \
  -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
  "${FILES[@]}" "$USER@$HOST:$REMOTE_ROOT/"

if [[ "$RUN_GPU_EVAL" != "1" ]]; then
  echo "SYNC_COMPLETE; paired evaluation not started (set RUN_GPU_EVAL=1)."
  exit 0
fi

sshpass -e ssh -p "$PORT" -o StrictHostKeyChecking=no "$USER@$HOST" bash -s -- "$REMOTE_ROOT" <<'REMOTE'
set -euo pipefail
cd "$1"
mkdir -p logs/evaluation
nohup bash launch_proofkg_ppo_standard_retrieval_paired_n100_remote.sh \
  > logs/evaluation/proofkg_ppo_standard_retrieval_paired_n100_seed42.launcher.log 2>&1 &
echo "PAIRED_EVAL_PID=$!"
REMOTE
