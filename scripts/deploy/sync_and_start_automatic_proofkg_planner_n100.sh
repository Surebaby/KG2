#!/usr/bin/env bash
# Sync the frozen n=100 planner package and start gold-free generation remotely.
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
  configs/training/query_planner_learned_scale_v1_1_seed42.yaml
  kgproweight/eval/query_planner.py
  kgproweight/kg/kg_filter.py
  kgproweight/training/query_planner.py
  kgproweight/utils/logging.py
  scripts/eval/generate_query_plans_unseen.py
  scripts/prepare/audit_query_planner_supervision.py
  scripts/prepare/build_query_planner_supervision.py
  outputs/audits/automatic_proofkg_2wiki_unseen_n100_seed20260830_preregistration/cohort.question_only.jsonl
  outputs/audits/automatic_proofkg_2wiki_unseen_n100_seed20260830_preregistration/protocol.json
  checkpoints/query_planner_learned_scale_v1_1_seed42/final/adapter_config.json
  checkpoints/query_planner_learned_scale_v1_1_seed42/final/adapter_model.safetensors
  checkpoints/query_planner_learned_scale_v1_1_seed42/final/tokenizer.json
  checkpoints/query_planner_learned_scale_v1_1_seed42/final/tokenizer_config.json
  checkpoints/query_planner_learned_scale_v1_1_seed42/final/special_tokens_map.json
  launch_automatic_proofkg_planner_n100_remote.sh
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
COHORT=outputs/audits/automatic_proofkg_2wiki_unseen_n100_seed20260830_preregistration/cohort.question_only.jsonl
PROTOCOL=outputs/audits/automatic_proofkg_2wiki_unseen_n100_seed20260830_preregistration/protocol.json
ADAPTER=checkpoints/query_planner_learned_scale_v1_1_seed42/final/adapter_model.safetensors
test "$(sha256sum "$COHORT" | cut -d' ' -f1)" = ddbd751f332a99430a4c58559fb2e9083614f0474419e133e25961f1824da35a
test "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" = afe7d3ed95e730bb475409a8621d8ada490c9d3cc1c16b3a974ca8f9abde4f7c
test "$(sha256sum "$ADAPTER" | cut -d' ' -f1)" = 0bd41d01140b00413c7d8a908d7d4482c4d955dd772072f2f3c74c8fe1c2c776
test ! -e outputs/validation/automatic_proofkg_2wiki_unseen_n100_seed20260830_plans
test ! -e logs/validation/automatic_proofkg_2wiki_unseen_n100_seed20260830_plans.log
mkdir -p logs/validation
nohup bash launch_automatic_proofkg_planner_n100_remote.sh \
  > logs/validation/automatic_proofkg_2wiki_unseen_n100_seed20260830_plans.launcher.log 2>&1 &
echo "PLANNER_PID=$!"
echo "LOG=$ROOT/logs/validation/automatic_proofkg_2wiki_unseen_n100_seed20260830_plans.log"
REMOTE
