#!/usr/bin/env bash
set -euo pipefail

# Prepared launcher only.  The development utility result must be copied to the
# remote workspace after it completes locally; otherwise this script fails.
ROOT=/root/autodl-tmp/kgpaper
CONFIG=configs/training/phase3_sft_saeg_v1_balanced_epoch4860_seed42.yaml
UTILITY=outputs/validation/saeg_v1_development_strong_sft_npdf_v2_attempt2/report.json
PREFLIGHT=outputs/audits/saeg_v1_sft_balanced_epoch4860_preflight_v2/report.json
LOG=logs/training/sft_saeg_v1_balanced_epoch4860_seed42.log
EXPERIMENT_ID=SAEG-V1-SFT-BALANCED-EPOCH4860-SEED42

cd "$ROOT"
test -s "$CONFIG"
test -s "$UTILITY"
test -s "$PREFLIGHT"

python - "$UTILITY" "$PREFLIGHT" <<'PY'
import json
import sys

utility = json.load(open(sys.argv[1], encoding="utf-8"))
preflight = json.load(open(sys.argv[2], encoding="utf-8"))
if utility.get("status") != "PASS_ZERO_TRAINING_UTILITY":
    raise SystemExit(f"SFT blocked by utility status: {utility.get('status')}")
if preflight.get("status") != "PASS_NOT_TRAINED":
    raise SystemExit(f"SFT blocked by preflight status: {preflight.get('status')}")
PY

test ! -e checkpoints/sft_saeg_v1_balanced_epoch4860_seed42
test -d /root/autodl-tmp/models/llama3-8b
test -s checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final/adapter_model.safetensors

mkdir -p logs/training "/root/tf-logs/$EXPERIMENT_ID"
export KGPW_LLAMA3_PATH=/root/autodl-tmp/models/llama3-8b
export PYTHONPATH="$ROOT:$ROOT/flashrag_src"
export CUDA_VISIBLE_DEVICES=0

python scripts/train/phase3_sft.py --config "$CONFIG" 2>&1 | tee "$LOG"
