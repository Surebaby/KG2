#!/usr/bin/env bash
set -euo pipefail

# Formal Proof400 paired run: PPO-T followed by PPO-TK.  This launcher is
# append-only and fail-closed.  It cannot enter either training command until
# the separately generated v3 GPU postflight report+manifest bundle exists at
# the exact frozen paths, has the exact runtime-only success status, binds the
# report hash, and the full CPU lock preflight passes again on the remote host.
KGPW_RUN_ROOT=${KGPW_REMOTE_ROOT:-/root/autodl-tmp/kgpaper}
KGPW_RUN_PYTHON=${KGPW_REMOTE_PYTHON:-/root/autodl-tmp/kgpw_env/bin/python}
cd "$KGPW_RUN_ROOT"

export PYTHONPATH="$KGPW_RUN_ROOT:$KGPW_RUN_ROOT/flashrag_src"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export KGPW_LLAMA3_PATH=${KGPW_LLAMA3_PATH:-/root/autodl-tmp/models/llama3-8b}
export KGPW_REARAG_PATH=${KGPW_REARAG_PATH:-/root/autodl-tmp/models/rearag-9b}

PAIR_MANIFEST=outputs/audits/mixed3_rearag_proof400_ppo_pair_7200_seed42_v2/pair_manifest.json
GPU_POSTFLIGHT=outputs/audits/mixed3_rearag_runtime_wiring_probe_v3_proof400_seed42_gpu_postflight/postflight.json
GPU_POSTFLIGHT_MANIFEST=outputs/audits/mixed3_rearag_runtime_wiring_probe_v3_proof400_seed42_gpu_postflight/manifest.json
PREFLIGHT=logs/training/ppo_mixed3_rearag_v2_proof400_paired7200_seed42.remote_preflight.json
CONFIG_T=configs/training/phase3_ppo_mixed3_rearag_v2_proof400_text7200_seed42.yaml
CONFIG_TK=configs/training/phase3_ppo_mixed3_rearag_v2_proof400_text_kg_v2_1_7200_seed42.yaml
OUT_T=outputs/ppo_mixed3_rearag_v2_proof400_text7200_seed42
OUT_TK=outputs/ppo_mixed3_rearag_v2_proof400_text_kg_v2_1_7200_seed42
LOG_T=logs/training/ppo_mixed3_rearag_v2_proof400_text7200_seed42.log
LOG_TK=logs/training/ppo_mixed3_rearag_v2_proof400_text_kg_v2_1_7200_seed42.log
TB_T=/root/tf-logs/ppo_mixed3_rearag_v2_proof400_text7200_seed42
TB_TK=/root/tf-logs/ppo_mixed3_rearag_v2_proof400_text_kg_v2_1_7200_seed42

test -x "$KGPW_RUN_PYTHON"
test -s "$PAIR_MANIFEST"
test -s "$GPU_POSTFLIGHT"
test -s "$GPU_POSTFLIGHT_MANIFEST"
test -s "$CONFIG_T"
test -s "$CONFIG_TK"
test -d "$KGPW_LLAMA3_PATH"
test -d "$KGPW_REARAG_PATH"
test ! -e "$PREFLIGHT"
test ! -e "$OUT_T"
test ! -e "$OUT_TK"
test ! -e "$LOG_T"
test ! -e "$LOG_TK"
test ! -e "$TB_T"
test ! -e "$TB_TK"

# Exact bundle gate requested by the frozen pair.  This status proves one K4
# runtime update per route, not model-quality improvement.
"$KGPW_RUN_PYTHON" -c 'import hashlib,json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); m=json.load(open(sys.argv[2], encoding="utf-8")); h=hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest(); s="PASS_RUNTIME_WIRING_NOT_EFFECT_EVIDENCE"; assert p.get("status")==s, p.get("status"); assert m.get("status")==s, m.get("status"); assert (m.get("run") or {}).get("postflight_sha256")==h; assert (m.get("run") or {}).get("effect_evidence") is False' "$GPU_POSTFLIGHT" "$GPU_POSTFLIGHT_MANIFEST"

nvidia-smi
mkdir -p logs/training /root/tf-logs

"$KGPW_RUN_PYTHON" scripts/prepare/preflight_mixed3_rearag_proof400_ppo_pair.py \
  --pair_manifest "$PAIR_MANIFEST" \
  --report_path "$PREFLIGHT" \
  --run_tests
"$KGPW_RUN_PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); assert p.get("status") == "PASS_CPU_PREFLIGHT_GPU_POSTFLIGHT_BOUND", p.get("status")' "$PREFLIGHT"

run_arm() {
  local config_path=$1
  local output_path=$2
  local log_path=$3
  local tensorboard_path=$4
  test ! -e "$output_path"
  test ! -e "$log_path"
  test ! -e "$tensorboard_path"
  mkdir -p "$tensorboard_path"
  export KGPW_TB_DIR="$tensorboard_path"
  "$KGPW_RUN_PYTHON" scripts/train/phase3_ppo.py \
    --config "$config_path" 2>&1 | tee "$log_path"
  test -s "$output_path/final/adapter_model.safetensors"
  test -s "$output_path/history.jsonl"
  test -s "$output_path/manifest.json"
}

run_arm "$CONFIG_T" "$OUT_T" "$LOG_T" "$TB_T"
run_arm "$CONFIG_TK" "$OUT_TK" "$LOG_TK" "$TB_TK"
