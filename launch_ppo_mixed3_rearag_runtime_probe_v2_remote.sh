#!/usr/bin/env bash
set -euo pipefail

# Recommended probe launcher: exactly two one-update arms (4+4 trajectories).
# It never invokes the formal 7,200-trajectory configurations.
KGPW_RUN_ROOT=${KGPW_REMOTE_ROOT:-/root/autodl-tmp/kgpaper}
KGPW_RUN_PYTHON=${KGPW_REMOTE_PYTHON:-/root/autodl-tmp/kgpw_env/bin/python}
cd "$KGPW_RUN_ROOT"
export PYTHONPATH="$KGPW_RUN_ROOT:$KGPW_RUN_ROOT/flashrag_src"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export KGPW_LLAMA3_PATH=${KGPW_LLAMA3_PATH:-/root/autodl-tmp/models/llama3-8b}
export KGPW_REARAG_PATH=${KGPW_REARAG_PATH:-/root/autodl-tmp/models/rearag-9b}

PROTOCOL=outputs/audits/mixed3_rearag_runtime_wiring_probe_v2_seed42_freeze/protocol.json
LOCAL_PREFLIGHT=outputs/audits/mixed3_rearag_runtime_wiring_probe_v2_seed42_local_preflight/preflight.json
REMOTE_PREFLIGHT=outputs/audits/mixed3_rearag_runtime_wiring_probe_v2_seed42_remote_preflight
POSTFLIGHT=outputs/audits/mixed3_rearag_runtime_wiring_probe_v2_seed42_gpu_postflight
CONFIG_T=configs/training/phase3_ppo_mixed3_rearag_runtime_probe_v2_t_noneligible_k4_seed42.yaml
CONFIG_TK=configs/training/phase3_ppo_mixed3_rearag_runtime_probe_v2_tk_eligible_k4_seed42.yaml
OUT_T=outputs/ppo_mixed3_rearag_runtime_probe_v2_t_noneligible_k4_seed42
OUT_TK=outputs/ppo_mixed3_rearag_runtime_probe_v2_tk_eligible_k4_seed42
LOG_T=logs/training/ppo_mixed3_rearag_runtime_probe_v2_t_noneligible_k4_seed42.log
LOG_TK=logs/training/ppo_mixed3_rearag_runtime_probe_v2_tk_eligible_k4_seed42.log
TB_T=/root/tf-logs/ppo_mixed3_rearag_runtime_probe_v2_t_noneligible_k4_seed42
TB_TK=/root/tf-logs/ppo_mixed3_rearag_runtime_probe_v2_tk_eligible_k4_seed42

test -x "$KGPW_RUN_PYTHON"
test -s "$PROTOCOL"
test -s "$LOCAL_PREFLIGHT"
test -s "$CONFIG_T"
test -s "$CONFIG_TK"
test -d "$KGPW_LLAMA3_PATH"
test -d "$KGPW_REARAG_PATH"
test ! -e "$REMOTE_PREFLIGHT"
test ! -e "$POSTFLIGHT"
test ! -e "$OUT_T"
test ! -e "$OUT_TK"
test ! -e "$LOG_T"
test ! -e "$LOG_TK"
test ! -e "$TB_T"
test ! -e "$TB_TK"

mkdir -p logs/training /root/tf-logs
"$KGPW_RUN_PYTHON" scripts/prepare/preflight_mixed3_rearag_runtime_probe_v2.py \
  --protocol "$PROTOCOL" --report_dir "$REMOTE_PREFLIGHT"
nvidia-smi

run_arm() {
  local config_path=$1 output_path=$2 log_path=$3 tensorboard_path=$4
  test ! -e "$output_path"
  test ! -e "$log_path"
  test ! -e "$tensorboard_path"
  mkdir -p "$tensorboard_path"
  export KGPW_TB_DIR="$tensorboard_path"
  "$KGPW_RUN_PYTHON" scripts/train/phase3_ppo.py --config "$config_path" 2>&1 | tee "$log_path"
  test -s "$output_path/final/adapter_model.safetensors"
  test -s "$output_path/history.jsonl"
  test -s "$output_path/manifest.json"
  find "$tensorboard_path" -type f -name 'events.out.tfevents.*' -print -quit | grep -q .
}

run_arm "$CONFIG_T" "$OUT_T" "$LOG_T" "$TB_T"
run_arm "$CONFIG_TK" "$OUT_TK" "$LOG_TK" "$TB_TK"
"$KGPW_RUN_PYTHON" scripts/prepare/verify_mixed3_rearag_runtime_probe_v2.py \
  --protocol "$PROTOCOL" --report_dir "$POSTFLIGHT"

