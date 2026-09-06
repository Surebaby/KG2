#!/usr/bin/env bash
# Dedicated remote release entry; never source the old checkout's environment.
set -euo pipefail
sourcegate_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export KGPW_PROJECT_ROOT="$sourcegate_root"
export KGPW_FLASHRAG_ROOT="$sourcegate_root/flashrag_src"
export KGPW_LLAMA3_PATH="$sourcegate_root/models/llama3-8b"
export KGPW_REARAG_PATH="$sourcegate_root/models/rearag-9b"
export KGPW_KG_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# AutoDL AutoPanel recursively discovers each experiment below its official root.
export KGPW_TB_ROOT="${KGPW_TB_ROOT:-/root/tf-logs/kgpaper}"
export PYTHONPATH="$sourcegate_root:$sourcegate_root/flashrag_src${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$sourcegate_root"
exec /root/autodl-tmp/kgpw_env/bin/python "$@"
