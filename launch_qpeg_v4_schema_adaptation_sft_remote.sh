#!/usr/bin/env bash
# 96GB remote: approved QPEG-v4 train-only schema-adaptation continued-SFT.
set -euo pipefail
KGPW_ROOT=${KGPW_ROOT:-/root/autodl-tmp/kgpaper}
KGPW_PYTHON=${KGPW_PYTHON:-/root/autodl-tmp/kgpw_env/bin/python}
cd "$KGPW_ROOT"

export PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=configs/training/phase3_sft_qpeg_v4_schema_adaptation_n2400_seed42.yaml
DATA=data/silver_data/qpeg_v4_schema_adaptation_n2400_seed42_v2/silver_curriculum.jsonl
PROTOCOL=outputs/audits/qpeg_v4_schema_adaptation_protocol_v1/protocol.json
OUT=checkpoints/sft_qpeg_v4_schema_adaptation_n2400_seed42
LOG=logs/training/sft_qpeg_v4_schema_adaptation_n2400_seed42.log
mkdir -p logs/training
test ! -e "$OUT"
test ! -e "$LOG"
test -s "$CONFIG"
test -s "$DATA"
test -s "$PROTOCOL"
test -d checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final
test "$(sha256sum "$CONFIG" | cut -d' ' -f1)" = e050d3bc100a6f4e6fc0ac835bd2af07deab8d8e8f725838088d64c7648c529c
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = 0107dcd8847e316127b24f490db8328edd3698d7cdbd64194856be6381f2d3a6
test "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" = 1f53b0957f3e82bd2b5fdc43992b4f7b7a8012215b079731cbb1424ee5a07c2f

"$KGPW_PYTHON" -m pytest -q \
  tests/test_qpeg_v4_schema_adaptation_data.py \
  tests/test_phase3_sft_config_forwarding.py \
  tests/test_silver_split.py

exec "$KGPW_PYTHON" scripts/train/phase3_sft.py \
  --config "$CONFIG" \
  2>&1 | tee "$LOG"
