#!/bin/bash
# Seeds 42 and 2024 for both arms, --rerank 10 (no truncation, ~9 min/run).
#
# Seed 13 is already on disk:
#   outputs/split_sft_rerank10        split ckpt, EM 0.420
#   outputs/elite_baseline_curenv     elite ckpt, but topk50 -> NOT comparable
#     to a rerank10 run. So elite needs seed 13 at rerank10 too, hence 3 seeds
#     for elite and 2 for split below.
#
# All runs share one env, one config, one retrieval stack; the only variable is
# (checkpoint, seed). That is what makes the paired bootstrap in
# scripts/_summarize_attrib.py attributable to the checkpoint.
set -uo pipefail
cd /home/zjulab/kgpaper

run () {
  local out="outputs/$1"; shift
  mkdir -p "$out"
  echo "=== $out starting $(date +%H:%M:%S) ==="
  python scripts/eval/run_kg_proweight.py \
    --config configs/eval/kg_proweight.yaml \
    --datasets hotpotqa --split dev --test_sample_num 100 --gpu_id 0 \
    --rerank 10 --save_root "$out" "$@" > "$out/run.log" 2>&1
  local rc=$? m
  m=$(find "$out" -name metric_score.json | head -1)
  echo "=== $out exit=$rc $( [ -f "$m" ] && tr -d '\n ' < "$m" || echo NO_METRIC ) trunc=$(grep -c 'greater than the maximum length' "$out/run.log") ==="
}

SPLIT=checkpoints/sft_student_split/final
ELITE=checkpoints/sft_student_elite/final

# elite at rerank10 — seed 13 included, it has no rerank10 run yet
run elite_rerank10_s13   --checkpoint "$ELITE" --seeds 13
run split_rerank10_s42   --checkpoint "$SPLIT" --seeds 42
run elite_rerank10_s42   --checkpoint "$ELITE" --seeds 42
run split_rerank10_s2024 --checkpoint "$SPLIT" --seeds 2024
run elite_rerank10_s2024 --checkpoint "$ELITE" --seeds 2024

echo "=== sweep done $(date +%H:%M:%S) ==="
