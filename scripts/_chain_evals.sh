#!/bin/bash
# Chain the two attribution runs behind the in-flight one (pid passed as $1).
#
# Run 1 (already running): new checkpoint, default topk 50 -> truncated prompts.
# Run 2 (here): sft_student_elite, SAME env + SAME config as run 1.
#     Isolates the checkpoint: run1 - run2 is attributable to weights alone,
#     because the local env has drifted since ablation_v6 (torch 2.4.1->2.6.0,
#     transformers 4.48->4.49, peft 0.19.1->0.20.0) and the historical 0.37
#     therefore cannot serve as the control.
# Run 3 (here): new checkpoint + --rerank 10 -> no truncation.
#     run3 - run1 quantifies what the 6144 right-truncation costs in EM.
set -uo pipefail
cd /home/zjulab/kgpaper

WAIT_PID="${1:?usage: _chain_evals.sh <pid-of-run1>}"

# Wait on the pid directly. Do NOT pgrep for the script name here: the pattern
# would match this watcher's own command line and it would wait on itself.
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
echo "=== run1 (pid $WAIT_PID) exited $(date +%H:%M:%S) ==="

for i in 1 2 3 4 5 6; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 2000 ] && break
  echo "  GPU still at ${used} MiB, waiting"; sleep 15
done

run () {
  local name="$1"; shift
  local out="outputs/$name"
  mkdir -p "$out"
  echo "=== $name starting $(date +%H:%M:%S) ==="
  python scripts/eval/run_kg_proweight.py \
    --config configs/eval/kg_proweight.yaml \
    --datasets hotpotqa --split dev --test_sample_num 100 --seeds 13 --gpu_id 0 \
    --save_root "$out" "$@" > "$out/run.log" 2>&1
  local rc=$?
  local m
  m=$(find "$out" -name metric_score.json | head -1)
  echo "=== $name exit=$rc  $( [ -f "$m" ] && tr -d '\n ' < "$m" || echo NO_METRIC ) ==="
  echo "    truncation warnings: $(grep -c 'greater than the maximum length' "$out/run.log")"
}

run elite_baseline_curenv --checkpoint checkpoints/sft_student_elite/final
run split_sft_rerank10    --checkpoint checkpoints/sft_student_split/final --rerank 10

echo "=== chain done $(date +%H:%M:%S) ==="
