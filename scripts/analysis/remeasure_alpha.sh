#!/bin/bash
# ===========================================================================
# Re-measure α across the three datasets under ONE fixed state (P6).
#
#   bash scripts/analysis/remeasure_alpha.sh
#   N=300 bash scripts/analysis/remeasure_alpha.sh
#
# WHY a script instead of three commands: the whole point is that every arm sees
# an IDENTICAL (gate, bias, logprobs, entity-cache) state. The old numbers in
# statistics.md §2 differ because those four drifted between runs -- hotpotqa
# 0.292 ran with an EMPTY entity cache (f_confidence ≡ 0) and pre-D1
# f_entropy ≡ 1.0, while musique 0.854 ran post-D1 with real logprobs. Typing the
# commands by hand is exactly how that happened. See:
#   python scripts/analysis/alpha_diagnose.py --run <old run>/intermediate_data.json
#
# THE MOVING TARGET: EntityLinker.link() persists every newly linked mention via
# cache.set() (entity_linker.py:516) -- and it does so even under
# KGPW_KG_OFFLINE=1, since only the network SEARCH is gated, not the write. So
# whichever dataset runs first warms the cache for the ones after it, and the
# three alphas are then not comparable no matter how carefully the flags match.
# Fix: copy the cache to a per-dataset scratch file, so all three start from the
# same snapshot and their writes cannot reach each other.
#
# RUN IT TWICE. BIAS defaults to 0.78, the production value, but +0.78 was
# justified in 03_method.md:77 as compensating for a hardcoded f_entropy = 1.0 at
# inference -- and D1 (commit e6b2198) replaced that with real logprobs, so the
# premise is gone. W2 is negative, so f_entropy dropping from 1.0 to the measured
# ~0.56-0.67 already pushes the α-logit up by +0.19..+0.26, i.e. 25-33% of what
# +0.78 was added to supply. The two now partly double-count. To separate the
# dataset effect from the double-count, run both arms:
#   N=300 bash scripts/analysis/remeasure_alpha.sh
#   N=300 BIAS=0.0 OUT=outputs/alpha_remeasure_bias0 bash scripts/analysis/remeasure_alpha.sh
# A high α (>0.9) in the first but not the second is the correction talking, not
# the KG.
# ===========================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

N="${N:-300}"                 # match statistics.md §2's n=300
SEED="${SEED:-42}"
CKPT="${CKPT:-checkpoints/sft_student_split/final}"
ALPHA="${ALPHA:-checkpoints/prm_alpha_gate_v1reann_negfix/alpha_gate.pt}"
BIAS="${BIAS:-0.78}"          # the pipeline default; pass 0.0 to disable
RERANK="${RERANK:-10}"        # see memory: 6144 right-truncation cuts the KG block without this
OUT="${OUT:-outputs/alpha_remeasure}"
SNAP="$OUT/_entity_cache_snapshot.jsonl"

export PYTHONPATH="$ROOT:$ROOT/flashrag_src"
export KGPW_KG_OFFLINE=1      # no live Wikidata: it has been SSL-timing-out since 2026-08-16
                              # and a partial outage would silently vary KG per arm.

[ -d "$CKPT" ] || { echo "MISSING checkpoint: $CKPT"; exit 1; }
[ -f "$ALPHA" ] || { echo "MISSING alpha gate: $ALPHA"; exit 1; }

mkdir -p "$OUT"
# Freeze the cache ONCE. Every arm gets a private copy of this exact file.
if [ ! -f "$SNAP" ]; then
  cp indexes/entity_cache.jsonl "$SNAP"
  echo "froze entity cache snapshot: $(wc -l < "$SNAP") entries -> $SNAP"
else
  echo "reusing existing snapshot: $(wc -l < "$SNAP") entries"
fi
echo "snapshot md5: $(md5sum "$SNAP" | cut -d' ' -f1)"
echo

RUNS=()
for DS in hotpotqa 2wikimultihopqa musique; do
  CACHE="$OUT/${DS}_entity_cache.jsonl"
  cp "$SNAP" "$CACHE"                      # identical starting state, isolated writes
  echo "=== $DS (n=$N, seed=$SEED) ==="
  python scripts/eval/run_kg_proweight.py \
    --config configs/retrieval/hybrid_rrf_top50.yaml \
    --checkpoint "$CKPT" \
    --alpha_gate_path "$ALPHA" \
    --alpha_bias_correction "$BIAS" \
    --entity_cache_path "$CACHE" \
    --datasets "$DS" \
    --seeds "$SEED" \
    --test_sample_num "$N" \
    --rerank "$RERANK" \
    --save_root "$OUT/$DS"
  # Confirm the isolation actually held: a grown cache is fine (this arm's own
  # writes), a cache that differs from the snapshot BEFORE the run would not be.
  echo "  $DS cache: $(wc -l < "$SNAP") -> $(wc -l < "$CACHE") entries"
  RUNS+=("$OUT/$DS")
done

echo
echo "=== diagnose all three under the same gate ==="
ARGS=()
for d in "${RUNS[@]}"; do
  f=$(find "$d" -name intermediate_data.json | sort | tail -1)
  [ -n "$f" ] && ARGS+=(--run "$f")
done
python scripts/analysis/alpha_diagnose.py "${ARGS[@]}" \
  --gate "$ALPHA" --bias_correction "$BIAS" \
  --output "$OUT/alpha_diagnose.json"

echo
echo "READ THE VERDICT ABOVE. If it still says NOT COMPARABLE, the arms differed"
echo "in something this script does not control -- do not tabulate the numbers."
echo "If it says comparable, the α spread (if any) is a real dataset effect and"
echo "statistics.md §2 can be rewritten from $OUT/alpha_diagnose.json."
