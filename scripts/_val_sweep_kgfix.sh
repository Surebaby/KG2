#!/bin/bash
# ===========================================================================
# Re-run the 6-checkpoint val selection under the FIXED KG index.
#
#   bash scripts/_val_sweep_kgfix.sh
#
# Identical to the 2026-08-07 03:07-06:07 sweep (outputs/val_select/) in every
# argument. The ONLY difference is what indexes/kg_cache/question_kg_index_v2.json
# resolves to:
#   before: question_kg_index_v2.json.parked  -> 0% dev coverage, 64% live
#           Wikidata fallback, 36% empty KG
#   now:    question_kg_index_devfix.json     -> 90% prebuilt, 0% fallback,
#           10% empty (measured on the n=20 smoke run at 13:09)
# So this sweep minus outputs/val_select/ isolates the KG supply mode.
#
# Writes to outputs/val_select_kgfix/ — the pre-fix numbers in
# outputs/val_select/ are NOT overwritten; both are needed for the comparison.
#
# Why every flag matters:
#   --alpha_gate_path   run_kg_proweight.py:60 defaults to prm_alpha_gate/, a
#                       DIFFERENT gate than training loaded. Never omit this.
#   --rerank 10         without it the 6144 right-truncation cuts the KG block
#                       and the assistant anchor, making KG numbers meaningless.
#   --split dev         PPO's val fold is hash-bucketed from silver and has no
#                       .jsonl; --split val would look for a nonexistent file.
#                       dev is cleanly held out anyway: dev∩silver = 0/22398.
#   --test_sample_num 300   SE ~2.5pp/point. The pre-fix sweep's 4.3pp spread
#                       across 6 checkpoints was inside noise at this n.
#
# Cost: MEASURED 36 min/checkpoint on the pre-fix sweep (03:07 -> 06:07 for six),
# so ~3.6 h. Not 17 min — the GPU idles during CPU-bound retrieval/reranking, so
# GPU-util-based estimates undershoot by 2x.
# ===========================================================================
set -uo pipefail
cd /home/zjulab/kgpaper

ALPHA=checkpoints/prm_alpha_gate_v1reann_negfix/alpha_gate.pt
ROOT=outputs/val_select_kgfix

# Fail before burning 3.6 h rather than after.
[ -f "$ALPHA" ] || { echo "MISSING alpha gate: $ALPHA"; exit 1; }
[ -f data/hotpotqa/dev.jsonl ] || { echo "MISSING data/hotpotqa/dev.jsonl"; exit 1; }
KGIDX=$(readlink -f indexes/kg_cache/question_kg_index_v2.json)
case "$KGIDX" in
  *question_kg_index_devfix.json) ;;
  *) echo "KG index v2 resolves to $KGIDX, expected question_kg_index_devfix.json"; exit 1 ;;
esac
echo "kg index: $KGIDX"
echo "alpha gate md5: $(md5sum "$ALPHA" | cut -d' ' -f1)   (training used 6d1a1756...)"
echo

for ck in step_500 step_1000 step_1500 step_2000 step_2500 final; do
  echo "########## $ck ##########"
  date +"start %H:%M:%S"
  python scripts/eval/run_kg_proweight.py \
    --checkpoint "checkpoints/ppo_r10_split/$ck" \
    --alpha_gate_path "$ALPHA" \
    --datasets hotpotqa \
    --split dev \
    --test_sample_num 300 \
    --seeds 42 \
    --rerank 10 \
    --save_root "$ROOT/$ck" 2>&1 \
    | grep -E "^\{|α summary|KG source|ERROR|Traceback|OutOfMemory|greater than the maximum length"
  date +"end   %H:%M:%S"
  echo
done
echo "ALL_DONE"
