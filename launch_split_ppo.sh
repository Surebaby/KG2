#!/bin/bash
# ===========================================================================
# Phase 3b PPO — on the train fold, anchored to the split SFT checkpoint
#
#   bash launch_split_ppo.sh
#
# Run launch_split_sft.sh FIRST: this resumes from its output.
#
# Treat this as a search with checkpoints, not a run whose endpoint you keep.
# Pick the best on the VAL fold afterwards, never on test — choosing a
# checkpoint on test is tuning on test.
#
# R10 (2026-08-06) — the "collapse at step 500" folklore was a units error.
# total_ppo_steps counts TRAJECTORIES SEEN, not optimiser steps
# (phase3_ppo.py:914 `n_seen += cfg.batch_size`, :821 `while n_seen <
# cfg.total_steps`). At batch_size=8 the old 2000 was 250 gradient updates and
# the reported peak near "step 500" was 63 updates in. At lr=1e-6 with
# clipfrac≈0.001 that is noise in an 8-trajectory batch, not policy
# degradation: r9_v5 logged reward 7.68 -> -3.29 -> 9.50 -> 0.76 with no trend,
# and the swings track valid=3/8..8/8 because the invalid penalty was ±10.
#
# R10 FINAL SCHEDULE, as it actually RAN (completed 2026-08-07): 3000
# trajectories at batch_size=4 = 750 optimiser updates in 3.73 h, mean
# 17.91s/update over 750. Checkpoints land every 500 seen (= every 125 updates),
# i.e. 6 of them at step_500 .. step_2500 plus final. Divide the directory number
# by 4 for the update count.
#
# The numbers this header carried before the run are superseded and were
# consistently pessimistic, so do not plan against them:
#   predicted 375 updates @ 39.8s = 4.1 h   ->  MEASURED 750 @ 17.91s = 3.73 h
# The 39.8s was measured at batch_size=8, and the OOM fix (8 -> 4, see
# configs/training/phase3_ppo.yaml) cut rollout 19->9s and reward 6.6->2.5s
# nearly proportionally, while the ppo stage's ~6.5s fixed cost did not double.
# Halving the batch therefore bought speed as well as safety — it doubled the
# update count and still finished sooner.
#
# Provenance of the pre-run 39.8s figure, kept because the decomposition is still
# the right way to reason about the stages: measured over 20 updates at bs=8 as
# rollout ~19.0s, reward ~6.6s, ppo ~14.2s, from two smoke-verified changes —
# batched rollout (48s -> 19s) and ppo_epochs 2 -> 1 (23.5s -> 14.6s). Baseline
# was 80.8s/update.
#
# Memory: three 8B/9B models are resident at once —
#   policy 14.96 + ref_model 14.96 + rearag-9B 17.21 + lora 0.44 = 47.6 GB of
#   weights before any activation.
#
# Recalculated 2026-08-06 against the MEASURED Phase 3a peak (55.60 GB allocated
# at bs4 x 6144, logged in checkpoints/sft_student_split/manifest.json), which
# put activations+workspace at 40.20 GB — the from-first-principles derivation
# had said 46 GB total and undershot by 17 GB, so it is not trusted here.
# Scaling that measured figure: KV cache 6.25 GB (bs8, 6400 tok, GQA) + backward
# logits 0.37 GB (mini_batch_size=1 backprops over the 256-token response only,
# not a 6144-token supervised forward) + scoring forward 3.66 GB (no stored
# graph) => ~58 GB, i.e. ~37 GB of headroom, not the ~18 GB assumed earlier.
#
# max_input_length still stays at 6144: it must equal phase3_sft.yaml's
# sft_max_length or the two phases see different passage counts.
#
# MEASURED, and the ~58 GB above is WRONG — do not plan against it. PPO holds the
# policy, a deepcopied reference model, the ReaRAG-9B scorer, a value head and the
# optimiser state simultaneously, and scaling Phase 3a's supervised peak does not
# capture that. Peaks of 97887 MiB total:
#   bs=8, two smoke runs .... 92.4 / 93.0 GB
#   bs=8, 500-traj run ...... 95681 MiB
#   bs=8, run that OOM'd .... 95683 MiB
#   bs=4, COMPLETED run ..... 96083 MiB   <- the config that ships
#
# Note bs=4 peaks HIGHEST. Halving the batch did not free memory; it reduced how
# often the run draws a long sequence at the 3.06 GB worst-case logits allocation
# that killed the bs=8 run. Headroom is ~1.8 GB regardless.
#
# Practical consequence: mini_batch_size and batch_size cannot be raised on this
# card. The remaining ladder steps, if a future change needs room, are
# ppo_max_kg_triples 30->20 then passages — both weaken what the paper measures.
# Watch nvidia-smi rather than trusting any estimate here.
# ===========================================================================
set -euo pipefail

REMOTE_ROOT=/root/autodl-tmp/kgpaper
cd "$REMOTE_ROOT"

export PYTHONPATH="$REMOTE_ROOT:$REMOTE_ROOT/flashrag_src"
export KGPW_FLASHRAG_ROOT="$REMOTE_ROOT/flashrag_src"
export KGPW_PROJECT_ROOT="$REMOTE_ROOT"
export KGPW_DATA_DIR="$REMOTE_ROOT/data"
export KGPW_INDEX_DIR="$REMOTE_ROOT/indexes"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset HTTP_PROXY HTTPS_PROXY

# model_path("rearag") falls back to the HuggingFace id "THU-KEG/ReaRAG-9B",
# which would try to DOWNLOAD 18 GB mid-run. Pin the local copy.
# Verified on the box 2026-08-06: it lives in /root/autodl-tmp/models/, NOT in
# the repo's own models/ dir (which has llama3-8b, corag, r1-searcher, e5, bge).
export KGPW_REARAG_PATH=/root/autodl-tmp/models/rearag-9b

PY=/root/autodl-tmp/kgpw_env/bin/python
SILVER="$REMOTE_ROOT/data/silver_data/silver_v1_reannotated.jsonl"
SFT="$REMOTE_ROOT/checkpoints/sft_student_split/final"
# alpha_gate only — the trained 3-way PRM head is NOT consumed by Phase 3b
# (composite_reward.py uses the rule-based prm_annotator.label()), so this is
# the only Phase 2 artefact that matters here.
ALPHA="$REMOTE_ROOT/checkpoints/prm_alpha_gate_v1reann_negfix/alpha_gate.pt"
OUT="$REMOTE_ROOT/outputs/split_ppo"

for p in "$SILVER" "$ALPHA"; do
  [ -e "$p" ] || { echo "MISSING: $p"; exit 1; }
done
[ -d "$SFT" ] || { echo "MISSING SFT checkpoint: $SFT — run launch_split_sft.sh first"; exit 1; }
[ -d "$KGPW_REARAG_PATH" ] || { echo "MISSING rearag: $KGPW_REARAG_PATH (would silently download)"; exit 1; }

mkdir -p "$OUT"

nohup "$PY" scripts/train/phase3_ppo.py \
  --config configs/training/phase3_ppo.yaml \
  --silver_data "$SILVER" \
  --sft_checkpoint "$SFT" \
  --alpha_gate_path "$ALPHA" \
  --text_reward_backend rearag \
  --output_dir "$OUT" \
  --split train \
  > "$OUT/train.log" 2>&1 &

echo "Launched Phase 3b PPO (pid $!)"
echo "  log:    tail -f $OUT/train.log"
echo "  split:  grep 'Phase 3b PPO split' $OUT/train.log    # must say fold=train"
echo "  reward: grep -o 'step=[0-9]* .*reward=[-0-9.]*' $OUT/train.log | tail"
echo "  board:  tensorboard --logdir $OUT/tensorboard"
echo "  expect: 3000 trajectories = 750 optimiser updates, 3.73 h at 17.91s/update"
echo "          (MEASURED on the completed run, not projected);"
echo "          checkpoints step_500 .. step_2500 + final (divide by 4 for updates)"
echo
echo "WATCH, in this order:"
echo "  1. the truncation warning 'right-truncation will drop the trailing KG"
echo "     block' — should be ABSENT now that ppo_max_kg_triples=30. If it"
echo "     fires, the KG context is being cut and the run is uninformative."
echo "  2. custom/kg_reward_share — should sit near 0.10. It was 0.009 under"
echo "     the old outcome_weight=10 / step_reward_scale=0.3, i.e. the KG"
echo "     process reward was numerically noise."
echo "  3. objective/kl vs kl_coef — KL is a PER-SEQUENCE SUM (28-69 measured"
echo "     at 256 tokens), which is why target is now 40.0, not 8.0."
echo "  4. custom/valid_rate and ppo/policy/entropy together. Do NOT lower the"
echo "     rollout temperature to fix valid_rate: sampling must stay at"
echo "     temperature=1.0/top_p=1.0/top_k=0 or TRL's logp recomputation no"
echo "     longer matches the sampling distribution and KL goes negative."
echo "  5. peak GPU: MEASURED 96083 MiB of 97887 on the completed bs=4 run."
echo "     Halving batch_size did NOT free memory — the bs=8 run peaked at"
echo "     95683, so bs=4 is 400 MB HIGHER. Headroom is ~1.8 GB either way."
echo "     What bs=4 bought is fewer draws at the 3.06 GB worst-case logits"
echo "     alloc, not headroom (see the yaml's OOM comment). The ~58 GB in"
echo "     the header is wrong. Do NOT raise batch_size or mini_batch_size."
echo "  6. TIMING upd=N rollout=..s reward=..s ppo=..s — MEASURED mean"
echo "     17.91s/update over 750 updates at bs=4 (rollout ~9s, reward ~2.5s,"
echo "     ppo ~6.5s). Was 39.8s at bs=8, 80.8s serial. If a stage regresses,"
echo "     that line says which one before the ETA drifts."
