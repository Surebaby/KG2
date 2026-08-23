#!/bin/bash
# ===========================================================================
# Verdict on the R10 smoke test. Run AFTER launch_split_ppo_smoke.sh finishes.
#
#   bash check_ppo_smoke.sh
#
# Prints PASS/FAIL per check so the R10 changes are confirmed by the log rather
# than by reading the config back.
# ===========================================================================
set -uo pipefail

OUT="${1:-/root/autodl-tmp/kgpaper/outputs/split_ppo_smoke}"
LOG="$OUT/train.log"
[ -f "$LOG" ] || { echo "no log at $LOG"; exit 1; }

fail=0
PYBIN="${PYBIN:-python3}"
say() { printf '%-46s %s\n' "$1" "$2"; }

echo "=== PPO smoke verdict — $LOG ==="
echo "    checks 0-7:   R10 reward rescaling"
echo "    checks 8-10:  2026-08-22 step-collapse fixes"
echo "    checks 11-13: 2026-08-23 量纲 fix (R_Text DC removal)"
echo

# 0. did it even run
# R10 inserted "(upd=N)" between step= and reward= on the log line, so the old
# "step=[0-9]+ reward=" pattern no longer matches and this check reported FAIL
# on a run that had completed successfully. Make (upd=N) optional so the script
# reads both the pre- and post-R10 log format.
STEP_RE="step=[0-9]+ (\(upd=[0-9]+\) )?reward="
if grep -qE "$STEP_RE" "$LOG"; then
  n=$(grep -cE "$STEP_RE" "$LOG")
  say "0. produced reward lines" "PASS ($n)"
else
  say "0. produced reward lines" "FAIL — no training steps; see traceback below"
  tail -30 "$LOG"; exit 1
fi

# 1. schedule reported in updates, and matches the flags
grep -oE "Phase 3b PPO schedule:.*" "$LOG" | tail -1

# 2. KG block not truncated
if grep -q "right-truncation will drop the trailing KG block" "$LOG"; then
  say "1. KG block intact (no truncation)" "FAIL — KG context is being cut"
  grep -m2 "right-truncation" "$LOG" | sed 's/^/     /'
  fail=1
else
  say "1. KG block intact (no truncation)" "PASS"
fi

# 3. kg_reward_share moved off ~0.009
# MEAN over all batches, not the last one. At batch_size=8 a single batch swings
# wildly -- the 2026-08-06 smoke run logged 0.011 then 0.209, and judging on
# either alone is a coin flip. The mean is what the design target refers to.
shares=$(grep -oE "kg_share=[0-9.]+" "$LOG" | grep -oE "[0-9.]+")
if [ -n "$shares" ]; then
  read -r share nb <<<"$(echo "$shares" | awk '{s+=$1;n++} END{printf "%.3f %d", (n?s/n:0), n}')"
  ok=$(awk -v s="$share" 'BEGIN{print (s>=0.04 && s<=0.25)?1:0}')
  rng=$(echo "$shares" | sort -g | awk 'NR==1{min=$1} {max=$1} END{printf "%s..%s", min, max}')
  [ "$ok" = 1 ] && say "2. kg_reward_share mean in [0.04,0.25]" "PASS ($share over $nb batches, range $rng)" \
                || { say "2. kg_reward_share mean in [0.04,0.25]" "FAIL ($share over $nb batches, range $rng; expected ~0.10)"; fail=1; }
  [ "$nb" -lt 3 ] && echo "     NOTE: only $nb batches -- mean is noisy, prefer SMOKE_TRAJ=160+"
else
  say "2. kg_reward_share" "UNKNOWN — not in log, check tensorboard"
fi

# 4. KL scale sane and coef stable
# -? matters: a NEGATIVE kl is the single most diagnostic failure here (rollout
# distribution != TRL's raw-logit scoring distribution), and the old unsigned
# pattern silently skipped those lines instead of flagging them. Anchor on " kl="
# so kg_share=/approxkl= cannot be picked up by accident.
kl=$(grep -oE " kl=-?[0-9.]+" "$LOG" | grep -oE "\-?[0-9.]+" | tail -1)
if [ -n "$kl" ]; then
  ok=$(awk -v k="$kl" 'BEGIN{print (k>5 && k<200)?1:0}')
  [ "$ok" = 1 ] && say "3. objective/kl in (5,200) per-seq sum" "PASS ($kl)" \
                || { say "3. objective/kl in (5,200)" "FAIL ($kl)"; fail=1; }
  awk -v k="$kl" 'BEGIN{if(k<0) print "     NOTE: negative KL => rollout/scoring distribution mismatch"}'
fi

# 5. reward components present (alpha not pinned, r_kg not identically zero)
last=$(grep -oE "α=[0-9.]+ r_kg=[0-9.]+ r_text=[0-9.]+" "$LOG" | tail -1)
say "4. reward components" "${last:-UNKNOWN}"
nz=$(grep -oE "r_kg=[0-9.]+" "$LOG" | grep -oE "[0-9.]+" | awk '$1>0' | wc -l)
tot=$(grep -coE "r_kg=[0-9.]+" "$LOG")
say "   r_kg nonzero" "$nz/$tot batches"

# 6. checkpoint landed
ck=$(ls -d "$OUT"/step_* 2>/dev/null | wc -l)
[ "$ck" -gt 0 ] && say "5. checkpoint written" "PASS ($ck: $(ls -d "$OUT"/step_* | xargs -n1 basename | tr '\n' ' '))" \
               || { say "5. checkpoint written" "FAIL — none under $OUT"; fail=1; }

# 7. peak GPU
# Nothing in phase3_ppo.py prints a "peak GPU memory" line, so this only ever
# reported "not logged". Read it from torch directly instead -- the value is
# per-process, so it is only meaningful while the run is alive; once it exits we
# fall back to the current card total.
pk=$(grep -oE "peak GPU memory: allocated [0-9.]+ GB \| reserved [0-9.]+ GB" "$LOG" | tail -1)
if [ -n "$pk" ]; then
  say "6. peak GPU" "$pk"
elif pgrep -f phase3_ppo.py >/dev/null 2>&1; then
  say "6. peak GPU" "$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader) (live)"
else
  say "6. peak GPU" "run exited; card now $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  echo "     NOTE: peak was not captured. Watch nvidia-smi during the full run;"
  echo "     the 2026-08-06 smoke run held 92.4/97.9 GB, not the ~58 GB estimated."
fi

# 8. valid rate over the run
vr=$(grep -oE "valid=[0-9]+/[0-9]+" "$LOG" | awk -F'[=/]' '{a+=$2;b+=$3} END{if(b) printf "%.3f (%d/%d)", a/b, a, b}')
say "7. valid_rate over run" "${vr:-UNKNOWN}"

# ---------------------------------------------------------------------------
# 2026-08-22: the two checks this smoke run actually exists for.
#
# Everything above was written for R10 (reward RESCALING). This round changes
# what the reward REWARDS -- min_valid_steps 2->3 plus shortfall_coef 0.25 --
# and neither of the two failure modes it can produce was detectable above. A
# run could print "ALL CHECKS PASS" while the step collapse continued.
# ---------------------------------------------------------------------------

# 9. ABORT CRITERION (phase3_ppo.yaml, min_valid_steps comment).
# R7-A already tried min_valid_steps=3: valid_rate stuck at 9.2% with no upward
# trend over 1600 steps. If that repeats, revert min_valid_steps to 2 and rely
# on shortfall_coef alone -- do NOT wait it out, the flat curve does not recover.
# Judged on updates 40-60 so it is measured where the criterion is stated
# (update ~50) rather than over the whole run, which the early updates drag down.
vr50=$(grep -oE "\(upd=[0-9]+\) .*valid=[0-9]+/[0-9]+" "$LOG" \
       | sed -E 's/.*\(upd=([0-9]+)\).*valid=([0-9]+)\/([0-9]+).*/\1 \2 \3/' \
       | awk '$1>=40 && $1<=60 {a+=$2;b+=$3} END{if(b) printf "%.3f %d %d", a/b, a, b}')
if [ -n "$vr50" ]; then
  read -r v va vb <<<"$vr50"
  ok=$(awk -v v="$v" 'BEGIN{print (v>0.5)?1:0}')
  [ "$ok" = 1 ] && say "8. valid_rate > 0.5 by upd ~50" "PASS ($v = $va/$vb over upd 40-60)" \
                || { say "8. valid_rate > 0.5 by upd ~50" "FAIL ($v = $va/$vb) -- ABORT CRITERION HIT"
                     echo "     R7-A repeated. Set min_valid_steps back to 2 in"
                     echo "     configs/training/phase3_ppo.yaml and keep shortfall_coef=0.25."
                     fail=1; }
else
  say "8. valid_rate > 0.5 by upd ~50" "UNKNOWN -- run did not reach update 40 (need SMOKE_TRAJ >= 240)"
fi

# 10. STEP COLLAPSE, the thing shortfall_coef exists to price.
# n_steps in the log is the BATCH TOTAL parsed steps, not per trajectory, so
# divide by batch_size. Silver teaches 3.36; PPO(1) went 2.84 -> 2.04 and PPO(2)
# 3.13 -> 2.01, both converging on the old min_valid_steps=2. Compare the FIRST
# and LAST thirds: the absolute value matters less than the trend, because a run
# that starts at 3.4 and ends at 2.0 has collapsed even though its mean looks OK.
BS=$(grep -oE "batch_size=[0-9]+" "$LOG" | head -1 | grep -oE "[0-9]+")
BS="${BS:-4}"
steps=$(grep -oE "n_steps=[0-9]+" "$LOG" | grep -oE "[0-9]+")
nst=$(echo "$steps" | grep -c .)
if [ "$nst" -ge 6 ]; then
  read -r first last <<<"$(echo "$steps" | awk -v bs="$BS" -v n="$nst" '
    {v[NR]=$1/bs} END{t=int(n/3); for(i=1;i<=t;i++)a+=v[i]; for(i=n-t+1;i<=n;i++)b+=v[i];
                      printf "%.2f %.2f", a/t, b/t}')"
  drop=$(awk -v a="$first" -v b="$last" 'BEGIN{printf "%.2f", a-b}')
  # The pathology is CONVERGING ON THE GATE, not "being low". PPO(1)/(2) ended at
  # 2.04/2.01 against min_valid_steps=2; the same behaviour under the new gate
  # ends at ~3.0, which a naive ">= 2.7" test would happily pass. So the test is
  # stated against the gate: the last third must sit clear of min_valid_steps by
  # a margin, and must not be trending down toward it.
  MVS=$(grep -oE "min_valid_steps[=: ]+[0-9]+" "$LOG" | head -1 | grep -oE "[0-9]+")
  if [ -z "$MVS" ]; then
    MVS=$("$PYBIN" -c "import yaml;print(yaml.safe_load(open('configs/training/phase3_ppo.yaml'))['training']['ppo']['min_valid_steps'])" 2>/dev/null || echo 3)
  fi
  margin=$(awk -v b="$last" -v m="$MVS" 'BEGIN{printf "%.2f", b-m}')
  ok=$(awk -v mg="$margin" -v d="$drop" 'BEGIN{print (mg>=0.2 && d<0.4)?1:0}')
  [ "$ok" = 1 ] && say "9. steps/traj clear of the gate, no collapse" "PASS (first third $first -> last third $last, drop $drop, gate $MVS, margin +$margin)" \
                || { say "9. steps/traj clear of the gate, no collapse" "FAIL (first third $first -> last third $last, drop $drop, gate $MVS, margin $margin)"
                     echo "     Target is silver's 3.36; the gate is min_valid_steps=$MVS."
                     echo "     Ending within 0.2 of the gate = writing the MINIMUM, which is the same"
                     echo "     pathology as PPO(1)/(2) ending at 2.04/2.01 against a gate of 2 --"
                     echo "     the policy found the cheapest passing trajectory, just at a higher floor."
                     echo "     Consider raising shortfall_coef above 0.25, or target_steps above 3."
                     fail=1; }
else
  say "9. steps/traj trend" "UNKNOWN -- only $nst batches, need >= 6"
fi

# 11. The rebuilt question-KG index must actually be in use (R-2).
# A 0% hit rate is now a hard error, but a run that silently fell back to the
# DEFAULT index would still start -- so confirm which file was loaded.
qk=$(grep -oE "Loaded [0-9]+ question→KG entries from [^ ]+" "$LOG" | tail -1)
if [ -n "$qk" ]; then
  say "10. question-KG index loaded" "$(echo "$qk" | sed 's/.*from //')"
  echo "     $(echo "$qk" | grep -oE 'Loaded [0-9]+ [^ ]+')"
  case "$qk" in
    *question_kg_index_v2_train.json*) : ;;
    *) echo "     WARNING: not the train-fold index. The shipped v2 index is built"
       echo "     from the DEV splits and is ABSENT for 100% of PPO prompts."
       fail=1 ;;
  esac
  am=$(grep -oE "[0-9]+ ABSENT from the index \([0-9.]+%" "$LOG" | tail -1)
  [ -n "$am" ] && echo "     index miss breakdown: $am)"
else
  say "10. question-KG index loaded" "UNKNOWN -- no load line in log"
fi

# ---------------------------------------------------------------------------
# 2026-08-23: the 量纲 fix (retraining_plan §9.4-1 / D2).
#
# Everything above was written for R10 (reward rescaling) and the 2026-08-22
# step-collapse fixes. This round removes R_Text's DC offset, and neither of its
# two failure modes is detectable above: a run whose config never reached the
# dataclass, and a run where the sign of dR/dalpha stayed negative anyway.
# ---------------------------------------------------------------------------

# 12. Did the centering actually RUN?
# The specific trap: schemas.py sets extra="allow", so a YAML key that is not
# explicitly forwarded in scripts/train/phase3_ppo.py is accepted and silently
# ignored -- exactly how ppo_max_kg_triples stayed at its default while the YAML
# "set" it. So this is judged on the LOG, not on the config.
#
# Read both channels: r_text is the RAW scorer output and must stay near its
# measured 0.63, while "used" is the centered value that entered r_total and must
# sit near 0. That pair distinguishes the two ways this can go wrong --
#   used ~ 0.63  => the flag never reached the dataclass (centering not running)
#   r_text ~ 0   => the SCORER changed, which is a different problem entirely
# -- which a single number could not.
cu=$(grep -oE "r_text=-?[0-9.]+ \(used -?[0-9.]+ base -?[0-9.]+\)" "$LOG" | tail -1)
if [ -n "$cu" ]; then
  read -r rt us bs <<<"$(echo "$cu" | grep -oE -- "-?[0-9.]+" | tr '\n' ' ')"
  # Judge `used` on the MEAN over the run, not the last line: a single batch of 4
  # trajectories is ~12 step samples and swings widely (the same reason check 2
  # takes a mean).
  um=$(grep -oE "\(used -?[0-9.]+" "$LOG" | grep -oE -- "-?[0-9.]+" \
       | awk '{s+=$1;n++} END{if(n) printf "%.4f %d", s/n, n}')
  read -r umean un <<<"$um"
  # Two conditions, not one. `used` near 0 says the centering ran; but a DUMMY
  # text backend also returns 0.0 for everything, which centers to 0 trivially
  # and would pass a used-only test while the text channel was absent entirely.
  # So the RAW value must also be a plausible scorer output. `RearagPromptScorer`
  # returns tanh((2.5-nll)/1.5), measured mean 0.6284; require it clear of 0.
  rm_=$(grep -oE " r_text=-?[0-9.]+" "$LOG" | grep -oE -- "-?[0-9.]+" \
        | awk '{s+=$1;n++} END{if(n) printf "%.4f", s/n}')
  ok=$(awk -v u="$umean" -v r="$rm_" 'BEGIN{
         print (u<0.15 && u>-0.15 && (r>0.15 || r<-0.15))?1:0}')
  [ "$ok" = 1 ] && say "11. r_text centered (量纲 fix live)" "PASS (raw mean $rm_, used mean $umean over $un, baseline $bs)" \
                || { say "11. r_text centered (量纲 fix live)" "FAIL (raw mean $rm_, used mean $umean over $un, baseline $bs)"
                     awk -v u="$umean" -v r="$rm_" 'BEGIN{
                       if (u > 0.4) {
                         print "     used ~ raw => centering did NOT run. center_text_reward did not";
                         print "     reach Phase3PPOConfig: check the explicit forwarding in";
                         print "     scripts/train/phase3_ppo.py (schemas.py extra=allow hides this).";
                       } else if (r < 0.2) {
                         print "     raw r_text is also near 0 => the SCORER changed, not the centering.";
                         print "     Check text_reward_backend: a dummy backend returns 0.0.";
                       } else {
                         print "     Centering ran but the residual is off-center: the baseline is";
                         print "     tracking a drifting scorer. Check reward/text_baseline in tensorboard.";
                       }}'
                     fail=1; }
else
  say "11. r_text centered (量纲 fix live)" "UNKNOWN -- no '(used ... base ...)' in log; pre-2026-08-23 build?"
fi

# 13. THE BUG the centering exists to fix: the sign of dR/dalpha.
# MEASURED at -0.148 over PPO(1) (r_kg 0.0896, r_text 0.6284, c_text 0.3,
# c_step 1.5): the reward paid the policy to LOWER alpha, and since alpha rises
# with f_density = |E|/(|V|+eps), "lower alpha" means "cite a sparser subgraph".
# A KG-grounding reward was rewarding less KG grounding.
#
# Judged on the MEAN over the run and on the FRACTION of batches that are still
# negative -- not on the last line. Individual batches may legitimately go
# negative (r_kg varies); a PERSISTENTLY negative mean means the KG channel is
# still being outbid and centering alone was not enough.
dr=$(grep -oE "dR/dα=-?[0-9.]+" "$LOG" | grep -oE -- "-?[0-9.]+")
if [ -n "$dr" ]; then
  read -r drm drn drneg <<<"$(echo "$dr" | awk '{s+=$1;n++; if($1<0)k++} END{if(n) printf "%.4f %d %.2f", s/n, n, k/n}')"
  ok=$(awk -v m="$drm" -v f="$drneg" 'BEGIN{print (m>0 && f<0.5)?1:0}')
  [ "$ok" = 1 ] && say "12. dR/dα > 0 (gate points at the KG)" "PASS (mean $drm over $drn batches, ${drneg} negative)" \
                || { say "12. dR/dα > 0 (gate points at the KG)" "FAIL (mean $drm over $drn batches, ${drneg} negative)"
                     echo "     The reward is still paying the policy to LOWER alpha, i.e. to cite a"
                     echo "     sparser subgraph. Pre-fix this measured -0.148. If check 11 PASSED,"
                     echo "     centering is running and r_kg itself is simply too small to outbid"
                     echo "     the residual text variation -- that is a text_reward_scale question"
                     echo "     (§9.5: retune it ALONE, in a separate run, never alongside centering)."
                     fail=1; }
else
  say "12. dR/dα sign" "UNKNOWN -- not in log"
fi

# 14. §9.6: direct evidence for the "r_kg is sparse" claim.
# Not pass/fail -- it is the measurement §9.6 flagged as MISSING. The claim that
# r_kg_mean 0.0896 is driven by citation FREQUENCY rather than accuracy rested on
# the batch mean plus a 13/13 precision reading; this reports the fraction of
# step records on the PRM's NEUTRAL branch (r_kg exactly 0) directly. A high value
# confirms the diagnosis and says the lever is upstream citation rate (§12 P1-a),
# NOT a tighter matcher.
# NOTE: do NOT pipe this through `grep -oE "[0-9]+"` -- the field NAME contains
# a 0 ("r_kg_0="), so that yields two numbers per match and halves the mean.
zf=$(grep -oE "r_kg_0=[0-9]+%" "$LOG" | sed -E 's/^r_kg_0=([0-9]+)%$/\1/' \
     | awk '{s+=$1;n++} END{if(n) printf "%.1f%% over %d batches", s/n, n}')
say "13. r_kg == 0 fraction (§9.6, FYI)" "${zf:-UNKNOWN -- not in log}"

echo
if [ "$fail" = 0 ]; then
  echo "ALL CHECKS PASS — safe to run: bash launch_split_ppo.sh"
else
  echo "SOME CHECKS FAILED — do not start the full run yet."
fi
exit "$fail"
