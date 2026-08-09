#!/bin/bash
# R10 sync verification — runs ON the remote box.
#
# Exists because sync_split.sh's inline verification block streams the 1.37 GB
# silver file for its fold table, and everything after that step (the R10
# assertions, the CLI-override check, the rearag presence check) never got a
# chance to print. This script does those checks and nothing slow.
set -u
cd /root/autodl-tmp/kgpaper
export PYTHONPATH=/root/autodl-tmp/kgpaper:/root/autodl-tmp/kgpaper/flashrag_src
PY=/root/autodl-tmp/kgpw_env/bin/python

echo "=== 1. the 6 files R10 touched: md5 + mtime ==="
for f in configs/training/phase3_ppo.yaml kgproweight/training/phase3_ppo.py \
         scripts/train/phase3_ppo.py launch_split_ppo.sh \
         launch_split_ppo_smoke.sh check_ppo_smoke.sh; do
  if [ -f "$f" ]; then
    printf '  %s  %s  %s\n' "$(md5sum "$f" | cut -c1-12)" \
      "$(date -r "$f" '+%m-%d %H:%M')" "$f"
  else
    printf '  MISSING       %s\n' "$f"
  fi
done

echo
echo "=== 2. R10 values as the config loader actually sees them ==="
$PY - <<'PYEOF'
from kgproweight.config import ProjectConfig, load_config
from kgproweight.training.phase3_ppo import Phase3PPOConfig
p = load_config('configs/training/phase3_ppo.yaml',
                validate=ProjectConfig).training.ppo
exp = {'total_ppo_steps': 16000, 'save_every_steps': 2000, 'target_kl': 40.0,
       'outcome_weight': 4.0, 'step_reward_scale': 1.5}
bad = {k: (getattr(p, k), v) for k, v in exp.items() if getattr(p, k) != v}
print('  YAML:', 'all 5 applied' if not bad else 'MISMATCH %r' % bad)
t = Phase3PPOConfig(silver_path='s', output_dir='o').ppo_max_kg_triples
print('  ppo_max_kg_triples = %s %s' % (t, '' if t == 30 else '<-- WANT 30'))
bs = p.batch_size
print('  schedule: %d traj / bs %d = %d updates; ckpt every %d (= %d upd)'
      % (p.total_ppo_steps, bs, p.total_ppo_steps // bs,
         p.save_every_steps, p.save_every_steps // bs))
a, rkg, rtx, n = 0.75, 0.15, 0.50, 22 / 8
skg = a * rkg * p.step_reward_scale
stx = (1 - a) * rtx * p.text_reward_scale * p.step_reward_scale
print('  kg_reward_share = %.1f%%  (was 0.9%% before R10)'
      % (100 * skg * n / ((skg + stx) * n + p.outcome_weight)))
PYEOF

echo
echo "=== 3. CLI overrides wired (they were silently dropped with --config) ==="
$PY - <<'PYEOF'
src = open('scripts/train/phase3_ppo.py').read()
for label, needle in [
    ('--total_steps', 'ppo_cfg.total_ppo_steps if args.total_steps is None'),
    ('--save_every_steps', 'if args.save_every_steps is None'),
    ('--seed', 'tcfg.seed if args.seed is None else args.seed'),
]:
    print('  %-20s %s' % (label, 'ok' if needle in src else 'NOT WIRED'))
PYEOF

echo
echo "=== 4. kg_share/upd on the step log line ==="
if grep -q 'kg_share=%.3f' kgproweight/training/phase3_ppo.py; then
  echo "  ok — kg_reward_share will appear in train.log, not just tensorboard"
else
  echo "  MISSING — kg_share only reachable via tensorboard/history.jsonl"
fi

echo
echo "=== 5. smoke scripts executable + rearag weights present ==="
for f in launch_split_ppo_smoke.sh check_ppo_smoke.sh; do
  [ -f "$f" ] && echo "  present: $f" || echo "  MISSING: $f"
done
R=$(grep -oP 'KGPW_REARAG_PATH=\K\S+' launch_split_ppo_smoke.sh | tail -1)
if [ -f "$R/config.json" ]; then
  echo "  rearag ok: $R ($(du -sh "$R" 2>/dev/null | cut -f1))"
else
  echo "  rearag MISSING at $R -> PPO would download 18 GB mid-run"
fi

echo
echo "=== 6. SFT anchor checkpoint + alpha gate ==="
for p in checkpoints/sft_student_split/final \
         checkpoints/prm_alpha_gate_v1reann_negfix/alpha_gate.pt \
         data/silver_data/silver_v1_reannotated.jsonl; do
  [ -e "$p" ] && echo "  present: $p" || echo "  MISSING: $p"
done

echo
echo "=== 7. GPU ==="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null \
  || echo "  no GPU visible (card not started — expected before you power it on)"
