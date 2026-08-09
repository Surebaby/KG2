#!/bin/bash
# ===========================================================================
# Sync the train/val/test split work to the AutoDL box.
#
#   export KGPW_SSH_PASS='...'      # not stored in the repo
#   bash scripts/deploy/sync_split.sh
#
# Only the 21 files the split touches (~252 KB) — NOT the 1.3 GB silver file or
# the model weights, which are already on the remote and unchanged.
#
# Why an explicit list rather than `rsync .`: the working tree also holds
# scratch scripts (_kgt/, _t/, o/, launch_silver_*.sh) and third_party/, and
# pushing those would overwrite remote state that is not part of this change.
#
# Verification runs after the copy — a sync that silently drops a file is how
# the remote ends up running a half-applied split, where e.g. phase3_sft honours
# --split but silver_dataset.py does not know the flag.
# ===========================================================================
set -euo pipefail

: "${KGPW_SSH_PASS:?set KGPW_SSH_PASS first (do not hardcode it)}"
# sshpass -e reads the password from SSHPASS specifically, so the documented
# usage above (export KGPW_SSH_PASS) failed with "no password was set" unless
# SSHPASS happened to be exported too. Bridge them here. Still an env var, never
# argv: `sshpass -p <pw>` would expose the password in ps output to any local
# user for the lifetime of the transfer.
export SSHPASS="$KGPW_SSH_PASS"
HOST="${KGPW_SSH_HOST:-connect.bjb1.seetacloud.com}"
PORT="${KGPW_SSH_PORT:-41354}"
USER="${KGPW_SSH_USER:-root}"
REMOTE_ROOT=/root/autodl-tmp/kgpaper

LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$LOCAL_ROOT"

FILES=(
  kgproweight/data/silver_split.py
  kgproweight/data/silver_dataset.py
  kgproweight/data/__init__.py
  kgproweight/config/schemas.py
  kgproweight/training/phase2_prm.py
  kgproweight/training/phase3_sft.py
  kgproweight/training/phase3_ppo.py
  kgproweight/training/phase3_grpo.py
  kgproweight/eval/data_efficiency.py
  scripts/train/_split_args.py
  scripts/train/phase2_train_prm.py
  scripts/train/phase3_sft.py
  scripts/train/phase3_ppo.py
  scripts/eval/run_data_efficiency.py
  scripts/utils/inspect_split.py
  configs/training/phase2_prm.yaml
  configs/training/phase3_sft.yaml
  configs/training/phase3_ppo.yaml
  tests/test_silver_split.py
  # R10 speed: proves batched rollout (rollout_chunk_size>1) returns exactly what
  # the old one-prompt-at-a-time loop did -- unpadded queries, right per-row
  # logprobs, trimmed responses. Runs on CPU, so the remote can check it too.
  tests/test_r10_batched_rollout.py
  launch_split_sft.sh
  launch_split_ppo.sh
  # R10 (2026-08-06): smoke test for the PPO reward/schedule changes.
  launch_split_ppo_smoke.sh
  check_ppo_smoke.sh
  scripts/deploy/_verify_r10.sh
  # Feasibility check that needs no GPU: run it on a no-card instance to clear
  # config/data/index/prompt-length before paying for a card.
  scripts/deploy/_feasibility_cpu.py
)

for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "MISSING locally: $f"; exit 1; }
done

command -v sshpass >/dev/null || { echo "need sshpass: apt-get install -y sshpass"; exit 1; }

echo "Syncing ${#FILES[@]} files to $USER@$HOST:$PORT$REMOTE_ROOT"

# --relative preserves each file's directory structure under the remote root.
# -p keeps the executable bit on the launch scripts.
sshpass -e rsync -avzp --relative \
  -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
  "${FILES[@]}" "$USER@$HOST:$REMOTE_ROOT/"

echo
echo "=== verifying on the remote ==="
sshpass -e ssh -p "$PORT" -o StrictHostKeyChecking=no "$USER@$HOST" bash -lc "
set -e
cd $REMOTE_ROOT
export PYTHONPATH=$REMOTE_ROOT:$REMOTE_ROOT/flashrag_src
PY=/root/autodl-tmp/kgpw_env/bin/python

echo '--- split module imports'
\$PY -c 'from kgproweight.data.silver_split import SplitSpec, assign_split; print(\"  ok\", SplitSpec())'

echo '--- all phases agree on one SplitSpec'
\$PY - <<'EOF'
from kgproweight.training.phase2_prm import Phase2Config
from kgproweight.training.phase3_sft import Phase3SFTConfig
from kgproweight.training.phase3_ppo import Phase3PPOConfig
s = {C(silver_path='s', output_dir='o').build_split_spec()
     for C in (Phase2Config, Phase3SFTConfig, Phase3PPOConfig)}
assert len(s) == 1, s
print('  ok, identical:', s.pop())
EOF

echo '--- YAML caps match (SFT max_length must equal PPO max_input_length)'
\$PY - <<'EOF'
from kgproweight.config.loader import load_config
from kgproweight.config.schemas import TrainingConfig
sft = TrainingConfig(**load_config('configs/training/phase3_sft.yaml')['training'])
ppo = TrainingConfig(**load_config('configs/training/phase3_ppo.yaml')['training'])
print('  sft_max_length=%s  ppo max_input_length=%s' % (sft.sft_max_length, ppo.max_input_length))
assert sft.sft_max_length == ppo.max_input_length, 'CAP MISMATCH -> passages get dropped in SFT only'
print('  ok, aligned')
EOF

echo '--- split invariants (remote has no pytest, so assert inline)'
\$PY - <<'EOF'
from kgproweight.data.silver_split import SplitSpec, assign_split
class T:
    def __init__(s, q, a): s.qid, s.question, s.accepted = q, q, a
spec = SplitSpec()
folds = [assign_split(T('q%d' % i, i % 3 == 0), spec) for i in range(3000)]
assert set(folds) == {'train', 'val', 'test'}, set(folds)
# Purity: same key -> same fold, regardless of when it is asked.
assert all(assign_split(T('q%d' % i, i % 3 == 0), spec) == folds[i] for i in range(3000))
print('  ok, deterministic over 3000 keys')
EOF

# ORDER MATTERS: every cheap, decisive check runs BEFORE the fold table, which
# streams the whole 1.37 GB silver file and takes minutes. On 2026-08-06 the
# table sat at the top of this section and the R10 assertions below it never got
# to print, so the truncated output read as a sync that had not finished.
# Slow, informational steps go last; assertions go first.
#
# NOTE: this whole section is one double-quoted argument to bash -lc. An
# unescaped double quote here ends that argument, and every later line then runs
# in the LOCAL shell (the 2026-08-06 symptom: \$PY: command not found at the
# first R10 assertion). Keep comments in this block quote-free.
echo '--- R10 PPO values landed, and the CLI can still override them'
\$PY - <<'EOF'
from kgproweight.config import ProjectConfig, load_config
from kgproweight.training.phase3_ppo import Phase3PPOConfig
d = load_config('configs/training/phase3_ppo.yaml', validate=ProjectConfig)
t = d.training; p = t.ppo
# 16000/2000 updated to 3000/500 on 2026-08-06 once the measured 39.8s/update
# fixed the run length by wall clock (3000 traj = 375 updates = 4.1 h). This
# assertion is a tripwire for the config not syncing, so it has to track the
# value that is actually meant to run -- otherwise it fails on every correct
# sync and gets ignored, which is how the 16000 here outlived the decision.
exp = {'total_ppo_steps': 3000, 'save_every_steps': 500, 'target_kl': 40.0,
       'outcome_weight': 4.0, 'step_reward_scale': 1.5}
for k, v in exp.items():
    got = getattr(p, k)
    assert got == v, 'R10 NOT APPLIED: %s = %r, expected %r' % (k, got, v)
assert Phase3PPOConfig(silver_path='s', output_dir='o').ppo_max_kg_triples == 30, \\
    'ppo_max_kg_triples still 50 -> KG block risks right-truncation'
bs = p.batch_size
print('  ok: %d trajectories / bs %d = %d updates; ckpt every %d (= %d updates)'
      % (p.total_ppo_steps, bs, p.total_ppo_steps // bs,
         p.save_every_steps, p.save_every_steps // bs))
a, rkg, rtx, nst = 0.75, 0.15, 0.50, 22 / 8
skg = a * rkg * p.step_reward_scale
stx = (1 - a) * rtx * p.text_reward_scale * p.step_reward_scale
print('  kg_reward_share (at measured alpha/r_kg) = %.1f%%  [was 0.9%%]'
      % (100 * skg * nst / ((skg + stx) * nst + p.outcome_weight)))
EOF

echo '--- --total_steps actually overrides the YAML (it used to be ignored)'
\$PY - <<'EOF'
import subprocess, sys
# --help is enough to prove the flags exist; the override path is asserted by
# reading the source, since a real run needs a GPU.
src = open('scripts/train/phase3_ppo.py').read()
assert 'ppo_cfg.total_ppo_steps if args.total_steps is None else args.total_steps' in src, \\
    '--total_steps is still dropped when --config is passed'
assert 'if args.save_every_steps is None' in src, '--save_every_steps not wired'
assert 'tcfg.seed if args.seed is None else args.seed' in src, '--seed not wired'
print('  ok: --seed / --total_steps / --save_every_steps override --config')
EOF

echo '--- rearag weights at the path launch_split_ppo.sh pins'
REARAG=\$(grep -oP 'KGPW_REARAG_PATH=\K\S+' $REMOTE_ROOT/launch_split_ppo.sh | tail -1)
echo \"  pinned: \$REARAG\"
if [ -f \"\$REARAG/config.json\" ]; then
  echo \"  ok, \$(du -sh \"\$REARAG\" 2>/dev/null | cut -f1) of weights present\"
else
  echo \"  MISSING -> PPO would download 18 GB. Candidates on this box:\"
  ls -d /root/autodl-tmp/models/* $REMOTE_ROOT/models/* 2>/dev/null | grep -i 'ea\|rag' || echo '    none'
fi

# LAST, because it streams the full 1.37 GB silver file — minutes, not seconds.
# --stream, NOT the eager reader: no-GPU mode caps this container at a 2 GiB
# cgroup (/sys/fs/cgroup/memory.max), so SilverDatasetReader gets SIGKILLed
# (exit 137, prints nothing) before the table appears.
# tests/test_silver_split.py pins --stream to the eager output.
# Set KGPW_SKIP_FOLD_TABLE=1 to skip it: nothing above depends on it, and the
# fold assignment is already proven deterministic by the invariant check.
if [ -z \"\${KGPW_SKIP_FOLD_TABLE:-}\" ]; then
  echo '--- real fold table (streamed; ~minutes, Ctrl-C is safe from here on)'
  \$PY scripts/utils/inspect_split.py data/silver_data/silver_v1_reannotated.jsonl --stream 2>/dev/null | sed -n '7,20p'
else
  echo '--- real fold table: skipped (KGPW_SKIP_FOLD_TABLE=1)'
fi
"

echo
echo "Sync + verification done. Next: bash launch_split_sft.sh (on the remote)"
