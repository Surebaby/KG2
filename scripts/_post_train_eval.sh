#!/usr/bin/env bash
# Post-training evaluation chain for the Phase 2 NEG fix — REMOTE execution.
#
# The downlink from the AutoDL box measures ~0.1 MB/s, so fetching the 129 MB
# adapter would take ~20 min and the 1.28 GB enriched silver ~3.4 h. Both the
# 8B base and the checkpoint already live on the remote box, so we run the eval
# there and pull back only the JSON results (a few KB).
#
# Assumes the fixed eval scripts have been uploaded (MAXLEN=1024 +
# PRM_EVAL_PREFIX); the remote copies were stale at 512 with no prefix switch.
set -u

# The base conda env is BROKEN: transformers 5.12.1 there fails to import with
# "NameError: name 'nn' is not defined" and cannot see torch at all. The training
# run used the llama env (torch 2.4.1 / transformers 4.45.2 / peft 0.12.0), so
# evaluation must use the same one or G5 dies on `from peft import PeftModel`.
PY=/root/miniconda3/envs/llama/bin/python
RROOT=/root/autodl-tmp/kgpaper
CKPT=prm_alpha_gate_v1reann_negfix
LOG=logs/post_train_eval.log

mkdir -p logs
: > "$LOG"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# G5 on the new checkpoint: prefixed input at max_len 1024 — the format and cap
# it was trained with. The old checkpoint's result is already on local disk
# (bare_step @512, matching ITS training), so it is not re-run.
say "=== 1/3 G5 泛化评测 (远程, prefixed, max_len=1024) ==="
python scripts/deploy/_ssh.py "cd $RROOT && PRM_EVAL_BS=16 PRM_EVAL_MAXLEN=1024 \
  $PY scripts/_phase2_prm_holdout.py checkpoints/$CKPT \
  data/silver_data/silver_v1_reannotated.jsonl 3000 \
  > /root/autodl-tmp/g5_negfix.log 2>&1; \
  echo EXIT=\$?; tail -32 /root/autodl-tmp/g5_negfix.log" 3000 2>&1 | tee -a "$LOG"

say "=== 2/3 目标测试 G1-G4 (远程) ==="
python scripts/deploy/_ssh.py "cd $RROOT && \
  $PY scripts/_phase2_goal_test.py checkpoints/$CKPT 1200 \
  > /root/autodl-tmp/goal_negfix.log 2>&1; \
  echo EXIT=\$?; tail -20 /root/autodl-tmp/goal_negfix.log" 1800 2>&1 | tee -a "$LOG"

say "=== 3/3 回传 JSON 结果 ==="
KGPW_SSH_PASS="$KGPW_SSH_PASS" python - <<PYEOF 2>&1 | tee -a "$LOG"
import os, paramiko, posixpath
from pathlib import Path
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect("connect.bjb1.seetacloud.com", port=41354, username="root",
            password=os.environ["KGPW_SSH_PASS"], timeout=30)
s = cli.open_sftp()
local = Path("checkpoints/$CKPT"); local.mkdir(parents=True, exist_ok=True)
base = "$RROOT/checkpoints/$CKPT/"
for name in ("prm_holdout.json", "goal_test.json", "manifest.json",
             "history.jsonl", "alpha_gate.pt", "text_reward_head.pt"):
    try:
        s.get(base + name, str(local / name))
        print("  got %-22s %8d B" % (name, (local / name).stat().st_size))
    except IOError:
        print("  missing (remote): %s" % name)
s.close(); cli.close()
PYEOF

say "=== 新旧对比 ==="
python scripts/_phase2_compare.py checkpoints/prm_alpha_gate_v1reann \
  "checkpoints/$CKPT" 2>&1 | tee -a "$LOG"

say "=== 完成, 日志 $LOG ==="
