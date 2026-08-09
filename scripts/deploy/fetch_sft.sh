#!/bin/bash
# ===========================================================================
# Pull the Phase 3a SFT adapter from the AutoDL box to this machine.
#
#   export KGPW_SSH_PASS='...'
#   bash scripts/deploy/fetch_sft.sh
#
# Only the LoRA adapter + tokenizer (~130 MB), NOT a merged 16 GB model: the
# base llama3-8b already lives in models/llama3-8b locally, and PeftModel loads
# the adapter on top of it.
#
# The downlink from this box measures ~0.1 MB/s (documented in
# scripts/_post_train_eval.sh, which is why the Phase 2 eval ran remotely), so
# budget ~20 min for the adapter. --partial keeps a dropped transfer resumable
# instead of restarting from zero, and -P shows throughput so a stall is visible.
#
# Refuses to run before training finishes: an adapter copied mid-save is a
# truncated safetensors file that fails to load with a confusing header error
# rather than an obvious "incomplete download".
# ===========================================================================
set -euo pipefail

: "${KGPW_SSH_PASS:?set KGPW_SSH_PASS first (do not hardcode it)}"
HOST="${KGPW_SSH_HOST:-connect.bjb1.seetacloud.com}"
PORT="${KGPW_SSH_PORT:-41354}"
USER="${KGPW_SSH_USER:-root}"
REMOTE_ROOT=/root/autodl-tmp/kgpaper
REMOTE_CKPT="$REMOTE_ROOT/checkpoints/sft_student_split"

LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOCAL_CKPT="$LOCAL_ROOT/checkpoints/sft_student_split"

command -v sshpass >/dev/null || { echo "need sshpass"; exit 1; }
SSH=(sshpass -e ssh -p "$PORT" -o StrictHostKeyChecking=no "$USER@$HOST")

echo "=== checking the run finished ==="
# Three conditions, because each fails differently:
#   process gone   -> still training, or died
#   done line      -> trainer.save_model returned
#   final/ exists  -> the adapter is actually on disk
STATUS=$(SSHPASS="$KGPW_SSH_PASS" "${SSH[@]}" bash -s <<EOF
L=$REMOTE_CKPT/train.log
pgrep -f phase3_sft.py >/dev/null && echo "RUNNING" && exit 0
grep -qa 'Phase 3a SFT done' "\$L" 2>/dev/null || { echo "NO_DONE_LINE"; exit 0; }
[ -d "$REMOTE_CKPT/final" ] || { echo "NO_FINAL_DIR"; exit 0; }
echo "READY"
EOF
)
STATUS=$(echo "$STATUS" | tr -d '\r' | tail -1)

case "$STATUS" in
  READY) echo "  training finished, adapter present" ;;
  RUNNING) echo "  STILL TRAINING — refusing to copy a half-written checkpoint."; exit 2 ;;
  NO_DONE_LINE) echo "  process gone but no 'SFT done' line — the run FAILED. Check train.log."; exit 3 ;;
  NO_FINAL_DIR) echo "  'SFT done' logged but $REMOTE_CKPT/final is missing."; exit 4 ;;
  *) echo "  unexpected status: '$STATUS'"; exit 5 ;;
esac

mkdir -p "$LOCAL_CKPT"

echo
echo "=== copying adapter + loss curve (~130 MB, ~20 min at 0.1 MB/s) ==="
# Exclude checkpoint-*/ intermediates: save_strategy=epoch with 1 epoch means
# final/ is the same weights, and each intermediate is another 130 MB over a
# slow link for no additional information.
sshpass -e rsync -avzP --partial \
  -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
  --exclude 'checkpoint-*' \
  "$USER@$HOST:$REMOTE_CKPT/final" \
  "$USER@$HOST:$REMOTE_CKPT/sft_loss.jsonl" \
  "$USER@$HOST:$REMOTE_CKPT/train.log" \
  "$USER@$HOST:$REMOTE_CKPT/manifest.json" \
  "$LOCAL_CKPT/" || true

echo
echo "=== verifying the local copy ==="
python - <<PY
import json, sys
from pathlib import Path

ck = Path("$LOCAL_CKPT")
final = ck / "final"
ok = True

# Adapter weights must exist and be non-trivial. A 0-byte or few-KB file is what
# a truncated transfer leaves behind, and it only errors much later at load time.
w = list(final.glob("adapter_model.safetensors")) + list(final.glob("adapter_model.bin"))
if not w:
    print("  FAIL: no adapter_model.* in", final); ok = False
else:
    mb = w[0].stat().st_size / 1024**2
    print("  adapter: %s  %.1f MB" % (w[0].name, mb))
    if mb < 50:
        print("  FAIL: adapter is implausibly small (expected ~100 MB for r=32)"); ok = False

cfg = final / "adapter_config.json"
if cfg.exists():
    c = json.loads(cfg.read_text())
    print("  lora r=%s alpha=%s targets=%s" % (c.get("r"), c.get("lora_alpha"),
                                               sorted(c.get("target_modules", []))))
else:
    print("  FAIL: adapter_config.json missing"); ok = False

# The loss curve is the only record of whether the run actually learned.
lc = ck / "sft_loss.jsonl"
if lc.exists():
    rows = [json.loads(l) for l in lc.read_text().splitlines() if l.strip()]
    print("  loss curve: %d points" % len(rows))
    if rows:
        print("    first: step %s loss %.4f" % (rows[0]["step"], rows[0]["loss"]))
        print("    last:  step %s loss %.4f" % (rows[-1]["step"], rows[-1]["loss"]))
else:
    print("  WARN: sft_loss.jsonl missing"); ok = False

# split_info is what proves this checkpoint did not train on val/test.
# dump_manifest nests whatever it is handed as `extra` under a "run" key, so
# reading a top-level "split_info" silently yields {} and the fold check reports
# a failure that is really a bug in this reader. Fall back to top level in case a
# future dump_manifest flattens it.
mf = ck / "manifest.json"
if mf.exists():
    m = json.loads(mf.read_text())
    si = m.get("run", {}).get("split_info") or m.get("split_info", {})
    print("  split_info: %s" % si)
    if si.get("split") != "train":
        print("  FAIL: checkpoint was NOT trained on the train fold -> val numbers "
              "would not be held out"); ok = False
else:
    print("  WARN: manifest.json missing — cannot prove the fold")

sys.exit(0 if ok else 1)
PY

echo
echo "Fetched to $LOCAL_CKPT"
echo "Next: python scripts/eval/validate_sft.py --n 200"
