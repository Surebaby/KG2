#!/usr/bin/env bash
# =============================================================================
# 启动训练(后台运行,断开 SSH 也不会停)。在服务器终端直接运行:
#   bash scripts/deploy/start_train.sh
# 然后用 watch_train.sh 看进度。
# =============================================================================
set -uo pipefail

ROOT=/root/autodl-tmp/kgpaper
LOG=/root/autodl-tmp/train.log
PIDFILE=/root/autodl-tmp/train.pid

# 已经在跑就不重复启动
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "训练已在运行中 (PID $(cat "$PIDFILE"))。"
  echo "看进度:  bash $ROOT/scripts/deploy/watch_train.sh"
  exit 0
fi

# 检查 GPU(无卡模式跑不了)
if ! nvidia-smi >/dev/null 2>&1; then
  echo "❌ 检测不到 GPU —— 现在是无卡模式,无法训练。请先在 AutoDL 控制台切换到「有卡模式」再运行。"
  exit 1
fi

cd "$ROOT"
echo "启动训练流水线(SFT → PPO → 评估),后台运行..."
# 无缓冲输出:让 HF Trainer 的 loss 行实时刷到日志(否则 nohup 块缓冲会
# 把 SFT 的 {'loss':...} 攒到进程退出才落盘,实时面板看不到 loss 数字)。
export PYTHONUNBUFFERED=1
nohup bash scripts/deploy/run_retrain_eval.sh > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
sleep 2
echo ""
echo "✅ 已启动,PID $(cat "$PIDFILE")"
echo "   日志: $LOG"
echo ""
echo "看实时进度,运行:"
echo "   bash $ROOT/scripts/deploy/watch_train.sh"
echo ""
echo "(训练约 7-8 小时;关掉终端不影响,它在后台继续跑)"
