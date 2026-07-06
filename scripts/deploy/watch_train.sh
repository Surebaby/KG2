#!/usr/bin/env bash
# 实时查看训练进度(每 8 秒刷新)。服务器终端运行:
#   bash scripts/deploy/watch_train.sh
# Ctrl+C 只退出查看,不影响训练。
ROOT=/root/autodl-tmp/kgpaper
LOG=/root/autodl-tmp/train.log
PIDFILE=/root/autodl-tmp/train.pid

while true; do
  clear
  echo "============================================================"
  echo "  KG-ProWeight 训练进度    $(date '+%H:%M:%S')"
  echo "============================================================"

  # ---- 进程 + 是否在干活(CPU%)----
  RUNPID=""
  [ -f "$PIDFILE" ] && RUNPID=$(cat "$PIDFILE")
  WORKPID=$(pgrep -f "scripts/train/phase3" | head -1)
  if [ -n "$WORKPID" ]; then
    read CPU ET RSS <<<"$(ps -o %cpu=,etime=,rss= -p "$WORKPID" 2>/dev/null)"
    RSSG=$(awk "BEGIN{printf \"%.1f\", ${RSS:-0}/1024/1024}")
    echo "状态: 运行中 ✅   训练进程PID $WORKPID   CPU ${CPU}%   内存 ${RSSG}GB   已运行 $ET"
  elif grep -q "PIPELINE_DONE" "$LOG" 2>/dev/null; then
    echo "状态: 全部完成 🎉"
  elif grep -q "PIPELINE_FAILED" "$LOG" 2>/dev/null; then
    echo "状态: 失败 ❌"
  elif [ -n "$RUNPID" ] && kill -0 "$RUNPID" 2>/dev/null; then
    echo "状态: 流水线运行中(阶段切换/准备中)"
  else
    echo "状态: 未运行"
  fi

  # ---- GPU ----
  if nvidia-smi >/dev/null 2>&1; then
    read GMEM GUTIL <<<"$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ',')"
    echo "GPU : 显存 ${GMEM}MiB   利用率 ${GUTIL}%"
  fi
  echo "------------------------------------------------------------"

  [ ! -f "$LOG" ] && { echo "还没有日志。先 bash scripts/deploy/start_train.sh"; sleep 8; continue; }

  # ---- 当前阶段 ----
  STAGE="(准备中)"
  grep -q "PHASE 3a: SFT"  "$LOG" && STAGE="Phase 3a — SFT"
  grep -q "phase3_sft OK"  "$LOG" && STAGE="Phase 3a 完成 → 进入 PPO"
  grep -q "PHASE 3b: PPO"  "$LOG" && STAGE="Phase 3b — PPO"
  grep -q "phase3_ppo OK"  "$LOG" && STAGE="Phase 3b 完成 → 评估"
  grep -q "EVAL:"          "$LOG" && STAGE="评估 (base/SFT/PPO)"
  grep -q "PIPELINE_DONE"  "$LOG" && STAGE="全部完成"
  echo "阶段: $STAGE"

  # ---- 最近一条 HF 训练进度条 (step/total + ETA) ----
  BAR=$(tr '\r' '\n' < "$LOG" 2>/dev/null | grep -aoE "[0-9]+%\|[^|]*\| [0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+,[^]]*\]" | tail -1)

  # ---- SFT 指标 ----
  if [[ "$STAGE" == Phase\ 3a* ]]; then
    LOSS=$(grep -aoE "'loss': [0-9.]+" "$LOG" 2>/dev/null | tail -3 | sed "s/'loss': //" | tr '\n' ' ')
    if [ -n "$LOSS" ]; then
      echo "SFT 训练中 | 最近 loss: $LOSS"
      [ -n "$BAR" ] && echo "  进度: $BAR"
    elif grep -q "dropped .* passages" "$LOG" 2>/dev/null; then
      echo "SFT: 数据准备完成,模型即将开始训练(等首个 loss)..."
    else
      echo "SFT: 正在 CPU 上准备数据集(对 9839 条轨迹分词+截断,无 loss 属正常)"
      echo "     —— CPU 在跑 = 正常推进;约需 10-20 分钟,完后 GPU 利用率会涨上来"
    fi
  fi

  # ---- PPO 指标 ----
  if [[ "$STAGE" == Phase\ 3b* ]]; then
    PPO=$(grep -aE "step=[0-9]+ mean_reward" "$LOG" 2>/dev/null | tail -1 | sed 's/.*:: //')
    if [ -n "$PPO" ]; then
      echo "PPO: $PPO"
      [ -n "$BAR" ] && echo "  当前rollout生成: $BAR"
    else
      echo "PPO: 加载模型/首批生成中(首个 step 较慢,要生成完一批才出指标)"
    fi
  fi

  # ---- 评估指标 ----
  if [[ "$STAGE" == 评估* || "$STAGE" == 全部完成 ]]; then
    for m in re_base re_sft re_ppo; do
      f=$(find "$ROOT/outputs/$m" -name metric_score.txt 2>/dev/null | head -1)
      [ -n "$f" ] && echo "  $m: $(tr '\n' ' ' < "$f")"
    done
    [ -n "$BAR" ] && echo "  当前生成: $BAR"
  fi

  # ---- 报错 ----
  ERR=$(grep -aiE "OutOfMemory|PIPELINE_FAILED|Traceback|Error:|RuntimeError" "$LOG" 2>/dev/null | grep -vi "label_names" | tail -2)
  [ -n "$ERR" ] && { echo "------------------------------------------------------------"; echo "⚠️  $ERR"; }

  echo "------------------------------------------------------------"
  echo "日志最后一条(已过滤进度条):"
  tr '\r' '\n' < "$LOG" 2>/dev/null | grep -avE "^\s*$|it/s\]|s/it\]|Loading checkpoint shards|examples/s" | tail -1 | sed 's/^/  /'
  echo ""
  echo "(每 8 秒刷新;Ctrl+C 退出查看,不停训练)"
  sleep 8
done
