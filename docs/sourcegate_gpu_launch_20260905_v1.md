# 完整 PPO-A 的首次 GPU 启动记录

研究者于2026-09-05明确授权远程启动。当前实例为 RTX PRO 6000 Blackwell Server Edition，
97887 MiB 总显存，CUDA/BF16 实测通过，PyTorch 2.11.0+cu128。当前正在生成 α 校准候选；
研究策略尚未进行 PPO 更新。

远端冻结工作目录：
`/root/autodl-tmp/kgpaper_releases/source_gated_mixed4_emf1_v1_20260905_tensorboard_v1`。

启动器：`scripts/train/launch_sourcegate_preparation_v1.py`，独立 SHA 写入 launch manifest。
未修改冻结的模型、奖励、PPO 参数、输入银行或 release 清单。
线程环境明确为 OMP/MKL/OpenBLAS=4，修复实例启动环境中无效的 OMP_NUM_THREADS 值。

后台运行 ID：`sourcegate_a_gpu_20260905_v1`；最初 runner PID=1267，生成 PID=1307。
PID可能在实例重启后失效；实际状态以运行目录的 status.json、日志和进程检查为准。

## 监控命令

```bash
ssh -p 30481 root@connect.bjb1.seetacloud.com
cd /root/autodl-tmp/kgpaper_releases/source_gated_mixed4_emf1_v1_20260905_tensorboard_v1

# 总进度：每30秒记录阶段、条数和基于已完成工作的ETA
tail -n 50 -F outputs/launches/sourcegate_a_gpu_20260905_v1.pipeline.log

# 模型加载、告警和错误详情
tail -n 50 -F outputs/launches/sourcegate_a_gpu_20260905_v1/generate.log

# 即时阶段状态
cat outputs/launches/sourcegate_a_gpu_20260905_v1/status.json

# GPU显存与利用率
watch -n 2 nvidia-smi
```

后续阶段日志为同一运行目录的 `score.log` 和 `calibrate.log`。`tail -F` 可等待尚未创建的文件。
这些命令只读取日志，按 Ctrl+C 退出查看不会停止后台任务。
TensorBoard 已在GPU开机后重新启动：AutoPanel → TensorBoard，端口6007，日志根为 `/root/tf-logs`。
PPO尚未更新期间，没有新PPO训练曲线；`_diagnostics` 仅为上轮日志通路检查。

## 自动执行边界

启动器先完成全release SHA、GPU kernel和候选绑定检查，再顺序执行：
830题×K2候选生成 → ReaRAG评分 → 新α校准。
每阶段独立进程退出后再开始下一阶段，已有输出目录拒绝覆盖；失败保留现场。
即使校准CLI退出码为0，也额外检查实际 `training_clearance`。

校准通过后状态为 `AWAITING_PROCESS_UTILITY_CHECK`，不会直接误触发大规模PPO。
还需检查真实过程奖励正误排序、α对照、归一化分层饱和与引用操纵反事实，再进入已授权的
A-probe12 → A-smoke600/开发集检查 → A-full12000。
这是当前实施协议已有的科研检查，不是再次索取训练许可；O-only不是前置。

权威实时产物在远端 `outputs/launches/sourcegate_a_gpu_20260905_v1/`：
`manifest.json`、`events.jsonl`、`status.json` 与各阶段日志。
本地启动快照保存在 `outputs/audits/sourcegate_a_gpu_launch_20260905_v1/`；快照不替代实时状态。
