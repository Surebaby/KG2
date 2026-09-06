# 本地4090完整候选重生成

2026-09-05，研究者明确要求停止远端，并在本地从零重新生成。
远端runner1267/生成1307经命令行身份与cwd核验后停止，原328条完整候选、日志与启动记录保留。
远端停止记录（保存在远端release内）：`outputs/audits/sourcegate_remote_stop_for_local_regenerate_20260905_v1/report.json`。
原结果未覆盖、不合并进入此次本地银行。本地停止事实观察另保存在
`outputs/audits/sourcegate_local_regenerate_20260905_v1/remote_stop_observation.json`。
该本地记录来自成功的远端停止命令输出；之后SSH断开，尚未下载远端报告原文件，实例当前开关机状态未确认。

本地主机：RTX4090 24GB，使用沙箱外CUDA；冻结最长输入实测生成通过。
环境：`/home/zjulab/anaconda3/envs/kgpaper/bin/python`，Torch2.4.1+cu121、Transformers4.48.0、PEFT0.19.1。
硬件和库版本与远端不同已明确登记，不声称相同seed跨平台逐token相同。
模型、输入、K2、每候选seed、BF16、batch1、temperature1/top_p1/top_k0、max_new_tokens384均使用原生成合同。
原 `source_quality_candidate_bank_v1.py` 及其冻结源码绑定不变；新增launcher仅管理进程、环境与进度日志。
启动器21项CPU mock测试通过；实际启动后已确认两个后台进程存活，4090利用率98%、显存17604MiB，输出持续增长。

- 运行ID：`sourcegate_local_regenerate_4090_20260905_v1`
- Experiment ID：`SOURCE-QUALITY-CANDIDATE-BANK-V1-GENERATE-SEED42-LOCAL4090-V1`
- 最初runner PID：2065046（实时PID以manifest/status和进程为准）
- 输入银行：`outputs/audits/source_quality_candidate_bank_v1_inputs_seed42_tensorboard_v1`
- 新生成目录：`outputs/audits/source_quality_candidate_bank_v1_generated_seed42_local4090_v1`
- 目标：830题×K2=1660条，全量本地重生成，远端候选复用数=0。

## 本地监控

```bash
cd /home/zjulab/kgpaper
tail -n 50 -F outputs/launches/sourcegate_local_regenerate_4090_20260905_v1.pipeline.log
```

模型加载和错误详情：

```bash
tail -n 50 -F /home/zjulab/kgpaper/outputs/launches/sourcegate_local_regenerate_4090_20260905_v1/generate.log
```

```bash
cat /home/zjulab/kgpaper/outputs/launches/sourcegate_local_regenerate_4090_20260905_v1/status.json
watch -n 2 nvidia-smi
```

主日志每30秒更新进度；退出tail不停止后台任务。完成后状态为
`LOCAL_GENERATION_COMPLETE_SCORING_PENDING`，仍不是PPO训练结果。
ReaRAG本地评分需要单独验证24GB是否足够；α校准、过程效用检查后才进入远端完整PPO-A。
停止计算进程不等于关闭AutoDL实例；GPU计费与实例开关机状态须分别确认。
