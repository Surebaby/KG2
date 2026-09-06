# PPO TensorBoard — AutoDL 配置与指标说明

> 2026-09-06 启动前遥测检查：默认前3个PPO batch均写histogram，之后每10个batch写一次；每批scalar和flush不变。
> 这覆盖probe12的HotpotQA→2Wiki→MuSiQue次序，使第2批合法有图轨迹也能显示α和六维分布。没有合法有图轨迹时仍不伪造α分布。
> 旧远端连通性诊断只有`diagnostics/event_write_read_ok`一张图；诊断事件不是训练。新只读审计见`outputs/audits/ppo_tensorboard_prelaunch_readonly_20260906_v1/`。

> 2026-09-06 本地继任：source-credit-v2 的实际writer已支持六维α特征，并将Text硬裁剪、softsign软饱和和标准分尾部比例分开。
> 两组32条真实SFT候选的零更新事件已生成并读回复核，目录为
> `outputs/audits/source_credit_v2_representative_cached_and_utility_20260906_v1/tensorboard_zero_update_diagnostic/`。
> 新增 `reward/all/text_softsign_saturation_frac`、`reward/all/text_raw_z_outside_unit_frac`，分层视图同样提供；
> `gate/eligible_valid/feature_source_edge_coverage_*` 与 `feature_min_step_citation_precision_*` 记录新增两维。
> 这些事件明确标记 `diagnostic/optimizer_updates=0`；本地修改本轮尚未同步远端，不能将远端旧release或诊断事件当作v2 PPO训练。

2026-09-05。此次修改仅增加遥测和日志目录配置；不修改奖励、loss、训练数据、采样或评估协议。
官方依据：https://www.autodl.com/docs/tensorboard/ 。AutoPanel 默认读取 `/root/tf-logs`，服务端口为 6007。

## 使用方式

远端当前监控版本目录：
`/root/autodl-tmp/kgpaper_releases/source_gated_mixed4_emf1_v1_20260905_tensorboard_v1`。

在该目录执行 `bash scripts/sourcegate_python.sh scripts/deploy/setup_autodl_tensorboard.py`。
脚本复用同目录的现有 TensorBoard；无服务时后台启动 6007，已有其他服务/目录时报告冲突并保留。
不执行批量 kill，不删除历史 events。实例重启后如 AutoPanel 服务没有自动恢复，可再次运行此命令。

`scripts/sourcegate_python.sh` 设置 `KGPW_TB_ROOT=/root/tf-logs/kgpaper`。
完整 PPO-A 原有 probe/smoke/full 命令不变，仍需先完成真实校准与 GPU 检查。
日志路径为 `/root/tf-logs/kgpaper/<Experiment ID>/<UTC时间_唯一标识>/events.out.tfevents.*`。
训练输出目录的 `tensorboard_run.json` 保存实际日志位置。每次运行独立，避免不同实验曲线合并。
本地直接执行 Python 时默认在 `<output_dir>/tensorboard/<session>`，不要求写 `/root`。
显式 `KGPW_TB_DIR` 可覆盖到新目录；已有 events 时拒绝混写。

开机后，在 AutoDL 控制台进入 AutoPanel → TensorBoard，选择 `kgpaper/` 下对应 run。
`_diagnostics/` 只验证 event 写入/读取，不是训练结果；历史根目录 events 保留。

## 图表及含义

横轴 **Step = 已完成 rollout 轨迹数**，每个 K4 batch 前进 4；与 step_200 等 checkpoint 对齐。
`progress/ppo_batches` 给出 PPO batch 调用计数，不是 mini-batch optimizer step。
每个 batch 写 scalar 并 flush；前3个及之后每10个 batch 写 histogram。

| 标签 | 内容 |
|---|---|
| `ppo/loss/*` | TRL 实际提供的 policy、value、total loss |
| `ppo/policy/*`、`ppo/val/*` | clip fraction、approx KL、entropy、value 诊断 |
| `objective/*`、`custom/adaptive_kl_coef` | reference KL、非分数奖励、动态 KL 系数 |
| `reward/all/*` | rollout EM/F1、合法率、答案/文本/图/总奖励 |
| `reward/dataset/<dataset>/*` | 三数据集分别统计，不用单批均值冒充三数据集 macro 分数 |
| `reward/m_graph/{0,1}/*` | 来源硬门控分层；也有 dataset×mask 交叉分层 |
| `gate/all/*` | 全体实际 α 信用；普通和无效样本可为0 |
| `gate/eligible_valid/*` | 真正评分的有图合法轨迹 α 分布及六个特征，排除未评分零占位 |
| `reward/.../text_clip_frac`、`graph_clip_frac` | 冻结归一化的裁剪饱和率 |
| `custom/advantage_*`、`value_*`、`return_*` | trainer 提供的 mask-aware advantage、value、return 与 explained variance |
| `custom/sft_replay_*` | replay loss 和实际样本比例 |
| `custom/response_tokens_mean`、`length_capped_frac` | 输出长度与生成触顶比例 |
| `optimizer/learning_rate_group_*` | 当前实际优化器学习率 |
| `runtime/*`、`system/gpu_*/*` | batch耗时、每秒轨迹/响应token、进程峰值RSS、GPU分配/保留/峰值显存 |
| `run/config`、`run/metadata`、`config/*` | Text 中的实际配置/运行身份，以及数值超参数 |
| `telemetry/nonfinite/*` | 原始统计中被跳过的 NaN/Inf 数量；不伪造为0 |

没有该类样本的 batch 不写对应均值；缺失点不是零分。
TRL 数组的 `raw_mean/raw_std/raw_histogram` 可能包含上游 padding；不能当作有效 token 统计。
训练 EM/F1 来自当前策略采样的训练 rollout，**不是开发集或 canonical baseline 评估结果**。
α 在 PPO 中冻结，图和文本评分仍是过程代理；曲线本身不证明方法提高正确率。

## 版本与验证

新的代码/文档清单：`outputs/audits/source_gated_mixed4_emf1_v1_release_tensorboard_v1/manifest.json`。
新候选输入银行：`outputs/audits/source_quality_candidate_bank_v1_inputs_seed42_tensorboard_v1`。
它与旧 `gitless_fix1` 的输入和评分配置逐字节相同，仅将遥测源码纳入绑定；旧银行与 release 保留。
新远端检查：`outputs/audits/source_gated_mixed4_emf1_v1_remote_validation_tensorboard_v1/`。
事件单元测试使用合成 fixture，生产数据/模型不做更新。真实 PPO 曲线只会在校准完成、GPU训练开始后出现。
