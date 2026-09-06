# 远端 PPO 与 TensorBoard 开卡前检查

2026-09-06；部署版本 `source_credit_v2_probe12_20260906_v1`。本记录区分监控检查、限额入口准备与实际 GPU 训练，不把诊断事件当成 PPO 结果。

## 已确认的远端状态

原实例已由研究者无卡开机，SSH `connect.bjb1.seetacloud.com:30481` 已接通。初次无卡检查中 CUDA 不可用；没有执行模型加载或 PPO 更新。

TensorBoard 开机后没有进程、6007 拒绝连接，现已恢复。服务使用 `/root/autodl-tmp/kgpw_env/bin/python -m tensorboard.main --port 6007 --logdir /root/tf-logs --host 0.0.0.0 --reload_interval 10`，本次启动 PID851。PID 会随重启变化。

按 [AutoDL 官方说明](https://www.autodl.com/docs/tensorboard/)，通过 AutoPanel → TensorBoard 访问 `/root/tf-logs`。没有删除旧 events，也没有批量终止其他服务。

## 上次只有一张图的原因与本次修复

远端 HTTP 标签核验确认：`kgpaper/_diagnostics/tensorboard_wiring_20260905T105506Z` 只有 `diagnostics/event_write_read_ok` 一项。这是事件连通性检测，不能代表 PPO 指标配置。根目录 run `.` 另有65项历史指标，也不是本轮 PPO。

当前真实 PPO writer 已连接 policy/value loss、KL、entropy、clip fraction、答案/格式/过程奖励、训练 rollout EM/F1、动态 α 和六维特征、replay、学习率、吞吐与显存。实际配置解析为 `log_with=tensorboard`、完整 A、12 trajectories、K4，`KGPW_TB_ROOT=/root/tf-logs/kgpaper`。

本次只改分布图频率：前3个 PPO batch 都写 histogram，之后每10批写一次。旧频率只在第1批写，而 probe 的有图 2Wiki batch 是第2批，可能看不到 α 分布；现在可覆盖该批。训练参数、奖励和 α 数值未因此改变。

本地与远端各通过16项真实事件写入/读回测试。测试中的 GPU 计数使用替身，不能称为 GPU 训练验证。代码 SHA 一致的证据见 [远端检查](../outputs/audits/ppo_remote_prelaunch_20260906_v1/remote_inspection.json)；完整标签、产生条件和本地审计见 [监控审计](../outputs/audits/ppo_tensorboard_prelaunch_readonly_20260906_v1/report.json)。

为验证 AutoPanel 实际可见多项指标，已复制原真实缓存事件到：

`kgpaper/_diagnostics/source_credit_v2_cached_SFT_A_20260906_v1`

HTTP 已读回257项 scalar、98项 histogram。它们来自32条现有 Strong SFT 候选的真实缓存 ReaRAG 评分，`diagnostic/optimizer_updates=0`，没有模拟 PPO loss/KL，也没有生成新的 PPO 训练数据。

## 正式 probe 图表的预期

probe 是3题各K4，共12条轨迹。全局 batch 指标横轴通常是4、8、12；H/W/M每域只有一批。α 只在有图且合法、来源获信用的候选上有分层观测；replay loss只在实际执行 replay 时出现。不要为了画连续曲线补零。

正式训练的 run 会位于 `kgpaper/<实验ID>/<session>`，输出目录 `tensorboard_run.json` 记录实际位置。开发集与 canonical baseline EM/F1沿用原评估规则；训练 rollout 的 EM/F1不能冒充独立测试结果。

## 环境与操作命令

独立工作目录：`/root/autodl-tmp/kgpaper_releases/source_credit_v2_probe12_20260906_v1`。现有模型、Strong SFT 与正式 PPO/replay 数据通过只读使用的链接复用；旧版本保留。

依赖实测：PyTorch `2.11.0+cu128`、TRL `0.11.4`、Transformers `4.49.0`、PEFT `0.19.1`、Accelerate `1.13.0`、TensorBoard `2.21.0`。

实例重启后恢复或复用 TensorBoard：

```bash
ssh -p 30481 root@connect.bjb1.seetacloud.com
cd /root/autodl-tmp/kgpaper_releases/source_credit_v2_probe12_20260906_v1
bash scripts/sourcegate_python.sh scripts/deploy/setup_autodl_tensorboard.py
```

此命令只维护监控服务，不启动训练。

## 最终验收：无卡准备完成，等待 GPU

当前状态 `REMOTE_CPU_READY_SCOPED_A_PROBE12_TENSORBOARD_VERIFIED_PPO_NOT_STARTED`。

- 远端89项模型/Strong SFT/正式PPO与replay/来源资产SHA通过，总计37,479,869,245字节；原mask仍为800图题中671 PASS，另有30个ordinary身份。[资产验收](../outputs/audits/ppo_remote_prelaunch_20260906_v1/remote_assets.json)
- 完整12条配置、实际adapter基座路径与ReaRAG路径检查通过；默认无配置加载、diagnostic旗标、600/12000预算、学习率或门控模式覆盖均被拒绝。[限额入口验收](../outputs/audits/ppo_remote_prelaunch_20260906_v1/remote_probe_scope.json)
- 相关scope/v2 gate/runtime的108项测试在本地及远端通过；远端另已通过16项TensorBoard事件测试。全部为CPU测试，尚非真实GPU PPO验证。
- 新child仅改变gate路径与输出目录，答案/过程/α、温度、步数、学习率与mini-batch保持原定值。原parent三个旗标仍false；child仅允许完整A-probe12，保留fresh132健康FAIL及600/full未放行。

冻结配置：`configs/training/phase3_ppo_mixed4_answer_format_v2_a_probe12_scoped_seed42.yaml`。
冻结产物：[scope与gate manifest](../outputs/calibration/source_credit_gate_v2_probe12_scoped_20260906_v1/manifest.json)。
最终运行目录：`outputs/ppo_mixed4_answer_format_v2_a_probe12_scoped_seed42_20260906_v1`。

可在AutoDL切换到原96GB GPU模式。开GPU后先恢复上述TensorBoard服务，再执行限额训练。以下为准备好的命令，本轮没有执行：

```bash
cd /root/autodl-tmp/kgpaper_releases/source_credit_v2_probe12_20260906_v1
mkdir -p outputs/launches
(
  set -o noclobber
  nohup env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    bash scripts/sourcegate_python.sh scripts/train/phase3_ppo.py \
    --config configs/training/phase3_ppo_mixed4_answer_format_v2_a_probe12_scoped_seed42.yaml \
    > outputs/launches/ppo_a_probe12_scoped_20260906_v1.log 2>&1 < /dev/null &
)
```

日志与训练图表位置：

```bash
tail -n 80 -F outputs/launches/ppo_a_probe12_scoped_20260906_v1.log
cat outputs/ppo_mixed4_answer_format_v2_a_probe12_scoped_seed42_20260906_v1/tensorboard_run.json
watch -n 2 nvidia-smi
```

日志文件已存在时上述启动命令拒绝覆盖；训练输出目录本身也拒绝混写。TensorBoard指针在真正启动writer后才产生。工程probe用于验证完整训练链路，不能据3个batch宣称EM/F1提升。

## 当前版本的兼容范围

此release仅用于上面的A-v2-probe12。收尾只读复核发现：新scope前置分支将旧source-credit-v1的通用`training_clearance_scope`误识别为v2，旧v1配置在新release会报schema mismatch；原v1 loader和旧release不受影响。当前A-v2入口已在真实远端通过，不需要据此重做本轮确认。

后续兼容修复应另版缩窄scope识别条件，并补旧v1 dispatch正测，保留v2限额/删除scope的负测；本轮已冻结代码不再改动。历史v1训练复现继续使用原冻结release，不能直接套用本次新目录。
