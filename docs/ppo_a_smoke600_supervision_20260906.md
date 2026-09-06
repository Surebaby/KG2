# PPO-A smoke600 运行与停止记录

2026-09-06，当前状态：`A_SMOKE600_STOPPED_AT300_PRESET_VALID_GUARD_AUDITED`。

本次获授权的 A-smoke600 已真实执行，在 **300/600 条轨迹、75 个 PPO 批次**后触发预设有效率保护规则停止，**600 未完成**。北京时间16:44:32启动，17:02:35训练结束；supervisor于17:02:39观察到exit 1、GPU显存占用和利用率均为0，训练manifest保留为`FAILED`，没有`final/`。

[独立健康核对报告](../outputs/audits/ppo_a_smoke600_stopped300_independent_health_20260906_v1/report.json)复现了首次触发点：最后15批共60条中有效41条，有效率 **41/60=0.6833 < 0.7**；长度截断3/60=0.05、平均KL=1.3282均未越线。该窗口19条无效中，9条为两步短缺、10条为其他严重无效，不能把停止原因全部归于最低步数要求。此前从200条开始的保护检查均通过；记录中没有NaN/Inf或OOM。

全300条中216条格式有效、55条两步短缺、29条其他严重无效；实际记录61条非零α轨迹、674次Text步骤评分和30条replay。上述为训练诊断，不是独立综合baseline成绩，也不证明α优越性。

远端保留`step_200/`与`aborted_step_300/`。本轮终态归档位于[terminal300目录](../outputs/audits/ppo_a_smoke600_gpu_supervision_terminal300_20260906_v1/)，原失败、启动/过程快照和此前200条归档全部保留。20份产物共254,806,764字节已完整备份并与远端SHA逐项一致。[独立参数与事件验收](../outputs/audits/ppo_a_smoke600_stopped300_independent_tensor_events_20260906_v2/report.json)完成：7452项事件/history/契约对照一致，638项scalar、133项histogram；两份checkpoint各256/256 LoRA张量与value head均实际更新且有限。**终态证据验收完成，训练FAILED及600未完成状态不变。**默认不重启、不降低阈值、不扩展F/T或full。建议按原评估协议评估保存的checkpoint；自然≥2步规则如需验证，应作为单独版本实验，尚未启动评估或该验证。

以下启动与早期进度为历史快照。

研究者在完整A-probe12独立工程验收通过后明确回复“进入”，授权本次完整A-smoke600。远端独立release `source_credit_v2_smoke600_20260906_v1` 的910份文件/约37.49GB SHA全部通过，实际base/ReaRAG路径检查与10项远端负向检查通过。

训练已于北京时间2026-09-06 16:44:32（UTC `2026-09-06T08:44:32Z`）启动，supervisor PID **2846**、training PID **2847**。截至16:44:48，监督状态为RUNNING，处于初始化阶段，已完成轨迹 **0/600**，尚无首批PPO更新或实际TensorBoard run；不能把进程已启动写成已完成更新。主代理持续按分钟检查，不预测训练效果或完成时间。

**16:46:38进度快照：** 已完成2批、**8/600**条，初始显式SFT reference KL为0；真实TensorBoard writer已启动。初始化0条为上面的历史快照。实时指标继续保存于`outputs/audits/ppo_a_smoke600_gpu_supervision_20260906_v1/snapshots/`，后续进度以监督记录为准。

本地149项相关测试和独立真实loader的13项负向检查已完成。此前probe12只证明完整工程链路可运行；原fresh132健康FAIL与所有历史失败保留，不把用户授权改写成自动统计放行。

## 冻结身份与预算

- 训练 Experiment ID：`ppo_mixed4_answer_format_v2_a_smoke600_scoped_seed42_20260906_v1`。
- 监督 Experiment ID：`SOURCE-CREDIT-V2-A-SMOKE600-GPU-SUPERVISED-20260906-V1`。
- [用户授权记录](../outputs/audits/ppo_a_smoke600_authorization_20260906_v1/authorization.json)。仅授权完整A的600条，不自动扩展到F/T、full12000或失败重启。
- [新配置](../configs/training/phase3_ppo_mixed4_answer_format_v2_a_smoke600_scoped_seed42.yaml)：相对原A-smoke600只改变gate与output路径。
- [冻结gate](../outputs/calibration/source_credit_gate_v2_smoke600_scoped_20260906_v1/gate.json)，SHA `7b95bbdaaac93e814214c376bf41c22f15d1a8a4cbf144172031ccea80ea410a`。
- [冻结scope](../outputs/calibration/source_credit_gate_v2_smoke600_scoped_20260906_v1/scope.json)，SHA `f3aecfa797bc86dbd6f1a20683e3a7f438190e0b8aa235d290fc9c93ec33ace2`。

起点仍为原Strong SFT：`checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final`，不从probe12 checkpoint继续。显式冻结SFT reference、真实冻结ReaRAG、答案/格式v2、Text/ProofKG过程奖励、六维learned α、原800图题/671 PASS来源mask及replay均保留。

原计划固定150题×K4=600条轨迹，HotpotQA/2Wiki/MuSiQue各200条；batch4、mini-batch1、PPO epochs2、lr1e-6、BF16、seed42、最大生成384 tokens。计划共150个PPO批次，第200、400条保存中间checkpoint，正常完成后保存`final/`；本次实际只执行75批，在300条停止，保存`step_200/`与`aborted_step_300/`，未产生400条或final checkpoint。Replay按原10%样本比例与anchor权重0.1执行。

原成本保护规则不变：从第200条开始检查最近15个PPO批次，平均有效率低于0.7、长度截断比例高于0.2或平均KL高于10时停止并保留失败产物；NaN/Inf立即失败。该规则负责运行健康与成本保护，不等于模型性能评估门槛。

PyTorch张量峰值70.18184GiB，缓存预留峰值93.74023GiB；监督采样的整卡最高占用94.41699GiB，三者口径不同。停止时GPU显存已释放；17:12:46的最新SSH检查被远端关闭，当前实例电源模式无法核实，所有产物及events此前已完整备份到本地。

## 日志与 TensorBoard

远端工作目录为 `/root/autodl-tmp/kgpaper_releases/source_credit_v2_smoke600_20260906_v1`。以下只查看运行状态，不启动或重启训练：

```bash
ssh -p 30481 root@connect.bjb1.seetacloud.com
cd /root/autodl-tmp/kgpaper_releases/source_credit_v2_smoke600_20260906_v1
tail -n 100 outputs/launches/ppo_a_smoke600_scoped_20260906_v1.log
```

查看已停止任务的监督终态和实际TensorBoard路径：

```bash
cat outputs/launches/ppo_a_smoke600_scoped_20260906_v1_supervision/status.json
cat outputs/ppo_mixed4_answer_format_v2_a_smoke600_scoped_seed42_20260906_v1/tensorboard_run.json
```

通过AutoPanel→TensorBoard访问`/root/tf-logs`的6007服务。在Runs中选择本次已实际生成的run：

```text
kgpaper/ppo_mixed4_answer_format_v2_a_smoke600_scoped_seed42_20260906_v1/20260906T084601Z_805b7946
```

完整远端events目录为：

```text
/root/tf-logs/kgpaper/ppo_mixed4_answer_format_v2_a_smoke600_scoped_seed42_20260906_v1/20260906T084601Z_805b7946
```

该路径与本次`tensorboard_run.json`对应。选择这个smoke600 run查看实际训练曲线，不要将旧probe12或`_diagnostics`缓存事件并入本次结果。

关注真实loss/KL、答案与过程奖励、有效率/截断率、eligible-valid α及六维特征、replay、显存和吞吐。横轴为已完成轨迹数，每批增加4；前3批和之后每10批记录histogram。无有效图样本的批次不补造α分布。

## 验收边界

预设保护规则、完整日志、history、事件、checkpoint参数变化与冻结身份的终态验收均已完成，失败和中间checkpoint均保留。[汇总报告](../outputs/audits/ppo_a_smoke600_gpu_supervision_terminal300_20260906_v1/report.json)绑定完整证据。训练rollout EM/F1只用于运行诊断。最终EM/F1是否改善仍须冻结开发选模与独立综合baseline评估；本次既未完成600条，也尚无独立评估结果，不能据此作论文方法结论。

前序记录：[A-probe12执行与归档](ppo_a_probe12_supervision_20260906.md)。
