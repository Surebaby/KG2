# Fresh132 独立过程确认：现有 Strong SFT 的 PPO 路线

Experiment ID：`SOURCE-CREDIT-V2-FRESH-CONFIRMATION-K4-GREEDY-SEED42-20260906-V1`。

## 最终结果：独立效用通过，格式健康未通过

2026-09-06，生成、评分与恢复分析均已完成。当前权威结果为[恢复分析报告](../outputs/audits/source_credit_v2_fresh_confirmation_local4090_20260906_v1/analysis_recovery_v1/report.json)，SHA `a469aa370567ea3020e9af6c4cee5bafd10f2e2472281e54f295d9222c4c7516`；[恢复协议](../outputs/audits/source_credit_v2_fresh_confirmation_analysis_recovery_20260906_v1/protocol.json) SHA `286e89606fa574e5b2f856c052602f08be65583d4b5d0c4ef6bcb39bbbd2533a`。修正副本只有一处分值范围检查变化，额外19项测试通过；原代码、失败目录、生成与评分都保留且未重做。

最新执行状态保存在[latest_result.json](../outputs/audits/source_credit_v2_fresh_confirmation_local4090_20260906_v1/latest_result.json)；原`status.json`保留首次控制器失败时的历史快照，不表示恢复分析未完成。GPU已回落至显示占用约666MiB、利用率0%。

`independent_utility_status=PASS`，`health_status=FAIL`，因此`overall_status=FAIL`；`engineering_probe_eligibility=true`，`matched600_investment_clearance=false`。这表示有条件推进限额完整PPO-A的工程probe，不表示已经签发生产gate或证明α优越；当前SFT/PPO更新仍0。

下表是固定96道新图题的**冻结候选过程排序**，不是训练后的baseline成绩。A/F/T共用K4候选且仅按过程分选择，greedy是单独参考；全无效题保留赋零。

| 选择方式 | EM | F1 |
|---|---:|---:|
| 原Strong SFT greedy | 83.33%（80/96） | 88.73% |
| A：动态α图文过程排序 | 84.38%（81/96） | 88.03% |
| F：固定α图文过程排序 | 84.38%（81/96） | 88.72% |
| T：Text-only过程排序 | 66.67%（64/96） | 72.62% |

A−greedy EM为+1.04pp，95%CI为[−5.21,+7.29]pp；不构成稳定提升证据。A与F逐题EM相同，选择候选仅3/96题不同，A的F1点估计略低；不能声称动态α已有额外收益。F−T EM为+17.71pp，95%CI为[+10.42,+25.00]pp，支持这批新图题上的图文混合排序信号；F同时改变Text权重，因此不是保持Text系数不变后纯增加Graph的实验。

在79道来源PASS题中，39个独立family含有效正确/错误候选，产生114个相关crosspairs。按family等权的pairwise为A94.87%（95%CI89.74–99.15）、F97.01%、T48.93%；不能把114对当作114个独立问题。Graph96的valid-K4 oracle−greedy EM为3.125pp，刚超过冻结的3pp信息门，CI仍跨0。

格式健康：sampled有效471/528=89.20%，95%CI86.74–91.48%，该项INCONCLUSIVE；三域等权macro为82.64%，95%CI76.70–87.96%，未达到90%而FAIL。H为37/48=77.08%，W为396/432=91.67%，M为38/48=79.17%；H/M各只有12题，保留小样本边界。greedy有效130/132=98.48%，不能与采样分母混用。

完整132题A过程top1 EM/F1为68.94/73.64%，greedy为71.97/78.14%；普通36题Text回退top1 EM为27.78%，greedy41.67%。这些结果与图题优势并列保留，不把graph96/PASS79收益外推成跨域总体提升，也不把ordinary36当作PPO之后的独立测试。

独立复核确认5个输出SHA、2651个唯一封存文件SHA、132/660身份、1584个候选→题级指标关联及1206个分层点估计一致；未重新读取Gold或重跑bootstrap。下一步仍为同Strong SFT的限额完整A-probe12；需要从原训练gate/671信用mask派生绑定本次确认的版本，原远端恢复连接后执行。600预算须另据健康风险与工程结果明确决策，不能自动越过本次冻结结果；本轮不扩样、不重训SFT、不调奖励或温度。

以下为启动及故障恢复历史记录。

2026-09-06 13:00（UTC+8）已在本地启动，控制器PID3077412、生成子进程初始PID3077413。当前状态为`FRESH132_CONFIRMATION_PIPELINE_RUNNING_PPO_NOT_STARTED`；即时状态以运行目录`status.json`为准。145项新测试全部通过（0失败/错误/跳过），记录保存在协议目录`validation.json`与`tests.junit.xml`。

已冻结[protocol](../outputs/audits/source_credit_v2_fresh_confirmation_execution_20260906_v1/protocol.json)，SHA256：`d1bf882764c825e8e14de9a83e7219ed68ea83b8c480096e23c952f260b8fc97`。实际执行代码及全部模型/输入身份已绑定；[launch记录](../outputs/audits/source_credit_v2_fresh_confirmation_execution_20260906_v1/launch.json)保存启动命令与进程。

## 14:11 收尾检查与恢复

生成已于14:01:26完成，660条文件及逐条payload SHA独立复验通过；生成阶段约3623秒。评分也完成660条，实际1927个ReaRAG步骤，约579秒；采样471/528格式有效，greedy130/132有效。这里分开报告两个生成条件，不把660条混成PPO采样有效率。

首次分析在Gold读取前停止，原`analysis/failed.json`保留`gold_boundary_entered=false`。原因是分析器对所有raw Graph统一要求非负，但原ProofKG v2.3对结构无效输出合法地返回−1诊断值。实际39条负分全部是无效轨迹，proof.score均为−1，六个视图的实际过程分与α均0，不进入排序。

这是分析校验器未覆盖无效诊断值的工程缺陷，生成、评分、奖励和排序未改变。恢复只允许修正版副本将该诊断范围按valid/invalid分开，并保留CPU精确重算及全部原判定规则。原代码、协议、失败目录不可覆盖，修正副本与恢复执行须另存绑定；此处尚未报告EM/F1或独立效用结论。

研究者已批准优先用现有 Strong SFT 获得有限预算 PPO 结果。新版 SFT 与付费 teacher 继续暂停。本实验先检查冻结的过程奖励和 α 是否具有独立于答案奖励的排序效用，再进行完整 PPO-A 的工程 probe。A 包含答案、Text、Graph 和 learned α；F 为固定 α，T 为 Text-only 过程对照，三者仍共享答案奖励。

## 固定的计算与分析

- 原 132 题不替换：Graph96（三种类型各32）、ordinary36（H/M/W各12）。79 个 Graph 来源 PASS、11 UNVERIFIED、6 FAIL；后两类保留并关闭图信用。没有 fresh inference 类型。
- Graph96 与正式 PPO3000 的 qid、question hash、family 均隔离。ordinary36 属于未来 PPO train，只用于训练前格式、Text 和回退诊断，不能在 PPO 后称独立测试。
- 每题 K4 采样加 1 条 greedy，660 条。现有 Strong SFT/BF16、batch1、384 输出上限、seed42、原提示和 EOS；采样 temperature/top_p=1、top_k=0。
- 先完成全部生成，再单独加载原 BF16 ReaRAG。每条合法轨迹的每步真实 Text 只评分一次，共用原 Graph v2.3 原始分数；N-only/N+F × A/F/T 六个视图在 CPU 重算。
- 过程 top1 只在 K4 合法候选中按过程分排序，同分取较小 candidate_index；greedy不参与K4排序。全无效题保留且 top1 EM/F1为0。排序不含答案奖励。
- 全部过程记录和排序封存后才读取只读 Gold，使用当前 canonical 答案抽取、EM/F1与原标签词面规则。N+F 为唯一主版本，N-only完整报告，不按结果择优。
- pairwise 先在题内对正确/错误候选交叉配对取均值，再按 family 等权。CI为20,000次固定构成的分层 family bootstrap，seed42，95%。报告完整132、Graph96、source-PASS79、三种图类型和各数据集，不能把图题偏重的132题汇总冒充均衡 baseline。

格式健康与独立过程效用分别判断。90% sampled有效率（micro及三域macro）仍是健康目标；有效混合结果 family 至少25、graph96的valid-K4 oracle−raw greedy EM至少3pp是信息量参考；主过程效用看source-PASS79 A pairwise至少0.65，以及graph96 A top1不低于greedy和F。点估计未达标而区间跨阈值时保留INCONCLUSIVE，不能补采样直到通过。F−T完整报告，不能称为固定Text权重后纯加Graph的效应。机器判定以执行前冻结的 protocol 为准。

格式风险不会直接改名为“α无效”，工程probe也不要求统计显著A>F；完整确认通过不等于论文中的α优势成立。后续600预算与正式训练仍需根据相应健康、效用和工程结果推进。

若独立效用确认通过，生产gate应从原representative calibration gate另版派生，附确认报告绑定并保留原训练800身份/671图信用mask；fresh wrapper的96题mask仅用于此次确认，不能复制进正式PPO。

## 执行产物与监控

协议目录：`outputs/audits/source_credit_v2_fresh_confirmation_execution_20260906_v1`。

执行目录：`outputs/audits/source_credit_v2_fresh_confirmation_local4090_20260906_v1`。

```bash
cd /home/zjulab/kgpaper
cat outputs/audits/source_credit_v2_fresh_confirmation_local4090_20260906_v1/status.json
tail -f outputs/audits/source_credit_v2_fresh_confirmation_local4090_20260906_v1/generation.*.log
```

切换到评分后查看同目录 `scoring.*.log`；最后结果为 `analysis/report.json`。`status.json` 是运行状态，正式科学产物以各阶段不可覆盖的 manifest 与 SHA 为准。自动流水线串行执行生成→评分→判定，任一步故障即停止；断点恢复使用原候选 seed，不按格式或答案结果重采样。

本地4090可分别完成两个模型的推理，完整PPO需同卡加载policy/reference/ReaRAG，沿用此前96GB远端环境。2026-09-06本轮只读远端检查在SSH握手阶段返回 `Connection closed by remote host`，尚未重新接通或同步。确认流水线不启动SFT/PPO、不重新拟合α、不修改任何gate放行标志；本轮尚无新模型指标。

## 验证与边界

新增生成器测试覆盖固定输入/模型/代码/seed/提示/EOS、原子提交及中断恢复；评分器测试覆盖真实共享评分函数的CPU替身调用、来源屏蔽、两步规则、非有限分数、越界输入、负有效分与全无效排序、greedy隔离及评分恢复。控制器另检查失败停止、并发锁、SIGTERM转发和不自动重做Gold分析。

全部生成前冻结模型、代码、标签源字节SHA及判定规则；原 Gold、正式PPO数据、baseline、奖励参数和已有失败实验均保留。源码工作区已有大量未提交变更，因此记录Git HEAD并保存本次实际执行文件快照，不将HEAD冒充完整执行版本。
