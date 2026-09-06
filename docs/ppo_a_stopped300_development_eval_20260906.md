# PPO-A 停止后 checkpoint 开发评估

2026-09-06，当前状态：`EVAL_DEVELOPMENT150_COMPLETE_SFT_SELECTED_AUDITED`。

评估于北京时间 **18:42:44** 完成。[终态status](../outputs/evals/ppo_a_stopped300_development150_20260906_v1/status.json)为`COMPLETE_DEVELOPMENT_COMPARISON_NOT_MAIN_TABLE`：900/900条预测、六项生成、六项评分和一次selection共13项任务全部完成。先生成全部回答，再评分与选模的顺序保持不变。

[原规则选模结果](../outputs/evals/ppo_a_stopped300_development150_20260906_v1/selection/report.json)选择 **Strong SFT**，排名为`strong_sft → ppo_a_step200 → ppo_a_aborted300`。下表直接汇总六份原评分报告，单位均为百分比。[独立结果验收](../outputs/audits/ppo_a_stopped300_development150_results_independent_20260906_v1/report.json)已通过：900条结果从原raw输出及原Gold独立复算一致，六项生成/六项评分/selection的SHA与先全部生成后评分的时序一致。审计PASS表示产物、评分和原规则执行一致，不代表PPO取得性能提升。

## 实际开发结果

三域各50题，macro为三个域均值。legacy是预先冻结的主选模视图，no_graph作为另一固定视图完整报告。

| 视图 | 模型与原评分报告 | Macro EM（%） | Macro F1（%） |
| --- | --- | ---: | ---: |
| legacy | [Strong SFT](../outputs/evals/ppo_a_stopped300_development150_20260906_v1/scores/strong_sft__legacy/report.json) | 24.00 | 34.70 |
| legacy | [PPO step_200](../outputs/evals/ppo_a_stopped300_development150_20260906_v1/scores/ppo_a_step200__legacy/report.json) | 22.67 | 35.34 |
| legacy | [PPO aborted_step_300](../outputs/evals/ppo_a_stopped300_development150_20260906_v1/scores/ppo_a_aborted300__legacy/report.json) | 22.00 | 32.79 |
| no_graph | [Strong SFT](../outputs/evals/ppo_a_stopped300_development150_20260906_v1/scores/strong_sft__no_graph/report.json) | 25.33 | 36.66 |
| no_graph | [PPO step_200](../outputs/evals/ppo_a_stopped300_development150_20260906_v1/scores/ppo_a_step200__no_graph/report.json) | 24.67 | 35.25 |
| no_graph | [PPO aborted_step_300](../outputs/evals/ppo_a_stopped300_development150_20260906_v1/scores/ppo_a_aborted300__no_graph/report.json) | 24.67 | 36.50 |

各域结果如下，每格为 **EM / F1（%）**，每域分母均为50题：

| 视图 | 模型 | HotpotQA | 2Wiki | MuSiQue |
| --- | --- | ---: | ---: | ---: |
| legacy | Strong SFT | 32.00 / 51.64 | 28.00 / 36.44 | 12.00 / 16.00 |
| legacy | PPO step_200 | 30.00 / 50.58 | 30.00 / 41.18 | 8.00 / 14.28 |
| legacy | PPO aborted_step_300 | 34.00 / 52.31 | 26.00 / 34.78 | 6.00 / 11.28 |
| no_graph | Strong SFT | 36.00 / 54.98 | 30.00 / 39.51 | 10.00 / 15.48 |
| no_graph | PPO step_200 | 34.00 / 52.91 | 30.00 / 37.05 | 10.00 / 15.80 |
| no_graph | PPO aborted_step_300 | 32.00 / 52.31 | 34.00 / 42.79 | 8.00 / 14.40 |

相较Strong SFT，step_200在legacy的F1增加 **0.65个百分点**，EM下降 **1.33个百分点**；aborted_step_300的EM下降 **2.00个百分点**、F1下降 **1.91个百分点**。差值根据未四舍五入的原分数计算。主选模规则优先EM，因此不能用step_200的局部F1增加替代原规则提升它的排名。

这组开发结果不支持PPO带来综合改善，也不支持α的独立贡献。legacy的EM差异对应150题净少2题和3题，不能据此宣称显著退化；各域和视图的变化并不一致，尚不能归因为α或最低步数规则。本次仍是150题固定上下文开发代理，900条预测不是正式canonical900题主表。原训练300条guard `FAILED`及所有checkpoint保留，不自动重启、新训练、扩full/F/T或开展正式canonical900评估。

[逐题配对诊断](../outputs/audits/ppo_a_stopped300_development150_paired_diagnostic_20260906_v1/report.json)已封存：相对SFT，legacy的step200有6题错转对、8题对转错；step300有5题错转对、8题对转错。两个PPO checkpoint的no_graph均为5题错转对、6题对转错。该报告只做事后描述，未执行显著性检验或bootstrap，也未用于改选模型或调整原协议。

独立结果验收报告SHA：`2e466259364630b88da4a3404ec5c9ce67b1cbcbcd66dc1b158940fab1b33a45`；配对诊断报告SHA：`dad9be879ea13a0caacd3b060f1f003cf0e3237084f592d7b43615b11f164ef6`。

## 启动历史与固定协议

以下为启动时的历史记录，当前状态以上方完成记录和终态产物为准。

2026-09-06，启动时状态：`EVAL_DEVELOPMENT150_RUNNING`。

研究者明确回复“行 启动评估吧”后，本地评估于北京时间 **17:29:19**（UTC `2026-09-06T09:29:19.805170+00:00`）真实启动，supervisor PID **3338466**。本次只比较原Strong SFT、PPO `step_200`与`aborted_step_300`，使用原A开发银行的150题和两个固定视图，共计划 **3×2×150=900条预测**。

**17:30:29状态快照：** `RUNNING`，当前任务为`generate__strong_sft__legacy`，子进程PID3338467；`status.json`记录已生成12/900条，完整任务数0。尚无评分或选模结果。此计数只对应该时刻，后续以[实时状态](../outputs/evals/ppo_a_stopped300_development150_20260906_v1/status.json)为准。

执行 Experiment ID：`PPO-A-STOPPED300-DEVELOPMENT150-SEED42-20260906-V1`。[冻结执行协议](../outputs/evals/ppo_a_stopped300_development150_20260906_v1/execution.json) SHA为`8b0a49028ce6cf6fb52ad267e34d8a66b539325a37cb7d9dfe10937f9bc9223b`。

## 比较对象与固定输入

| 模型ID | checkpoint | 已完成训练轨迹 |
| --- | --- | ---: |
| `strong_sft` | 原`checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final` | 0条PPO |
| `ppo_a_step200` | 本轮终态归档下的`step_200` | 200 |
| `ppo_a_aborted300` | 本轮终态归档下的`aborted_step_300` | 300 |

两份PPO checkpoint均已通过[训练终态归档验收](../outputs/audits/ppo_a_smoke600_gpu_supervision_terminal300_20260906_v1/report.json)，LoRA和value head实际发生更新且参数有限。原训练在300/600条时触发预设有效率guard，`FAILED`与600未完成的状态保持；开展评估不会将失败训练改写为完成。

原A银行为`outputs/audits/source_gated_mixed4_emf1_v1_development_a_smoke`。本次另版注册实际三个checkpoint；150题的qid顺序、`legacy.inputs.jsonl`、`no_graph.inputs.jsonl`及独立labels文件与原银行保持原字节，原银行未覆盖。每域HotpotQA/2Wiki/MuSiQue各50题，每题10 passages；`legacy`是原KG视图，`no_graph`保留相同问题与passages。

[独立输入审计](../outputs/audits/ppo_a_development150_inputs_independent_20260906_v1/report.json)确认300行input/prompt哈希、渲染、顺序与token长度一致，生成输入无顶层Gold字段；最长提示legacy2640、no_graph2326 tokens。与原Strong SFT完整24998条源文件、PPO3000、replay2000和既有confirmation/canonical集合的qid、question、current family重叠均为0。34项相关测试通过，[独立协议审阅](../outputs/audits/ppo_stopped300_development_protocol_independent_review_20260906_v1/report.json)及身份核查结果保留。

## 生成、评分与选模顺序

沿用原评估配置：**greedy、最多512新tokens、BF16、batch1、seed42、输入上限6144**，三个模型均使用原银行冻结的Strong SFT tokenizer与chat template。这里512是既定开发评估预算；原PPO训练的384生成预算未改变。三个checkpoint对全部300个提示的编码与特殊token语义一致，tokenizer的EOS/PAD均为128009；生成沿用基座generation_config的EOS列表`[128001, 128009]`。PPO保存时的padding元数据差异不改变本次共用tokenizer的输入。

先完成三个模型×两个视图的全部六项生成，共900条预测；全部生成完成后才执行六项Gold评分，再按原规则选模。主排序仍是legacy三域macro EM降序、macro F1降序，完全同分优先SFT，其后较早training step和model ID；no_graph作为固定的另一视图一并报告。未改变答案提取、EM/F1或选模规则。

[注册与执行独立复核](../outputs/audits/ppo_a_stopped300_successor_execution_independent_review_20260906_v1/report.json)于17:31完成，17/17项通过：三份checkpoint、119份代码绑定和冻结银行一致；各生成任务必须正常退出并通过完整150条封存核验，六项生成全部完成后才允许评分与选模。

本次是**150题开发集的固定上下文代理评估**。900是三个模型、两个视图的预测总数，不是900个独立问题；结果不能写成canonical Scheme-A正式主表或独立测试成绩，也不能据此宣称α优越性。本次不自动启动新训练、失败重启、F/T/full PPO或正式canonical900评估。

## 本地查看命令

在本地主机运行：

```bash
cd /home/zjulab/kgpaper
tail -n80 -F outputs/evals/ppo_a_stopped300_development150_20260906_v1/supervisor.log
```

查看首项生成的历史日志：

```bash
cd /home/zjulab/kgpaper
tail -n80 -F outputs/evals/ppo_a_stopped300_development150_20260906_v1/logs/generate__strong_sft__legacy.log
```

查看实际任务、已生成条数与完成任务列表：

```bash
cd /home/zjulab/kgpaper
cat outputs/evals/ppo_a_stopped300_development150_20260906_v1/status.json
```

这些命令只读取已完成任务的日志，不启动或重启任务。生成、评分和选模产物已经保留，终态独立复核已完成。训练阶段记录见[A-smoke600运行与停止](ppo_a_smoke600_supervision_20260906.md)。
