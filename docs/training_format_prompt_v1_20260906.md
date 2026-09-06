# 训练格式提示 v1：固定开发集配对验证

2026-09-06。研究者在讨论“格式无效怎么解决”后确认执行：先明确训练专用提示，在已消费的固定60题上验证，再决定是否需要独立版本的格式SFT。本阶段不更新PPO参数。

## 固定实验

实验 ID：`TRAINING-FORMAT-PROMPT-V1-PAIRED-20260906-V1`。

旧384→512长度试验仅使有效率从96/120变为98/120，22条仍无效；其中21条主动EOS结束。因此本次保留384输出预算，只试一个system提示版本。仍使用原Strong SFT、BF16、逐candidate seed、temperature=1、top-p=1、top-k=0、batch=1，复用同60题、K2的原384候选作为配对参考。

新模块 `kgproweight/data/training_format_prompt_v1.py` 为显式opt-in，不修改原SFT/RL/inference提示。新提示明确最低步数至5步、连续编号、每步三个字段、Reasoning至少20字符，以及在Final标题后写完答案才结束。同时加入不凑步重复、不引用无关图三元组的指导；这些是提示中的质量指导，不冒称都是现validator的硬校验条件。

最低步数直接从原format-v2 validator读取，使用source hardgate原值、在信用mask之前计算。60题中9题至少2步、51题至少3步。独立CPU审查确认60/60原user消息、passages、KG、完整source spec与身份不变。原reward、−4无效惩罚、α、归一化、checkpoint均保持冻结。

提示模块SHA256：`0ecd28a9ad35448951916b6330d77adb69bcc204734bd6fe4cf47c62e0be617b`。

## 预先固定的评估

独立评分实验 ID：`TRAINING-FORMAT-PROMPT-V1-ASSESSMENT-20260906-V1`。

评分协议在新生成开始前固定。生产worker不读取Gold、不调用ReaRAG。独立评估必须确认120条完整、身份与文件SHA一致，先写入纯格式/重复诊断与代码冻结凭证，再读取封版train labels。

- 格式沿用当前format-v2，不放宽步数、字段或Final检查。
- 原始答案EM/F1使用既有主表的双次答案抽取及canonical EM/F1，primary+aliases取max。全部120条进入分母，包括格式失败与空答案。
- 单独报告format-valid×EM/F1，以及PPO现用first-line答案抽取的门控诊断，不混用三种口径。
- 每题先平均K2，再按题平均、按三数据集等权macro。95%差值区间使用seed42、10000次dataset分层family bootstrap，不把120条当独立问题。
- 分开报告轨迹内重复Reasoning/Conclusion/整步、同题K2全文相同率、规范化答案相同率；精确重复是代理指标，答案相同也可能表示稳定性。

90%有效率仅作健康参考，不是新增正式放行门。没有自动提升提示版本或checkpoint的规则；不得看EM/F1后继续搜索提示、换seed或补抽到通过。本次开发试验不能替代新family确认、综合baseline评估或PPO结果。

## 权威产物与状态

- 提示合同与60题独立检查：`outputs/audits/training_format_prompt_v1_contract_20260906_v1/`。
- 配对生成与运行日志：`outputs/audits/training_format_prompt_v1_paired_probe_20260906_v1/`。
- 预先冻结的评分协议与执行代码：`outputs/audits/training_format_prompt_v1_assessment_20260906_v1/`。

当前状态：`TRAINING_FORMAT_PROMPT_V1_PROBE_COMPLETE_NOT_ADOPTED_PPO_NOT_STARTED`。120条真实生成与独立评分均完成。

## 实际结果与决定

| 固定60题、K2、两侧384 | 原提示 | 训练prompt-v1 |
|---|---:|---:|
| 格式有效 | 96/120（80.00%） | 110/120（91.67%） |
| 全量canonical EM | 38.33% | 35.00% |
| 全量canonical F1 | 43.40% | 37.82% |
| valid × canonical EM | 35.00% | 34.17% |
| valid × canonical F1 | 39.28% | 36.92% |
| PPO first-line门控EM / F1 | 35.00% / 39.28% | 35.00% / 37.75% |
| 输出tokens总计 | 26,541 | 29,956 |
| 长度截断 | 6 | 7 |
| 最低步数不足 / 超过5步 | 11 / 4 | 1 / 0 |
| 重复Reasoning或整步的轨迹 | 0 / 0 | 0 / 0 |
| 重复Conclusion的轨迹 | 3 | 5 |

21条从无效恢复有效，7条从有效退化，3条持续无效，89条持续有效。输出tokens增加12.87%，输入每题增加235 tokens。未观察到精确重复整步，但不能用该代理指标否定语义重复或凑步。GPU真实生成738秒，peak allocated15.62GiB、reserved16.45GiB；任务已完成并释放GPU。

| 数据集（每侧40条） | 有效率原→新 | EM原→新 | F1原→新 |
|---|---:|---:|---:|
| HotpotQA | 80.0% → 92.5% | 50.0% → 50.0% | 53.62% → 55.54% |
| 2Wiki | 85.0% → 87.5% | 55.0% → 47.5% | 56.09% → 47.68% |
| MuSiQue | 75.0% → 95.0% | 10.0% → 7.5% | 20.51% → 10.25% |

按family分层bootstrap的95%差值区间：valid为+3.33至+20.00个百分点，EM为−12.50至+5.83个百分点，F1为−15.58至+3.96个百分点。区间用于开发诊断；EM/F1区间跨0，不能断言稳定退化，也不能据此宣布等效或无损。

**决定：保留prompt-v1作为已完成开发对照，当前不提升为正式PPO提示。** 它改善了步数与字段遵循，但没有提供答案质量改善的证据，且增加token成本。不能仅靠91.67%格式有效率宣布修复完成。此次没有进一步搜索提示、增加长度或换seed来追回分数，也没有用新132题选择提示。

44项相关CPU测试通过，含提示/producer18项、独立评分器26项；四个既有指标实现及此前98个核心代码/配置SHA不变。所有候选包括10条新无效输出均保留，原baseline checkpoint和评估协议未修改。

独立复算120条所有字段及12项macro/区间与主评分一致。额外抽取诊断确认：EM有16条由对变错、12条由错变对；净少4条中仅1条由Final首行之后继续解释造成。即便仅作首行诊断、不乘格式门，正确数仍46→43。因此不能靠改答案parser消除这次质量疑虑。独立入口：`outputs/audits/training_format_prompt_v1_independent_review_20260906_v1/manifest.v2.json`。

后续优先核查封版replay目标的真实format-v2健康，再设计独立、短程的格式SFT对照。若执行，将保留Strong SFT，另存继任checkpoint，A/F/T使用同一起点，并评估该继任本身的SFT对照。继续SFT会更新语言模型，不能把全序列交叉熵描述成“只学标点而不影响语义”；replay仅来自Hotpot，也不能宣传为三数据集均衡监督。当前没有启动SFT或PPO训练。

replay健康核查已完成：2000/2000原监督目标满足当前format-v2，原始行、渲染目标、train-fold来源SHA完整对齐；全部为Hotpot，原hardgate均为0、最低3步。两种system下assistant目标token序列2000/2000保持一致；原/new最大prompt4231/4466、最大full4753/4988，无删passages或超packing预算。这只证明现有监督目标的结构与装载健康，不证明自由生成能力或384输出预算已经对齐。审计入口：`outputs/audits/format_sft_replay_target_health_20260906_v1/`；没有生成新筛选集或修改监督。

独立预算补充复现全部2000条封版assistant token长度（含模板结束token）：median195、p95346、max1347；1941条不超过384，59条超过384（2.95%），其中23条超过512。超384分布为3步13/1593、4步18/336、5步28/71；本轮未过滤或截断这些目标。监督目标本身不存在本次validator硬格式缺陷，不能把继续SFT称为修复坏标签；未来若做格式适配，应先固定唯一的训练与输出预算，优先保留原prompt以隔离权重更新的影响。

新132题confirmation尚未生成，不能拿它选择提示。若采用新训练提示，后续A/F/T应使用同一版本；policy与reference共用同一query，replay需保留原目标与证据。ReaRAG的评分上下文继续沿用原canonical passage-only提示，不能随policy提示一起替换。现α与Text统计可保持数值冻结，在新提示下做fresh confirmation；旧提示下的开发结果不能挪作新确认。生产配置和未确认gate的训练禁用状态目前不变。
