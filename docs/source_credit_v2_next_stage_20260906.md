# Source-credit v2：下一阶段执行记录

2026-09-06。研究者要求“下一步”。在已完成v2修复的基础上，本阶段已完成生成长度的配对验证，以及新问题family的独立确认输入与来源准备。完整方法仍为答案奖励 + ReaRAG/Graph过程奖励 + learned α；当前没有PPO更新。

## 1. 本地生成长度对照

从代表性120题银行按固定身份hash选择60题：HotpotQA20、MuSiQue20、2Wiki有图16+ordinary4；每题K2，共120条。选择不读Gold、轨迹有效率或奖励分数。384-token基线来自已冻结的完整generation记录；新版本只将输出上限改为512。

冻结SFT、输入prompt/passages/KG、逐candidate seed、BF16、batch1、temperature1/top-p1/top-k0及EOS语义均一致。旧EOS结束样本必须整条输出相同；旧384长度上限样本必须前384个token相同。只有全部120条通过，才将结果解释为输出长度的单变量对照；任一不一致保留结果并标为前缀复现失败，不自动换seed、重抽问题或扩大样本。

检查沿用生产format-v2与现有最小步数规则，报告有效恢复/退化、cap/EOS原因、各数据集与有图/普通分层、token成本。本步骤不加载ReaRAG、不读取Gold、不修改−4非法奖励，也不按格式结果自动修改正式PPO配置。90%是原TODO中的建议健康参考，不能声称它已是既定强制门。

权威运行目录：`outputs/audits/generation_length_384_512_paired_probe_20260906_v1/`。

**实际结果：120/120条前缀完全一致，首尾模型、代码、输入与384基线SHA一致。**

| 指标 | 384 | 512 |
|---|---:|---:|
| 有效候选 | 96/120（80.00%） | 98/120（81.67%） |
| HotpotQA有效 | 32/40 | 32/40 |
| 2Wiki有效 | 34/40 | 35/40 |
| MuSiQue有效 | 30/40 | 31/40 |
| 长度截断 | 6 | 1 |
| EOS结束 | 114 | 119 |
| 总生成tokens | 26,541 | 26,899 |

恢复2条、退化0条、仍无效22条；token成本增加1.35%。输出长度不足只解释一部分格式失败，512仍远未达到建议90%健康参考。此次没有测EM/F1，也不根据这120条的结果继续搜索更大上限或改EOS约束。正式配置保留384；512保留为已测的候选预算，未来若采用，须在独立确认开始前与训练配置一起另版锁定。

实际生成约674秒，GPU peak allocated15.57GiB、reserved16.23GiB，进程已退出且GPU已释放。7项关键测试通过。独立CPU复核从原parent manifests核对384候选精确复用、重新decode与检查240侧format，并独立重算prefix及配对统计，见 `outputs/audits/generation_length_384_512_independent_review_20260906_v1/report.json`。

## 2. 独立确认的身份与来源准备

原830题候选银行已经覆盖正式训练集全部800道有图题；同题换seed或增加K不能称为新family独立确认。新身份来自official-raw n1500中未进入旧候选、代表性统计或保护账本的剩余样本，排除qid、question hash与family，并保持来源可追溯。

当前拟冻结132题身份提案：三个仍有fresh供给的图类型各32题，共96图题；另取HotpotQA、MuSiQue、2Wiki ordinary各12题，共36道文本通路对照。fresh inference供给为0，因此不能声称独立确认覆盖四种图问题。选题在新来源PASS检查之前完成；不得看PASS、答案质量或有效率后替换不通过的题。

132题身份已冻结，96道图题与正式3000题的qid、question hash、family交集均为0；36道ordinary题来自正式训练池，只能称α开发未见，不能用于声称PPO训练后独立测试。原正式数据与schedule不因此删题。

现有source-credit mask只绑定旧已审计身份。对新图题直接使用旧mask会得到缺失身份、α=0，不能据此检验Graph/α效用。新确认需要从原绑定store/cache派生来源证据，运行同一冻结`source_integrity_v1`规则，并生成确认专用mask与绑定封装。原671题图信用、门系数、feature标准化、Graph/Text统计均保持不变；来源未通过的确认题保留并关闭图信用。

身份提案目录：`outputs/audits/source_credit_v2_fresh_confirmation_capacity_20260906_v1/`。它不是已完成的confirmation，不改变正式3000题或K4训练schedule。

**132题输入与离线来源准备已完成。** 输入scope权威入口为 `outputs/audits/source_credit_v2_fresh_confirmation_inputs_20260906_v1/manifest.scope_v2.json`；每题10 passages，最大2262 input tokens，132个不同family，所有原始record与消息格式保留。初版manifest保留，scope补充没有修改输入字节。

96图题同一来源规则的结果为：

| 图类型 | PASS | UNVERIFIED | FAIL |
|---|---:|---:|---:|
| bridge_comparison | 18 | 9 | 5 |
| comparison | 30 | 1 | 1 |
| compositional | 31 | 1 | 0 |
| 合计 | 79 | 11 | 6 |

17题未获图信用，但问题和输入全部保留。没有联网或写缓存；新mask真实加载复验，两组确认专用门的权重、标准化及Graph/Text统计逐字段保持不变，96身份均无MISSING，非PASS的α为0。该空轨迹接线检查不是过程效用确认；正式训练加载仍拒绝这两组未确认门。13项来源准备测试通过，10个输出SHA独立复核通过。

来源权威目录：`outputs/audits/source_credit_v2_fresh_confirmation_source_20260906_v1/`；`scope_addendum.json`额外绑定输入scope_v2。旧800题/671图信用、原mask、原gate和正式3000题不变。

## 3. 确认完成后再推进PPO

在读取确认Gold前冻结输入、实际生成预算、评分代码及判定规则。过程排序只能用Graph/Text过程项，Gold用于事后评估；不能把包含答案奖励的总reward排序当作独立过程效用。K4的多个正确/错误组合共享同一个问题，样本量与区间必须按qid/family处理。

新family确认与同题新trajectory诊断分别报告；工程连线检查、候选健康建议、正式科研效用门也分别报告。不能把“没有显著A−F优势”自动解释为三batch工程probe不可执行，也不能把工程probe通过解释为正式训练或论文结论已通过。

既定后续为完整A-probe12 → A-smoke600 → 冻结development150选模 → 独立full12000与matched F/T。PPO参数仍为lr1e-6、batch4/mini-batch1、epochs2、γ1/λ.99、显式SFT reference和10% replay；本阶段不同时试调mini-batch、奖励权重或α结构。

上轮已冻结产物：[v2修复记录](source_credit_v2_repair_20260906.md)。本文件记录新的执行阶段，不覆盖上轮release快照。

当前待执行的是：冻结确认生成与分析协议（含K、greedy参考、过程排序、无Gold tie-break、family聚类及信息不足规则），生成固定132题的真实候选并调用一次ReaRAG，再对同一候选做A/F/T与N-only/N+F重排。该步骤尚未执行，不以本阶段的来源PASS或格式对照替代其结果。
