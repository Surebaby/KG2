# Proof-KG 零训练模型利用率结果

日期：2026-08-29  
Experiment family：`a1_fixed_context_kg_utilization_n30_20260829`  
冻结判定：`ADVANCE_TO_UNSEEN_CONFIRMATION_DESIGN`

## 结果

固定同一30题、同一2Wiki原始10段context、同一greedy解码和同一checkpoint，只替换
`kg_subgraph`。首次沙箱内GPU运行因NVML不可用、0题完成而中止，已保留为
`FAILED_LOCAL_GPU_RUNTIME`；沙箱外retry1未改变协议并完整成功。

| 模型 | legacy EM/F1 | proof EM/F1 | ΔEM | gained/lost | KG引用响应率 |
|---|---:|---:|---:|---:|---:|
| SFT | 0.467 / 0.526 | 0.633 / 0.650 | +16.7pp | 9 / 4，净+5 | 0.367→1.000 |
| hybrid PPO | 0.433 / 0.484 | 0.667 / 0.705 | +23.3pp | 11 / 4，净+7 | 0.433→1.000 |

SFT parse rate从0.933升至1.000，hybrid保持1.000；citation contract error分别
`0.067→0.033`和`0.033→0.000`。预注册的完整性、parse/no-harm、EM no-harm、citation
contract和model utility六项检查全部通过。

统计边界必须保留：SFT ΔEM bootstrap 95% CI=`[-6.7,+40.0]pp`，McNemar p=0.267；hybrid
ΔEM CI=`[0,+46.7]pp`，McNemar p=0.118。样本只有30题，不能声称统计显著或作为论文主表。

## 后置误差诊断（不改主指标）

SFT的4个exact-EM loss全部是同一日期的ISO格式与自然语言格式差异，例如
`1882-12-06`对`6 December 1882`。hybrid的4个loss中3个是该日期格式差异；唯一真实语义
退化是A1-014：KG已给出mother→mother的正确两跳链，模型在第二步已推出正确祖母，随后又
多走一跳输出了曾外祖母。这说明仍需限制“证据链已完成后的过度推理”。

用实验前已存在的日期/国籍alias `_value_match` 做**后置诊断**时，SFT为`0.600→0.900`
（净+9、0 loss），hybrid为`0.533→0.900`（净+11、1 loss）。该值未预注册，只能解释严格
EM低估，不能替代或改判正式EM。

在proof KG条件下，hybrid EM=0.667、SFT EM=0.633，仅多1/30题；F1高5.5pp。这个数值方向
符合“PPO可能在正确KG下优于SFT”，但样本太小且cohort已用于planner工程开发，不能据此
宣称PPO已经稳定超过SFT。

## 能回答与不能回答的问题

本轮可以回答：

- 模型确实读取并使用逐题Proof-KG，而不是完全忽略KG；
- 更聚焦的2.2条Proof-KG相较平均6.6条、8题为空的legacy KG产生了明显净收益；
- hybrid PPO没有在Proof-KG输入下退化，并取得本组最高EM/F1。

本轮不能回答：

- 自动planner能否在未见问题上稳定生成同等质量KG；
- 实时检索场景是否仍有相同收益；
- HotpotQA和MuSiQue是否成立；
- PPO是否在统计上稳定优于SFT。

29/30答案本来就在固定context可见，28/30答案值可由proof KG链获得。本轮是“正确证据的
模型利用率”测试，不是检索召回测试。

## 下一步

按冻结决策，只进入新的未见确认设计，不直接训练PPO。建议先做2Wiki n=100：排除全部A0、
A1、旧planner dev/confirmation和历史诊断family；在任何Gold审计前自动生成question-only
plan和property KG；同时冻结legacy/proof配对、结构门与模型利用率门。只有n=100同时证明
自动构建覆盖和零训练收益，才扩Hotpot/MuSiQue或申请PPO smoke。

## 关键文件

- 预注册：`outputs/audits/a1_fixed_context_kg_utilization_preregistration_20260829/protocol.json`
- 配对结论：`outputs/validation/a1_fixed_context_kg_utilization_paired_n30_20260829/report.json`
- SFT预测SHA256：`705ecc8cec6bc7494c286c29fb51e11a16546426fbe0011c7eb4d89c08421968`
- hybrid预测SHA256：`79568ca1a87739b2fcc0fa0f5f103f880c28207f43a5d3c760ba39b24b919dab`
- 配对报告SHA256：`40a059208cc6abb87d650ab9f8b16da2799d4991d08446fe722d1f33b363b396`
