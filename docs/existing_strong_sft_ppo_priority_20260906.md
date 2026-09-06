# 现有 Strong SFT 的 PPO 优先执行决定

2026-09-06；Decision ID：`EXISTING-STRONG-SFT-PPO-PRIORITY-20260906-V1`。

研究者希望尽快获得可用于论文分析的真实 PPO 结果，允许首版指标提升有限。新版三域 SFT 从必要主线降为可选后续对照；不把“可能有帮助”当作“已经证明必须重训”。本决定不修改原 Gold、baseline、评价协议、奖励公式、来源信用门或 α 参数。

现有 Strong SFT 为 `checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final`。它确有 Hotpot-only 监督及部分格式/跨域风险，但已有合法可评分输出，当前证据不能证明跳过新版 SFT 就不能进行 PPO。A/F/T 的起点与显式 reference 都继续使用同一冻结 checkpoint。

## 最小结果链

1. 以固定 fresh132 完成过程奖励和 α 的确认：先冻结生成和分析规则，随后真实生成/评分，不能把来源 PASS 或旧开发集重分析当确认。过程排序不含 Gold 答案奖励；不按结果换题、换门或选择较有利的 α 版本。现确认输入无需重建，仍未消费。
2. 完整 A-probe12 检查真实策略、参考、奖励、replay、梯度、checkpoint 与 TensorBoard。A 包括答案、Text/Graph 过程奖励与 learned α；没有把 answer-only 改称主模型。
3. A-smoke600 先得到训练和开发评估结果；随后 matched F-smoke600、T-smoke600。A/F/T 保持数据、checkpoint 起点、rollout/update 预算、答案目标和评价一致。A−F 才直接检验 learned α 相对固定 α 的贡献，A−SFT 不能单独证明门控贡献。
4. checkpoint 使用事先固定的 development150 规则选择；最终再按原 canonical900 协议评估。初版 seed42 是 pilot，最终重要结论须补多 seed 与不确定性，不能把单个偶然 checkpoint 当稳定发现。

即使结果有限，也可以据实分析图信用何时有帮助、何时无效、格式与证据问题如何影响学习。数据支持哪些范围，就把论文结论写到哪些范围；不承诺第一版一定提升，不删除无收益消融或为追分改评价。

## 当前阻塞与计算环境

当前 source-credit-v2 的 `training_clearance` 与 `independent_confirmation_clearance` 都为 false。真实生产 loader 会在模型分配前拒绝未确认 gate；不能手改布尔值放行。fresh132 的输入、离线 mask 和两个确认 wrapper 已就绪，但完整 generation/scoring/analysis 执行协议及流水线仍待完成。三种图类型有 79 个来源 PASS，另外 17 个图输入仍保留并关闭图信用；不含 inference 图类型。

当前代码没有独立限额的 optimizer 工程旁路，仅有诊断读取 gate 能力。若在确认前做零更新检查，必须明确零更新；不能用诊断注释把原样训练入口变成已授权、已确认的训练。

本地 24GB 4090 可分阶段进行 SFT 候选生成和 ReaRAG 评分。完整 PPO 当前会同卡加载 8B policy、8B 显式 reference 和约 9B ReaRAG，仅 BF16 权重约 47GiB，尚未计激活和优化器；旧 15–16GiB 生成峰值不代表完整 PPO。沿用已配置的远端 96GB 卡完成真实 PPO，避免为省事临时删 reference/ReaRAG、改量化或改方法。

## 已暂停的新版 SFT 准备

本轮已停止本地原 16,500 候选检索与自动组装进程，退出码 130 对应研究者改优先级后的主动 SIGINT；60 个图补充检索已正常完成。全部冻结候选、已完成批次、失败尝试、脚本和测试保留。DeepSeek 只做过免费模型列表核验，付费教师调用 0，合格新 SFT 轨迹 0，SFT/PPO 更新 0。

- [停止记录](../outputs/audits/sft_v3_pause_for_existing_sft_ppo_20260906_v1/report.json)
- [暂停前的新 SFT 数据准备](sft_v3_data_preparation_20260906.md)
- [现有确认规则与门的依据](source_credit_v2_confirmation_rules_review_20260906_v1.md)
- [确认输入与来源准备](source_credit_v2_next_stage_20260906.md)

新版 SFT 的 50 美元教师预算提案一并暂缓，不能将此前未提交的默认选项当成付费授权。

已补齐三份 `phase3_ppo_mixed4_answer_format_v2_{a,f,t}_smoke_seed42.yaml`，与现有对应 probe 同用 answer-format-v2，防止进入 smoke 时回到 legacy。8 项配置测试通过：除输出目录、固定 schedule、轨迹预算 12→600 和保存间隔外，两阶段 resolved 配置一致；A/F/T 只改变门控模式和实验输出。记录见[配置一致性审计](../outputs/audits/answer_format_v2_matched_probe_smoke_configuration_20260906_v1/manifest.json)。生产 gate 仍拒绝未确认加载，没有新绕行入口。

当前 PPO 的普通样本最低步数仍为 3；两步 sole-shortfall 沿用答案−1、过程为0。新版 SFT 准备中的2–5步 contract是独立模块，尚未改写正式 PPO 规则。为尽快得到可解释的首轮结果，本决定不再叠加步数、奖励权重或 mini-batch 的同时调整。
