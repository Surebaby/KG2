# 新版三域 SFT 数据准备（2026-09-06，持续执行记录）

> **最新：已按研究者新优先级暂停扩充。** 原 16,500 检索和自动组装进程已主动停止，60图补充检索已完成；所有资产保留，付费 teacher 调用 0。下方“后台运行”和“预算待确认”描述暂停前状态，预算提案同步暂缓。当前主线见[现有 Strong SFT 的 PPO 优先决定](existing_strong_sft_ppo_priority_20260906.md)。

当前状态为 **候选身份与检查标签已冻结，尚无合格训练轨迹**。候选池共 16,500 题：HotpotQA、2Wiki、MuSiQue 各预留 train 5,000、validation 500。目标合格集为 train 6,000、validation 300；候选数量不保证最终达标，当前 accepted=0，尚未训练。

已另行冻结[16,560 个组合候选](../data/silver_data/sft_v3_three_domain_graph_combined_n16560_seed42_20260906_v1_attempt2/manifest.json)：保留原 16,500，追加 60 个无 family 冲突的图补充；原池 169 个同题附图，合计 229 个 source-backed 图输入（train210/validation19）。此组合不含 inference 图类型。Teacher 调度先按冻结 rank 消费已核验图层，再消费普通层；不强制 teacher 引用不相关 KG。正式自动审核 release 至少需要 100 个训练、5 个验证的真实非空图引用目标，来源检查通过本身不计数。

首轮组合因一题原始问题末尾空格触发严格停止；失败与当时代码保留。按既有 question identity 的 strip 规范化确认 60/60 身份后形成 attempt2，原始答案未改变。[192 条真实就绪输入](../outputs/audits/sft_v3_combined_ready_inputs_snapshot_20260906_v1/manifest.json)已冻结，每域64，含3个图题；其余输入仍待检索。

新版保护账本含 12,735 个数据集内 qid、12,678 个全局问题哈希和 11,321 个全局词面 family，覆盖历史评估与开发、原 SFT 留出集、PPO 3,000 题及 α 开发和 fresh132。选样前按 seed42 固定全局 family 划分，每 family 仅选一题；三层保护重叠和 train/validation 重叠均为 0。较宽的 family 排除规则会改变 2Wiki 的模板分布，应在论文中说明。

检索与教师输入仅使用问题和可见证据；原始答案在问题选择落盘后忠实复制到独立 checker 标签，不能进入输入。后续仍须完成每题 10 passages、2–5 步有依据轨迹、独立模型逐步审核、KG 覆盖及独立全量审计。人工抽检尚未进行，是论文可靠性限制和后续建议；自动审核数据不冒充人工过程标签。API 预算尚待确认，不能把候选冻结写成数据已封版。

模型方案为同一 Llama3-8B-Instruct 基座新建 LoRA；旧 Strong SFT 和原 PPO 数据保留，本步骤不启动训练。

后续独立组合版本保留原 16,500 题顺序，附加 60 道 family 隔离图题，共 16,560 候选；229 个图分配已具备来源检查。真实 ready snapshot 的 192 题已完成生产与独立来源双重重放（各 192/192，含 3 个非空 KG），不代表教师轨迹已生成或接纳。检索摘要沿用原 canonical 默认空格 JSON，图记录与 API 沿用各自 compact JSON；首个集成失败版本保留。

- [候选池 manifest](../data/silver_data/sft_v3_three_domain_candidate_pool_n16500_seed42_20260906_v1/manifest.json)
- [保护账本 manifest](../outputs/audits/sft_v3_protected_identity_ledger_20260906_v1/manifest.json)
- 验证：账本与候选冻结共 25 项测试通过；候选 7/7 身份门、8/8 输出文件 SHA 复核通过。这些检查不代表教师轨迹质量通过。

## 教师数据接纳方法

新版 producer 为 `deepseek-v4-flash`，另用 `deepseek-v4-pro` 审查完整推理链，均显式关闭 thinking，避免旧别名与默认模式漂移。Producer 和 reviewer 只得到同一份可见 question、10 passages 和已经核验的 KG；原始 train 答案留在独立 checker 文件，在 producer 原始响应落盘后才用于 canonical max-alias EM 接纳检查。不向 teacher 反馈答案，不替换 Final，不修补错误轨迹后冒充原始响应。

每条接纳目标需依次通过严格 JSON、2–5 个连续步骤、每步唯一字段、唯一且简短的 Final、可见原文引用定位、真实 Llama 模板 assistant（含 EOT）≤384 tokens，以及逐步语义和完整答案链审核。Reviewer 的 uncertain/reject 均拒收；只检查答案正确不够。审查覆盖 Reasoning 与 Conclusion 的所有事实，防止把乐队总销量错误归属到某张专辑等问题。

这是同一提供商的两个不同模型/调用，不能声称它们的错误统计独立，也不能把自动审查意见当作人工 Gold。人工抽检和主论文的过程可靠性标注仍有价值，但本次不另设人工签字才能准备数据的审批环节。KG 来源 PASS 同样只认证绑定的来源/实体/执行一致性，不能替代 teacher 轨迹的语义审核。

所有调用有发送前 intent、返回后原始响应的 append-only WAL，SDK/传输/语义均不自动重试。请求中的 Gold 不由字段清空保证，而由 prompt 构造接口只接收 question/passages/KG 保证。费用按官方峰值 cache-miss 单价为在途请求预留；中断后无法确认的调用不重复发送，仍保留预留金额。峰值费用上界不是提供商账单。

## 已完成的真实输入检验

本地 RTX4090 已完成前 128 题 canonical 检索，H/W/M=43/43/42。每题均为 10 passages，真实 BGE 成功且无 fallback；两个 64 题批次分别用时 143.90、123.98 秒，含加载的运行时间 278.67 秒。完整原始 Wiki18 索引和模型已绑定 SHA，批次通过各自 seal 复验。

空 KG 长度探测下，三域 SFT prompt 平均约 1847/1895/1850 tokens，最大 2137；128/128 均可在 6144 全长内保留完整输入和 384 的输出预算。实际附加 KG 后须再次全量检验。Teacher prompt 的 Llama tokenizer 代理计数约 2256/2304/2259，不能把它当作 DeepSeek 实际计费 tokens。该检查没有调用 teacher、没有产生训练标签。

- [检索批次进度](../outputs/audits/sft_v3_canonical_retrieval_n16500_seed42_20260906_v1/progress_0001.json)
- [128 题真实输入 token 复验](../outputs/audits/sft_v3_retrieval_prefix128_token_preflight_20260906_v1/report.json)
- [独立候选复验](../outputs/audits/sft_v3_candidate_pool_independent_review_20260906_v1/manifest.json)
- [非空 KG 容量复验](../outputs/audits/sft_v3_source_backed_kg_capacity_20260906_v1/manifest.json)

独立复验重选全部 277,839 条 raw train 后，16,500 个候选完全重现，原始行与标签忠实一致。新版 validation 候选中有 23 个问题/25 个 family 在旧 Strong SFT 的 accepted train 出现过；这是允许复用旧 train 的范围，不污染新 SFT 自身 train/validation 隔离，但不能用这份 validation 声称 old/new 的独立公平对照。新旧模型正式比较继续使用已保护的 development150/canonical900 及原评估协议。

## 预算与尚未完成部分

目标 6300 条合格数据的教师 API 估算为 30–60 美元，包含不合格候选和第二模型核查，实际取决于证据完整率、答案匹配率、审查通过率和调用时段。若只按 1 美元≈7 元做预算换算，约 210–420 元；这不是实时汇率或账单。建议总费用上限 50 美元，当前尚未获得该上限的回复，因此实际付费调用仍为 0。官方定价已在 2026-09-06 核验：[DeepSeek Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing/)。

接下来完成非空 KG 版本化组合输入、剩余本地检索、真实 teacher 生成与审核、全量独立 release 审计。合格数量不足时保留不足状态和全部失败，不能重复样本或放宽事前质量门凑齐。当前训练配置仅为[准备规格](../configs/training/phase3_sft_v3_three_domain_from_base_seed42.preparation.yaml)，最终 accepted release 路径为空，不能直接交给旧 silver loader 启动。

本地原 16,500 候选的检索已在首 128 题检查完成后恢复，日志持续追加、逐批输出 seal；此任务不会触发任何 teacher 调用。按已测批次外推全池约 9–10 小时，磁盘/RAM 状态会影响实际速度。查看进度：

```bash
cd /home/zjulab/kgpaper
tail -f outputs/audits/sft_v3_canonical_retrieval_n16500_seed42_20260906_v1.launch.log
```

查看最近已完成的累计题数：

```bash
rg 'BATCH_COMPLETE' outputs/audits/sft_v3_canonical_retrieval_n16500_seed42_20260906_v1.launch.log | tail -5
```
- 严格检索投影另有 5 项测试；[独立复验](../outputs/audits/sft_v3_candidate_pool_independent_review_20260906_v1/manifest.json)自行重选全部原始 train，16,500 候选与原样标签完全复现。新版 validation 候选与旧 Strong SFT train 有 23 个 exact-question、25 个 family 交集：旧训练复用符合当前协议，但新版 validation 仅用于新版 checkpoint 选择，old/new 对照仍使用冻结 development150/canonical900。
- [真实 192 题来源集成复验](../outputs/audits/sft_v3_ready192_independent_source_integration_20260906_r2/manifest.json)核验 45 个文件 SHA 与 26 个已完成全 SHA 冻结的检索资产状态。独立 release audit 另有 12 项 fixture 测试，通过不计作真实 API 或 SFT 数据结果。
