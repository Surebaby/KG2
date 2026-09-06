# AGENTS.md — Research Project Rules

本文件是本项目 AI Agent 的全局行为规范。

在执行重要科研任务前，应阅读：
- `RESEARCH_WORKFLOW.md`
- 与当前任务相关的实验配置、日志和结果

---


## 1. 文献

- 不得编造论文、作者、DOI、年份或实验结果。
- AI 推荐的文献必须经过真实来源核验后才能正式引用。
- 优先使用原始论文和官方来源。

## 2. 数据

- 原始数据不得覆盖。
- 数据清洗、过滤和转换必须生成新的版本。
- 数据处理必须尽可能保留来源和处理记录。
- Gold labels、测试集和人工标注不得擅自修改。

## 3. 实验

- 每个正式实验必须具有唯一 Experiment ID。
- 实验应记录代码版本、配置、数据版本、模型版本、seed 和 evaluation protocol。
- 默认一次实验只改变一个主要研究变量。
- 多变量同时修改必须明确标记为组合实验。
- 失败、无效和被淘汰的实验不得删除。

## 4. Baseline 与 Evaluation

- Baseline 默认只读，不得覆盖。
- Evaluation protocol 必须版本化。
- 不得为了获得更好结果而事后修改评估规则。
- 不同 evaluation protocol、数据版本或实验设置的结果不得直接当作同一条件比较。

## 5. ML / LLM 实验

- Reward、loss、数据筛选、模型结构等核心科研变量的修改必须可追踪。
- Checkpoint 必须能够追溯到训练配置和代码版本。
- 重要结论不得仅依赖单个 seed 或单个偶然 checkpoint。
- 不得通过修改代码、数据或评估逻辑人为提高指标。

## 6. Code

- 优先最小修改，不进行无关重构。
- 修改代码后运行相关测试。
- 影响实验结果的代码修改必须产生新的实验记录。
- 尽可能使用 Git commit 绑定实验结果。

## 7. AI 权限

AI 可以直接：
- 阅读代码、数据统计、日志和实验结果
- 分析已有结果
- 运行测试
- 进行小规模验证
- 修改普通代码
- 生成图表和分析报告

涉及以下操作必须先获得人类确认：
- 删除或覆盖科研数据
- 修改 Gold labels
- 修改 Baseline
- 修改 Evaluation protocol
- 修改核心 Reward / Loss
- 启动大规模训练
- 删除 Checkpoint 或实验记录
- 发布或提交论文

## 8. 科学结论

所有重要论文结论应能够追溯到：

`Claim → Evidence → Experiment → Evaluation → Data/Model`


## 9. 不确定时停止

如果发现：
- 数据来源不明
- 实验记录冲突
- 结果与日志不一致
- Baseline 不明确
- Evaluation protocol 不明确
- 需要删除/覆盖科研资产
- 无法判断某个科学结论是否成立

应停止操作并请求研究者确认，而不是自行猜测。