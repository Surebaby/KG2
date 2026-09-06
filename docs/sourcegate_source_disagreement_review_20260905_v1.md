# α 候选的来源分歧复核（2026-09-05）

**结论：已定位到真实的 QID→显示名称投影缺陷，以及一例根实体绑定冲突。当前图满分不能作为来源正确性的保证。正式 α 校准应等待新的来源或奖励信用版本；现有候选可以保留，是否需要重新生成取决于是否修改 prompt。**

Experiment ID：`SOURCEGATE-SOURCE-DISAGREEMENT-REVIEW-20260905-V1`。
来源复核只读冻结输入、原始历史实体缓存、官方别名表、store 和既有候选；未改 Gold、KG、候选、baseline、既有评分代码或训练配置，未使用网络。
随后按研究者修复授权新增了独立的来源校验模块及测试，见第 8 节；它不修改旧 runtime。
审计目录：`outputs/audits/sourcegate_source_disagreement_review_20260905_v1`。

## 1. 37 条满图分但 EM=0 的候选究竟是什么

按原诊断条件 `valid && m_graph && raw_graph≈0.85 && EM=0` 选择，只用于诊断，绝不用于筛选 α 训练样本。
26 题、37 条候选的记录在 `cases.jsonl`。Gold 在这个独立报告中后接入；来源范围审计脚本完全不读取 Gold。

| 类别 | 问题 | 候选 | 本地证据可以支持的判断 |
|---|---:|---:|---|
| 日期表面形式不同 | 17 | 24 | 原有日期区间解析器判为同精度同区间；canonical EM 不变 |
| 同 QID 支持的别名差异 | 2 | 4 | Santiago/Santiago de Chile；University of Maryland/University of Maryland, College Park 均在对应 QID 官方 aliases 中 |
| 地名/国籍词形差异 | 2 | 4 | English/England、Roman/Rome；官方表明确区分 aliases 与 demonyms，当前投票未保留这种类型区别 |
| QID 的显示名称错配 | 3 | 3 | Q881 被显示为 New Zealand；两题中的 Q35 被显示为 American |
| 显示名称错配并破坏同国比较 | 1 | 1 | Q713165 被显示为 Enrique Carreras，Q30 被显示为 Peru，造成相同实体比较为不同 |
| 根实体绑定冲突 | 1 | 1 | 请求 Leopold Joseph（1682–1684），实际解析与执行为 Leopold I |

“官方 aliases 支持”是同一 QID 下的表面形式证据，不构成对人物事实的独立考据，也不改变正式 EM/F1 的定义。

## 2. New Zealand 的来源已经查清

问题：`2wikimultihopqa::2de52eba0bde11eba7f7acde48001122`，询问 Đèo Văn Long 父亲的出生地。

1. 冻结历史原缓存中，Q3023449 的 P22 指向 Q1767814，后者 label 为 Đèo Văn Trị。
2. Q1767814 的历史修订 `1260812065`（`2020-08-19T19:23:10Z`）的 P19 原值是 **Q881**；不是一个 New Zealand 字符串。
3. 官方 `id_aliases.json` 的 Q881 aliases 包含 `Vietnam`、`Viet Nam`，demonyms 包含 `Vietnamese`。
4. `VersionedEvidenceStore` 由训练来源表面频次投票决定 QID 显示名称。v6 store 中 Q881 的 `New Zealand=1`、`Russia=1`，其余 Vietnam/Vietnamese 等为 0；按分数再字典序，选中了 New Zealand。
5. `HistoricalWikidataPropertyRetriever._label()` 优先采用这个 store 返回值，覆盖了使用实体身份的机会；实际候选 prompt 因此写成 `(Đèo Văn Trị, place of birth, New Zealand)`。

因此，此例的直接原因是**尾实体显示名称投影**。不需要假定父实体链接错误，也不需要查 Gold 来识别这一缺陷。当前本批历史缓存没有 Q881 自己的实体修订，不能声称已经从该批历史修订取到了 Q881 canonical label；官方 QID 别名表提供了可独立核查的名称证据。

## 3. 仅把国家显示名称改好仍不足以修好执行合同

两道同国比较题 `c94da00008b211ebbd85ac1f6bf848b6`、`ffebd38108f211ebbdaaac1f6bf848b6` 的原始尾实体分别为 Q30、Q35。
Q35 的 store 票数 American/Danish/Denmark 各为 5，字典序选 American；Q30 本来就显示 American。
原始身份不同，表面被折叠为相同，现有 `singleton_equality` 因而推导 yes。

`4ba6da5208c811ebbd91ac1f6bf848b6` 则相反：

- A Prince for Christmas 的历史 P57 原值 Q713165；该电影原历史描述明确写 `2015 film by Fred Olen Ray`，官方 Q713165 aliases 也包含 Fred Olen Ray。
- Q713165 的 store 票数 Enrique Carreras/Fred Olen Ray 各为 1，选中了前者。
- store 的 `Q713165::P27` 边保留 `tail_qid=Q30`，却写 `tail_value=Peru`；另一个导演的 citizenship 尾实体同样为 Q30，显示 American。
- 相同身份被投影成不同表面，现有图执行因此推导 no。

这三题说明：修复应保留并使用 **typed QID operands**，而不应把最终字符串相等当成实体相等。当前 v2.2/v2.3 的执行 trace 没有完整保留这一身份映射。还需要继续对多值、qualifier/rank 与歧义执行保持 abstain，不能简单新增一次字符串替换。

## 4. 根实体问题是另一条来源链

`fb22d59c0bae11ebab90acde48001122`：root-resolution 文件明确记录，原请求补全为 `Archduke Leopold Joseph of Austria (1682–1684)`，但 exact-title 结果为 **Q150494、Leopold I, Holy Roman Emperor**。历史 Q150494 原修订也标为 Leopold I，且一致地执行 `father→Ferdinand III→mother→Maria Anna of Bavaria`。

因此，图执行完整只说明它完整执行了已绑定根实体的关系。它没有证明该根实体就是题中那个人。现存 title cache 只保留 label/QID，没有保留 redirect/section 等更细来源；本次无法在本地裁定正确替代实体，不硬编码修例外。日期、人物名、标题限定词必须纳入新版本的身份绑定检查，并保留缺证据时的 abstain。

## 5. 对全部 830 输入的 Gold-free 范围审计

`audit_source_labels.py` 不读取 Gold，只重放 store-first/historical-fallback，再把实际显示名称与同 QID 的官方 aliases、demonyms、历史英文 label/aliases 做规范化精确成员检查。

| 统计 | 数量 |
|---|---:|
| 有图问题 | 800/830 |
| trace 中表面 matches | 2141 |
| 可唯一追溯为一条 typed edge 的 matches | 2121 |
| 不同尾 QID 折叠为同一表面，不能唯一还原 | 20 条/20 题 |
| 可读 head/tail 未获得同 QID 本地名称支持 | 82 条/62 题 |
| 其中来自 store / historical fallback | 53 / 29 条 |
| 直接显示不透明 QID 的尾值 | 270 条/235 题 |
| 单本批历史缓存缺少尾 QID 英文 label | 1053 次/590 题 |
| 连同 QID 官方名称也没有的尾值 | 32 次/27 题 |

**82/62 是待审查范围，不是确认错误率。** 未获精确别名支持可能包含未收录别名，根实体同样可能有标点/标题格式差异。不透明 QID 也不等于身份错误，故没有把它们算入 82 条可读名称风险。

20 条非唯一还原主要是 Italy/Kingdom of Italy 被显示为同一个 Italian，或 United Kingdom/历史国家实体被显示为同一个 British。是否应在任务语义上归并必须有事先固定的规则；当前表面去重不保留这一决策所需的身份，不应冒称严格单值。

额外只读盘点了本地 25 个同 cutoff 的历史实体缓存：1105 个唯一尾 QID 中，仅 701 个能在任一缓存中找到英文 label；Q881/Q35/Q713165/Q220/Q503415/Q2887 仍没有。这些其他缓存还包含开发/保护 cohort 的来源，**不能直接合并成训练来源**。详细盘点在 `historical_label_inventory.json`。所以“只切到本地 canonical historical label，就能无网完整重建”目前并不成立。

## 6. 最小、可泛化的修复方案（本次未实施）

1. **独立的新 source-policy 版本，显式 opt-in。** 保留 legacy store/retriever 默认行为与原 baseline。给新 PPO 来源构建增加 QID-bound label 选择与来源记录；禁止让未验证的训练频次别名覆盖 QID 对应名称。
2. **分开处理三件事。** 根实体绑定验证、边的 `(head_qid,pid,tail_qid/literal)` 身份、用于阅读的显示名称分别建合同。store 边的 head/tail 显示都要检查，仅修改 historical `_label` 无法覆盖已存在的 53 条 store 边风险。
3. **优先有来源的实体 label。** 使用当前已批准来源中精确 QID 的缓存 label；缺失时可把官方 aliases 作为已版本化显示候选，但不得称它为 canonical label，且要区分 aliases 与 demonyms。不自动选频次最高的未知名称；缺乏可靠身份/显示时保留 QID 或 fail closed。若需要补取实体 labels，应针对全部所需 QID 统一物化、绑定 cutoff 和来源，不能按 Gold/EM 选择实体。
4. **typed operand 执行与表面展示解耦。** 保留尾 QID、原修订、PID、rank/qualifier、原始值，禁止在去重时把不同 QID 因同名合为一个单值。比较逻辑在受支持的确定性条件下使用实体身份；有歧义时 abstain。
5. **建立新的 prompt/source 与生成记录。** 显示或根绑定变化会改变 KG block、prompt/input hash；受影响题的旧候选不能改绑为在新 prompt 上生成。可保留未变题并按显式迁移合同复用，变化题重新生成；若没有实现严格逐项迁移合同，则整批重生成最清楚。现有1660条保留，不覆写，不删错例。
6. **α 和论文结论。** 现有 ReaRAG 可以完成旧来源的诊断对照。重新评分不能被称为“修复了 prompt 中的源 KG”；若保留旧 prompt，可按第 9 节路线 B 建立新的奖励信用拒绝策略，再做分 family 校准与过程效用检查。现阶段可以写“识别并审计来源投影错误”，不能写“严格来源门已保证事实可靠”或“learned α 已能修复这种错误”。

本次统计没有改变数据配额或评估规则，也没有产生新来源数据。若新来源 gate 会减少 strict 800，必须如实记录新有效图分布与版本，而不是继续引用旧 800 strict 的含义。

## 7. 复现与完整性

审计机器可用 Python：`/home/zjulab/anaconda3/envs/kgpaper/bin/python`，`PYTHONPATH=/home/zjulab/kgpaper`。

- `audit_source_labels.py`：Gold-free 全银行来源重放。原始结果在 `source_label_report.json`、`edge_source_audit.jsonl`、`root_source_audit.jsonl` 和 `unmatched_replay.json`。
- `assemble_diagnostic.py`：独立读取 frozen train labels，接入既有候选诊断，生成 26 题的 `cases.jsonl` 与 `report.json`。程序断言 37 条/26 题与 24 日期等价条目，并确认 20 个非唯一 match 都有至少两个 typed 来源。
- 输入、历史 cache、官方 aliases、store、根解析、执行代码与报告文件 SHA256 记录在各报告 bindings 中。原有科研文件未被写入。
- 首次标签审计尝试遇到历史实体 `aliases=[]` 的序列化形态，随后以空集合兼容后完成。该调整仅在审计副本，未更改训练/retriever 实现。

所有带 Gold 的本目录产物均为诊断，`calibration_input_eligible=false`，禁止作为 α ratio-target 或样本筛选输入。

## 8. 已实施：独立的 fail-closed 来源校验

新增 `kgproweight/reward/source_integrity_v1.py::validate_source_integrity_v1(record, evidence)`，输入原 record 与 hash-bound QID 名称/typed-edge 证据，返回 `PASS / FAIL / UNVERIFIED`、逐项理由、来源绑定及严格布尔 clearance。
模块无文件和网络 I/O，不读取 Gold，不修改原记录，不改变 legacy gate。调用者必须验证证据文件 SHA；模块本身只验证绑定格式与证据内容的一致性。

- 原始 anchor、entity.surface 与 resolved_surface 分别接受同 QID 的名称支持检查；不能因解析后名称合法就忽略原请求指向另一人的风险。
- typed edge 从原 store/cache 离线重放；对 edge head/tail 名称使用官方/历史 QID 证据，绝不把 store 的投票 label 当权威。
- 不同尾 QID 在同一条或不同 hops 中折叠成相同显示名称时返回 FAIL；typed head 与检索输入 QID 不同也 FAIL。
- 缺少证据/来源绑定、表面未获支持、根身份无法确认等返回 UNVERIFIED，不声称该事实必然错误。
- 检查可见 KG 与 execution matches 的集合一致性，只允许每个字段首尾空白 trim；没有加入别名或日期等价来改变可见三元组。
- PASS 仅表示本合同的身份/显示/投影检查通过，仍需原 legacy hard gate 和全部来源审计；不保证事实真值、实体解歧完整性、关系语义或自然语言推理正确。

全 830 输入的最终结果在 `integrity_clearance_v2/report.json`（Experiment ID `SOURCEGATE-SOURCE-INTEGRITY-CLEARANCE-20260905-V2`）：

| 原始路由 | PASS | UNVERIFIED | FAIL |
|---|---:|---:|---:|
| 原有图 800 | 671 | 100 | 29 |
| 原无图 30 | 0 | 30 | 0 |
| 合计 830 | 671 | 130 | 29 |

29 题 FAIL 均存在不同尾 QID 的显示折叠，其中 20 题在同一表面 match 已无法唯一还原。其余无同 QID 名称支持与证据缺失统计有重叠，见逐题 `question_checks.jsonl`，不能把它们解释成互斥的语义错误类别。
`full_bank_clearance=false`。本次没有把 671 个 PASS 写成新的源数据门，也没有重写原 800 的身份或 Graph 资格。

首版 `integrity_clearance/` 为 670 PASS / 100 UNVERIFIED / 30 FAIL，其额外 FAIL 是字段首空格差异；它和当时源码快照完整保留。采用已存在的显示 trim 合同后另出 v2，消除该非语义误报。v2 首次 metadata 误沿用 V1 ID，已分配独立 V2 ID；更正前 metadata 保留，逐题判定与统计未变。

独立新增 `tests/test_source_integrity_v1.py` 的 36 项合成 CPU 回归全部通过，覆盖根请求/解析错人、typed identity 折叠、名称与来源缺失、未知项拒绝、显示图一致性与 trim、输入不变及禁止读取 Gold。全量身份、计数与原始来源 SHA 也已复核，最终记录在审计根目录 `verification.json`。

## 9. 两条后续路线必须分开命名

**路线 A：修复模型输入。** 新版本重建可信根实体、QID 显示名称和 typed execution，产生新的 KG/prompt hash。变化题必须重新生成候选，不能把旧候选改绑为在新 prompt 上采样；未变题若复用，需要显式迁移/逐项 hash 合同。这条路线能处理输入图误导，但来源物化和生成成本更高。

**路线 B：保留当前输入，拒绝不确定图的奖励信用。** 用新发布的 Gold-free 来源校验为原 830 问题生成冻结 source-credit mask；仅 PASS 图允许 Graph 信用，其余令 `α_eff=0`，保持原 prompt、Gold、passages 和既有候选。该路线符合原方法“来源不可信时 fail closed”的目标，**无需重新生成未改变 prompt 的候选**，但它没有修复 prompt 中错误的图。Text scorer 仍只能读原有 passages。

路线 B 是更小的可执行后继实验，不应声称 α 自动学会了修正实体。它必须另立 reward/source-credit 版本，把新 mask、评分、family split、ratio target、归一化与 gate 权重共同绑定，并与 PPO runtime 使用同一 mask。不能把当前全银行 clearance=false 直接改为 true，也不能复用按旧 mask 拟合的 gate。
原有“800 strict”仍是上游封版供给定义；新的 671 PASS 只是此项检查下可授予图信用的上界，还会受格式/目标 abstain 等约束，不能称为“671 道事实完全正确”。

当前仅实施检查与报告，路线 A/B 的新数据或 mask 版本均未由本子任务发布。正式放行仍需主流程完成对应版本、校准和独立效用检查。
