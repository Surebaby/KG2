# 证据供给与答案目标修复执行记录

日期：2026-09-06。研究者已同意按上一轮依据执行；本记录替代策略复核中的“尚未批准”作为当前执行授权说明。历史复核及失败结果保留。

当前阶段为小规模开发验证，PPO/SFT优化器更新数为0。主线仍是答案奖励、Graph/Text过程奖励和α同时启用。原3000题封版数据、Gold、baseline、评估parser与fresh132确认输入保留；本轮未消费fresh132。

**本轮状态：`STAGED_REPAIRS_VALIDATED_EVIDENCE_V1_NOT_ADOPTED_PPO_NOT_STARTED`。** 奖励v2接线和验证完成，证据v1的完整Reader对照与独立复算完成；当前证据不足以采用该供给启发式替换正式数据。

## 实验分离

1. `EVIDENCE-SUPPLY-CONSUMED-MUSIQUE20-20260906-V1`：只改变数据准备阶段的证据供给。固定此前消费的20道MuSiQue题，原问题与原top3可见材料抽取最多3个新专名，查询为“原问题＋专名”，每路沿用E5@100、BM25@100、RRF60@50和BGE。先保留原排序中前4个不同ID及规范化内容的段落，再轮询补充新段至10。全部20题保留，禁止Gold、官方support、旧生成答案或语义审阅标记进入查询规划。最终Reader段数/输入上限不变，但最多新增3路检索，不能称同上游预算。
2. `EVIDENCE-SUPPLY-V1-READER-CONSUMED20-20260906-V1`：原Strong SFT、原system prompt、K2、每candidate原seed、batch1、BF16、max384。生成全部40条后才读取冻结train Gold。使用原canonical双抽取和max-alias EM/F1，全40为分母；20family成对bootstrap仅描述开发不确定性。词面覆盖只搜索真正展示的每段前1200字符，不能等同完整证据链。
3. `ANSWER-FORMAT-OBJECTIVE-V2`：独立训练目标消融。原数据、提示、模型、α和过程有效性规则均保留，配置新增`answer_format_reward_version: v2`，缺省`legacy`。不与证据替换合并归因。

供给协议：[protocol.json](../outputs/audits/evidence_supply_v1_consumed20_20260906_v1/protocol.json)。Reader预冻结协议：[protocol.json](../outputs/audits/evidence_supply_v1_reader_consumed20_20260906_v1/protocol.json)。供给资产约59.43GB在冻结时完整SHA一次，运行首尾核对device/inode/bytes/mtime及代码SHA；不宣称每次运行独立重hash全部资产。

## 奖励v2的精确定义

记原答案项`O=4*(EM+0.1*F1)`，PPO答案抽取仍取原parser的首行，并沿用冻结Gold aliases。原format-v2合法轨迹严格保留：

`R = O + 0.3*(1-alpha_eff)*TextN + 0.2*alpha_eff*GraphN`。

只有以下单因最低步数短缺获得例外：原要求3步、恰2个完整顺序Step、每个Reasoning至少20字符、每字段恰1、非空Conclusion、唯一非空末尾Final、无原始空/多余Step标记、无完全重复Reasoning、无已识别语法的未知KG/越界passage引用。原validator唯一违规须为最低步数/内容总检查，辅助检查再排除内容损坏。

这些输出仍计为**格式无效**，只获得`R=O-1`；Text/Graph均为0，α不预测，奖励放在真实末尾token。其余无效输出仍为−4。旧合法输出不被这个新辅助检查重新筛选。

过程混合绝对值上界0.3，因此相同正确答案下，合法轨迹最坏4.1，合格两步短缺为3.4；合格短缺的正确答案又高于合法错误答案最高0.7。这是有限奖励排序性质，不保证PPO提高EM，也不解决tokenwise KL、省略过程或语义伪造。显式引用检查仅覆盖规定语法，不是事实验证。

最初“任何唯一Final都保留答案”的宽松提案已在任何真实候选重算或训练前淘汰；裸Final仍−4。没有放宽评估parser或原format-valid口径。

## 已完成结果

真实缓存240条包括旧提示120与此前prompt-v1的120。原format/violations、canonical与PPO答案指标全部复现。206条原合法输出答案项不变；34条无效中只救回旧提示8条，其中4条首行EM正确；其余26条仍−4。

| 答案项＋格式项均值 | legacy目标 | shortfall-only v2 |
|---|---:|---:|
| 原提示120条 | 0.757119 | 1.106008 |
| prompt-v1 120条 | 1.217667 | 1.217667 |

表中不包含Graph/Text、KL或任何训练更新，EM/F1没有变化。新目标仍不能消除此前prompt-v1较高组件均值与较低答案分的现象，不能表述为完全对齐EM。

入口：[缓存组件复算](../outputs/audits/answer_format_objective_v2_cached_20260906_v1/manifest.json)。

**生产复验现已完成。** [生产缓存v2](../outputs/audits/answer_format_objective_v2_production_cached_20260906_v2/manifest.json)覆盖原120条及新提示10条无效输出，共130个真实输出、A/F/T×两版本780次调用。96条原有效输出的token数组及奖励字段与不可变旧runtime精确相等；34条无效输出的204次调用均禁止Text/Graph/α预测且未触发。576次复用已有真实ReaRAG分数，独立token分配最大误差1.89e−7。新提示110条有效输出没有真实Text缓存，明确不纳入、不伪造。审计v1在任何reward调用/Gold join前出现prediction哈希比较对象错误，失败保留；v2只修审计适配，不改变奖励。

## TensorBoard和配置

新增`reward/{分组}/answer_component_mean`、`format_component_mean`、`shortfall_salvage_rate`、`severe_invalid_rate`、`answer_signal_applied_rate`、`canonical_em_mean`、`canonical_f1_mean`，覆盖all/dataset/m_graph及交叉分组。原`outcome`为实际施加的净答案＋格式项；原`em/f1`仍是训练可用答案指标，不能与新增全量canonical指标混称。事件读回测试验证真实TensorBoard文件与分母。

新建A/F/T三份probe12配置，均继承原六维source-credit-v2配置，仅显式切换答案目标及独立输出目录。A保留完整α及过程；F/T是matched消融。原18份配置缺省legacy且未被覆盖。source-credit的独立确认和PPO放行仍false，新增配置不绕过这一门。

实际生产缓存另写A/F/T三个诊断event，各130行、48项scalar读回通过，每run368scalar tags与138histogram tags，明确`optimizer_updates=0`、`cached_reward_not_training=1`。本地查看命令（这些是奖励诊断事件，尚无PPO学习曲线）：

```bash
/home/zjulab/anaconda3/envs/kgpaper/bin/tensorboard \
  --logdir /home/zjulab/kgpaper/outputs/audits/answer_format_objective_v2_production_cached_20260906_v2/tensorboard \
  --port 6007 --host 127.0.0.1
```

350项相关测试全部通过，记录为`outputs/audits/evidence_and_objective_execution_20260906_v1/tests.junit.xml`。新A/F/T配置pre-model校验通过，实际生产gate loader仍拒绝未确认的α门；配置解析本身未调用训练器。

## 检索排序边界核查

检查发现`MemmapSearch.search`用argpartition选top-k后没有显式降序排序，而RRF依赖输入顺序。历史部分权威检索确实使用该路径，但当前NumPy2.2.6多规模toy的k100恰好返回降序，尚不能据缺少显式排序断言原数据排名已错。单独运行真实20原问题排序诊断；若无差异，回到已冻结供给实验；若出现差异，则先做排序单变量对照，保留扩展协议但不混合执行。

**真实诊断结果：20/20分数降序，0逆序、top1均为最高分。** 显式docid tie-break会使10题的完全同分项调换顺序，但20题的分数序列逐值不变。未采用这个新增tie政策，未更改原索引实现，不声称修复了已有排名错误。权威入口：[真实排序检查](../outputs/audits/dense_rank_contract_consumed20_20260906_v1/manifest.json)、[历史路径追溯](../outputs/audits/dense_rank_historical_lineage_20260906_v1/manifest.json)。

供给实验随后按原冻结协议执行完成：60查询、3000个BGE pairs、每题恰6个新文档，总输入tokens35441→35340、最大1904；耗时148.95秒，CUDA峰值allocated/reserved为2.71/4.33GiB。两次Gold-free重构均通过，独立审计入口：[manifest.json](../outputs/audits/evidence_supply_v1_independent_review_20260906_v1/manifest.json)。新供给固定dense→sparse拼接，旧router并发完成顺序在RRF等分时可能不同；因此只称共享检索组件，不能称严格复现旧RRF的tie顺序。该顺序已在新代码冻结，并非事后改动。

## Reader完整结果与采用决策

40条新输出已生成、GPU已释放，独立复算80条新旧raw tokens与EOS、40组身份/seed/model/protocol绑定、全部12项指标和10000次成对family bootstrap一致。主评估脚本与评分协议在供给生成之前冻结；第二个agent的审计是完成后独立复算，读新输出/Gold前冻结代码，但发生在收到完成通知后，不冒充预生成盲评。

| 固定20题/K2 | 原10段证据 | 供给v1 |
|---|---:|---:|
| canonical EM | 4/40（10.00%） | 5/40（12.50%） |
| canonical F1 | 20.5064% | 19.5221% |
| 原format-v2有效 | 30/40（75.00%） | 33/40（82.50%） |
| 格式门控canonical EM | 4/40 | 4/40 |
| 可见材料答案词面覆盖 | 6/20 | 10/20 |
| 平均输出tokens | 227.675 | 236.975 |
| 达输出上限且未EOS | 3/40 | 4/40 |

EM差值+2.5pp，95%成对family区间[−10.0,+17.5]pp；F1差值−0.9843pp，区间[−12.9191,+12.1667]pp。4条新增答对、3条原正确变错，净增1条；不是稳定质量增益证据，也不是证明方法无效。可见答案词面增加不能当作已补齐4条完整证据链。

新生成耗时229.87秒，含收尾绑定检查记录237.62秒，峰值allocated/reserved约15.52/15.91GiB。与供给阶段分开；没有加载ReaRAG或运行优化器。

结果入口：[主评估](../outputs/audits/evidence_supply_v1_reader_consumed20_20260906_v1/assessment/manifest.json)、[独立复算](../outputs/audits/evidence_supply_v1_reader_independent_review_20260906_v1/manifest.json)。**不将证据v1并入正式3000题，不用这个20题结果选择PPO checkpoint。** 已预定检查所有EM变化题和新增词面覆盖题，作为事后诊断，不能据此逐qid修改查询后再称同一实验。

## 后续条件

事后按固定规则审阅所有EM变化题及新增词面覆盖题，不反馈改查询或再采样。根agent额外读取`train_11940`和`train_3018`的全部新10段材料及K2输出，确认两个具体边界：

- `11940`新两条Final都为“Hurricane Maria”，与材料的事件一致，原冻结Gold是“Maria”。按原canonical规则严格计分，不改Gold aliases或parser来追回EM；这条下降不能自动解释为证据变差。
- `3018`新材料明确Megadeth全球销量38 million，两条Final均匹配。k0的四步过程却把总销量归给`Peace Sells`及`Rust in Peace`单张专辑；k1两步直接使用乐队销量，原format-v2判最低步数不足。**答案正确不保证过程正确，形式完整也不是语义证明。** 现v2只缓解两步短缺的−4问题，仍在相同答案下给原合法过程更高奖励；它没有解决这个事实归属错误。尚未计算这对输出的真实ReaRAG分值，不能据此宣布Text scorer无效。

完整有界核查覆盖7题、140段及28个输出，43条引用通过原文及可见性核验，入口：[事后案例报告](../outputs/audits/evidence_supply_v1_posthoc_case_review_20260906_v1/manifest.json)。其中新增词面还包含无关年份命中、仍缺桥接及地域/人物指代不确定的情况。该报告是单agent暂定语义诊断，引用精确不等于分类已获人工确认。

因此下一步优先做事先固定的、证据支持与实体/归属错误的匹配步骤诊断，检验ReaRAG代理分对事实错误是否敏感，并单独审视最低步数规则；不能用凑足三步代替推理质量。再据证据支持情况准备少量、身份隔离的MuSiQue/2Wiki监督可行性验证。上述诊断不是新的人工Gold，不作为自动训练标签。

本轮不默认扩充3000题或重复Hotpot replay；正式SFT起点与提示确定后，才消费fresh132作一次独立过程分/α效用确认，再进入完整PPO-A probe与matched对照。
