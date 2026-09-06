# KG-ProWeight 当前 TODO

> 活动版：2026-09-03
> 历史完整版本：`archive/project_plans/20260902/todo.full.md`
> `[ ]` 未开始，`[~]` 进行中或部分完成，`[x]` 已完成/已冻结。

## 1. 启动前批准

- [x] 批准并冻结版本化 QPEG evaluation protocol；legacy protocol/index/result 保持只读；
- [x] 批准开始SAEG continued-SFT的数据/配置准备；实际冻结为152 updates，但仍受development utility门约束；
- [ ] 批准同预算PPO-O/PPO-K各1,200 trajectories；
- [ ] 批准PPO-K使用归一化process-v2.1、权重`.2`；PPO-O process权重为0。

当前活动主线更正（2026-09-03）：停止继续迭代SAEG/Passage-QPEG数据分支；不重训SFT，正式主模型
改为HotpotQA/2Wiki/MuSiQue均衡混合PPO-O/PPO-K。2Wiki hard-contrastive配对只作机制消融，不能冒充
跨数据集通用训练。这里reward-only仅指新两臂互相之间，不能与历史hybrid PPO作单变量归因。

- [x] 保留SAEG/P-QPEG失败实验并冻结为`FAIL_STOP_DATA_GATES`，不删除、不反转；
- [x] 核对2Wiki hard cohort、共同1,200-trajectory schedule、两份v2 lock与CPU preflight均已通过；
- [x] 核实当前strong-SFT和历史hybrid PPO训练都只有HotpotQA；三数据集指标此前属于跨集泛化；
- [x] 核实可直接复用的family-safe身份：Hotpot600、MuSiQue599、2Wiki hard ProofKG208；
- [ ] 冻结三数据集均衡mixed-smoke身份与100/100/100、K=4共同schedule；
- [ ] 新建派生mixed silver/question-KG records；禁止失败QPEG/P边进入训练；
- [ ] 实现默认关闭的统一outcome + eligible-only process路径并补回归测试；
- [ ] 研究者明确批准新reward后，远程顺序启动mixed PPO-O/PPO-K各1,200 trajectories；
- [ ] 下载两臂checkpoint/history/manifest/TensorBoard event并做训练健康审计；
- [ ] 标准legacy三数据集n300评估strong-SFT/mixed PPO-O/mixed PPO-K；另做2Wiki ProofKG机制表；
- [ ] 仅在K相对O方向为正且两臂训练健康时，成对冻结中程预算；禁止只延长有利臂；
- [ ] 论文主表将三数据集legacy同资源结果与2Wiki额外Wikidata资源结果分栏报告。

## 2. Day 1：冻结与实现

- [x] 在任何图构建前冻结HotpotQA/2Wiki/MuSiQue各pilot50、confirmation100、final300；
- [x] qid与question-family隔离，排除已消费评估题；
- [x] 固化question/passages/order/hash、seed42、greedy和canonical scorer；
- [x] 实现QPEG-v1 schema、builder和≤12边确定性排序；
- [x] 测试identity/hash join=1.0、passage provenance=1.0、gold_access=false；
- [x] 测试空图显式no-graph，不静默混入legacy/Wikidata边；
- [x] 为协议、代码和输入生成manifest与SHA256；
- [x] 审计外部checkpoint baseline兼容性：三数据集n=300 qid/question/Wiki18/解码通过，
  所有既有逐题EM/F1与canonical scorer重算完全一致；
- [x] 物化三数据集共1,350题QPEG；结构门通过，但语义utility仍为UNKNOWN。

## 3. Day 2：pilot50×3

- [x] 构建150题pilot QPEG并随n=1,350总物化生成结构覆盖报告；
- [x] 同qid/同passages运行strong SFT no-QPEG/QPEG A/B；
- [x] 报每数据集EM/F1、gained/lost/tied、parse/valid和covered分片；
- [x] 已使用唯一一次通用修正：删除自环与verbatim伪三元组；未做逐题补丁；
- [x] 修正版macro ΔEM=−0.67pp且2Wiki净损3/50，触发最终停止门；
- [x] 状态冻结为`FAIL_STOP_FINAL_NO_CONFIRMATION`，不再修改QPEG-v1.1。

## 4. Day 3：confirmation与4.8k数据

> **BLOCKED BY FROZEN PILOT GATE**：QPEG-v1.1 未通过 Day 2；以下任务不得在当前协议下执行。

- [ ] 在不改协议的条件下运行unseen confirmation100×3；
- [ ] 至少2/3数据集供给效应正向才进入训练数据构建；
- [ ] 每数据集生成目标800条QPEG-grounded正确轨迹；
- [ ] 每数据集抽取800条既有accepted silver replay；
- [ ] 新轨迹只接受EM=1、格式合法、引用全部属于可见QPEG；
- [ ] 每数据集合格新轨迹至少600，否则记录FAIL且不得复制凑数；
- [ ] 冻结派生数据、来源、过滤原因、family split、manifest和hash。

上述清单是已关闭的 QPEG-v1.1 路径，不再作为活动待办。

## 4A. QPEG-v2 独立选择器路线（已完成并停止）

- [x] 用三数据集各1,000条 train-only 样本审计 selector 监督量；
- [x] 在任何拟合前冻结 family-disjoint train/dev/holdout、特征、模型和门；
- [x] train-only holdout 通过：AUC=.8761、edge precision=.7006、qid route=.9307；
- [x] 物化 QPEG-v2 selector 输出，fail-closed 且不回退 legacy/Wikidata；
- [x] 打开一次 family-disjoint confirmation100×3；
- [x] confirmation macro ΔEM=−1.67pp，MuSiQue净损3题，判`FAIL_STOP_FINAL`；
- [x] append-only 更正 evaluator 遗留的“confirmation remains unopened”错误元数据；
- [x] 冻结表示损失审计：passage答案可见64/59/34%，selected sentence 25/25/13%，
  injected tail仅11/6/3%；
- [x] 不在已消费 confirmation 上调阈值或修改 v2，不构建4.8k，不启动SFT/PPO。

## 4B. QPEG-v3 证据句图（已完成并停止）

- [x] 冻结 train-only sentence selector 的 family split、特征、模型、阈值和 holdout 门；
- [x] 构建全句候选，不再以 regex semantic tail 作为监督载体；
- [x] 输出 typed edge=`(passage_title, evidence_sentence, full_source_sentence)`并保留句级hash；
- [x] 原报告保留：因未应用runtime top-4而错误记precision=.509/FAIL；
- [x] append-only纠错：不重训/不改阈值，按冻结runtime top-4重算precision=.566并通过全部门；
- [x] final900 answer-free图物化：nonempty=.830/.877/1.000，max edges=4，provenance全通过；
- [x] 核验 final300×3 的历史 no-KG prompt 可复用：SHA256精确一致900/900；
- [x] 实现final输入冻结工具：强制approval ID、answer-free资产先验门、Gold后置join、只写新目录；
- [x] 实现final评估工具：A臂逐题prompt重建+重评分后复用，只生成B臂900条；
- [x] final协议/复用/决策门相关测试通过（当前相关集合24 passed）；
- [x] final300×3 A/B evaluation protocol获批并冻结，approval ID已写入协议；
- [x] A复用900条精确prompt历史生成，B新生成900条；共1800条且唯一键完整；
- [x] 报告三数据集EM/F1、paired CI、McNemar、gained/lost/tied及图覆盖分层；
- [x] 独立复核19/19冻结hash、900/900非图字段、空图生成一致；相关测试12 passed；
- [x] final结果：Hotpot ΔEM=−2.0pp、2Wiki=+1.67pp、MuSiQue=−3.33pp；
  macro ΔEM=−1.22pp、ΔF1=−2.01pp；
- [x] 四项冻结晋级门未通过，状态=`FAIL_STOP_FINAL_VERIFIED`；
- [x] 不在final集继续调selector，不构建4.8k数据，不启动本路线continued-SFT/PPO。

## 5. Day 4：continued-SFT（QPEG-v3 路线由 final 门阻断）

> **BLOCKED BY FINAL SUPPLY GATE**：以下原计划任务不再执行；保留作为未启动记录。

- [ ] 从`checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final`启动；
- [ ] 使用lr=2e-6、effective batch=32、最多120 updates，保存step40/80/120；
- [ ] 本地并行评估legacy replay n=200和三数据集QPEG dev100；
- [ ] 选择最早满足parse≥.995、replay EM≥.780、hidden EM≥.545的checkpoint；
- [ ] QPEG dev macro EM/F1不得低于strong SFT，单数据集EM退化≤1pp；
- [ ] 无checkpoint过门则SFT分支记FAIL，回到strong SFT，不追加训练步数。

## 5A. 当前活动任务：QPEG-v4 train-only 图格式适配

- [x] 获得研究者批准，登记独立Experiment ID；不重开已消费QPEG final900；
- [x] 冻结train 600×3、development 50×3、confirmation 100×3；排除旧评估qid/family；
- [x] 物化2400条训练数据：1800 graph + 600 balanced no-graph replay；
- [x] 验证2400/2400 parser合法、引用精确、轨迹3–5步、Teacher API调用为0；
- [x] 冻结全新450题canonical retrieval；每题10 passages、identity唯一、gold_access=false；
- [x] 物化answer-free QPEG图：nonempty Hotpot=.847、2Wiki=.833、MuSiQue=1.0；
- [x] 固化SFT配置：strong-SFT起点、lr=2e-6、effective batch32、75 updates、step25/50/75；
- [x] CPU preflight全部通过；远程同步/启动脚本通过bash语法与本地文件检查；
- [x] 远程96GB服务器已开启；GPU/磁盘/基础模型/起点checkpoint检查通过；
- [x] 两次pytest收集期同步依赖失败均保留日志且训练步为0；补齐builder/QPEG导入链；
- [x] 第三次远端38项测试通过并完成75 updates；manifest=`COMPLETE`，四个adapter均完整；
- [x] 训练健康性：loss `.3766→.3005→.2465`，grad norm有限，峰值allocated/reserved=`39.25/43.42GB`；
- [x] step25/50/75、final、manifest、sft_loss、主日志及全部失败/launcher日志已下载并核验；
- [x] 冻结development50×3配对输入；150题A/B除图块外完全一致，confirmation未打开；
- [x] strong-SFT A/B完成：Hotpot EM `.36/.36`、2Wiki `.30/.30`、MuSiQue `.10/.08`；
- [x] checkpoint-25 C/D完成但FAIL：macro interaction EM/F1=`−.0267/−.0321`，0/3数据集为正；
- [x] checkpoint-50已评估且FAIL；confirmation未打开；
- [x] checkpoint-75恢复评估完成：macro interaction EM/F1=`+.01333/+.00643`，但仅1/3数据集为正且parse/no-graph门失败；
- [x] 三checkpoint选择=`FAIL_STOP_DEVELOPMENT`，selected=null；confirmation未打开，本路线停止且不追加步数。

## 5B. 提案任务：SAEG-ProWeight 双源证据图

> 详细方案：`docs/source_adaptive_evidence_graph_plan.md`。当前仅方向获批，正式protocol、核心
> reward/loss和新大训练仍需逐阶段确认。

- [x] 定义P=Passage-QPEG、W=Wikidata-ProofKG、N=no-graph三分支研究框架；
- [x] 写明同资源主表与额外Wikidata增强表必须分开；
- [x] 写明SFT四臂interaction与PPO-K−PPO-O唯一变量归因；
- [x] 完成QPEG-v4 checkpoint-25/50/75；三者均FAIL，作为旧P-only伪三元组schema的边界；
- [x] 运行CPU-only QPEG/ProofKG qid、family、context overlap审计；v1跨namespace family字段已append-only纠错，v2有效；
- [x] 量化候选池：P-only=1800、W-only=1231、P+W=1231、N=600，共4862 variants；
- [x] 验证1299/1299 ProofKG题可重建P分支且passage标题/正文集合完全一致；
- [x] 排除当前QPEG-v4 development/confirmation family后保留1231个安全fused候选；
- [x] 研究者指示继续后冻结source/dataset-balanced采样权重；图资产与4860条family-disjoint SFT targets均已物化；
- [x] 明确保留canonical语义检索；旧alpha checkpoint不直用，alpha-v2需重新校准与消融；
- [x] 实现`saeg-question-record-v1`/edge schema、P/W provenance、graph/passages hash与确定性融合；
- [x] 物化4862条train-only图资产（非SFT targets）：schema/hash/identity=1.0，P边7551、W边6434；
- [x] 冻结两级采样权重：dataset均衡，整体P/W/fused/N=`.6667/.10/.1333/.10`；训练长度未冻结；
- [x] P/W answer-free对齐审计：3041/3082 hops一对一对齐，1191/1231 fused题全部hop对齐；
- [x] 按1191题P+W联合、40题W-only fail-closed策略物化target；family过滤后联合target=1190；
- [x] 纠正citation contract：W只用标准三元组；P用`[P<n>]`+`Passage Used`，不进入`Knowledge Used`；
- [x] 新增SAEG prompt/parser与silver sidecar，legacy prompt/parser保持不变；
- [x] 物化train master4862；排除1个跨数据集held-out family对应的2个variant，release-v2=4860；
- [x] 冻结/物化evaluation：development150、sealed confirmation300、canonical reporting900；Gold物理隔离；
- [x] 完成release总审计：qid/family overlap=0、Gold泄漏=0、伪KG三元组=0、citation错误=0；
- [x] token门通过：train max3287、eval max2743，全部<4096；
- [x] fresh 2Wiki150 ProofKG闭包保留负结果：nonempty=.760/complete=.600，fail-closed不注入；
- [x] 冻结实际epoch size=4860与精确分层采样日程；CPU preflight 31/31和真实tokenizer/data-loading smoke均通过；
- [x] 固化continued-SFT配置：strong-SFT起点、lr=2e-6、effective batch32、1 epoch=152 updates、step38/76/114/final；
- [x] 实现eval runner的N/P/W/F arm转发与confirmation sealed guard；v1空P题误删已在首题前中止，v2严格配对协议已冻结且13项测试通过；
- [ ] 完成单题device/latency探针后运行v2 development N/P/W/F零训练utility；通过后才允许continued-SFT；
- [ ] SFT interaction confirmation通过且reward rankability通过后才允许PPO-K。

本阶段产物：

- `data/silver_data/saeg_v1_sft_balanced_epoch4860_seed42_v1/`；
- `configs/training/phase3_sft_saeg_v1_balanced_epoch4860_seed42.yaml`；
- `outputs/audits/saeg_v1_sft_balanced_epoch4860_preflight_v1/`；
- `scripts/prepare/freeze_saeg_v1_sft_epoch.py`、`scripts/prepare/preflight_saeg_v1_sft.py`；
- `tests/test_freeze_saeg_v1_sft_epoch.py`。

下一动作：先完成development的N/P/W/F零训练utility；不得因SFT数据已经可加载就跳过供给门。

当前运行说明（2026-09-03）：

- v1 utility没有产生prediction，append-only标记为`ABORTED_BEFORE_FIRST_GENERATION`；
- v2 population为A/B/D各150，C=0 N/E；P非空136，P空14精确回退A；
- v2首次尝试在模型加载后首题超过90秒无返回，已停止；先做单题设备/时延探针，不把运行异常当科学结果；
- 当前utility/SFT不加载ReaRAG是正确行为；PPO-O也不加载，未来PPO-K的`R_text`分支必须显式加载
  `models/rearag-9b`并fail-hard；旧alpha gate不得直接冒充SAEG alpha-v2。
- [x] SAEG-SFT TensorBoard显式接入`/root/tf-logs/SAEG-V1-SFT-BALANCED-EPOCH4860-SEED42`；SFT只记录loss/lr/grad-norm，无KL；
- [x] 新增远程SFT硬门启动器；utility/preflight未PASS、起点缺失或输出目录已存在均拒绝启动；
- [x] 配置更新后preflight-v2 33/33通过，状态仍为`PASS_NOT_TRAINED`；
- [x] 完成PPO pool capacity-only审计：拟议5400 unique prompts（三数据集各1800）×K4=21600 trajectories/arm；
- [x] 排除全部SAEG eval与历史reward审计family后容量仍充足；现有3024 unique qid，需新增Hotpot/MuSiQue各1200；
- [ ] utility通过后再冻结精确5400 qid/source route与fresh reward dev/confirmation；capacity PASS不等于数据已物化或PPO已批准；
- [x] 已实现5400题身份冻结器并通过3项确定性测试；入口硬依赖utility PASS，当前未运行、未创建正式pool；
- [x] development utility v2完成：A/B/D各150、fallback14 exact；D−A macro EM/F1=`-.0667/-.0663`，状态`FAIL_STOP_BEFORE_SFT`；
- [x] 负效应分解：P非空136题gained/lost=`1/11`，citation使用率`.7533`，定位为自动P不完整与train Gold-P完整正例的质量错配；
- [ ] 当前SAEG continued-SFT、PPO pool冻结与PPO均保持BLOCKED；不得绕过失败门；
- [x] 研究者已批准`P hard-negative alignment v2`；冻结协议、同评估retrieval+selector路径、四类质量定义与development门；
- [x] 复核全SAEG评估集合并append-only排除19个重叠family，有效train cohort=1781，qid/family overlap=0；
- [x] 实现complete/partial/misleading/empty分型与选择性citation candidate builder，相关12项测试通过；
- [x] 本地1800题answer-free canonical retrieval完成：600×3、全部10 passages、Gold字段0；
- [x] 物化1781候选并判门：exact complete三集均0，状态`FAIL_STOP_DATA_GATES`，未采样/未训练；
- [x] 完成不反转原门的near-exact诊断：complete=Hotpot26/2Wiki26/MuSiQue2，逐边precision=.181/.237/.078；
- [ ] 等研究者批准`paired hard-negative curriculum v2.1`（Gold-complete正例 + same-qid automatic partial/misleading硬负例 + no-P replay）；
- [ ] v2.1若批准，先冻结配对身份/配额/开发门和小规模SFT配置，再单独请求启动训练；PPO与confirmation继续BLOCKED。

## 6. Day 5：PPO启动门（QPEG-v3 路线由 final 门阻断）

- [ ] 仅用train split构建QPEG nonempty、完整provenance的hard cohort；
- [ ] 冻结300 prompts×K4的两臂共同schedule；
- [ ] 零更新审计valid≥.90；
- [ ] oracle@4−greedy≥8pp；
- [ ] process pairwise≥.65；
- [ ] process-top1≥greedy；
- [ ] 任一失败则PPO-K保持STOP，不在confirmation/test上调公式。

## 7. Day 6：PPO-O / PPO-K（QPEG-v3 路线由 final 门阻断）

- [ ] 固化两个唯一Experiment ID和两份配置锁；
- [ ] 两臂保持起点、qid、seed、KL、replay、generation和更新预算一致；
- [ ] PPO-O使用`4*EM + 0.4*F1`；
- [ ] PPO-K仅增加process-v2.1权重`.2`；
- [ ] 96GB远端顺序训练，每臂1,200 trajectories；
- [ ] 下载checkpoint/history/manifest并检查valid、KL、critic、截断和replay。

## 8. Day 7：最终评估

- [ ] 完成A=strong-SFT/noQPEG；
- [ ] 完成B=strong-SFT/QPEG；
- [ ] 完成C=QPEG-SFT/QPEG；
- [ ] 完成D=QPEG-SFT/noQPEG；
- [ ] 完成E=PPO-O/QPEG；
- [ ] 完成F=PPO-K/QPEG及低成本F-noQPEG诊断；
- [ ] 三数据集统一n=300 EM/F1和macro；
- [ ] 候选模型统一n=100 IHR，固定judge/qid/template；
- [ ] 报Supply、SFT-learning、Outcome-PPO、KG-reward-PPO及Total method；
- [ ] 报paired bootstrap CI、McNemar、gained/lost/tied和单seed限制；
- [ ] 写机器可读result record和人类可读闭环报告。

## 9. 仍需补齐的既有证据文件

- [ ] `outputs/audits/hotpot_relation_graph_v2_root_cause_result_v1/request_level_audit.jsonl`；
- [ ] `outputs/audits/2wiki_confirmation270_semantic_audit/human_adjudication.jsonl`；
- [ ] 先冻结`outputs/audits/2wiki_proofkg_ihr_matched_v1/protocol.json`，再生成同目录
  `result_record.json`。

这些缺失不改变已完成的canonical EM/F1结果，但未补齐前不能把对应聚合解释或IHR写成正式论文结论。

## 10. 已关闭且不得重复投入的路线

- [x] 训练KG索引0%命中工程问题已修复；
- [x] 8k/50% ProofKG continued-SFT失败，结果保留；
- [x] light20 5k未同时通过replay/hidden门，不继续加步数；
- [x] 当前ProofKG PPO-smoke600未在canonical n300超过strong SFT；
- [x] Hotpot Wikidata-only relation路径结构门失败；
- [x] MuSiQue当前subquery conversion未过recognized门；
- [x] 最后一次claim-constrained Wikidata pilot已完成：Hotpot nonempty/complete=7/1，
  MuSiQue=4/1；同passages A/B均净增0、净损1，正式`FAIL_STOP`，不扩confirmation/训练；
- [x] retrieved-passage本地模型SRO pilot已完成并在结构门停止：Hotpot/MuSiQue均
  nonempty=13/30，parse=.667/.600；未跑utility A/B、不扩confirmation、不接训练；
- [x] reward-v2.1与L0 verifier未通过“top1稳定超过greedy”主门；
- [x] 新建hard-contrastive curriculum：421个完整ProofKG候选中筛出208个mixed qid，
  recovery/stability=25/183，reserve82与其qid/family交叉为0；
- [x] reserve82 greedy+K4零更新门全部通过：valid=.9878、mixed=45、pairwise=.6961、
  reward-top1−random=+19.44pp；状态仅为`PASS_READY_TO_PREPARE_PAIRED_PPO`；
- [ ] 研究者批准后才准备PPO-O/PPO-K两份锁定配置；准备不等于启动远程训练；
- [x] 2Wiki额外Wikidata ProofKG `.640/.694`只作为resource-augmented结果单列。

## 11. 禁止事项

- [x] 不覆盖Gold、raw data、legacy baseline/index或旧实验；
- [x] 不删除失败实验、checkpoint、日志和原始评估产物；
- [x] 不在同一实验同时改变数据、reward和优化参数后作单变量归因；
- [x] 不事后降低门、换qid或挑checkpoint制造正结果；
- [x] 不把2Wiki额外Wikidata结果冒充三数据集同资源主表。
- [x] 不把passage-SRO负结果扩写成“DBpedia/训练型关系抽取器也无效”；二者尚未测试。

详细历史任务与决策D1–D36保存在`archive/project_plans/20260902/todo.full.md`。

## 12. 当前冻结执行清单：mixed3 ReaRAG PPO-T / PPO-TK（2026-09-03）

> 本节为最新活动状态，append-only取代上文“mixed数据尚未物化/PPO-O不加载ReaRAG”的旧计划描述；
> 历史失败、BLOCKED项和负结果均保留。当前总状态：`CPU_PASS_GPU_NOT_STARTED`。

### 已完成并冻结

- [x] 强SFT起点固定为`checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final`；
  本轮不重训SFT，不使用任何失败的continued-SFT checkpoint；
- [x] 物化mixed population 1799 unique：Hotpot/2Wiki/MuSiQue=`600/600/599`；1591行为无图
  outcome+text训练，失败QPEG/SAEG passage edges为0；
- [x] 冻结1800 prompt groups、每数据集600组、K=4，即每臂7200 trajectories；
- [x] 冻结208个identity-safe complete 2Wiki ProofKG eligible问题；日程重采样为300 groups/
  1200 KG-eligible trajectories；
- [x] 冻结alias-aware判分：合法轨迹
  `R_out=4*(max_alias canonical_EM + .1*max_alias canonical_F1)`，invalid=`-4`；1799行中163行
  有多原始alias、149行规范化后仍有多个alias，aliases不进入prompt；
- [x] 冻结共享ReaRAG文本项：更新前因果EMA中心化、`clip[-1,1]`，每个step-end放置
  `.3*c_t/n`；显式`rearag`后端且fail-hard，禁止dummy/旧head回退；
- [x] 冻结PPO-T=`R_out+R_text`，PPO-TK仅增加
  `I_eligible*.2*R_ProofKG-v2.1`；两臂有效配置唯一科研差异为
  `proofkg_process_reward=false/true`（另有各自输出ID）；
- [x] mixed路径禁用旧alpha gate、旧Phase2 PRM与旧composite reward；outcome/全局KG项在final
  generated token，ReaRAG是真正step-local credit；
- [x] 两臂共享稳定参数：lr1e-6、batch4/mini1、2 PPO epochs、KL `.25`/target`8`/horizon`2000`、
  gamma/lambda`.95`、clip/value clip`.2`、critic zero-init/dropout0、max input/new=`6144/384`、
  max steps5、每600保存及step200后的健康门；
- [x] 两臂共享10% HotpotQA-only SFT replay与`.10` anchor weight；明确它是防遗忘锚而非三数据集
  均衡replay；
- [x] CPU preflight通过：`outputs/audits/mixed3_rearag_ppo_pair_7200_seed42_v2/local_preflight.json`
  状态=`PASS_NO_GPU_PREFLIGHT`，相关测试全过，1799/1799 alias转发、KG join=1.0、7200日程和配置
  diff均通过；旧data report的blocked状态不覆盖，由v2后续记录解除；
- [x] 固化配置：
  `configs/training/phase3_ppo_mixed3_rearag_v1_text7200_seed42.yaml`与
  `configs/training/phase3_ppo_mixed3_rearag_v1_text_kg_v2_1_7200_seed42.yaml`。

### 启动前与正式训练

- [ ] 在96GB远端重复fail-closed preflight：模型/adapter/ReaRAG/data hashes、磁盘、GPU、两个输出目录、
  两个`/root/tf-logs/<ExperimentID>`目录必须满足锁定条件；
- [ ] 先跑≤8 trajectories真实GPU连线探针；核对policy/reference/ReaRAG并存、alias reward、
  ReaRAG raw/EMA/centered/clip telemetry、step-end与final-token reward位置；失败则不启动正式训练；
- [ ] 按固定顺序训练PPO-T后训练PPO-TK，每臂7200；保存完整stdout日志、history、TensorBoard events、
  每600 checkpoint、final及manifest，并同步回本地；
- [ ] 以final-7200为预注册主终点；中间checkpoint仅作训练健康/曲线诊断，不事后挑最优点；
- [ ] 验证valid、KL、critic、截断、replay和各数据集outcome/text/process/total reward遥测；出现non-finite
  或健康门失败时保留失败记录并停止。

### 评估、论文边界与复现

- [ ] 标准legacy同资源主表：相同n/qid/decoding评估strong-SFT、PPO-T、PPO-TK三数据集EM/F1；
  `T−SFT`解释outcome+ReaRAG后训练，`TK−T`解释eligible ProofKG过程奖励；
- [ ] HotpotQA/MuSiQue没有直接KG process监督，其变化只报共享策略迁移/保持，不宣称直接KG监督；
- [ ] 2Wiki ProofKG `.640/.6944`及matched `+23.7pp EM`只进额外Wikidata资源增强表，显式区别于
  legacy baseline资源，禁止混成同条件主表；
- [ ] 对同qid报告EM/F1、paired bootstrap CI、McNemar、gained/lost/tied与按dataset/eligible分层；
- [ ] 当前仅seed42。重要提升若方向成立，至少按同一冻结协议补额外seeds；若资源不足，必须显式写
  single-seed限制，不能从一次或中间checkpoint选择性宣称成功；
- [ ] 生成最终可追溯记录：`Claim -> Evidence -> Experiment -> Evaluation -> Data/Model`，绑定配置、
  pair locks、数据hash、checkpoint hash、seed及evaluation protocol。

## 13. 最新执行版本：mixed3 Proof400 v2（2026-09-03）

> 状态：`DATA_CONFIG_PASS`。本节append-only取代第12节的v1数据选择；GPU探针和训练尚未启动。

- [x] 撤回v1“family overlap=0”口径：旧hash来自不兼容namespace；统一lexical-family-v1重算为
  51个dataset-scoped families/83行重叠，仅旧family结论被supersede，v1资产和结果全部保留；
- [x] 冻结v2共1799 unique：Hotpot/2Wiki/MuSiQue=`600/600/599`；
- [x] 2Wiki改为400个完整ProofKG + 200个普通空图问题；400=`125 safe hard + 275 expansion`；
- [x] ProofKG四类各100：inference/comparison/compositional/bridge-comparison；
- [x] 用统一`answer-free-lexical-family-v1`排除canonical main900和未开confirmation300；按
  `(dataset,family)`检查，population/proof与A类qid及family overlap均为0；
- [x] 锁定7200预算：三数据集各600 groups、K=4；1799身份全部调度，仅MuSiQue重复1题；
- [x] eligible从v1的208 unique/1200 trajectories扩为400 unique/1600 trajectories；
- [x] 全量物化门通过：question-KG identity join=1.0、400 execution完整、1399空图、Gold仅作train
  outcome label、失败QPEG/SAEG-P边为0；相关回归测试42项通过；
- [x] 固化v2数据与配置：
  - `data/silver_data/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42/`；
  - `configs/training/phase3_ppo_mixed3_rearag_v2_proof400_text7200_seed42.yaml`；
  - `configs/training/phase3_ppo_mixed3_rearag_v2_proof400_text_kg_v2_1_7200_seed42.yaml`；
- [ ] 为v2生成新的正式pair lock/preflight（不得复用v1 lock）；
- [ ] 运行最多8 trajectories的v2 GPU wiring probe，确认policy/reference/ReaRAG、reward telemetry、
  TensorBoard和显存；
- [ ] probe通过且研究者批准后，才顺序启动PPO-T/PPO-TK各7200；失败则保留记录并停止。

## 14. Proof400 runtime probe v3（CPU完成，GPU未运行）

- [x] 冻结独立微型数据：T非eligible 1题×K4，TK完整ProofKG eligible 1题×K4，总计8；
- [x] 绑定当前Proof400数据、family/code lock、两臂配置及90个runtime代码文件；
- [x] CPU preflight通过：真实CLI差异、identity join、K4、alpha=null、ReaRAG与v2.1 execution均正确；
- [x] 新增只读遥测：pre-update EMA、unclipped centered、process-applied count；reward公式未改；
- [x] postflight设为fail-closed：explicit reference、initial KL≤1、合法ReaRAG步骤及有限统计、T process=0、
  TK process-applied≥1且加权非零、critic/PPO、adapter/log/TensorBoard/manifest必须全过；
- [ ] 研究者批准并提供远端GPU后，运行
  `launch_ppo_mixed3_rearag_runtime_probe_v3_proof400_remote.sh`；
- [ ] 仅当v3 GPU postflight为PASS，才解锁正式PPO-T/PPO-TK 7200；当前二者均未启动。

## 15. 正式PPO前的强SFT探索空间门（Proof400 fill275 n100）

- [x] 从新增fill275中排除safe-hard125，按四种qtype各25条冻结n100；100题属于100个不同family；
- [x] 将n100标为train-side `development/consumed`，禁止以后充当独立confirmation；
- [x] 冻结同一强SFT、mixed-v2原passages与完整ProofKG；prompt与train-only Gold outcome标签分离；
- [x] 冻结生成口径：每题greedy 1个 + sampled K4，384 tokens、temperature/top-p=1、seed42，共500次；
- [x] 冻结三门：valid≥0.90、oracle@4−greedy EM≥5pp、mixed-outcome qid rate≥0.20；
- [x] 首版v1在GPU前发现模型hash只记录未复核，保留并标记superseded；v2已补强模型文件fail-closed；
- [x] CPU物化与相关测试通过；未加载模型、未分配CUDA、未生成候选、未训练；
- [ ] 获研究者批准后可运行`launch_proof400_fill275_sft_headroom_n100_local.sh`；
- [ ] 若三门失败，只能新冻cohort/配额，不能改门或声称PPO无效；三门通过也只解锁训练，不证明PPO有效。

## 16. 强SFT探索空间门v3修复

- [x] 保留v2失败manifest/log：错误发生在生成前，0 candidate、0 optimizer update，标记无科学结果；
- [x] 修复采样参数层级为`generation.sampled.temperature/top_p`，不改cohort、预算或决策门；
- [x] v3主动hash并复核Llama config/index/4 shards/tokenizer及强SFT manifest/loss/adapter/tokenizer共19项；
- [x] code closure补全question-KG、ProofKG eligibility、logging等直接依赖，并冻结软件包/CUDA build版本；
- [x] 15项CPU测试通过，含四类25题、100 unique families、原passages/ProofKG、Gold分离和完整模型hash；
- [ ] 后续如获批准，只运行v3 launcher；当前未重跑GPU、未生成候选、未启动PPO。

## 17. 子问题拆解到依赖检索 feasibility pilot（2026-09-03）

- [x] 核对完整本地Wiki18三资产：corpus/dense/BM25均21,015,324篇，ID和维度一致；
- [x] 实现独立的plan-once依赖检索器、bridge替换、确定性10篇merge、Gold-free物化、Gold后附加及
  paired evaluator；未改canonical pipeline；
- [x] 57项相关测试通过；HotpotQA/MuSiQue真实冻结plan dry-run均30/30；
- [x] append-only保留v1/v2检索前superseded和v3运行效率中止（0行、无科学结果）；
- [x] 冻结并运行v4 layer-batched等价版本；两数据集均30/30执行、0错误/0回退，依赖查询非空率均1.0；
- [x] 冻结A/B scoring输入并在本地强SFT跑完60题配对评估；
- [x] 记录结果：pooled EM `.3167 -> .3500`、F1 `.4610 -> .4716`、gained/lost=`3/1`，两个数据集
  EM均`+3.3pp`；但95% CI含0；
- [x] 按预注册门判`FAIL_STOP_DEVELOPMENT_FEASIBILITY`：净增2题未达3题，且MuSiQue出现1题parse
  退化；不事后降低门、不打开fresh confirmation、不宣称方法已验证；
- [x] 完成逐题机制审计：存在有效依赖检索正例，也存在bridge类型错误和固定预算误删旧证据的反例；
- [ ] 如继续，另冻development-only改进：bridge entity/type约束或bridge rerank、避免低置信新passage
  挤出高置信A passage，并先在已消费pilot复测；
- [ ] 只有新版本通过既定效用门，才准备fresh family/QID-disjoint confirmation，并加入相同检索调用
  预算的generic-bridge对照，隔离“结构化拆解”与“更多检索调用”；
- [ ] 不把gold字符串可见率写成support recall，不据本pilot调整PPO/SFT/reward。

## 18. 子问题依赖检索 v5 状态与下一步（2026-09-04）

- [x] 冻结v5组合实验：typed bridge admission + A前8篇保护/完整问题CE严格替换；明确不能做组件归因；
- [x] 冻结60道已消费development题、完整Wiki18资产、强SFT/base/CE模型、v4 scorer及全部直接代码hash；
- [x] 109项相关测试通过，正式Gold-free物化完成，0 runtime error、fallback精确、无root注入或越权置换；
- [x] 按预注册机制门判`FAIL_STOP_GOLD_FREE_MECHANISM_GATE`：Hotpot依赖查询11/24、新文档覆盖3/30；
  MuSiQue依赖查询22/30、新文档覆盖8/30，均未达到`.80/.50`；
- [x] finalizer在Gold读取前停止，未运行强SFT答案生成，EM/F1为`NOT_AVAILABLE`；
- [x] 冻结Gold-free失败分解：bridge admission卡住Hotpot/MuSiQue 13/8题；依赖检索成功但CE拒绝8/14题；
  被拒新文档相对A尾部平均margin为`-0.2230/-0.1412`；
- [ ] 另冻v6 development-only协议：完整问题锚定的多候选bridge查询扩展，保留相同10篇预算和A8保护；
- [ ] v6不得降低v5门或选题；先跑Gold-free安全/机制门，失败即停止，成功后才允许既定EM/F1效用评估；
- [ ] 仅当v6通过开发效用门，才一次性打开fresh family/QID-disjoint confirmation，并加入调用预算匹配的
  generic query-expansion臂，区分子问题结构收益与额外检索预算收益；
- [ ] 本线在fresh confirmation前不修改SFT/PPO/reward，也不把v4方向性提升或v5机制失败写成正式主表结论。

## 19. 子问题依赖检索 v6 状态（2026-09-04）

- [x] 冻结v6组合实验：最多两个bridge query hints、完整问题前缀、variant-balanced top2；保留A8、
  top10、最多替换2篇及完整问题CE严格胜出门；
- [x] 锁定same60输入、完整Wiki18三资产、E5/CE/强SFT/base、代码和三个唯一Experiment ID；
- [x] v5/v6相关回归测试通过，真实60题Gold-free dry-run通过；
- [x] 正式Gold-free机制门通过：Hotpot query nonempty=`24/24`、changed=`15/30`；MuSiQue
  query nonempty=`30/30`、changed=`18/30`；两者plan=`1.0`，安全/一致性门全部通过；
- [x] 机制门通过后独立附加Gold并完成强SFT A/B：pooled EM `.3167 -> .3000`、F1
  `.4610 -> .4215`、gained/lost=`1/2`；Hotpot退化，MuSiQue EM持平/F1方向为正；
- [x] 按预注册判`FAIL_STOP_DEVELOPMENT_FEASIBILITY`，不打开fresh confirmation、不启动训练；
- [x] 核验本地CoRAG/ReaRAG/R1-Searcher/Search-R1/Self-RAG，并记录可借鉴机制与边界到
  `docs/multistep_retrieval_reference_notes.md`；
- [x] 完成v6已解锁结果的只读逐题诊断并落盘：33个changed问题EM/F1均退化，47篇替入文档
  虽44篇命中子查询词但仅8篇含Gold字面；确认主因是证据语义角色/决定性hop不足，不是查询未执行；
- [ ] 不在same60继续调CE阈值或挑例；若推进新版本，先提出新的证据支持性变量和独立开发设计，
  获研究者确认后再冻结，当前不得直接启动v7、SFT或PPO。

## 20. 子问题回答驱动依赖检索 v7（2026-09-04）

- [x] 冻结development-only A/B/C：A为canonical一次检索，B为确定性top-1实体，C为强SFT严格JSON
  子答案；B/C逐question/depth/hop匹配逻辑检索预算；
- [x] append-only澄清递归估计量：根query同则passages逐字节同；bridge分叉后的上下文差异是策略中介，
  C−B解释为完整递归策略效应，总模型计算量不匹配；
- [x] 绑定强SFT/base完整文件树、Wiki18三资产、模型/代码/父协议/stage descriptor/scorer闭包hash，并实现
  Gold递归禁字段、state parent链、预算、fallback和模型身份fail-closed检查；
- [x] planner retry1在HotpotQA20/MuSiQue20上均20/20可执行；Gold-free root物化成功并冻结41个depth-1任务；
- [x] 强SFT生成41条：strict JSON为Hotpot `12/19`、MuSiQue `18/22`，机械验证仅`3/19`和`3/22`；
  失败主因为`answer_type=other` 24条、`abstain`非布尔10条、citation数量1条；
- [x] 记录四次执行异常：前两次无日志且仅作operator-supplied/unverified历史；后两次由日志hash佐证，均未
  产生可计入结果的新stage；修复runner终止状态与same-depth sibling重复query，但不以新代码续跑旧锁链；
- [x] 完成CPU-only Gold-free单调上界审计：最乐观Hotpot=`3/19=.1579`、MuSiQue=`4/23=.1739`，均
  低于冻结`.40`门；状态`FAIL_STOP_BEFORE_GOLD_MONOTONIC_UPPER_BOUND`；
- [x] 停止后续递归检索和最终答案评估；Gold未打开，EM/F1/IHR=`NOT_AVAILABLE`，不启动SFT/PPO；
- [x] v7相关回归测试全绿（91项），early-stop与recursive-addendum定向测试10项通过，核心脚本py_compile通过；
- [ ] 若继续，先预注册v8 Gold-free单变量schema适配并对冻结41条raw response做反事实parser审计：忽略模型
  自报type，由程序确定性推断类型，保持citation/locality/subject拒绝规则不变；
- [ ] 只有反事实潜在机械验证率在两个数据集都达到`.40`，才从头建立新implementation/plan/stage锁并跑v8；
  否则停止，不降低门、不打开Gold、不把已消费development题当confirmation；
- [ ] v8若过Gold-free门，再按原冻结顺序评估C−B与C−A；通过development utility后才设计fresh
  family/QID-disjoint confirmation。本线通过前不修改训练数据、reward、alpha、SFT或PPO配置。
- [ ] 若v8接口反事实仍未过门，停止固定PID-plan补丁，另行预注册IRCoT/DecomP启发的动态
  observation-conditioned controller；A=canonical one-shot、B=预算匹配generic expansion、C=动态拆解，
  固定corpus/模型/query/passage/token预算；已有trace属于baseline，多轮检索本身不得作为本文创新。
- [ ] 只有动态controller的Gold-free机制和development utility先后通过，才考虑构建
  `state->query / query+docs->subanswer / history->final answer`链式SFT；PPO阶段再在同一controller backbone上
  比较outcome-only与alpha-gated KG process reward，禁止把backbone变化归因于KG reward。

## 21. 动态问题拆解 v8（development90 Gold-free失败并停止）

- [x] 阅读并核对本地IRCoT、Decomposed Prompting、CoRAG、Search-R1/R1-Searcher、ReaRAG与Self-RAG；
- [x] 形成`docs/subquestion_decomposition_v8_test_plan.md`；当前状态为 Phase 0 已完成、容量审计进行中，
  A/B/C新检索、Gold评估和训练仍未授权；
- [x] 明确主可证伪问题：同corpus、调用上限、final passage预算和reader下，answer/observation-conditioned
  q2策略是否优于不看q1证据/答案的静态q2；该估计量是整个证据状态conditioned policy effect，不缩写成
  “只改变answer字符串”；
- [x] 未注册只读重放确认type-only单变量有潜力：Hotpot `11/19`、MuSiQue `17/22`；正式值仍须另写
  append-only审计复现，不能把探索数写成论文结果；
- [x] 发现并记录trace预算口径冲突：历史输出累计11–69篇，不能作为固定top-10 matched control；
- [x] 冻结并运行Phase 0 Gold-free接口审计；72项相关测试通过，0 GPU/model/retrieval，P0完整复现；
  P1将Hotpot从`3/19`提升至`11/19`、MuSiQue从`3/22`提升至`17/22`，两者均过`.40`机械门；
  状态`PASS_P1_TYPE_ONLY_INTERFACE_DIAGNOSIS`，不等于语义正确或EM/F1提升；
- [x] 追加Phase 0独立复核披露：现有数值有效，但聚合P0门顺序和直接hash闭包需加固；保留原结果，
  后续若重跑必须新协议/新Experiment ID；
- [x] 完成capacity-only审计：严格Scope A可冻结family为Hotpot`6213`、2Wiki`331`、MuSiQue`1644`；
  三数据集`development30+prospective300`均可行，但统一`reserve1000×3`因2Wiki缺999而失败；Scope B因
  training ledger不完整已标无效；审计只用identity投影但源可能含Gold，未生成任何fresh row；
- [x] 研究者已裁决严格Scope A：每数据集`development30+prospective300`，取消本轮
  `reserve1000`；
- [x] 已实现identity-only custodian export并一次性冻结90/900题；每行仅
  `dataset/qid/question`，与history/training/raw-train及两split间的dataset-scoped qid/family overlap全为0；
- [x] 已实现development-only锁定loader；默认拒绝prospective role/path且无unlock开关。
  后续materializer必须只经该loader，并在runtime protocol锁定其hash；
- [x] 已完成v8 Gold-free纯函数核心与测试：query/subanswer契约、唯一surface provenance绑定、
  static/dynamic action选择、`root6+q1 novel2+q2 novel2`固定10篇合并、Unicode/数值边界与
  fail-closed完整性；
- [x] 已完成历史Trace/IRCoT-style n300×3 append-only预算审计；累计文档均值
  `26.19/28.7433/28.6367`，不得作为fixed-10 matched control，无法恢复的runtime项均记`UNKNOWN`；
- [x] **研究者已裁决q2 call accounting**：2-call方案为B slot2只看
  `Q+q1+NO_VERIFIED_SUBANSWER`，C只a1有效时增加`a1+provenance`；a1无效子集严格byte-identical，
  dynamic-invalid则C回退Q并作ITT单列；批准时间为2026-09-04；
- [ ] 在打开Gold前定位并版本化官方Hotpot supporting facts与MuSiQue decomposition；当前L3 scorer来源
  为`UNKNOWN`；
- [x] 已实现A/B/C正式runner与call ledger：B/C固定root/q1/q2三次logical检索、
  2个logical controller slots、1次q1 reader和固定10篇；append-only driver与首版implementation freeze
  也已完成，相关回归测试全绿；
- [x] 已启动真实n=4×3 smoke attempt001；模型和索引加载成功，但逐query dense全库扫描为67--83秒，
  串行路径预计smoke 50--66分钟且development需数小时，故在0行科学输出时停止并保留FAILED manifest；
- [x] 将完全相同的root/q1/q2查询改为三层batch执行，保持logical/cache账本和逐题输出语义不变；重跑测试、
  生成新implementation freeze，并以append-only attempt002重跑engineering smoke；
- [x] 等价batch、implementation V2、133项相关测试与attempt002真实smoke均完成；smoke为12题、
  84/84 logical retrieval、3次full-index passes，16项Gold-free工程门全部PASS；
- [x] 已运行冻结development90并保留逐题输出；q1有效率=`1.000/.933/.967`，a1 admissible=
  `0/.133/.067`，B q2有效率=`.678`，C dynamic transition ITT=`0/.033/.033`，四个核心门失败；
- [x] 已做无Gold失败分解并append-only登记：a1跨文档歧义42、未找到29、q1回声6、sentinel 7、通过6；
  B q2重复原问题/q1共29题；同时更正3次backend batch实际对应5次memmap遍历的遥测命名；
- [x] 按预注册停止：不打开Gold、不计算EM/F1/IHR、不读取prospective900、不在same90调prompt/阈值；
- [ ] 若继续动态路线，必须使用独立train-side数据训练controller/subanswer reader，并另冻新的未见验证集；
  当前90题只能作consumed development，不能再次用于晋级；
- [ ] Gold-free机制门失败即停止；通过后先生成并hash无Gold predictions，再由独立scorer附加
  support/answer标签并跑development L2/L3/EM/F1；
- [ ] development要求C−B ITT EM至少+5pp、F1为正、L2/L3机制门通过且无明显单数据集退化；否则不打开
  prospective validation；
- [ ] prospective-validation n300/数据集只作周内前瞻验证；不得根据development临时扩n。论文级确认仅能
  使用预先独立封存的reserve1000×3，且不得与n300选择性合并；
- [ ] 零训练prospective validation通过前，不修改controller-SFT、PPO、reward、alpha或KG供给配置。

## 22. canonical 子问题回答 v9（Phase 0 已失败停止）

- [x] 冻结单变量协议：复用已消费v8 dev90的相同q1与top-10，只把一行短答案接口换为正式
  `build_inference_messages()` + 512-token greedy trace + `extract_final_answer()`；不改binder、不检索、
  不读Gold、不打开sealed prospective900；
- [x] 新增`kgproweight/retrieval/canonical_subqa_v9.py`、冻结器、append-only runner和单测；相关v8/v9
  测试90项通过；
- [x] 本地4090完成90次生成：三个数据集Final Answer解析率和step-trace率均为1.0，证明canonical接口
  消除了输出格式错配；
- [x] 按预注册门判FAIL_STOP：唯一文档binder接纳率Hotpot=`2/30=.067`、2Wiki=`5/30=.167`、
  MuSiQue=`5/30=.167`，均低于`.40`；未计算EM/F1/IHR；
- [x] 记录剩余瓶颈：55/90因同一答案surface在多篇文档出现而被拒，11/90未找到，其他12/90为
  boolean/null/echo；至少一篇文档surface命中的诊断上界为67/90，不能当作正确率；
- [x] 研究者随后已批准v9.1；唯一文档门单变量替换为“任一文档命中+最高rank确定性 provenance”，并在
  全新train-side、family-disjoint pilot30×3上运行完整q1→a1→q2→检索→final链，结果见§23；
- [x] v9.1在冻结与运行中均未打开sealed prospective900；其q1 binder门虽通过，但q1 schema与B静态q2门
  失败，因此没有预称整条动态链成立，也没有打开Gold。

## 23. rank-first v9.1 与论文并行写作（2026-09-04）

- [x] 获得研究者批准后，将v9.1唯一绑定变量冻结为“多文档surface命中时选择最高检索排名文档”；明确
  该规则只证明lexical locality，不证明semantic support；
- [x] 冻结全新、identity-only、family-disjoint pilot30×3与正式协议；未打开或hash sealed prospective900；
- [x] 完成runner/binder/driver和回归测试，相关测试118项通过；engineering smoke n=4×3全部工程门通过；
- [x] 完成fresh pilot90 Gold-free运行：a1 admissible=`.767/.600/.833`，C dynamic ITT=
  `.600/.533/.633`，两组核心binder/dynamic门通过；
- [x] 按冻结协议判`FAIL_STOP_GOLD_FREE_GATES`：q1 schema valid=`.900/.933/1.000`，前两数据集未过
  `.95`；B static q2 pooled valid=`.633<.90`，33个无效输出全部为重复原问题或q1；
- [x] 保留rows/report/FAILED manifest并补append-only诊断记录；Gold、EM/F1/IHR、sealed prospective均未打开；
- [x] 停止在same pilot90上继续改binder、prompt、parser或门槛；
- [ ] 若继续动态检索，先提交专门query-controller的train-side SFT/蒸馏协议、训练数据身份隔离和新
  Gold-free验证集方案；获得批准前不训练、不复用本轮90题调参；
- [x] 完成`docs/paper/polished_draft.md`首版重写：中英双语全文、RQ式结果、讨论/局限、9条本地核验
  参考文献；same-resource与额外Wikidata资源结果分开，失败结果保留；
- [ ] 补跑所有本地baseline checkpoint的统一canonical protocol主表；
- [ ] 完成PPO-T/PPO-TK成对多seed实验、统一盲化IHR、当前五特征α跨数据集测量后，再替换稿件中的TBD；
- [ ] 后续从原始/官方来源核验并补齐数据集、Wikidata、PPO与Llama-3的标准参考文献；在此之前不得
  从旧稿复制无法核验的引用元数据。

## 24. 方法论文重构与结果填充清单（2026-09-04）

- [x] 新建`docs/paper/PAPER_REWRITE_BLUEPRINT.md`，冻结唯一主故事与各章节职责：
  Strong SFT→matched PPO-T/PPO-TK是训练主线；ProofKG判过程，额外Wikidata单列，Controller只作独立推理扩展；
- [x] 重构`docs/paper/polished_draft.md`为方法优先的中英双语稿；摘要/引言不再由失败版本驱动，开发诊断
  集中到Appendix A；
- [x] 主奖励公式与预注册的mixed PPO目标设计对齐：`R_out + R_text + 0.2 I_eligible R_Proof`；由于
  temporal derivation实现缺陷，修复、版本升级和重新冻结paired配置仍是正式训练前的P0，不把历史
  `precision×relevance`或learned alpha冒充当前ProofKG-v2.1；
- [x] 数据章写入并区分可追溯单位：Reader SFT=`4,751`条实际train trajectories；PPO=`1,799` unique qids、
  `1,800` groups、`K=4`、`7,200` scheduled rollouts；ProofKG=`400` eligible 2Wiki groups / `1,600`
  scheduled eligible rollouts；
- [x] 主结果表填入已有canonical baseline与Strong SFT历史参考EM/F1，补齐可直接计算的macro并加
  `$\dagger$`口径；fresh Scheme-A Strong SFT仍须与paired PPO同批重跑。2Wiki extra-Wikidata matched结果
  继续单表报告，不当作PPO收益；
- [x] 更正Gold边界：Gold不进入prompt/ProofKG/execution；125个hard/A-safe入组使用train-only Gold outcome
  分层，275个automatic qids的图构建与入组为Gold-outcome-free；
- [x] 完成论文/蓝图末轮一致性审计：移除Controller作为“完整KG-ProWeight”的混写；统一“核心reward方法 +
  独立Controller扩展”的命名，并清除prepared schedule被写成已训练更新的表述；
- [ ] **P0，正式PPO前需研究者批准核心reward修改**：修复`proofkg_process_v2.py` temporal derivation
  对tail字符串做`len(...)`检查的错误，新增四位年份earlier/later与数值比较测试，升reward版本并重冻pair lock；
- [ ] **并行、非PPO前置**：物化、训练并冻结独立Controller release；最终manifest必须报告三数据集accepted
  qids/actions、q1/q2_dynamic比例、拒绝原因、teacher版本、qid/family隔离和reader regression；
- [ ] Controller通过机制门后，单独冻结matched-budget inference contexts，并以同一Controller版本评估
  Strong SFT/PPO-T/PPO-TK；不得改写现有7.2k PPO release。未来若用Controller contexts训练PPO，必须新建
  evaluation/training protocol并重新批准；
- [ ] reward P0修复、升版、补测试并重冻pair lock后，启动正式PPO-T/PPO-TK配对；该训练不依赖Controller
  release。先完成单seed主实验，再按论文主张补多seed与统一IHR；
- [ ] 只用冻结结果填充论文中的HTML result slots/破折号；不得使用历史smoke或development结果代填；
- [ ] 从原始论文/官方文档补齐数据集、Wikidata、PPO/GAE、LoRA、Llama 3、FlashRAG、检索与reranker引用；
- [ ] 投稿前移除全部内部HTML状态注释，并对每个claim完成
  `Claim -> Evidence -> Experiment -> Evaluation -> Data/Model`核对。

## 25. 完整成功态目标论文与实验回填（2026-09-04）

- [x] 按研究者最新口径将`docs/paper/polished_draft.md`改为完整方法成功态，而不再只围绕当前已冻结的
  `Strong SFT -> PPO-T/PPO-TK`最小实验讲故事；
- [x] 将摘要与引言重构为“多跳证据获取困难 -> outcome-only信用分配缺口 -> KG机会与不完整风险 ->
  query control/evidence graph/reliability weighting统一方法 -> SFT+PPO训练原则 -> 三项贡献”；
- [x] 从摘要、引言和方法总览中移除实验臂命名、样本量、版本号、路径、smoke与工程门；匹配PPO臂仅保留在
  实验设计的因果消融位置；
- [x] 修正目标公式：检索轮次/推理步骤索引分离；Controller结构化subgoal与中间答案验证闭环；连续可信标签+
  BCEWithLogits graph gate闭合；标准三元组允许tail entity或value；P/H/O/G/A图分数按完整轨迹聚合；
  Controller与Reader使用独立SFT target/adapter；Reader使用正式clipped PPO目标；
- [x] 中英文逐节镜像加入成功态结果槽位：总体效果、过程监督、Controller、跨证据结构泛化与机制分析；
  槽位不含虚构数值，顶部明确投稿前必须以冻结实验替换；
- [x] 保留并补足2Wiki已验证matched-control口径：dev n=300、seed42、greedy、同checkpoint、同10篇passages、
  仅替换KG block，并明确它是extra-Wikidata supply effect；
- [ ] 在未参与reward/alpha选择的held-out 2Wiki split上复制legacy-vs-ProofKG matched control；完成前正文
  §5.5只标为exploratory supporting analysis，不作为最终确认性主结果；
- [x] 补正Self-RAG年份并加入已核验Wikidata原始引用；
- [ ] 物化并训练专门Controller-SFT release：输出`query/anchor/relation/dependencies/output_slot/source_action`，
  中间答案必须绑定passage/triple provenance后才能写回状态；完成matched-call/controller消融，回填§5.3；
- [ ] 冻结三数据集统一双源训练release，使HotpotQA/MuSiQue主要使用passage/provenance分支、2Wiki同时使用
  canonical KG分支，并完成数据质量与family/QID隔离审计；
- [ ] 实现并冻结learned-alpha版本：仅用train-only连续`citation precision * required-hop relevance`目标训练，
  在独立calibration split选模并于PPO期间冻结；做`fixed coefficient vs validity-masked learned alpha`单变量
  matched消融，回填§5.2/§5.4；
- [ ] 在最终统一证据供给上完成Reader SFT与PPO主实验、多seed EM/F1/IHR、validity及检索成本，回填§5.1；
- [ ] 用正式冻结结果替换所有`Target result slot/目标结果槽位`。若某环节无正效应，删除对应成功态句子并收缩
  摘要、贡献和结论，不得用历史smoke或额外资源结果代填；
- [ ] 投稿前删除顶部内部draft-status说明，并完成逐claim的
  `Claim -> Evidence -> Experiment -> Evaluation -> Data/Model`审计。

## 26. 专门 Query Controller v4.4 / eval-e1（2026-09-04）

- [x] 冻结 2Wiki+MuSiQue action release：train=2,400、dev=240，q1/q2_dynamic 成对，train/dev qid 与
  family overlap=0；最终答案 Gold、confirmation 和 prospective 均未进入模型可见状态；
- [x] 完成 exact-20 QLoRA 可训练性 probe：20/20 optimizer steps、train loss=`3.5022`、末步
  loss=`1.5319`、loss/gradient 全有限且梯度非零、GPU reserved peak=`10.01 GiB`；
- [x] 保留 v4.2/v4.3 `FAIL_STOP`，定位两次均为 PEFT adapter reload 校验路径问题而非权重损坏；v4.4
  对齐真实 FP32 推理重载且仍坚持 256/256 tensor `torch.equal` 零容差，三份 dtype inventory 一致；
- [x] 保留 E0 CUDA 前参数转发失败（0 predictions），冻结并完成不重训的 eval-only E1；E1 只修两个
  已验证 protocol SHA 的调用点转发，不改变 checkpoint、dev240、decoding 或 scorer thresholds；
- [x] 完成 240 条 teacher-forced Gold-free 机制评分：2Wiki q1=`1.000`、q2=`0.9667`，分槽门通过；
  MuSiQue q1=`1.000`、q2=`0.8500`，q2 门失败；联合状态=`FAIL_STOP_ACTION_MECHANICS`；
- [x] 明确边界：当前结果证明“2Wiki 上专门 Controller 的动作机制可学”，不证明在线动态检索或 QA
  EM/F1 已提升；MuSiQue 尚未解决，HotpotQA 精确 action coverage 仍为 `UNKNOWN`；
- [ ] 冻结正式 Controller-SFT（使用全部 2,400 train actions，而非 128 条 probe subset）；训练前先按
  dataset/slot 报告 MuSiQue q2 的 9/60 失败类型，禁止通过降低 `.95` 门或修改同一 dev 来晋级；
- [ ] 正式 Controller 需在新的 family-disjoint Gold-free mechanism split 上同时通过 2Wiki/MuSiQue q1/q2
  门；HotpotQA 只有在独立 exact-action release 和覆盖审计通过后才能加入，不得用迁移假设代替数据；
- [ ] 机制门通过后冻结 matched-budget 在线 A/B：A=canonical one-shot，B=Controller 动态两轮；同 corpus、
  retriever、最终 passage budget、Reader checkpoint、decoding 与 scorer，只允许 query policy 不同；
- [ ] 在线链必须使用 Reader 预测且 provenance-bound 的中间答案，不得用 annotation intermediate；报告
  EM/F1、valid transition、retrieval calls、latency、gained/lost/tied 和按数据集分层；
- [ ] 只有在线 B 相对 A 的预注册 QA utility 门通过，才把 Controller 写入论文主结果或考虑用于后续 PPO
  contexts；否则保留为“2Wiki mechanism pass / MuSiQue underfit”的开发结论。

## 27. HotpotQA Controller 银标覆盖 pilot30（2026-09-05）

- [x] 审计 HotpotQA raw train 容量与两跳方向：90,447 条 raw；冻结 v1 严格链候选 14,137；排除当前已枚举
  consumed union 后可用 family 9,981，足以支持 pilot 与后续 600/60/30；旧 checkpoint 完整输入账本仍为
  `UNKNOWN`，不得宣称全局未见；
- [x] 冻结固定 pilot30：easy/medium/hard 各 10，qid/family overlap=0，输出仅 dataset/qid/question，sealed
  prospective 未打开，生成/审核失败不得换题；
- [x] 补齐 Gold 披露：train final answer 与 supporting facts 仅用于候选筛选、两跳定向和后续 Gold-aware
  label 审核；这是 Gold-screened silver supervision，不是 Gold-free evaluation；
- [x] 实现 Hotpot companion action builder：q1 无 bridge/final，q2-template 恰含一个 `#1`，
  `source_action=text`、`pid=null`、Hotpot 原生 provenance；不修改 frozen v4.4；
- [x] 修复通用别名/完整性问题：前导冠词与重音折叠防泄漏、扩展中间名有序掩码、proposal hash 与
  q1/q2/action 重算绑定；相关回归 83 项通过；
- [x] 完成固定30题预调用门：29/30 可安全生成，1/30 因原问题已含 bridge 扩展别名而拒绝；计入分母且不
  替换；
- [x] 冻结 DeepSeek execution protocol：Flash 单候选 producer + Flash blind review + Pro Gold-aware review，
  temperature=0、opaque nonce、上下文隔离、无 best-of/人工改写/语义重试；protocol SHA256=
  `f690063157f8818a79168f2f511fd7ec1becb2a431e81dcf26d372336a48360d`；
- [ ] 执行 29 个 producer 调用及对应双审；30×3 semantic slots 必须全部记账，预调用失败/上游跳过也必须
  留行；API key/base URL 值不得写入产物；
- [ ] 仅对双审通过项冻结 answer-free runtime projection：dataset/qid/question/q1/query-template及 hashes；
  运行器不得读取含标注 observation 或 Gold-bound q2 query 的完整训练 action；
- [ ] 另冻并执行 canonical Wiki18 + 强 SFT Reader pilot：q1 Top-10 -> Reader 预测并 rank-first provenance
  绑定 -> 用预测值替换 `#1` -> q2 Top-10 -> 固定10篇合并 -> 同一 Reader final；失败不得用 Gold bridge
  纠错；
- [ ] 固定分母至少 24/30 且所有 accepted 样本机械、双审、检索支持、Reader provenance 门均为 1.0 才扩
  HotpotQA 600/60/30；不通过则保留 FAIL_STOP，不调低门或换题；
- [ ] 扩量通过后，新建三数据集 successor schema/validator/trainer，再由研究者批准正式 Controller-SFT；
  当前 companion projection 通过不等于旧 v4.4 trainer 可直接消费；
- [ ] Controller 正式在线评估必须用强 SFT Reader 的预测 observation；train annotation intermediate 仅可用于
  标签物化和 scorer，不得作为运行时中间观察注入。
