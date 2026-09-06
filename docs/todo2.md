# KG-ProWeight 主线 TODO v2

> **2026-09-06 丢分诊断完成：`PPO_A_STOPPED300_REGRESSION_DIAGNOSIS_COMPLETE_NO_RETRAIN`。**
> 900输出及15个独立丢分问题/28次事件排查完成；新增丢分均唯一Final、非截断、统一min3有效。比较/关系/Final问题与缺证据下补全改变并存，原Gold别名与评分保留。实际训练仅75题，61条非零α只在2Wiki；无已确认奖励落点故障，不能归因α。详见[完整诊断](ppo_a_stopped300_regression_diagnosis_20260906.md)。

- [x] 核查900条输出提取/格式/长度，原EM/F1复算不变。
- [x] 审阅全部15个丢分问题，覆盖21次跨域和7次MuSiQue丢分；保留增益及共同错误案例。
- [x] 复核实际训练覆盖、奖励量级、两步答案信号、末token奖励/GAE；独立复算必要训练统计。
- [x] 完成仅修改Final的CPU Text输入不变性探针，不运行GPU或新预测。
- [ ] 按研究者后续决定，先冻结训练侧关系/比较/Final的过程评分效用小实验；自然两步另版验证，尚未改变核心变量或启动训练。

> **2026-09-06 评估终态（18:42:44完成，随后独立复核）：`EVAL_DEVELOPMENT150_COMPLETE_SFT_SELECTED_AUDITED`。**
> [终态status](../outputs/evals/ppo_a_stopped300_development150_20260906_v1/status.json)确认900/900条与13项生成/评分/选模任务完成。[原selection](../outputs/evals/ppo_a_stopped300_development150_20260906_v1/selection/report.json)选择StrongSFT，排序SFT/step200/aborted300。
> legacy宏EM/F1（%）为24.00/34.70、22.67/35.34、22.00/32.79；200相对SFT的F1+0.65pp但EM−1.33pp，300的EM−2.00pp/F1−1.91pp。no_graph为25.33/36.66、24.67/35.25、24.67/36.50。[六项原报告及分域表](ppo_a_stopped300_development_eval_20260906.md)完整保留，差值按未四舍五入分数计算。
> [独立结果验收](../outputs/audits/ppo_a_stopped300_development150_results_independent_20260906_v1/report.json)已通过900条评分复算及全部产物SHA/先生成后评分时序核验；[配对诊断](../outputs/audits/ppo_a_stopped300_development150_paired_diagnostic_20260906_v1/report.json)已完成，仅事后描述，不改选模。开发结果不支持综合改善或α独立贡献，亦不据小样本点估计宣称显著退化。当前仍是150题固定上下文代理，不是正式900题主表；训练300条guard FAILED保留，不自动新训练/重启/扩full/F/T或正式900评估。下方运行快照与待办是历史记录。

- [x] 完成全部六项生成，900条预测均保留。
- [x] 在全部生成结束后完成六项原EM/F1评分与原selection，选中StrongSFT。
- [x] 完成终态独立结果复核与逐题配对诊断；评分、产物与原协议执行一致，验收通过。
- [ ] 根据完整开发证据讨论下一步，不自动改变奖励/阈值或开启新训练、正式canonical900评估。

> **2026-09-06 17:30:29 当前状态：`EVAL_DEVELOPMENT150_RUNNING`。**
> 用户授权的本地评估已于17:29:19启动，supervisor3338466。原StrongSFT/step_200/aborted_step_300在原A银行150题×两视图上比较，共900条预测；固定输入/labels原字节、greedy512 BF16 batch1、原SFT tokenizer及原selection保持不变。
> 实际status在17:30:29记录generate__strong_sft__legacy运行中，12/900条预测、完整任务0，尚未评分；后续以[实时状态](../outputs/evals/ppo_a_stopped300_development150_20260906_v1/status.json)为准。34项相关测试与独立输入/三层隔离检查通过；全部六项生成先完成后再六项评分和原选模。
> 两份训练checkpoint真实更新已验收，原300条guard FAILED保留。当前仅150题固定上下文开发代理，不是正式canonical主表；不自动训练、重启、扩full/F/T或正式canonical900评估。[本地运行与日志入口](ppo_a_stopped300_development_eval_20260906.md)。下方“评估未启动”待办保留为历史快照。

- [x] 独立核对原A开发银行、三层训练/保护身份隔离及三个checkpoint/tokenizer身份。
- [x] 冻结实际SFT/200/aborted300注册与执行协议，完成34项测试并按用户授权启动本地评估。
- [ ] 完成全部六项生成、900条预测，保留每项输出和失败记录。
- [ ] 全部生成完成后运行六项原EM/F1评分与原selection，独立核对完整结果。
- [ ] 据完整开发结果讨论下一步；当前不自动进入正式canonical900或新训练。

> **2026-09-06 终态验收：`A_SMOKE600_STOPPED_AT300_PRESET_VALID_GUARD_AUDITED`。**
> 获授权的完整A-smoke600已于16:44:32启动，17:02:35在300/600条、75批时按预设有效率guard停止；supervisor17:02:39确认exit1/GPU占用与利用率0，600未完成，FAILED记录保留。
> [独立健康核对](../outputs/audits/ppo_a_smoke600_stopped300_independent_health_20260906_v1/report.json)：末15批有效41/60=0.6833<0.7，cap3/60=0.05、平均KL1.3282未越线；19条无效分为9条两步短缺+10条其他严重无效。全300条216有效/55两步短缺/29其他严重无效，61条非零α、674次Text评分、replay30，无NaN/Inf或OOM。
> step_200与aborted_step_300保留；[终态归档](../outputs/audits/ppo_a_smoke600_gpu_supervision_terminal300_20260906_v1/)20文件SHA与7452项事件/契约对照已通过，638 scalar/133 histogram及两份checkpoint真实有限参数更新均确认，[汇总验收](../outputs/audits/ppo_a_smoke600_gpu_supervision_terminal300_20260906_v1/report.json)完成；训练FAILED与600未完成不变。原失败、200条快照和fresh132健康FAIL保留，默认不重启、不降阈值、不扩full/F/T。[运行与停止记录](ppo_a_smoke600_supervision_20260906.md)。下方运行中/未启动待办均为历史快照。

- [x] 按用户授权执行独立完整A配置，保持原StrongSFT、奖励/α、数据、replay与预算保护规则。
- [x] 记录300条预设guard停止及独立健康复核，保留step_200/aborted_step_300和FAILED记录。
- [x] 完成terminal300全部checkpoint下载、20文件SHA核对及参数张量/TensorBoard事件独立终态验收；保留训练FAILED。
- [ ] 验收后先按原评估协议评估保存checkpoint；当前尚未启动评估，不把训练诊断当baseline成绩。
- [ ] 如需验证自然≥2步规则，单独冻结版本实验；当前未改变规则或阈值，未重启或扩展训练。

> **2026-09-06 16:44:48 当前状态：`A_SMOKE600_RUNNING_SUPERVISED`。**
> 完整A-smoke600于北京时间16:44:32（UTC08:44:32）真实启动；supervisor PID2846、training PID2847。远端910份文件/约37.49GB SHA、实际base/ReaRAG路径与10项负向检查均通过。
> 截至16:44:48仍处初始化，0/600条，尚无首批更新或实际TensorBoard run；按分钟监督，不预测结果。原StrongSFT起点、完整答案/过程/α、replay、固定600预算和原成本保护规则保持不变。
> [当前运行记录与查看命令](ppo_a_smoke600_supervision_20260906.md)。probe12工程PASS和fresh132健康FAIL保留；下方“600准备中/未启动”为历史状态。

- [x] 600独立scope/config/release及远端资产、模型路径、预算拒绝检查完成。
- [x] 在用户明确授权范围内启动唯一完整A-smoke600，并启用监督进程。
- [ ] 监督初始化、首批真实更新与TensorBoard，随后检查第200/400条checkpoint及原健康保护规则。
- [ ] 实验结束后归档并独立验收；性能结论仍须冻结开发选模与综合baseline评估。

> **2026-09-06 当前状态：`A_PROBE12_ENGINEERING_PASS_A_SMOKE600_AUTHORIZED_PREPARING_NOT_STARTED`。**
> 完整A-probe12已正常完成并独立验收：3批/12条轨迹、34次真实Text评分、3条有效Graph轨迹、10/12格式有效、1条replay；原StrongSFT、显式SFT reference与真实ReaRAG均参与。256/256 LoRA张量及从零初始化的value head确有更新，参数/loss/KL均有限。
> TensorBoard实际606 scalar/133 histogram，314项对照通过；α六维特征在第2批出现。PyTorch峰值allocated69.55585GiB，监督期间整卡最高观测约78.743GiB。12份训练文件共126,619,933字节已下载且全部SHA匹配。[执行归档与日志/TB入口](ppo_a_probe12_supervision_20260906.md)、[独立验收](../outputs/audits/ppo_a_probe12_independent_acceptance_20260906_v1/report.json)。
> 研究者已明确回复“进入”，授权下一版完整A-smoke600；当前准备中，**尚未启动600**。probe的在线EM/F1不等于综合baseline结果，原fresh132健康FAIL保留。下方未启动/未授权待办均按历史快照保留。

- [x] 完整A-probe12正常exit0，确认PPO actor/critic更新、replay、真实Text/Graph/α覆盖。
- [x] 下载并逐项核验全部12份训练产物SHA，归档日志、TensorBoard与独立CPU参数验收。
- [x] 获得研究者对下一版完整A-smoke600的明确授权。
- [ ] 完成独立A-smoke600配置/scope/发布及远端检查；准备完成后按已授权范围执行并监督。
- [ ] A-smoke600完成后按预先冻结协议验收；综合baseline与正式full预算仍须各自完成前置，不从probe推断效果。

> **2026-09-06 开卡前状态：`REMOTE_CPU_READY_SCOPED_A_PROBE12_TENSORBOARD_VERIFIED_PPO_NOT_STARTED`。**
> 原实例无卡SSH已恢复；TensorBoard已恢复`/root/tf-logs:6007`，旧单图为连通性诊断。新版监控已同步，本地/远端各16项事件测试通过，短probe α histogram已补齐。
> HTTP实际可读零更新缓存诊断257 scalar/98 histogram；没有新PPO训练结果。限额A-probe12及资产闭包已完成；远端89项/37.48GB SHA通过，108项scope/v2 runtime测试及真实模型路径检查通过。可开原96GB GPU跑probe；原健康FAIL与600未放行保留。[当前远端入口](ppo_remote_prelaunch_20260906.md)。

- [x] 恢复无卡远端连接与官方TensorBoard服务，核验实际日志根和HTTP标签。
- [x] 同步新版遥测并在远端复验16项事件测试；保留历史日志与旧release。
- [x] 完成来源/模型资产绑定、限额A-probe12入口的远端CPU复验；预算和关键配置覆盖均拒绝。
- [ ] 切换原96GB GPU模式，执行完整A-probe12并检查真实梯度、checkpoint、replay及TensorBoard。
- [ ] 后续另版修复scope前置分支对旧source-credit-v1通用scope的误识别；当前release限A-v2-probe12，旧v1复现沿用原release，不影响本轮已验收入口。

> **2026-09-06 最新结果：`FRESH132_COMPLETE_UTILITY_PASS_HEALTH_FAIL_PROBE_ELIGIBLE_PPO_NOT_STARTED`。**
> 660条生成、评分和最终分析全部完成；额外19项测试后另版修复Gold前分析范围错误，原失败与所有资产保留，未重跑GPU。
> 独立效用PASS、健康/整体FAIL；probe资格true、matched600自动放行false。图96过程top1 EM A=F84.38%、T66.67%、greedy83.33%；动态α额外优势尚未显示，不能冒称PPO成绩。
> sampled valid89.20%、三域macro82.64%未达90%，greedy98.48%；sourcePASS79的39个mixed family：A pairwise94.87%、F97.01%、T48.93%。[权威报告](../outputs/audits/source_credit_v2_fresh_confirmation_local4090_20260906_v1/analysis_recovery_v1/report.json)、[完整结果与边界](source_credit_v2_fresh_confirmation_execution_20260906.md)。

- [x] 完成660条生成、1927次真实Text评分、Gold前完整复验及原规则下的Gold分析。
- [x] 独立核验封存SHA、候选/题级指标及分层聚合；保留整体与ordinary负向结果。
- [ ] 从原训练gate/800身份/671信用mask派生绑定此次效用确认的限额probe版本；不能用fresh96 mask替换训练mask。
- [ ] 原远端恢复连接后运行完整A-probe12；根据实际工程结果和本次健康FAIL另行明确600预算决定。

> 生产gate尚未放行，SFT/PPO更新0；新版SFT继续暂停。下方“正在运行/恢复待定”等为历史记录。

> **2026-09-06 14:11 收尾更新：`GENERATION_SCORING_COMPLETE_ANALYSIS_RECOVERY_PENDING_PPO_NOT_STARTED`。**
> 660条生成和评分、1927步真实Text已完成。首次分析在Gold前误拒39个invalid的合法−1 Graph诊断值；实际过程分/α仍0、排名仍排除invalid。
> 只改单点校验范围后的真实Gold-free132/660复验PASS，另版恢复执行正在冻结；原失败与所有绑定保留，无奖励/排序/阈值变动、不重跑GPU。
> 当前独立效用与PPO资格尚未产生；[诊断](../outputs/audits/source_credit_v2_fresh_confirmation_range_diagnosis_20260906_v1/report.json)和[执行记录](source_credit_v2_fresh_confirmation_execution_20260906.md)为当前依据。

> **2026-09-06 13:00 最新执行：`FRESH132_CONFIRMATION_PIPELINE_RUNNING_PPO_NOT_STARTED`。**
> 用户已批准推进现有Strong SFT；145项相关新测试通过后，固定fresh132/K4+greedy共660条本地4090流水线已启动。顺序为生成→单次真实ReaRAG评分→排序封存→Gold独立判定。
> 当前入口：[fresh132执行与监控](source_credit_v2_fresh_confirmation_execution_20260906.md)；协议SHA `d1bf882764c825e8e14de9a83e7219ed68ea83b8c480096e23c952f260b8fc97`。
> 状态与进度在 `outputs/audits/source_credit_v2_fresh_confirmation_local4090_20260906_v1`；PID3077412。确认结果待产生，SFT/PPO参数更新仍0，teacher调用0，原gate未放行。
> 健康目标与独立效用分开，信息不足保留INCONCLUSIVE，不补抽到通过。后续按结果推进限额完整A-probe12，再决定matched600。远端目前SSH握手关闭，须恢复原环境连接。

- [x] 冻结生成、统计、三态判定、模型/代码/来源与原标签源字节SHA。
- [x] 新生成/评分/分析流水线及断点恢复；145项测试通过，132/96真实输入对齐通过。
- [x] 启动本地固定660条确认任务。
- [ ] 完成独立确认并复核实际效用/健康结果；从原训练gate另版派生，保留原800/671来源mask，禁止使用fresh96确认mask替换训练mask。
- [ ] 在原大显存远端完成限额完整A-probe12及相应matched600；不自动启动12000全量训练。

> **2026-09-06 最新执行：`EXISTING_STRONG_SFT_PPO_PRIORITY_SFT_EXPANSION_PAUSED`。**
> 用户优先要现有 Strong SFT 的真实 PPO 证据；新版 SFT 不是硬前置。本轮已停止 SFT 检索/自动组装，保留产物，teacher API 0；预算提案暂缓。
> 当前入口：[PPO 优先决定](existing_strong_sft_ppo_priority_20260906.md)。下一步是固定 fresh132 确认，再完整 A-probe12 和 matched smoke600；原 gate 仍 false，PPO 未训练。
> 下方新版 SFT“正在运行/待预算”条目保留准备历史，当前不执行。重要结论依真实评估和消融，首版不要求预先承诺涨分。

> **2026-09-06 最新执行：`SFT_V3_DATA_PREPARATION_IN_PROGRESS_NOT_TRAINED`。**
> 用户已要求准备新版 SFT 全部训练数据，选择同一基座新建 LoRA。原 Strong SFT 保留作比较，后续 A/F/T 必须共用选定继任及重新校准的 α。
> 随机候选 16,500（每域 train5000/validation500）已冻结；独立图补充 60 后组合为 16,560。目标成品 6000+300，当前不能宣称成品数据 ready。
> 当前入口：[新版 SFT 数据准备](sft_v3_data_preparation_20260906.md)。下方“当前不重训 SFT”等为历史记录。

- [x] 冻结全局 question/family 保护账本，排除正式 PPO3000、α 开发/确认与原 SFT 留出集。
- [x] 冻结三域原始 train 候选、生成前的 family 划分和 checker-only 原始答案；7/7 身份门通过。
- [x] 新版严格数据 contract、盲 teacher/盲 reviewer 和带费用上限的 API WAL 完成模块测试。
- [x] 新 loader 固定 10 passages、2–5 步、原 teacher target、assistant-only mask、含 EOT≤384；超预算拒收，不裁剪补格式。
- [x] 复核非空 KG 供给：335 候选中 281 来源 PASS；去除 family 冲突后最多 229 个可用输入，不等于 229 条合格 teacher 轨迹。
- [x] 首批 128 题 canonical 检索与真实 token 检查通过，全部 10 passages、无 fallback，运行约 279 秒。
- [x] 冻结 16,560 组合与 192 个真实就绪输入快照（每域 64，含 3 个图题），标签独立保存。
- [ ] 剩余本地检索与所有输入绑定（后台运行；自动组装进程不调用 API）。
- [ ] 确认教师 API 总费用上限后生成并逐步核查；当前没有付费调用。
- [ ] 全量格式、答案、语义审核、非空 KG 引用、token、身份与来源独立复验；按实际合格数封版，不用重复题凑数。
- [ ] 数据完成后另行冻结训练与 checkpoint 选择协议；从基座新建 LoRA，当前不启动训练。

> **2026-09-06 当前执行：`STAGED_REPAIRS_VALIDATED_EVIDENCE_V1_NOT_ADOPTED_PPO_NOT_STARTED`。**
> 用户已批准上一轮按依据执行；当前[执行记录](evidence_and_objective_execution_20260906.md)覆盖下方历史“尚未批准”状态。
> [x] 保守shortfall-only奖励v2与240缓存组件核验；[x] 新A/F/T probe配置和答案/格式/全量canonical TensorBoard拆分。
> [x] 真实20题排序未复现失序；[x] 60次Gold-free扩展＋40条Reader完成；[x] 主指标独立复算。EM4→5/40、F120.51%→19.52%，不采用供给v1。
> [x] 350项测试与780次真实缓存生产奖励复验；[ ] 优先检验“答案正确但归属错误”的过程分敏感性及最低步数规则，再有条件做身份隔离的跨域监督可行性验证。
> 原数据/Gold/baseline与fresh132保留，独立确认门未放行，PPO/SFT更新均0。下面prompt-v1结果为已完成上一阶段，仍不采用。

> 状态日期：2026-09-06
> **当前生效：`TRAINING_FORMAT_PROMPT_V1_PROBE_COMPLETE_NOT_ADOPTED_PPO_NOT_STARTED`。**
> 60题/K2真实对照完成：格式80.00%→91.67%，全量EM38.33%→35.00%、F143.40%→37.82%，输出tokens+12.87%；不据格式提高直接采用新提示。
> 44项相关测试通过；EM/F1差值区间跨0，结果仅为已消费train开发诊断，不能当独立确认或PPO收益。
> 当前格式修复入口：[训练提示验证](training_format_prompt_v1_20260906.md)。原baseline、奖励、α及PPO配置保持冻结。
> 最新只读[策略复核](pre_ppo_strategy_review_20260906.md)：保留主框架，先复核证据供给与答案/格式目标的关系，再决定局部奖励消融和定向补监督；尚未改变正式实验协议或批准实施新reward。
> 下一阶段准备完成：60题/K2长度对照120/120前缀一致，valid80.00%→81.67%，只恢复2条；正式配置仍384。
> 新确认132题输入已冻结：96图题独立于正式3000、36ordinary仅α开发未见；图信用79PASS/11UNVERIFIED/6FAIL，缺inference，失败不替换。
> 当前入口：[下一阶段执行记录](source_credit_v2_next_stage_20260906.md)。尚未生成确认候选或完成独立过程效用；PPO仍未启动。
> 下方v2修复为已完成的上一阶段结果：
> Text逐step统计与softsign、六维α特征、代表性H40/W40/M40补充银行已完成；240候选中192有效/654steps，48无效保留。
> 最终Text μ=.6359942727、s=.2092371033；旧来源mask冻结，800原有图题中仍671获信用、129题α_eff=0，输入KG未改写。
> 465项集成回归通过，最终192次cached真实奖励检查及两组全1660候选特征核对通过；TensorBoard六维与软饱和指标读回通过。
> 同114对过程排序：A/F=90.35%/88.60%，优势仅2对，N-only与N+F该项相同；没有新增特征额外排序收益或独立确认。
> 当前报告：[source-credit-v2修复记录](source_credit_v2_repair_20260906.md)。原候选EM/F1仍为55.12%/61.54%，不是PPO成绩。
> 新版gate的独立confirmation与训练放行均false；18份新配置已准备，本轮没有PPO更新或远端同步。
> [v1记录](source_credit_v1_repair_20260905.md)保留原9/9拟合检查、A=F 83.33%和Text硬裁剪49.94%的历史结果。
> 用户此前要求本地从零重生成：
> 远端生成/调度已停止，328条历史候选保留；本地4090新运行 `sourcegate_local_regenerate_4090_20260905_v1`。
> 本地不复用远端候选，监控见 `docs/sourcegate_local_regenerate_20260905_v1.md`；以下较早远端状态已被本条替代。
> 历史GPU执行状态（已替代）：`GPU_PREPARATION_RUNNING_PPO_NOT_STARTED`；当时96GB卡已启用，
> 后台运行 `sourcegate_a_gpu_20260905_v1` 正在执行候选生成→评分→α校准，日志命令见
> `docs/sourcegate_gpu_launch_20260905_v1.md`。PPO仍需完成既定的校准/过程效用检查。
> 本地节费可行性：4090 24GB 沙箱外CUDA/BF16正常；真实最长候选生成probe通过（peak reserved15.96GiB）。
> 当时尚未迁移、暂停远端；此后已按用户要求停止远端，在本地从零重生成；本地BF16 ReaRAG真实评分也已完成。
> 当前优先执行：**冻结 Strong SFT → 新 α 校准 → 答案 + 过程 + α 完整 PPO-A → 综合 baseline EM/F1**；
> PPO-F/T 用作 matched 消融。旧 mixed3 PPO-T/PPO-TK 只作为固定加法的工程底座与
> 历史对照，不代表新的动态门控方法。
> 本文件是当前执行入口；`docs/todo.md` 与 `docs/retraining_plan.md` 保留完整历史和失败记录，
> 不再作为“下一步做什么”的入口。

> **2026-09-05 执行更新：** 主 PPO 问题池已由历史 `600/600/599` 封版为均衡的
> HotpotQA/2Wiki/MuSiQue=`1000/1000/1000`。2Wiki 主集冻结目标为 `800` 个严格通过来源门的
> ProofKG 样本 + `200` 个普通 passage/outcome-only 样本；K=4 后总计 `12,000` 条 rollout
> 轨迹。2Wiki 先冻结 `1,500` 个 Gold-free 候选，再从 `1,287` 个 strict eligible 中选出 `800` 个 strict ProofKG；
> 未入选候选只作失败余量或 graph-heavy 消融，不加入主集，避免把数据集过采样
> 与方法收益混为一谈。final v4 data已完成且preflight=`PASS_NOT_TRAINED`。当前总状态为
> `DATA_READY_PPO_NOT_STARTED`。研究者随后已授权“开始修改”，并优先追求综合 baseline 下的 EM/F1；
> 最新澄清是全部奖励因素和 α 一起进入主模型，不能把 O-only 当主线。代码、环境和数据已同步，本地α候选生成已完成。

### 当前优先任务：完整 PPO-A（2026-09-05 研究者澄清，覆盖此前 O-first 误读）

- [x] 按 AutoDL 官方 `/root/tf-logs` / 6007 配置 TensorBoard：实验隔离、每batch刷新、PPO统计、
  分层 EM/F1/奖励、eligible-valid α 分布、归一化裁剪、replay、显存与吞吐；
  使用说明与独立验证记录见 `docs/ppo_tensorboard_autodl_20260905.md`，CPU诊断不代表PPO已开始。
- [x] 实现 `runtime_contract_version=v2`：生成期间关闭 dropout、保留真实 EOS、禁止奖励长度静默裁剪。
- [x] 保留全部输出后检查最多 5 步、唯一 Final Answer、每步三个字段各出现一次。
- [x] 可选 O 消融已经准备；它不是完整 A 方法的必跑前置，旧产物保留。
- [x] A/F/T 各三阶段配置：probe12 / smoke600 / full12000；γ=1、λ=.99、lr=1e-6、显式 SFT reference、10% replay。
- [x] `ppo_max_passages=15` 保留原 replay 渲染；v4 rollout 本身仍只有 10 passages。
- [x] 新建保守 Proof scorer v2.3 并保留 v2.1/v2.2；去除未验证语义的加分，禁止子串答案匹配；
  它仍无法区分所有伪造推理与正确推理，必须在真实候选上检查排序，不声称语义验证已解决。
- [x] 接通新版 trajectory α、hard mask、passage-only ReaRAG、Graph/Text 冻结统计；旧 α 不可直接加载。
- [x] 冻结 Gold-free train-only candidate 输入：830题/758families，K2=1660已生成，最长2399tokens。
- [x] 停止远端准备任务并保留328条历史候选，启动本地4090从零生成1660条候选（独立运行ID与目录）。
- [x] 完成本地1660条候选生成和CPU独立质量审计：来源/身份/EOS核验通过，6项新统计测试通过。
- [x] 空Final训练端format-v2修复：仅1条旧误判转为非法；原候选/旧格式/评价协议不改。
- [x] 完成来源分歧复核并建立Gold-free source-credit-v1：671图获信用；129题保留输入并关闭Graph奖励。旧KG未获完整性认证。
- [x] 完成1660条真实ReaRAG评分、4517有效steps、新gate拟合9/9通过、9份A/F/T配置解析通过。
- [x] 生产runtime真实缓存分数零更新检查：32条×A/F/T，96/96通过；不等于GPU PPO probe。
- [x] 正式PPO loader全人群9/9通过：3000题中671题、12000 schedule中2684条可获图信用，830个bank成员五项输入完全匹配。
- [x] 留出与同题特征诊断：新信用人口114对中94对四维特征相同，A=F 83.33%，1改善/1变差；未放宽既定门槛。
- [x] 建立Text normalization-v2：逐step拟合、问题/候选/step等权、固定softsign；硬裁剪与软饱和/尾部遥测分开。
- [x] H40/W40/M40归一化train补充银行240条完成：146新生成+94精确复用，192有效/654steps；身份选择及来源完整重放、holdout/保护账本重叠0。
- [x] α新增来源边覆盖率、最差步引用精确率；train307对相同特征251→160，重复引用/语义反事实边界已测，未用Gold选特征。
- [x] 冻结N-only/N+F两组继任与18份A/F/T配置；代表性Text重绑定保持α/Graph/mask及主候选不变。
- [x] 最终192/192真实缓存token奖励复算、两组各1660特征全量对齐、真实TensorBoard事件读回通过；465集成回归通过，追加相关测试单独记录。
- [x] 完成固定后验reward-top1/oracle诊断：114图pair的A/F=90.35%/88.60%，N-only=N+F；开发重分析不能冒充独立确认。
- [x] 审查剩余189格式失败：81长度上限、108主动EOS，无新EOS裁剪bug；未放宽validator、未统一提高min_length。
- [x] 完成60题/K2的384→512真实配对probe：120/120前缀一致，valid96→98/120，cap6→1，token成本+1.35%，原预算不变。
- [x] 冻结训练专用prompt-v1及固定60题配对协议：最低步数原规则2步9题/3步51题、user/证据不变；不覆盖全局提示或放宽validator。
- [x] 完成prompt-v1本地120条生成与全量EM/F1、格式、重复独立评估：有效110/120，但EM/F1点估计下降；不提升为正式PPO提示，新132题未用于选择。
- [x] 核查封版replay目标的现format-v2健康：2000/2000通过，原始行/目标/train-fold SHA一致，仅Hotpot；未生成筛选集或启动训练。
- [x] 完成答案/格式目标只读诊断：原8条只因最低步数无效，其中4条答对；当前答案项＋无效罚分均值上升不能替代raw EM/F1。
- [x] 完成已消费60题词面诊断与20题MuSiQue暂定证据审阅；仅为开发定位，未修改Gold或把单agent判断当正式标签。
- [ ] 复核明确证据缺口，准备保持资源预算的Gold-free供给修订提案；另行审查答案信号与格式惩罚分离的单变量reward消融，现规则保持冻结。
- [ ] 若诊断支持SFT适配，再冻结独立短程SFT的样本、prompt、更新/生成预算及评估对照；原Strong SFT不覆盖，继任具备自己的SFT评估与A/F/T共同起点。
- [x] 冻结132题新确认身份与输入：96图在三种可用类型中各32、ordinary H/M/W各12；每题10passages、max2262tokens，scope_v2明确独立性范围。
- [x] 离线物化新source evidence/mask及两组确认专用门：79PASS/11UNVERIFIED/6FAIL，原数值不变；13测试与输出SHA复核通过，未生成确认候选。
- [ ] 冻结新确认generation与分析协议，再实际生成/评分固定132题；过程重排不得包含Gold outcome，统计按family聚类，信息不足不得事后补抽到通过。
- [ ] 完成独立过程效用confirmation及新候选生成健康验证；原银行valid88.61%、代表性补充80%人群不同，均不等于格式问题解决。新版gate默认拒绝未确认PPO加载。
- [x] 完成独立远端 release：308 CPU tests通过，972文件完整SHA一致；发布时远端为无卡模式。
- [ ] 校准与零更新门通过后执行完整 A-probe12（3 个 K4 groups、一次 replay），再进入 A-smoke600。
- [x] 为 A 主模型独立冻结 development150 银行与 SFT/step200/step400/final600 registry；
- [ ] legacy/no-graph 两视图实际运行比较；
  主规则为 macro EM → macro F1 → 同分优先 SFT/更早 checkpoint，允许最终仍选择 SFT。
- [ ] smoke 后决定是否正式训练；full 在开始前单独冻结候选 registry，不根据 canonical900 追选。
- [ ] 模型选定后按冻结 Scheme A pipeline 做 canonical 三数据集综合评估；开发代理不是主表。

主线实施说明：`docs/ppo_sourcegate_execution_20260905_v1.md`；`ppo_emf1_execution_20260905_v1.md`
只保留 O 消融准备记录。共同 runtime 修复是组合工程变化，α 效果需 A−F matched 对照，未产生训练结果。

## 0. 已作出的方向决定

- [x] Reader 保持一次性回答原始多跳问题，不承担 search action、query controller 或子问题交互。
- [x] 输出协议固定为：

  ```text
  [Step 1]
  Reasoning: ...
  Knowledge Used: [(head, relation, tail), ...]
  Conclusion: ...

  [Step 2]
  Reasoning: ...
  Knowledge Used: ...
  Conclusion: ...

  [Final Answer]
  ...
  ```

- [x] 当前不重训 SFT；使用已验证的 Strong SFT 作为所有 PPO 臂的同一起点。
- [x] Controller、动态子问题拆解、QPEG、SAEG 不进入当前方法、数据或主表；产物保留为
  `PAUSED/FAIL_STOP` 历史研究分支。
- [x] 2Wiki ProofKG 的 `0.6400 EM / 0.6944 F1` 继续作为“额外 Wikidata 资源增强”结果，
  不与同资源 legacy 主表混算。
- [x] 后续方法研究问题为：**在固定 Reader、固定证据供给和固定训练预算下，学习型 α 能否在
  高质量图可用时偏向 KG、图不可靠或缺失时偏向 passage/ReaRAG，并使 PPO 提升 EM/F1/IHR？**

## 1. 当前可复用资产

### 1.1 Strong SFT 起点

- [x] Checkpoint：
  `checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final`
- [x] Manifest：`COMPLETE`；adapter 大小 `109,086,416 B`，MD5
  `6cf328bc9f634e0c2e0a8d355cf8e43d`。
- [x] LoRA：`r=32`、`alpha=64`、`dropout=0.05`、`q/k/v/o_proj`。
- [x] 实际 SFT train fold：4,751 条 accepted HotpotQA trajectories；1 epoch。
- [x] 标准 legacy pipeline（n=300，seed=42，greedy）：
  HotpotQA `0.3833/0.4898`、2Wiki `0.4267/0.4846`、MuSiQue
  `0.2467/0.3419`（EM/F1）。
- [x] 当前结论：Strong SFT 是冻结起点，不做 continued-SFT；只有完整性或格式回归失败时才重新训练。

### 1.2 已验证的 PPO 工程底座

- [x] 显式冻结 SFT reference，避免 KL 错锚到裸 base。
- [x] critic zero-init、value dropout=0、`ppo_epochs=2`。
- [x] `max_new_tokens=384`，历史 combined/hybrid smoke 的截断与 valid-rate 已恢复健康。
- [x] 保留 10% SFT replay，历史运行实际交付比例为 0.10。
- [x] 稳定配置候选：`lr=1e-6`、batch=4、mini-batch=1、`kl_coef=0.25`、
  `target_kl=8`、历史 `gamma=.95/lambda=.95`；新 PPO-O 单独升为 `gamma=1/lambda=.99`。
- [x] 历史 hybrid smoke 在 replay 上追平 SFT，但标准 pipeline 未超过 SFT；因此“训练不再崩”已解决，
  “奖励能否产生正增益”尚未解决。

### 1.3 可复用的三数据集 PPO 身份与调度

- [x] 数据目录：
  `data/silver_data/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42/`
- [x] 1,799 unique qids：HotpotQA 600、2Wiki 600、MuSiQue 599。
- [x] 1,800 prompt groups × K=4 = 7,200 scheduled trajectories；三数据集各 600 groups。
- [x] 每题有 10 passages、gold outcome label，source steps 已清空，适合作为 rollout 问题池。
- [x] 可复用范围仅限：K=4 工程机制，以及经 complete ledger 重新确认安全的父行/passages；v4 的
  identities、source records、sampling weights 与 12,000 schedule 必须重新冻结，不能原样复用 v3。
- [x] 该分布正好对应双源门控场景：同一个 record-level quality gate 的观测结果是 400 题有完整
  ProofKG、1,399 题没有合格图；前者当前恰好都来自 2Wiki，后者由 passage/ReaRAG 分支承担。
- [ ] **不能直接启动的原因**：现有 PPO-T/PPO-TK 只比较“是否额外加固定 Proof reward”，并强制
  `alpha_gate_path=null`；需要新增独立 successor 才能测试 α。

本轮新增的 append-only 数据 release：

- [x] `data/silver_data/mixed_ppo_three_dataset_v3c_source_gated_proof400_n1799_k4_seed42/`
  已复制并逐文件核对父版五个冻结资产，新增 1,799 条 `source_gate_records.jsonl`；
- [x] 同一数据无关硬门得到 HotpotQA `0/600`、2Wiki `400/600`、MuSiQue `0/599` 个
  `m_graph=1`，全部 1,799 题仍有可用 passages/ReaRAG；
- [x] 历史 `data/silver_data/sft_replay_legacy_strong_train_rendered3to5_n2000_seed42_v1c/`
  曾提供 2,000 条 replay，但 complete ledger 复核发现 138 条 identity/family overlap，现已 supersede；
- [x] 权威 replay 为
  `data/silver_data/sft_replay_legacy_strong_train_rendered3to5_n2000_seed42_v2/`：3/4/5 步分别
  `1593/336/71`，与 ledger/HM 三重零重叠，真实 PPO loader/tokenizer `2000/2000` 通过；
- [x] 独立 CPU/tokenizer preflight 全部通过，证据保存在
  `outputs/audits/source_gated_mixed3_v3c_preflight_v2/`；
- [x] 首次物化因二进制 git diff 被错误按 UTF-8 解码而在 manifest 阶段失败；失败目录保留并写入
  `FAILED_MATERIALIZATION.json`；随后发现 `v3b/v1b` 与失败尝试共用 Experiment ID，亦保留并标为
  `SUPERSEDED_NOT_FOR_TRAINING`。v3c 只保留为历史数据底座；v1c 已被 replay-v2 supersede，当前 v4
  final v4已另行封版；v3c仍只保留为历史数据底座，不能替代当前v4或冒充α实验。

当前 `mixed3` 不是已经包含 α 的最终配置，而是一套已冻结的数据与调度骨架：

| 项目 | 当前已冻结内容 |
|---|---|
| 问题池 | 1,799 unique qids；HotpotQA 600 / 2Wiki 600 / MuSiQue 599 |
| 训练调度 | 1,800 prompt groups × K=4 = 7,200 rollouts |
| 文本证据 | 每题 10 passages；所有题均可使用 ReaRAG 文本过程分 |
| 图证据 | 400 个 complete 2Wiki ProofKG；其余 1,399 个显式空图 |
| 旧 PPO-T | outcome + ReaRAG；`proofkg_process_reward=false`；α 为空 |
| 旧 PPO-TK | 与 PPO-T 只差 `proofkg_process_reward=true`，固定加权 `0.20`；α 仍为空 |
| 新主线 | 复用相同问题池、证据和调度，新增 `PPO-F/PPO-A`，用图有效性 mask 和 α 做双源选择 |

旧配置分别是
`configs/training/phase3_ppo_mixed3_rearag_v2_proof400_text7200_seed42.yaml` 与
`configs/training/phase3_ppo_mixed3_rearag_v2_proof400_text_kg_v2_1_7200_seed42.yaml`。
它们保留作历史固定加法消融，不能改名冒充 α 实验。

### 1.3.1 v4 均衡扩容（数据封版完成，PPO未启动）

- [x] 冻结主实验比例：HotpotQA/2Wiki/MuSiQue 各 `1,000` 个 prompt groups；K=4，目标
  `12,000` 条轨迹。
- [x] 冻结 2Wiki 组成：严格 ProofKG=`800`、ordinary=`200`，并要求四类问题各 `200`
  个 strict ProofKG；不得为了凑数降低 hard graph/source gate。
- [x] 冻结“候选池与训练暴露分离”：先从 official raw 冻结约 `1,500` 个 Gold-free、
  evaluation/family 隔离的 2Wiki 候选，最终主 schedule 仍只消费其中 `800` 个 strict ProofKG。
- [x] HotpotQA 新增 `417`、MuSiQue 新增 `401` 的身份与 family 隔离 cohort 已冻结；旧样本分别
  保留 `583/599`。
- [x] 2Wiki 新增主 cohort=`300`，另冻 reserve=`50`；reserve 不改变主集 1:1:1 比例。
- [x] 完整 protected identity ledger v2 已冻结并接入所有 final gate：`4,690` qid、`3,631`
  current families；旧的不完整保护清单不再允许正式消费。
- [x] official-raw、Gold-free 2Wiki 候选池 `n=1,500` 已冻结：
  bridge/comparison/compositional/inference=`390/390/389/331`；它只供择优，不改变 Proof800 主配额。
- [x] clean Strong-SFT replay v2 已冻结 `n=2,000`，与 protected ledger 和 v4 H/M population 的
  qid/question/current-family overlap 均为 0，真实 PPO loader/tokenizer dry-run `2000/2000` 通过。
- [x] 2Wiki ordinary200 successor 已冻结：保留 `148`、替换 `52`；与 ledger、replay 和所有 Proof
  候选池三重零重叠。
- [x] 历史中间版 H/M `818` 题 retrieval 已完成但含需退休身份；最终 successor 已在下方封版为
  823 rows，不得再消费旧818 release。
- [ ] 2Wiki 新 cohort 的 clean-store 闭包首轮只得到 `47/300` complete，已定位为根实体解析
  覆盖不足（`65/300` resolved），该结果作为负诊断保留。Gold-free 定向根解析后的 closure-v2
  已收敛至 nonempty=`241/300`、complete=`222/300`，但 anchor resolved=`232/300=0.773`
  仍低于预注册 `0.80`，因此结果保持 `FAIL_DIAGNOSTIC_STRUCTURE_RETAINED`，不得进入 Proof800。
- [x] H/M replacement retrieval 已封版为 823-row release：HotpotQA=`417`、MuSiQue=`406`，
  复用 `812`、新检索 `11`、退休污染记录 `6`；每题 10 passages，BGE/CUDA、无 fallback，
  final materializer 兼容门通过。
- [x] 新 2Wiki n1500 候选的 Gold-free planner 已完成：`1500/1500` 有输出、`1499/1500`
  schema valid、identity join=1、Gold violation=0、runtime error=0；唯一 invalid 原样保留。
- [x] n300 root projection 偏差已定位：旧协议错误地把另一 resolver 栈的 resolved bit 继承进
  clean consumer，导致预估 `281/300`、实际 `232/300`；并非 closure join/cache bug。旧失败结果
  保留，后续 n1500 必须对全部 roots 重解析并强制 projection=dry-run=runtime。
- [x] strict Proof800 的 answer-blind selector-v1 实现与预注册协议已完成：固定四类各 `200`
  的硬配额，并逐题复核 identity/current-family、完整 roots/hops、可追溯边、source gate、cutoff、
  10 passages 及其 hash；任一类型不足即 fail，不降低门。其 closure schema 仍绑定v2，而权威
  closure为v3，故v1保留但不可执行；selector-v2 P0REFRESH2现为权威完成版本。
- [x] n1500 全根实体解析通过：2116/2279 occurrences，1337/1500 all-root questions，
  projection/dry-run mismatch=0，Gold=false，runtime fail=0。
- [x] closure-v3已完成并通过：两轮物化后收敛，nonempty=`1365/1500=0.910`、
  complete/strict eligible=`1287/1500=0.858`、runtime error=0、max triples=7；四类eligible为
  `361/353/317/256`，均满足每类≥200；权威结果位于
  `outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_clean_closure_v3_result/`。
- [x] closure 结果前已冻结 official-raw canonical retrieval 的范围规则：只检索**全部且仅**通过
  answer-free strict structural predicate 的候选，而不是全1500或预先挑出的800；四类任一少于200即
  fail。scope policy 位于
  `outputs/audits/2wiki_official_raw_canonical_retrieval_v1_scope_policy_preregistration/`；正式exact scope
  已冻结为1287题，位于`outputs/audits/2wiki_official_raw_canonical_retrieval_v1_scope_preregistration/`，
  且使用独立schema/status，禁止冒用reserve50 release。
- [x] official-raw retrieval materializer、unified-v3 materializer与source contract已实现并冻结；旧
  unified-v2行为未修改。合同位于
  `outputs/audits/2wiki_unified_proofkg_official_raw_v3_contract/unified_contract.json`，SHA256=
  `3950ac3999d077569bdbe9fb08cddcccdb3ce74cfaa448ae9a69688b541c2ba9`。
- [x] canonical retrieval Attempt1在完成3批、第4批开始后未知退出且未形成release；失败记录保留于
  `outputs/audits/2wiki_official_raw_canonical_retrieval_v1_attempt1_failure/report.json`，exit code/root cause
  仍为`UNKNOWN`，无OOM/GPU故障证据。Attempt2已完成41/41批并通过正式validator：1287题、每题10
  passages、BGE load=true/fallback=false；权威目录
  `outputs/audits/2wiki_official_raw_canonical_retrieval_v1/`，contexts SHA=`a935351a...f9b4`。
- [x] unified-v3 candidate supply已完成：1287个silver/QKG/gate/wrapper四路identity join，13/13 checks
  通过；目录`data/derived/2wiki_unified_proofkg_official_raw_v3/`，report SHA=`f0677560...2468`。
- [x] selector P0REFRESH1因silver不存冗余question hash发生schema-adapter失败，0/1287 admitted，无result、
  未训练；append-only记录位于
  `outputs/audits/2wiki_proof800_strict_selection_v2_seed42_result_p0refresh1_failure/report.json`。
  修复只现场重算同一hash，不改quota/seed/ranking/predicate。
- [x] selector P0REFRESH2完成：1287 strict admitted，四类各200，Proof800 qid/hash各800唯一、current
  family=728；目录`outputs/audits/2wiki_proof800_strict_selection_v2_seed42_result_p0refresh2/`，
  proof SHA=`8e58d829...1fc3`，report SHA=`dcad7b6f...410c`。
- [x] final 3000×K4 v4已完成并独立preflight：H/W/M各1000、Graph=800、ordinary=2200、K4=12000、
  scheduled graph=3200、replay=2000；materializer 23/23 gates与preflight 46/46 checks均通过。
  数据目录`data/silver_data/mixed_ppo_three_dataset_v4_proof800_n3000_k4_seed42/`（report SHA
  `375e3b20...eefa`）；preflight目录
  `outputs/audits/mixed_ppo_three_dataset_v4_proof800_n3000_k4_seed42_preflight/`（`PASS_NOT_TRAINED`）。

### 1.4 α/PRM 现状

- [x] 现有五特征 α checkpoint 完整：
  `checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42_no_text_head/alpha_gate.pt`。
- [x] Hotpot held-out 1,886 steps 上，hard-verdict 校准 Brier=`0.0620`、
  R²-vs-constant=`0.5109`、α mean/std=`0.1180/0.2174`。
- [x] 但历史 hybrid PPO rollout 上 α mean≈`0.731`，且最终只追平 SFT；说明离线校准不等于 PPO 效用，
  并存在 teacher→policy、Hotpot→三数据集的分布差异。
- [x] Phase2 三分类 PRM head 当前从未进入 PPO reward path；**不需要重训 PRM head**。
- [ ] 需要新训/重校准的是轻量 α gate，而不是 8B PRM adapter。旧 α checkpoint 的 target、特征来源和
  粒度均与新双源问题不同，只能作为历史基线，不能直接装入新主实验。

## 2. 正式主实验定义

### 2.1 双源供给与公平比较边界

matched 训练比较中的 Strong SFT 与全部 PPO 模型必须共享：

- 同一 SFT 起点；
- 同一三数据集 train qid/family 与 rollout schedule；
- 同一 Wiki18 corpus、retrieval snapshot、passages 顺序和数量；
- 同一 source state：v4 主实验中 800 个 2Wiki groups 使用同一份通过 hard gate 的 ProofKG，
  其余 2,200 个 groups 使用同一 ordinary/empty-graph 状态；
- 同一 prompt、固定 Step schema 与 passage evidence；
- 同一 decoding、scorer、checkpoint 选择规则和训练预算。

不得按模型改变 evidence：不能只给 PPO-A 更好的图。标准 legacy baseline 主表另行在所有 checkpoint 上
使用同一 legacy pipeline fresh 评估；它与 source-adaptive system 表分开报告。QPEG、SAEG 和 Controller
均不进入两张表。

### 2.2 α 的含义与候选公式

α 是**生成之后、计算 PPO reward 时的证据—轨迹信用门**：prompt 仍包含冻结 passages，以及该题若
eligible 时的冻结 ProofKG；α 不改变检索、不增删输入，也不读取数据集名称。它只回答“这条已生成轨迹的
过程信用更应来自 graph 还是 passage/ReaRAG”。

\[
\alpha_i^{\mathrm{eff}}=m_i^{graph}\,\sigma(z_i/\tau),
\]

其中轨迹 `i` 的 `m_graph` 是不可学习的 fail-closed mask：ProofKG
schema/identity/provenance/complete-execution 全部合法时为 1；否则为 0。以当前 mixed3 资产为例，
400 个有合格 ProofKG 的样本可获得非零 α，其余 1,399 个样本自动回到 passage/ReaRAG。当前这 400 个
恰好都来自 2Wiki，是上游供给审计的结果，不是代码写了 `dataset == 2wiki`；未来任何数据集只要通过
同一图质量门，也应获得同样资格。

新版本可以沿用五类信号的概念，但必须重新版本化其来源和聚合：graph density；从冻结 Proof
provenance/execution trace 计算的 entity-link confidence；轨迹 token entropy；cite-any；cite-match。
禁止在新路径里偷偷调用 legacy live EntityLinker。由于当前 ProofKG-v2.1 是轨迹级 reward，本轮 α 也定义为
轨迹级，避免把一个全局分数伪装成逐步监督。

主候选 soft target 衡量 graph 相对 text 的可信度（**提案，尚未冻结或接入训练**）：

\[
y_\alpha=m_{graph}\frac{q_{graph}}{q_{graph}+q_{text}+\epsilon}\in[0,1],
\]

其中 `q_graph,q_text∈[0,1]`。`q_graph` 来自 Gold-free ProofKG 完整性、required-hop coverage、
citation precision/answer consistency；`q_text` 是冻结 ReaRAG 原始分数经 **train-only 固定映射**得到的
非负质量值，不是 PPO 中可能为负的 centered reward。若两者分母退化，则该记录从 α loss 中 abstain 并单独
报告，不能事后赋值。使用 `BCEWithLogitsLoss(logit, y_alpha)`；该 loss 支持 `[0,1]` soft target。

α 的 BCE、Brier、R² 和排序校准只在 `m_graph=1` 的 eligible 集合上计算，避免轻量 gate 仅重复学习“有无
图”；`m_graph=0` 记录只用于 fail-closed 测试并必须满足 `α_eff=0`。精确 target、归一化、聚合和
confirmation 隔离必须在核心 loss 修改前获得批准并冻结。

双源过程奖励恢复为 KG-ProWeight 的自适应混合，而不是旧 mixed3 的固定加法。对轨迹 `i` 的 `n_i` 个
合法步骤，ReaRAG 仍在各 step-end 提供文本信用，ProofKG 的轨迹级分数只放在 final token：

\[
r_{i,t}^{text}=\frac{(1-\alpha_i^{eff})\lambda_T}{n_i}\,\widetilde r_{i,t}^{ReaRAG},
\qquad
r_{i,final}^{graph}=\alpha_i^{eff}\lambda_G R_i^{Proof}.
\]

三臂共享完全相同的文本聚合、token credit placement、outcome 与 validity；PPO-F/PPO-A 只改变冻结的
α 值。文本分数来自冻结 ReaRAG，并用 causal EMA 的**更新前基线**中心化后裁剪。合法轨迹末步加入
`4*(EM + 0.1*F1)`；非法轨迹为 `-4`。公式、缩放和 mask 在实现前必须形成新版本并冻结，不得覆盖历史
reward。

### 2.3 最小充分对照

SFT 是静态基线，不算 PPO 臂。正式训练三臂：

1. `PPO-T`：`α_eff=0`，outcome + ReaRAG text；它是 text-process control，不等同于 outcome-only PPO。
2. `PPO-F`：只有 eligible graph 上使用常数 α；常数取冻结 α-train/calibration eligible 轨迹按正式
   schedule 加权的 learned-α conditional mean，并在 confirmation 前锁定；回答“图过程信号本身是否有效”。
3. `PPO-A`：只把常数换成冻结 learned α；回答“按证据质量动态选择 graph/passage 是否更好”。

决定性差值：`PPO-A − PPO-F` 是 α 的贡献，`PPO-F − PPO-T` 是 KG 过程信号的贡献，
`PPO-A − Strong SFT` 是最终系统训练收益。

历史 `PPO-TK-additive` 是 `PPO-T + 0.20×R_Proof`，不是 PPO-F，不能改名或复用其结果。若 α 是论文
核心主张，正式阶段 T/F/A 必须使用相同完整预算；只给 PPO-F 跑 smoke 时，A−F 只能作为 pilot，不能作为
正式因果结论。

注意：α 选择的是**训练奖励信号来源**，不会凭空构建 KG，也不会在评估时改变 passages/KG 输入。
因此供给层的 400/1,399 划分、训练层的 α 因果消融、推理层的 EM/F1/IHR 必须分别报告。

## 3. P0：正式训练前必须解决

### 3.1 固定 Step 与 replay 一致性

- [ ] 新增/运行 contract test：SFT prompt、RL prompt、eval prompt 的 system/schema hash 一致。
- [x] 核对 replay：当前抽样池 2,000 条中 128 条过滤后只有 1–2 步，与 PPO `min_valid_steps=3`
  冲突；另有 10 条超过 PPO `max_steps=5`。
- [x] 采用统一规则重新派生 replay：只保留 3–5 步、格式合法、Final Answer 可解析的条目；
  保持 10% 目标比例并生成新 manifest。不得修改原 Silver。

### 3.2 mixed3 双源数据完整性

- [x] 复核现有 1,799 个 records 的 `dataset::qid + question_sha256` join=1.0。
- [x] 复核 400 个 eligible 全部满足 complete ProofKG、execution trace、Gold-free graph 构建和 ≤12 triples；
  其余 1,399 个必须显式 empty，禁止静默回退低质量 graph。
- [x] 历史 v3 报告按 source/dataset 的 7,200 schedule：eligible 为 400 groups/1,600 trajectories；
  当前 v4已封版为12,000 schedule、Proof800，对应800 groups/3,200 graph trajectories。
- [ ] 新建 α successor config/lock，不覆盖现有 Proof400 data、PPO-T/PPO-TK 配置或 manifest。
- [ ] 标准 legacy evaluation 继续使用既有 frozen KG；不得把 mixed3 train qid 与 eval index 强行 join。

### 3.3 α calibration release

- [ ] 后续从 Strong SFT 在 exact v4 PPO prompt 上生成 train-only candidate trajectories；优先覆盖全部
  800 个 graph-eligible qids，K=2，形成约 1,600 个 eligible candidates；另取 passage-only controls 只验证
  `m_graph=0 ⇒ α_eff=0` 和 reward 接线，不把平凡零标签混入 α 拟合或主校准指标。
- [ ] 以 family 做 α-train/calibration 隔离；不得接触正式 eval qids。
- [ ] 离线计算版本化五特征、`R_Proof`、原始/归一化/centered `R_ReaRAG`、soft target、格式合法性和
  provenance；所有特征必须来自同一冻结输入与轨迹。
- [ ] 先检查 soft label 是否真的连续：报告 0、(0,1)、1 三段比例；若 fractional <5%，停止并重审 target，
  不把近似二分类称为连续门控。
- [ ] 训练 3 seeds 的轻量 α gate；PPO 期间冻结。
- [ ] eligible-only 校准门（正式冻结前写入 protocol）：Brier 优于 constant、R²-vs-constant 为正、α 有
  非退化跨度，且 α 随 `q_graph−q_text` 分桶单调；passage-only 只报告 mask 正确率。
- [ ] 当前 eligible 记录全部来自 2Wiki，因此跨 HotpotQA/MuSiQue 的 learned quality-gate 泛化仍为
  `UNKNOWN`；不能用代码无 dataset feature 代替跨数据集证据。

### 3.4 reward 与配置接线

- [ ] mixed PPO 目前明确拒绝 `alpha_gate_path != null`；新增默认关闭的 trajectory-level α successor 路径，
  不改变历史配置行为。
- [ ] `m_graph=0 ⇒ α_eff=0`、invalid/incomplete graph 回退文本、eligible graph、α 聚合均补单测。
- [ ] PPO-T/F/A 解析后的配置除 output、α mode/checkpoint 外必须逐字段一致。
- [ ] ReaRAG 加载 fail-hard；不得回退 dummy/未跟踪 head。
- [ ] 显式 SFT reference、critic、KL、10% replay、固定 schedule 全部进入 lock/manifest。

### 3.5 ProofKG reward P0（当前为硬阻断）

- [x] `proofkg_process_v2.py` temporal 分支当前错误地对已解包 tail 做 `len(t)==3` 并再取 `t[2]`；
  400 个 ProofKG eligible 中 46 个 temporal 问题受影响。
- [x] 保留 `proofkg-process-v2-1-frozen-1` 历史实现，新建
  `proofkg-process-v2-2-frozen-1`；已补 bare/full/ISO date、earlier/later、root 回溯、
  temporal/same-different/terminal multi-value、opaque QID、顺序确定性和 runtime dispatch 测试。
- [x] Proof400 静态审计：v2.1 temporal 可确定 `0/46`；v2.2 可确定 `45/46`，确定部分与事后
  gold surface `45/45` 对齐；冲突日期 1 题 fail closed；400/400 顺序确定。产物：
  `outputs/audits/proofkg_process_v2_2_static_audit_v3/`。
- [ ] 静态派生一致不等于 reward rankability；仍须在冻结候选池 append-only 重排并做新的
  family-disjoint confirmation，不能覆盖旧 v2.1 负结果。

## 4. 零更新门：先证明 reward 值得训练

- [ ] 在冻结 candidate bank 上同时重排 PPO-T/F/A reward，并分别报告 graph-eligible 与 passage-only 子集。
- [ ] 报 valid rate、greedy EM、oracle@K、reward-top1 EM/F1、correct-vs-wrong pairwise、
  Spearman、tie rate，并按数据集分层。
- [ ] 候选门建议为：valid≥0.90、eligible-only pairwise≥0.65、oracle−greedy≥3pp、PPO-A top1 不低于
  greedy，且 PPO-A 不低于 PPO-F；最终阈值必须在看 confirmation 前冻结。此门只决定是否值得训练，
  不能代替训练后的 A−F 统计检验。
- [ ] 若 oracle headroom 不足，归因于 exploration/data；不继续手调 α 权重。
- [ ] 若 pairwise 或 top1 失败，停止 PPO，先修 reward/α；不靠增加 PPO 步数碰运气。

## 5. GPU 连线与 smoke

### 5.1 8-trajectory runtime probe

- [ ] 在 96GB 远程 GPU 上分别跑 PPO-T/F/A 的最小 probe。
- [ ] 核对：同一 policy/reference、K=4 group、reward 分支、α checkpoint、ReaRAG、replay、
  TensorBoard、checkpoint/manifest、无 NaN/OOM。
- [ ] event 文件写入 `/root/tf-logs/<Experiment-ID>/`，至少记录 loss、KL、reward 分量、α、
  valid、critic EV、clip fraction、长度截断和 replay ratio。

### 5.2 600-trajectory matched smoke

- [ ] 三臂必须使用同一 schedule/seed/预算；不得只延长看起来更好的臂。
- [ ] 健康门：valid≥0.95、length-capped≤0.05、KL 有限且无持续越过健康阈值、critic EV 不持续恶化、
  replay 实际比例=`0.10±0.01`。
- [ ] 效用门：PPO-A replay EM 相对 SFT 退化不超过 1pp；标准三数据集 macro EM/F1 不低于 PPO-F，
  至少 2/3 数据集方向非负。A−F 的最小有意义效应和 paired CI 规则须在 smoke 结束前预注册。
- [ ] smoke 只作决策，不写论文 headline。

## 6. 正式 PPO 与 checkpoint 选择

- [ ] smoke 通过后才批准正式训练；v4 数据目标为每臂 3,000 groups×K4=`12,000` scheduled
  trajectories。精确更新次数/是否跑满一遍 schedule/early-stop 必须在正式配置中预先冻结。
- [ ] 若 α 保持论文核心主张，PPO-T/F/A 三臂必须使用同一完整预算；不能只跑满 T/A，也不能用历史
  `PPO-TK-additive` 结果替代 PPO-F。若成本只允许 T/A，必须把 α 降为探索性结果。
- [ ] 每 600 trajectories 保存 checkpoint；保留 final 和所有失败点。
- [ ] checkpoint 只按冻结 validation protocol 选择，不能按 test/confirmation 追选。
- [ ] seed42 有正向信号后，再补 seeds 13/2024；重要结论不得只依赖一个 seed。
- [ ] 所有训练完成后下载 adapter、value head、history、TensorBoard、manifest、resolved config 和日志到本地。

## 7. 最终评估与论文表格

- [ ] 表 A（标准 baseline）：冻结三数据集各 n=300 的 canonical Scheme A 标准 pipeline；同 qid、检索/KG
  配置、prompt、decode、scorer，并 fresh 跑 Strong SFT 与选定 PPO。历史 frozen prompt bank 不能代替
  Scheme A；当前源码升级需追加兼容性证据后才能按原规则出表，不能绕过旧 source hash gate。
- [ ] 表 B（source-adaptive system）：对每个 `dataset::qid` 使用同一 record-level graph quality gate；当前
  冻结资产观测到 eligible 题都来自 2Wiki，其余题为 passage-only。所有模型逐题共享完全相同输入，只比较
  checkpoint，不能在 evaluator 中按数据集名指定证据源。
- [ ] 主指标：EM、F1、macro；配对 bootstrap CI、McNemar、gained/lost/tied。
- [ ] 过程指标：格式有效率、引用 precision/coverage、IHR；IHR judge 模型、prompt、版本和样本必须冻结。
- [ ] α 只在训练中改变 reward；评估时 α telemetry 不能被描述成直接提高 EM/F1。
- [ ] 外部 checkpoint baseline 放表 A 作端到端参照，资源/模型差异必须标注；表 B 只作内部 matched 因果。
- [ ] 2Wiki supply-only matched causal +23.7pp 继续独立报告，不得当作 PPO 或 α 收益。

## 8. 明确暂停且本轮不做

- [ ] 不训练或扩充 Query Controller，不继续 v8/v9/v9.1 子问题拆解。
- [ ] 不生成 Hotpot Controller 银标，不调用 DeepSeek API。
- [ ] 不继续 QPEG-v1/v2/v3/v4、SAEG 或 passage-derived pseudo-KG。
- [ ] 不重训/continued-SFT，不把 50% ProofKG SFT 的失败 checkpoint 作为 PPO 起点。
- [ ] 不修改 Gold、baseline 或已冻结 evaluation protocol。
- [ ] 不删除失败实验、checkpoint、日志或 manifest。

## 9. 需要研究者批准的节点

- [x] 研究者已授权本轮 runtime/reward 修复和以 EM/F1 为目标的 PPO 准备，并明确无卡期间同步环境/代码。
  该授权不等于 α target 已冻结，也不等于大型训练已经执行；无需为已批准的修复重复询问。
- [x] 2026-09-05最新授权“可以按照你的想法进行修复”：已执行format-v2、source-credit-v1、真实评分与新α校准。
  该组合修复的公式、mask、clearance边界和代码绑定详见新修复记录；下列早期待批条目不重复阻断本次已授权修复。
- [ ] 批准 α successor 的精确 target、mask 和 reward 公式后，才修改核心 reward。
- [ ] 批准新的 source-gated v4 数据说明、α 配置和 reward 版本后，才冻结正式 schedule/config lock；
  H/M 已通过的 passages 可以复用，但 v4 identities/weights/12,000 schedule 必须重新生成并验门。
- [ ] 零更新门通过后，批准 600×3 matched smoke。
- [ ] smoke 通过后，单独批准每臂最多 12,000 scheduled trajectories 的正式 PPO。
- [ ] 发布/提交论文前单独批准。

## 10. 下一项立即执行

数据封版与严格 preflight 已经完成；当前依次执行：

1. [x] 修复并升版 ProofKG temporal/multi-value scorer；
2. [x] 重冻 complete-ledger-safe replay-v2、H/M retrieval、ordinary200 和 official-raw n1500 planner；
3. [x] 完成 n1500 full-root resolution、v6 closure、official-raw canonical retrieval/unified supply；
4. [x] 严格筛出 Proof800 并物化 final 3000×K4 v4 release，preflight=`PASS_NOT_TRAINED`；
5. [x] 实现 PPO-O 配置、v2 runtime/format 修复和保守 ProofKG v2.3；
6. [x] 同步独立远端 release、封版 v4 和 replay-v2，验证环境与 CPU 契约；证据见
   `outputs/audits/source_gated_mixed4_emf1_v1_remote_validation/`；
7. [x] 本地完成α候选/真实ReaRAG/format-v2/source-credit-v1校准与train-only排序、96次生产奖励零更新复核；
8. [ ] 处理Text尺度、ordinary分布、α区分及有效率，完成独立效用确认 → 完整 A-probe12 → A-smoke600 → 开发集选模；
9. [ ] 正式完整 PPO、综合 baseline 与同预算 T/F/A 消融支持性能和方法结论。

本轮 PPO-O 产物使用 `ppo_mixed4_emf1_v1` 独立命名空间；后续 α 产物沿用其单独 successor 命名；旧 mixed3 PPO-T/PPO-TK 配置、lock、日志和
失败结果保持只读，绝不原地改写。
