# KG-ProWeight 当前研究工作流

> **2026-09-06 丢分诊断完成：`PPO_A_STOPPED300_REGRESSION_DIAGNOSIS_COMPLETE_NO_RETRAIN`。**
> 用户授权“查找”后，900条输出契约诊断、全部15个独立丢分问题/28次事件审阅及实际300条训练信号核查完成。28次新增丢分均为单一完整Final、非cap、符合统一min3诊断；未发现提取或奖励落点故障。明确看到比较方向、关系对象与Final精度错误，也有SFT命中但缺桥/错链和FIFA法文别名边界，不能将全部EM损失称为可靠推理退化。
> 实际仅75个训练问题/每域25，61条非零α均来自2Wiki；34条训练口径答对的两步shortfall仍净获+3.4。Text仅改Final的CPU输入不变性已验证；答案奖励仍看Final，合格ProofKG另有答案检查。训练聚合经第二代理独立复算，未识别α因果影响。
> 当前建议先用训练侧固定银行验证关系/比较/Final过程评分辨别力，自然两步另版验证；当前无新训练、无核心Reward/Gold/评估修改，SFT原选模与300条FAILED保留。[完整诊断、逐题证据与后续建议](docs/ppo_a_stopped300_regression_diagnosis_20260906.md)。下方为历史状态。

> **2026-09-06 评估终态（18:42:44完成，随后独立复核）：`EVAL_DEVELOPMENT150_COMPLETE_SFT_SELECTED_AUDITED`。**
> 本地开发评估完成900/900条预测、六项生成+六项评分+原selection共13项任务；[终态status](outputs/evals/ppo_a_stopped300_development150_20260906_v1/status.json)为COMPLETE_DEVELOPMENT_COMPARISON_NOT_MAIN_TABLE。[原规则选模](outputs/evals/ppo_a_stopped300_development150_20260906_v1/selection/report.json)选择StrongSFT，排名SFT/step200/aborted300。
> legacy宏EM/F1（%）：SFT24.00/34.70、200为22.67/35.34、300为22.00/32.79；200的F1+0.65pp但EM−1.33pp，300为EM−2.00pp/F1−1.91pp。no_graph为25.33/36.66、24.67/35.25、24.67/36.50；完整分域表见[评估结果](docs/ppo_a_stopped300_development_eval_20260906.md)。差值用原始精度计算。
> [独立结果验收](outputs/audits/ppo_a_stopped300_development150_results_independent_20260906_v1/report.json)通过：900条独立评分复算、六项生成/六项评分/selection的SHA及先全生成后评分时序一致。[配对诊断](outputs/audits/ppo_a_stopped300_development150_paired_diagnostic_20260906_v1/report.json)已封存，只作事后描述、不改变选模。开发结果不支持综合改善或α独立贡献，小样本点估计也不能直接写成显著退化。
> 仍限150题固定上下文开发代理，不能当正式canonical900主表；原训练300条guard FAILED保留，不自动新训练/重启/full/F/T或正式900评估。下方RUNNING等为历史快照。

> **2026-09-06 17:30:29 当前状态：`EVAL_DEVELOPMENT150_RUNNING`。**
> 用户已明确授权评估；本地于17:29:19真实启动，supervisor3338466。比较原StrongSFT、PPO step_200与aborted_step_300，沿用原A银行150题、legacy/no_graph两视图，共3×2×150=900条预测；原输入/labels字节、greedy512、BF16、batch1、冻结SFT tokenizer和原选模规则不变。
> 17:30:29实际status快照：当前generate__strong_sft__legacy，子进程3338467，已生成12/900，完整任务0；尚无评分或选模结果，后续以[实时状态](outputs/evals/ppo_a_stopped300_development150_20260906_v1/status.json)为准。全部六项生成先完成，再六项评分与原selection。
> 34项相关测试与独立输入/三层身份隔离检查通过；两份PPO checkpoint的实际参数更新已验收。原300条guard停止、FAILED及fresh132健康FAIL保留。本轮是150题固定上下文开发代理，不能写成canonical主表；不自动训练/重启/F/T/full或正式canonical900评估。
> [评估协议、进度与本地查看命令](docs/ppo_a_stopped300_development_eval_20260906.md)、[冻结执行协议](outputs/evals/ppo_a_stopped300_development150_20260906_v1/execution.json)。下方“评估尚未启动”等为历史快照。

> **2026-09-06 终态验收：`A_SMOKE600_STOPPED_AT300_PRESET_VALID_GUARD_AUDITED`。**
> 用户授权“进入”的完整A-smoke600已真实执行：北京时间16:44:32启动、17:02:35结束，在300/600条、75批时按原有效率guard停止；supervisor17:02:39观察exit1，GPU占用/利用率均0，训练manifest为FAILED，600未完成。
> [独立健康报告](outputs/audits/ppo_a_smoke600_stopped300_independent_health_20260906_v1/report.json)复现末15批有效41/60=0.6833<0.7；cap3/60=0.05、平均KL1.3282未越线。末窗口无效为9条两步短缺+10条其他严重无效，不能全部归因于步数下限。全300条216有效/55两步短缺/29其他严重无效，61条非零α、674次Text评分、replay30；无NaN/Inf或OOM。
> step_200与aborted_step_300保留；[terminal300归档](outputs/audits/ppo_a_smoke600_gpu_supervision_terminal300_20260906_v1/)20文件/254,806,764字节已SHA复验一致，7452项事件/契约对照通过，638 scalar/133 histogram；两份checkpoint的LoRA及value head实际更新且有限。[汇总验收](outputs/audits/ppo_a_smoke600_gpu_supervision_terminal300_20260906_v1/report.json)完成，训练FAILED保留。原失败及200条快照保留，默认不重启、不降guard、不扩full/F/T。
> 下一步建议按原评估协议评估保存checkpoint；自然≥2步规则另版单独验证，当前均未启动。原StrongSFT、reward/α、数据/mask及fresh132健康FAIL保持。[运行与停止记录](docs/ppo_a_smoke600_supervision_20260906.md)。下方RUNNING/初始化/未启动均为历史快照。

> **2026-09-06 16:44:48 当前状态：`A_SMOKE600_RUNNING_SUPERVISED`。**
> 研究者回复“进入”授权的完整A-smoke600已于北京时间16:44:32启动（UTC08:44:32）；远端supervisor PID2846、training PID2847。910份文件/约37.49GB SHA、实际base/ReaRAG路径及10项远端负向检查通过。
> 截至16:44:48，RUNNING/初始化，完成0/600条，尚无首批PPO更新或TensorBoard run；主代理按分钟监督，不预测结果。起点仍为原StrongSFT，固定150题×K4、原奖励/α/replay及200条后window15成本保护规则不变。
> probe12工程PASS及原fresh132健康FAIL继续保留；600在线结果不能代替独立综合baseline评估，full/F/T未自动启动。[当前运行、PID与日志入口](docs/ppo_a_smoke600_supervision_20260906.md)。以下“600准备中/未启动”为此前历史快照。

> **2026-09-06 当前状态：`A_PROBE12_ENGINEERING_PASS_A_SMOKE600_AUTHORIZED_PREPARING_NOT_STARTED`。**
> 完整 A-probe12 已在远端 RTX PRO 6000 96GB 正常完成（exit 0）：原 Strong SFT policy、显式冻结 SFT reference、真实 ReaRAG、答案/过程奖励与 learned α 全部接通；3批/12条轨迹，34次真实 Text 步骤评分、3条有效 Graph 轨迹，格式有效10/12，replay执行1条。
> 真实 TensorBoard 为606 scalar/133 histogram；314项事件与history/冻结契约核对通过。256/256 LoRA张量发生变化，value head从零更新，loss/KL及参数均有限；中间2Wiki批实际记录α六维特征。PyTorch峰值allocated69.55585GiB，监督期间整卡最高观测约78.743GiB，两者口径不同。
> 12份训练产物共126,619,933字节已下载并与远端SHA逐项匹配；[执行与查看日志](docs/ppo_a_probe12_supervision_20260906.md)、[监督归档](outputs/audits/ppo_a_probe12_gpu_supervision_20260906_v1/report.json)、[独立验收](outputs/audits/ppo_a_probe12_independent_acceptance_20260906_v1/report.json)。本轮只成立工程链路通过，尚无综合baseline EM/F1提升或α优越性结论，原fresh132健康FAIL保留。
> 研究者随后明确回复“进入”，已授权下一版完整 A-smoke600；当前正在准备独立配置/scope/发布，**600尚未启动**，full12000与F/T训练不因本次授权自动开始。下面所有“PPO未开始/600未获授权”等状态为此前历史快照，保留原失败与旧产物，不作为当前状态。

> **2026-09-06 远端开卡前检查：`REMOTE_CPU_READY_SCOPED_A_PROBE12_TENSORBOARD_VERIFIED_PPO_NOT_STARTED`。**
> 原实例已无卡开机并成功SSH登录；TensorBoard重启后未运行，现已恢复`/root/tf-logs`、6007服务。旧单图run实际只有连通性测试标签。
> 新版遥测已同步独立release，本地/远端各16项event写入读回测试通过；首3批写histogram，覆盖probe第二批的α分布。HTTP已核验真实零更新缓存诊断257 scalar/98 histogram，不能当成PPO训练曲线。
> 限额A-probe12已冻结并同步：远端89项资产共37.48GB SHA复验通过，108项scope/v2 runtime测试通过，真实模型路径及12条正向加载通过；600/full、关键参数覆盖拒绝。无卡准备完成，可开原96GB GPU执行probe；当前SFT/PPO更新仍0，原health FAIL及600未放行保留。[远端检查与监控](docs/ppo_remote_prelaunch_20260906.md)。下方“SSH关闭/生产gate尚未签发”是本次限额child完成前的历史状态，不能解释为600/full已有许可。

> **2026-09-06 最新结果：`FRESH132_COMPLETE_UTILITY_PASS_HEALTH_FAIL_PROBE_ELIGIBLE_PPO_NOT_STARTED`。**
> 固定132题的660条生成、1927步真实ReaRAG评分及最终分析全部完成。首次Gold前分析校验错误已通过另版单点恢复修正；原代码/失败/评分保留，额外19项测试通过，未重跑GPU。
> 独立效用PASS，健康FAIL，整体FAIL；probe资格true、matched600自动投资false。新图96题过程top1 EM：A=F=84.38%，T66.67%，greedy83.33%；A相对F尚无额外EM收益，F1略低。这是冻结候选排序，不是PPO训练成绩。
> sampled有效89.20%，三域macro82.64%未达90%；greedy98.48%。sourcePASS79的39个mixed family上A pairwise94.87%，F97.01%，T48.93%。完整132与ordinary的未改善结果同时保留。
> 当前权威：[恢复分析报告](outputs/audits/source_credit_v2_fresh_confirmation_local4090_20260906_v1/analysis_recovery_v1/report.json)（SHA `a469aa370567ea3020e9af6c4cee5bafd10f2e2472281e54f295d9222c4c7516`）、[完整解释](docs/source_credit_v2_fresh_confirmation_execution_20260906.md)。独立SHA/聚合复核通过。
> 下一步限额完整A-probe12，保留现Strong SFT；尚未签发生产gate、未更新PPO、未放行600。远端本轮复查仍在SSH握手关闭。下方运行/恢复待办均为历史状态。

> **2026-09-06 14:11 收尾更新：`GENERATION_SCORING_COMPLETE_ANALYSIS_RECOVERY_PENDING_PPO_NOT_STARTED`。**
> 固定660条生成及评分已全部完成，1927次实际ReaRAG步骤；sampled有效471/528，greedy有效130/132。
> 首次分析在Gold读取前因校验器误拒39条invalid的合法−1诊断分而停止；这些轨迹六视图过程分及α均0，未进入排序。原失败、代码与产物保留。
> 单点修正范围后的真实132/660 Gold-free完整复验已通过；正在另版冻结恢复执行，不改变奖励、排序、阈值或数据，不重跑GPU。此刻尚无独立效用结论。
> 依据：[诊断记录](outputs/audits/source_credit_v2_fresh_confirmation_range_diagnosis_20260906_v1/report.json)、[执行记录](docs/source_credit_v2_fresh_confirmation_execution_20260906.md)。

> **2026-09-06 13:00 最新执行：`FRESH132_CONFIRMATION_PIPELINE_RUNNING_PPO_NOT_STARTED`。**
> 研究者已批准现有 Strong SFT 路线。本地4090的固定132题/K4+greedy共660条流水线已启动，依次生成、真实ReaRAG单次评分、封存过程排序后读Gold判定；新版SFT继续暂停。
> 145项新执行/评分/统计/隔离测试通过，实际132输入与96来源检查对齐通过；协议与执行代码快照已冻结，SHA `d1bf882764c825e8e14de9a83e7219ed68ea83b8c480096e23c952f260b8fc97`。
> 90%格式健康线与独立过程效用分开报告，工程probe资格只依据完整性和独立效用；不要求小样本统计显著A>F。尚无确认结果、PPO更新或新模型指标。
> 当前运行目录 `outputs/audits/source_credit_v2_fresh_confirmation_local4090_20260906_v1`；控制器PID3077412，状态以`status.json`为准。任何阶段失败均保留原记录并停止；不自动扩样、重拟合或启动PPO。
> 远端本轮SSH握手被关闭；完整PPO仍需原96GB环境恢复连接。执行与监控入口：[fresh132确认](docs/source_credit_v2_fresh_confirmation_execution_20260906.md)。下方“确认尚未开始”为此前历史状态。

> **2026-09-06 最新优先级：`EXISTING_STRONG_SFT_PPO_PRIORITY_SFT_EXPANSION_PAUSED`。**
> 研究者希望尽快获得真实 PPO 结果；新版 SFT 不再是前置，保留为后续可选对照。原 Strong SFT 继续作为 A/F/T 的共同起点和 reference。
> 本轮新 SFT 检索/自动组装已主动停止，产物保留，付费 teacher 调用 0。下方 SFT“后台运行”“预算待确认”为暂停前历史状态，预算提案同步暂缓。
> 当前顺序为固定 fresh132 确认 → 完整 A-probe12 → matched A/F/T-smoke600 → 冻结开发选模与原评价；不能伪造未确认 gate 放行。
> 本地可分阶段生成/评分；原样完整 PPO 仅 BF16 权重约 47GiB，使用此前 96GB 远端环境。此刻 PPO 尚未启动。
> 当前入口：[现有 Strong SFT 的 PPO 优先决定](docs/existing_strong_sft_ppo_priority_20260906.md)。

> **2026-09-06 最新任务更新：`SFT_V3_DATA_PREPARATION_IN_PROGRESS_NOT_TRAINED`。**
> 研究者已选择同一 Llama3-8B-Instruct 基座新建 LoRA，要求完整准备新版三域 SFT 数据；下方“当前不重训 SFT”是历史决定。
> 新保护账本与 16,500 个 train-only 随机候选已冻结，新增独立图补充 60 后组合为 16,560；按全局 family 提前划分 train/validation，目标合格集 6,000+300，不把候选数当成品数。
> 新版限定 10 passages、2–5 自然步骤、assistant（含 EOT）≤384 tokens；不再通过旧 loader 用 Gold 替换 Final 或裁减 passages。
> 旧三域大型 silver 为 dev 来源，不纳入训练。来源核验得到最多 229 个可用非空 KG 输入，仍须真实 teacher 引用和逐步证据审核。
> 首批 128 题真实 canonical 检索与长度检查通过；当前已冻结 192 个就绪输入快照，剩余本地检索和自动组装后台运行。尚无付费 teacher 生成，API 预算待研究者确认；SFT/PPO 参数更新仍为 0。
> 当前数据入口：[新版 SFT 数据准备](docs/sft_v3_data_preparation_20260906.md)。旧 Strong SFT、正式 PPO 数据、评估协议与 fresh132 保留。

> **2026-09-06 最新执行更新：`STAGED_REPAIRS_VALIDATED_EVIDENCE_V1_NOT_ADOPTED_PPO_NOT_STARTED`。**
> 研究者已授权按策略顺序执行：先固定20题Gold-free证据供给小实验，独立实现保守shortfall-only答案目标v2。
> 240条缓存组件复算完成，旧提示仅8条两步单因短缺保留答案−1（其中4条答对），其余无效仍−4、原合法输出不变；不是训练收益。
> 真实供给与Reader40条完成：可见答案词面6→10/20、EM4→5/40、F120.51%→19.52%、format30→33/40；独立复算一致，暂不采用供给v1。
> 350项测试、780次真实缓存生产奖励复验和48项TensorBoard读回通过。20题真实dense排序未复现失序，未改索引排序。
> 新目标默认legacy、仅新A/F/T probe配置显式v2，独立α放行仍false。
> 当前入口：[证据与目标执行记录](docs/evidence_and_objective_execution_20260906.md)。下方“未批准新reward”等为此前历史状态。

> **论文命名（2026-09-05，研究者确认）：TRACE-Gate: Learning to Balance Graph and Text Process Rewards for Multi-Hop RAG。**
> 中文：TRACE-Gate：面向多跳 RAG 的图文过程奖励平衡学习。
> 当前[中英对照稿](docs/paper/TRACE-GATE/manuscript_bilingual.md)已统一名称，并在同一目录持续修改。
> 摘要 v3 按 background → research problem → objective → methods → key results → contribution 重写，面向 ICLR 投稿；结果仍为占位符。
> 广泛文献检索后已重写引言、Related Work 和比较定位；第三节压缩后正式引用 32 篇，既有文献核验记录保留。本地 PILOTRAG 仅作写作风格参考。
> 2026-09-06论文修订：TRACE-Gate与Experiments正文按研究问题组织，数值配置移至附录A–D，并同步已实现的来源信用、六维α和softsign描述。
> 此项仅为论文写作更新，代码、实验 ID、训练设计与结果状态不变。

> 当前版本：2026-09-06（v4 数据封版；source-credit-v2 进一步修复与代表性 Text 校准完成）
>
> 当前总状态：`DATA_READY_PPO_NOT_STARTED`
> **最新执行状态：`TRAINING_FORMAT_PROMPT_V1_PROBE_COMPLETE_NOT_ADOPTED_PPO_NOT_STARTED`。**
> 固定60题/K2真实对照完成：格式80.00%→91.67%，全量EM38.33%→35.00%、F143.40%→37.82%，输出tokens+12.87%；当前不提升新提示为正式PPO配置。
> 模型、384预算、user/passages/KG/spec保持固定；新提示及评分协议预先冻结，完整生成后才读取train labels。EM/F1区间跨0，不能宣称稳定退化或无损；新132题未消费。
> 44项相关测试通过，原98个核心代码/配置SHA不变；奖励/α与baseline不变，GPU已释放。当前入口：[格式修复记录](docs/training_format_prompt_v1_20260906.md)。
> replay只读复验2000/2000目标通过当前format-v2，来源/渲染SHA一致；59条目标超过384，未筛除。格式SFT尚需独立固定训练协议，当前未训练。
> 最新方案复核（仅建议，正式协议未改）：优先检查证据供给和训练目标，格式SFT不再作为自动下一步。固定60题显示答案项＋无效罚分均值0.757→1.218而raw EM46→42/120；20题MuSiQue暂定逐题审阅发现供给、Reader及歧义多种问题，不能外推总体。详见[策略复核](docs/pre_ppo_strategy_review_20260906.md)。
> 2026-09-06下一阶段已完成：60题/K2的384→512长度对照，120/120前缀一致，valid80.00%→81.67%，仅恢复2条；正式预算仍384。
> 新确认132题输入已冻结：96图题与正式3000三层身份隔离、36ordinary仅α开发未见；96图来源检查79PASS/11UNVERIFIED/6FAIL，失败均保留。
> 新确认缺inference类型；确认候选生成与过程效用尚未执行，PPO未开始。当前执行记录：[下一阶段记录](docs/source_credit_v2_next_stage_20260906.md)。
> 以下为上一阶段已完成的v2修复结果：
> Text改为逐step拟合、问题/候选/step分层等权和固定softsign；H/W/M各40题补充银行完成240条候选，192有效/654steps。
> 最终Text μ=.6359942727、s=.2092371033，有效覆盖H38/W40/M40题；48条无效输出保留，格式健康仍待改善。
> α增加唯一来源边覆盖率与最差步引用精确率；train同题特征相同率251/307→160/307，未加入Gold或奖励scalar。
> 原800有图题仍671获图信用、129题α_eff=0；原输入KG未改写或获得完整性认证。正式3000题/K4/replay均不变。
> 465项集成回归通过；最终192次缓存真实分数的生产token奖励复算通过，两组各1660候选特征一致，TensorBoard真实事件读回通过。
> 同114对的过程排序A/F从83.33%/83.33%到90.35%/88.60%；N-only与N+F该项相同，尚未显示新增特征的额外排序收益。
> 这是已消费银行的开发重分析，优势仅2对；新版独立confirmation与PPO放行均false，没有PPO参数更新。
> 完整报告与当前产物入口：[source-credit-v2修复记录](docs/source_credit_v2_repair_20260906.md)。
> [v1修复记录](docs/source_credit_v1_repair_20260905.md)保留其9/9启发式校准与旧Text裁剪49.94%的历史结果。
> 原候选EM/F1仍为55.12%/61.54%，这是graph-heavy训练银行，不能冒充baseline或新模型成绩。
> 研究者此前明确要求停止远端、
> 在本地重新生成全部候选。远端runner1267/生成1307已停止，328条完整候选与原日志保留，不复用于新银行。
> 本地4090运行 `sourcegate_local_regenerate_4090_20260905_v1` 已从0生成1660条，结果目录
> `outputs/audits/source_quality_candidate_bank_v1_generated_seed42_local4090_v1`。
> 最新启动与监控说明：`docs/sourcegate_local_regenerate_20260905_v1.md`；下方远端运行/迁移可行性条目保留历史状态。
> 历史记录（已被上述本地重生成状态替代）：2026-09-05，96GB RTX PRO 6000 上线，完整A的候选生成→评分→α校准后台任务启动，
> 状态为 `GPU_PREPARATION_RUNNING_PPO_NOT_STARTED`。实时入口见 `docs/sourcegate_gpu_launch_20260905_v1.md`；
> 下文“无卡/未生成”是冻结准备阶段记录；本条远端运行状态也已被最新本地运行替代。
> 本地GPU复核：RTX4090 24GB 在沙箱外可正常执行CUDA/BF16；此前沙箱内 `cuda_available=false`
> 不能视为本地主机无GPU。最长候选输入2399 tokens＋输出384的真实生成已通过，峰值allocated15.63GiB、
> reserved15.96GiB。记录：`outputs/audits/sourcegate_local_4090_generation_probe_20260905_v1/report.json`。
> 当时仅作迁移可行性验证，未接续/替换候选银行、未停止远端；随后已按最新指令停止远端并本地重生成。
> ReaRAG本地验证现已完成：原BF16模型、最长2589tokens，完整候选真实评分已结束；上述生成显存不是ReaRAG评分显存。
>
> 当前执行入口：`docs/todo2.md`
> TensorBoard 监控：`docs/ppo_tensorboard_autodl_20260905.md`；AutoPanel 入口读取 `/root/tf-logs`，
> 远端历史遥测 release 为 `source_gated_mixed4_emf1_v1_20260905_tensorboard_v1`；本地v2新增六维与softsign监控已核验，本轮尚未同步远端，训练未开始。
> 历史实验、失败记录与旧决策保留在 `docs/todo.md`、`docs/retraining_plan.md`、各实验目录的
> `manifest.json/report.json` 中；它们不得删除，也不得与当前协议混作同条件结果。

## 0. 一页结论与实时状态

**2026-09-05 最新指令：** 研究者明确要求**答案奖励 + 过程奖励 + α 门控全部加入后**，
训练一版综合 baseline 下 EM/F1 较好的模型。此前将该要求理解为 O-only 优先是执行误读，现已纠正；
已生成的 PPO-O 配置、协议和开发银行保留为可选消融，绝不是完整方法的必跑前置。
当前主模型为 `PPO-A`，同时准备 matched `PPO-F/PPO-T` 对照。runtime 修复和 γ/λ 是共享的
组合工程底座；A−F 才隔离 learned α 的贡献。
数据仍为 `DATA_READY_PPO_NOT_STARTED`；远端准备进程已停止，本地已完成原1660条候选及新120题代表性Text统计银行。
当前A/F/T使用相同 source-credit-v1 mask与format-v2。Text逐step尺度、补充人口及六维α已实现，18份N-only/N+F配置已绑定新gate。
本地长度对照已完成；512仅恢复2/120，未解决主要格式失败。新确认132题输入和来源已就绪，下一步冻结生成/分析协议并测独立过程效用，之后进入GPU PPO probe；旧9/9 ratio校准及开发排序不代表独立效用放行。
当前没有发生 PPO 参数更新。不能将旧 α checkpoint 直接当作新版已校准门。

新执行顺序：冻结 SFT → train-only 候选生成/评分 → 新 α 校准与独立零更新检查 → A-probe12
→ A-smoke600 → development150 选模/效用判断 → 独立 full12000。F/T 同时具备配置，按论文消融计划运行。
合法轨迹 outcome=`4 × (max_alias EM + 0.1 × max_alias F1)`、非法轨迹总奖励=`−4`；
有效图上由 frozen learned α 混合 Graph 与 passage-only ReaRAG，缺图严格 α=0。
Graph/Text 均使用 train-only 冻结的中心/尺度；新门移除当前 policy entropy 特征，避免奖励随 policy 漂移。
Reader 消费相同封版证据：rollout 每题 10 passages，replay 保留原始 15 passages；显式 SFT reference、
10% replay、anchor .1 共同保留。ratio target 暂定位为启发式质量分配标签，不冒称独立可靠性监督。

开发集只用于选择 checkpoint：legacy 固定输入为主视图，no-graph 为保留能力诊断，按 macro EM、
macro F1、优先 SFT/更早 checkpoint 的顺序选择。canonical900 仅用于最终报告，必须按既有
Scheme A 标准 pipeline 重跑 fresh SFT 与选中的 PPO；开发代理结果不能冒充主表。
完整方法配置、奖励边界及运行方式见 `docs/ppo_sourcegate_execution_20260905_v1.md`；
`docs/ppo_emf1_execution_20260905_v1.md` 仅记录 O 消融准备，不能覆盖本节最新优先级。

当前主线不是继续扩 Query Controller，也不是重训 SFT。主线固定为：

```text
原始多跳问题 + 冻结 passages + 逐题冻结的可用图证据
                    ↓
         Strong SFT Reader 一次性生成
 [Step 1] ... [Step N] ... [Final Answer]
                    ↓
       PPO outcome reward + 双源过程 reward
                    ↓
  α 按“证据—轨迹质量”分配 Graph / Passage 信用
```

研究假设是：

- 当某题具有通过身份、来源与完整执行审计的高质量 Graph 时，过程奖励应更多依赖 ProofKG；
- 当 Graph 缺失或不可信时，必须 fail closed，令 `α_eff=0`，只使用 passage/ReaRAG；
- 路由依据是逐题证据质量，不能硬编码 `dataset == 2wiki`；
- α 只改变 PPO 训练时的奖励信用，不改变评估时的检索、passages 或 KG 输入。

历史 mixed3 资产中，400 个 graph-eligible 样本恰好全部来自 2Wiki，其余 1,399 个样本没有合格图。
这是上游供给审计得到的分布，不是方法预先按数据集分支。新的均衡 v4 已封版：三数据集各
1,000 个 prompt groups，其中 2Wiki 为 800 strict ProofKG + 200 ordinary；K=4 后每个 PPO 臂共
12,000 条 rollout。HotpotQA/MuSiQue 目前主要依靠 passage/ReaRAG；将来任何数据集只要通过同一
Graph hard gate，都能获得非零图信用。

当前实验故事分成三个必须分开回答的问题：

1. **供给层**：高质量、可追溯的 ProofKG 是否比 legacy KG 更有用？2Wiki 已有肯定证据；
2. **训练层**：在完全相同的输入与预算下，Graph 过程奖励是否让 PPO 优于 text-only PPO？尚未验证；
3. **门控层**：learned α 是否优于固定 α，并使最终 checkpoint 超过 Strong SFT？尚未验证。

不能用“ProofKG 供给让 SFT 提高”替代“PPO 学到了东西”，也不能把额外 Wikidata 资源结果混进
同资源 baseline 主表。

### 0.1 当前里程碑

| 模块 | 当前状态 | 核心事实 |
|---|---|---|
| Strong SFT Reader | `COMPLETE/FROZEN` | 4,751 条 accepted Hotpot 轨迹，1 epoch；所有 PPO 臂共用 |
| protected identity ledger v2 | `COMPLETE` | 4,690 qid / 3,631 current families；覆盖历史开发、确认和 verifier cohort |
| clean SFT replay v2 | `COMPLETE` | 2,000 条，3--5 steps，真实 PPO loader/tokenizer 2000/2000 通过 |
| HotpotQA/MuSiQue v4 retrieval | `COMPLETE` | 823 个 expansion contexts：H417/M406，10 passages/题，无 BGE fallback |
| 2Wiki ordinary | `COMPLETE/BOUND` | 200 条身份与来源已锁定并进入 final v4：旧148 + 无污染替换52 |
| 2Wiki official-raw candidates | `COMPLETE` | Gold-free n1500；只作候选，不全部进入正式训练 |
| n1500 QueryPlan | `COMPLETE` | 1500/1500 有输出，1499/1500 schema-valid，Gold violation=0 |
| n1500 全根实体解析 | `COMPLETE/PASS` | 2116/2279 occurrences；1337/1500 题 all-roots；projection mismatch=0 |
| n1500 v6 多跳闭包 | `COMPLETE/PASS` | 2轮物化后收敛；nonempty=1365/1500，strict complete=1287/1500，四类均≥200，runtime error=0 |
| n1500 canonical passages | `COMPLETE/PASS` | 只对1287个strict eligible项检索；41/41批，10 passages/题，BGE load成功且fallback=false；Attempt1未知退出记录保留 |
| strict Proof800 selector | `COMPLETE/PASS (P0REFRESH2)` | 1287 strict admitted；四类各选200，共800；P0REFRESH1 schema-adapter失败记录保留 |
| final v4 dataset | `COMPLETE/PREFLIGHT PASS` | H/W/M各1000，graph800、ordinary2200，3000 groups × K4=12000；46/46 preflight checks通过 |
| PPO-O successor config | `OPTIONAL_CONTROL_NOT_TRAINED` | 已准备的纯 outcome 消融；不是主线前置 |
| format-v2 / source-credit-v1 | `IMPLEMENTED/CPU_VERIFIED` | 有效1471/1660；800原有图题中671获图信用，129关闭；原输入KG未修复 |
| Text normalization v2 / 代表性银行 | `REPAIRED/REAL_SCORED` | H/W/M各40题，240候选中192有效/654steps；逐step softsign统计冻结，48无效保留 |
| α feature-v2 / PPO-T/F/A config | `DEVELOPMENT_CALIBRATED` | 同题特征相同率81.76%→52.12%；18份N-only/N+F配置绑定代表性统计；A−F诊断仅2对改善 |
| 生产奖励 / TensorBoard | `ZERO_UPDATE_VERIFIED` | 192/192缓存真实奖励复算；两组全1660特征一致，六维及软饱和事件读回通过 |
| 384→512生成长度对照 | `COMPLETE/PAIRED` | 60题/K2，120/120前缀一致；valid96→98/120，cap6→1，tokens+1.35%；正式预算未改 |
| fresh确认输入与source mask | `INPUTS_READY_NOT_CONFIRMED` | 132题输入，96新图中79PASS/11UNVERIFIED/6FAIL；缺inference，未生成确认候选 |
| PPO runtime probe / smoke / full | `NOT_STARTED` | 新版独立confirmation与训练放行false；格式有效率与新α泛化仍待验证 |
| 远端独立 release | `CPU_VERIFIED/TASKS_STOPPED` | 308 tests、972 文件完整 SHA 通过；远端328条候选保留，本地另行从零完成 |

### 0.2 当前唯一主流程

```text
保护账本 + H/M canonical passages + 2Wiki official-raw n1500
                              ↓
          QueryPlan → 全 roots clean 解析 → v6 property closure
                              ↓
  answer-free canonical retrieval → unified supply → strict Proof800 选择
                              ↓
 H1000 + (2W Proof800 + ordinary200) + M1000 + replay2000
                              ↓
              final 3000 groups × K4 schedule / preflight
                              ↓
               α/reward 零更新验证 → GPU runtime probe
                              ↓
                    matched PPO-T / PPO-F / PPO-A
                              ↓
         同资源 legacy 主表 + source-adaptive 表 + IHR
```

## 1. 固定的 Reader 与输出协议

### 1.1 当前起点

所有新 PPO 臂都从同一个 Strong SFT checkpoint 开始：

`checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final`

已核实：

- manifest 状态为 `COMPLETE`；
- adapter 大小 `109,086,416 B`，MD5 `6cf328bc9f634e0c2e0a8d355cf8e43d`；
- LoRA 为 `r=32`、`lora_alpha=64`、`dropout=0.05`、目标模块 `q/k/v/o_proj`；
- 实际 SFT train fold 为 4,751 条 accepted HotpotQA trajectories，训练 1 epoch；
- 当前不做重新 SFT 或 continued-SFT。

### 1.2 固定输出格式

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

Reader 一次性回答原始问题。它不生成 search action，不在每一步暂停等待新检索，也不把完整多跳 Reader
降格成“子问题回答器”。SFT、PPO rollout 与 evaluation 必须共享同一 system/schema contract。

### 1.3 Strong SFT 冻结基线

标准 legacy pipeline，三数据集各 `n=300`、`seed=42`、greedy：

| Dataset | EM | F1 |
|---|---:|---:|
| HotpotQA | 0.3833 | 0.4898 |
| 2WikiMultiHopQA | 0.4267 | 0.4846 |
| MuSiQue | 0.2467 | 0.3419 |

这些结果是后续 PPO checkpoint 的核心同协议基线。任何不同 KG 供给或不同检索协议的结果必须另表报告。

## 2. 训练数据：历史 mixed3 与当前 v4

### 2.1 历史 v3 数据底座

下面是**已完成的历史 v3**，用于复用工程经验和保留历史对照；它不是当前准备训练的最终数据。

冻结数据目录：

`data/silver_data/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42/`

| 项目 | 已冻结内容 |
|---|---|
| unique questions | 1,799：HotpotQA 600 / 2Wiki 600 / MuSiQue 599 |
| rollout schedule | 1,800 prompt groups × K=4 = 7,200 trajectories |
| passages | 每题固定 10 篇 |
| target use | 只用于 train-only outcome reward；不写入 prompt |
| source steps | 已清空，不蒸馏旧推理步骤 |
| eligible Graph | 400 个 complete、identity-safe 2Wiki ProofKG |
| passage-only | 1,399 个显式空图记录 |
| eligible schedule mass | 400 groups / 1,600 trajectories，即 22.2% |

该数据已经通过当时的 `dataset::qid + question_sha256` join、空图、qid/family 和固定 schedule 门，因而可
复用工程实现及经 complete ledger 重新确认安全的父行/passages。v4 **不会原样复用** v3 的全部 qids、
sampling weights 或 7,200 schedule：v4 identities、source records、weights 与 12,000 schedule 均须在
完整账本和 Proof800 确定后重新冻结。

### 2.2 为什么旧 mixed3 排除了 α

旧的两份配置是一个固定加法单变量实验：

- `configs/training/phase3_ppo_mixed3_rearag_v2_proof400_text7200_seed42.yaml`
  - 论文臂名 `PPO-T`；
  - valid outcome 为 `4 × (EM + 0.1 × F1)`；
  - 文本过程分为 `0.30 × centered ReaRAG`；
  - `proofkg_process_reward=false`；
  - `alpha_gate_path=null`。
- `configs/training/phase3_ppo_mixed3_rearag_v2_proof400_text_kg_v2_1_7200_seed42.yaml`
  - 历史臂应称 `PPO-TK-additive`；
  - 除 output identity 外，只把 `proofkg_process_reward=true`；
  - 在 eligible 轨迹上额外固定加 `0.20 × R_Proof`；
  - α 仍然为空。

代码在 mixed reward fast path 中会主动拒绝非空 `alpha_gate_path/alpha_override`。这样做是为了让旧实验只
改变一个变量，不是说 α 不重要。旧配置保持只读；新的 α 方法必须使用新的配置、reward 版本和 Experiment ID。

### 2.3 当前 v4 数据设计

v4 正式主集保持三数据集等权，避免将“2Wiki 数据更多”误认为“Graph 方法更强”：

| 数据集 | 正式问题组 | Graph 状态 | 主要过程信用 |
|---|---:|---|---|
| HotpotQA | 1,000 | 当前无 strict ProofKG | passage/ReaRAG |
| 2WikiMultiHopQA | 1,000 | strict ProofKG 800 + ordinary 200 | Graph + passage / passage-only |
| MuSiQue | 1,000 | 当前无 strict ProofKG | passage/ReaRAG |
| 合计 | 3,000 | m_graph=1 目标800 | K=4，共12,000 rollouts |

2Wiki 不直接把 1,500 个候选全放进训练。正确关系是：

```text
official raw 2Wiki n1500 候选
       ↓ planner / root resolution / property closure / provenance gates
strict eligible 漏斗
       ↓ 四类各确定性选择200
ProofKG 800 + ordinary 200 = 正式2Wiki 1000
```

候选分布为 `bridge_comparison/comparison/compositional/inference=390/390/389/331`。n1500 用于抵抗
解析、闭包和来源门失败；未入选项只能作为 reserve 或 graph-heavy 消融，不能偷换进三数据集等权主实验。

PPO rollout 数据与 SFT replay 不是同一批数据：

- `3000×K4` 是 PPO 采样 schedule；K=4 表示同一问题采样四条候选轨迹，不是四个模型臂；
- `replay2000` 是单独的 Strong-SFT 监督锚点池，训练时按冻结比例穿插，防止策略遗忘；
- train-only outcome aliases 只用于奖励计算，不进入 prompt；旧 teacher steps 不作为 PPO rollout target。

### 2.4 v4 数据生产流程与硬门

```text
完整保护账本 v2
  ├─ H/M 身份重冻 → 只补11条检索 → 合并823 expansion contexts
  ├─ 2Wiki ordinary200 重冻
  ├─ replay2000 重冻
  └─ 2Wiki Gold-free n1500 候选
             ↓
       relation-graph QueryPlan
             ↓
       对所有 planner roots 使用同一 clean resolver 全量解析
             ↓
       v6 store-first + historical fallback 的迭代 property closure
             ↓
       冻结 official-raw answer-free canonical retrieval scope
             ↓
       canonical 10 passages + unified candidate supply
             ↓
       strict Proof800 selector
             ↓
       final v4 materializer / fixed K4 schedule / source-gate sidecar
```

不可放宽的门：

- qid、question SHA256、current lexical family 与保护账本零重叠；
- planner/record identity join=1，`gold_access=false`，runtime error=0；
- root projection 必须与最终 production consumer dry-run 逐 occurrence 一致；
- strict Proof 必须 all roots resolved、all hops complete、KG nonempty、每条边可追溯且 cutoff 完整；
- 每题正好 10 passages，passage hash、retrieval backend 和 BGE no-fallback 可追溯；
- Proof800 四类各200；任一类不足即停止，不降低标准；
- 最终 H/2W/M 精确为1000/1000/1000，schedule 精确为3000×K4=12000；
- rollout、replay、保护账本之间 qid/hash/current-family 三重零重叠。

### 2.5 当前共同 PPO 工程底座

新 T/F/A 三臂拟共享：

- Strong SFT policy 与显式冻结的 Strong SFT reference；
- learning rate `1e-6`、batch `4`、mini-batch `1`、K=`4`、PPO epochs=`2`；
- `max_new_tokens=384`、`max_steps=5`；
- `kl_coef=0.25`、`target_kl=8`、新版 `gamma=1`、`lambda=.99`（历史底座为 .95/.95）；
- critic zero-init、value dropout=`0`；
- 10% SFT replay；
- ReaRAG 必须 fail-hard 加载，禁止 dummy 或未登记 fallback；
- TensorBoard 记录 loss、KL、critic、valid、reward 分量、α、截断率和 replay ratio。

这些值来自已修复稳定性的 combined/hybrid smoke。它们是当前候选配置，不代表新的 α 实验已经训练。

## 3. Graph / Passage 双源门控

### 3.1 Graph 可用性硬门

对每个问题记录计算同一个不可学习的 `m_i^graph`：

```text
m_graph = 1，当且仅当：
  schema 合法
  dataset::qid 与 question hash 精确匹配
  provenance/cutoff 完整
  gold_access = false
  runtime_error = 0
  QueryPlan 完整执行
  每条保留边能回溯到计划 hop

否则 m_graph = 0
```

硬门负责“有没有合格图”。它不参与 learned α 拟合。任何失败都必须回退 passage/ReaRAG，不能静默使用
partial/noisy graph。

### 3.2 learned α 的职责

α 是生成之后、计算 PPO reward 时的**证据—轨迹信用门**：

\[
\alpha_i^{eff}=m_i^{graph}\,\sigma(z_i/\tau).
\]

- `m_graph=0` 时，严格 `α_eff=0`；
- `m_graph=1` 时，learned gate 才比较 Graph 与 passage/text 对该轨迹的相对可靠性；
- α 不选择检索器、不改变 prompt，也不在评估时直接移动 EM/F1；
- 当前 ProofKG reward 是轨迹级标量，因此新 α 也先做轨迹级，不能伪称逐步 Proof 监督。

新版实现采用四项：graph density、Proof provenance 中的 link confidence、cite-any、cite-match。
移除旧候选中的 current-policy surprisal/entropy 特征；旧 legacy α checkpoint 不能直接接入。

### 3.3 连续 α target 提案

研究者已授权完整方法接线；当前 v1 使用明确标为 heuristic 的 ratio target，真实校准 artifact 尚未产生：

\[
y_i=m_i^{graph}\frac{q_i^{graph}}{q_i^{graph}+q_i^{text}+\epsilon},
\qquad q^{graph},q^{text}\in[0,1].
\]

- `q_graph=R_Proof_v2.3/.85`，`q_text=mean((raw_ReaRAG+1)/2)`；它们不是独立校准的事实正确率；
- PPO 中固定中心化的 ReaRAG reward 可以为负，不能拿来构造 `[0,1]` target；
- `max(qG,qT)≤.05` 或分母≤1e-8 时 abstain；invalid/ineligible 也不进入 α loss；
- loss 为 soft-label BCE，轻量 logistic 的 CPU numpy 实现；PPO 期间 gate 冻结；
- 同 current family 固定60/20/20划分 train/calibration/internal-confirmation，只在 train 拟合统计和权重；
- α 训练与 Brier/R²/排序校准只在 `m_graph=1` 内进行；`m_graph=0` 只验证 fail-closed。

当前 eligible 记录全部来自 2Wiki，因此 learned gate 在 HotpotQA/MuSiQue 的 graph-quality 泛化仍为
`UNKNOWN`。方法代码可保持 dataset-agnostic，但论文不能把“没有读取数据集名”写成已经验证的跨数据集
Graph 门控泛化。

### 3.4 新过程奖励的 credit placement

对轨迹 `i` 的 `n_i` 个合法步骤：

\[
r_{i,t}^{text}=\frac{(1-\alpha_i^{eff})\lambda_T}{n_i}
\widetilde r_{i,t}^{ReaRAG},
\qquad
r_{i,final}^{graph}=\alpha_i^{eff}\lambda_G R_i^{Proof}.
\]

新版公式中的 Graph/Text 都先用 train-only 固定中心/尺度标准化并 clip[-1,1]；Text 逐步标准化后取均值，
不使用在线 EMA。ReaRAG 在 step-end 提供信用；轨迹级 Proof 分只放 final token。所有正式臂必须共享完全相同的文本
聚合、token placement、validity 和 outcome 规则，F/A 之间只替换 frozen scalar α。

合法轨迹的末步再加：

\[
R_{outcome}=4\times(EM+0.1\times F1).
\]

非法轨迹总奖励为 `-4`。核心修改已经获研究者授权；真实校准、过程排序与 GPU 健康/效用仍待验证。

## 4. 正式对照与因果问题

Strong SFT 是静态起点。新的 PPO 主实验为：

| 臂 | Graph/Text 过程奖励 | 回答的问题 |
|---|---|---|
| PPO-T | `α_eff=0`，outcome + ReaRAG | text-process PPO 相对 SFT 是否有效 |
| PPO-F | eligible 内使用冻结常数 α | 加入 Graph 过程信用是否有效 |
| PPO-A | eligible 内使用冻结 learned α | 动态证据质量门是否优于固定权重 |

PPO-F 的常数必须取 α-train/calibration eligible 轨迹按正式 schedule 加权的 conditional mean，并在
confirmation 前锁定。

决定性差值：

- `PPO-F − PPO-T`：固定 Graph/Text 信用混合相对纯 Text 过程奖励的净效应；由于 F 同时将 text 权重
  乘以 `(1-α)`，不能把它简写成“只增加一个 Graph bonus”；
- `PPO-A − PPO-F`：learned α 的贡献；
- `PPO-A − Strong SFT`：整个 PPO 方法的净训练收益。

若 α 是论文核心，T/F/A 的正式训练预算必须相同。只跑 T/A 或只让 F 跑 smoke，不能支持 α 的正式因果
结论。旧 `PPO-TK-additive` 不是 PPO-F，历史结果不能替代新臂。

## 5. 已成立、未成立与未知

### 5.1 已成立

1. Strong SFT checkpoint 完整，固定 Step Reader 可以作为统一起点。
2. explicit SFT reference、critic、截断和 10% replay 的 PPO 稳定性修复有效；hybrid smoke 在 replay
   EM 上追平 SFT。
3. 2Wiki 的高质量 ProofKG **供给层**有效：
   - canonical ProofKG pipeline：SFT `EM=0.6400, F1=0.6944`；
   - matched same-passages `n=300`：legacy `0.4033/0.4582` → ProofKG `0.6400/0.6944`；
   - paired ΔEM=`+23.67pp`，95% CI `[+17.0,+30.3]pp`，McNemar `p=5.3e-11`。
4. 这证明优质 Graph 输入对该 2Wiki development n=300 cohort 的 Reader 有用，但它是额外 Wikidata
   资源效果，不是 PPO 或 α 收益；独立 held-out replication 仍为 `PENDING`，不得写成 untouched-test 结论。

### 5.2 已知负结果

1. hybrid PPO 标准 pipeline 为 HotpotQA/2Wiki/MuSiQue `0.360/0.423/0.243` EM，未超过 Strong SFT
   `0.383/0.427/0.247`。
2. 当前 legacy KG 的 PPO 利用率 2×2 interaction macro 为负，不能宣称旧 PPO 提高了 KG 利用。
3. HotpotQA Wikidata-only Proof 路径结构门失败；选择性 partial Proof pilot 也未带来收益。
4. MuSiQue 当前 subquery conversion recognized=`0.633`，未过结构门；零样本 relation-graph 为失败方向。
5. 50% Proof continued-SFT 造成 replay/pipeline 退化，不能作为 PPO 起点。
6. Controller、QPEG、SAEG 等分支没有通过各自冻结门，当前暂停。

### 5.3 尚未验证

- 新 trajectory-level α 是否校准良好；
- 新 PPO-F 是否优于 PPO-T；
- 新 PPO-A 是否优于 PPO-F；
- PPO-A 是否在标准 legacy pipeline 上超过 Strong SFT；
- PPO-A 是否降低 IHR；
- learned α 对 2Wiki 以外 eligible Graph 的泛化。

任何论文草稿不得把这些 `UNKNOWN` 写成已验证事实。

## 6. 正式训练前的 P0 阻断

### 6.1 ProofKG temporal scorer 错误（实现已修复，rankability 待验）

`kgproweight/reward/proofkg_process_v2.py` 的 temporal 分支在已经解包 tail 后又执行 `len(t)==3` 并取
`t[2]`，实际把 tail 当三元组处理。400 个 eligible 问题中有 46 个 temporal 问题受影响。

已完成：

- 保留旧 `proofkg-process-v2-1-frozen-1`，新建 `proofkg-process-v2-2-frozen-1`；
- 修复 full/bare/ISO date、比较结果向根实体回溯、冲突多值 fail-closed、same/different 多值
  abstain、opaque QID abstain 和顺序确定性；
- 相关测试 57 项通过；Proof400 静态审计中 temporal 从 `0/46` 可确定提升到 `45/46`，且确定的
  45 题与事后 gold surface 对齐，剩余冲突题 abstain；
- 第一版静态审计暴露 terminal 多值的顺序依赖并以 `FAIL_STATIC_AUDIT` 保留；修复后的 v2 审计为
  `PASS_IMPLEMENTATION_STATIC_AUDIT_RANKABILITY_PENDING`。

尚未完成：在冻结候选池上重新做零更新 rankability 与 family-disjoint confirmation。静态审计不能替代
该门，旧 scorer 的审计结果也不能拿来证明新 scorer。

### 6.2 replay 与固定 Step 冲突（已解决）

旧 v1c replay 在完整保护账本下又发现 138 个 identity/family 冲突，因此保留但禁止给 v4 使用。权威 successor
是 `data/silver_data/sft_replay_legacy_strong_train_rendered3to5_n2000_seed42_v2/`：

- 2,000 条 HotpotQA Strong-SFT accepted-train trajectory；
- 3/4/5 steps=`1593/336/71`；
- 与 complete ledger、H/M v4 population 的 qid/hash/current-family overlap 全为0；
- 2,000 个 current families 全唯一；
- 真实 PPO loader 与 Strong-SFT tokenizer dry-run 均为 `2000/2000`；
- max prompt/total=`4231/4753` tokens，15 passages 无需丢弃。

历史 v3c source-gated release 仍保留为已完成工程资产：H/2W/M=`600/600/599`、Graph eligible=`0/400/0`、
1,800×K4=7,200。它不能替代当前 v4 的 H/2W/M=`1000/1000/1000`、Proof800、12,000 rollout 主实验。

### 6.3 α successor 尚不存在

当前生产 mixed fast path 会拒绝 α。需要新增默认关闭的 successor：

- 老配置行为逐位不变；
- 新路径实现 `m_graph`、trajectory feature aggregation、PPO-T/F/A；
- ReaRAG 与 ProofKG 的量纲/credit placement 固定；
- 新 config、lock、manifest、Experiment ID；
- 单元测试覆盖 mask、回退、eligible、常数 α、learned α 和 dataset-name independence。

### 6.4 GPU postflight 未完成

正式配置需先在 96GB GPU 上做每臂最多 8 trajectories 的 runtime probe。当前状态不是“可直接跑
12,000”；final v4 data、α/reward 协议和配置目前都尚未同时就绪。

此外，SFT、PPO rollout 与 evaluation 的 system prompt、step schema 和 chat-template hash contract
尚未完成三路径一致性测试。第1节定义的是目标协议，不代表代码路径已经逐字节一致；该测试通过前同样不能训练。

## 7. 执行顺序与停止门

### Stage A：v4 身份、检索与 replay（完成）

1. complete protected-ledger v2：完成；
2. H/M 1000/1000 身份重冻及 823 expansion contexts：完成；
3. 2Wiki ordinary200 successor：完成；
4. clean Strong-SFT replay2000 v2：完成；
5. official-raw Gold-free 2Wiki n1500 候选及 planner：完成；
6. v6 evidence store、fail-closed materializer 与 final v4 dataset 均已完成，见 §12.1。

### Stage B：n1500 ProofKG 执行（完成）

1. `[COMPLETE]` 对1499个可识别计划的全部2279个root occurrences做clean解析；
2. `[COMPLETE]` 用最终title/entity cache + v6 clean alias做production-consumer dry-run；
3. `[PASS]` projection、dry-run逐root完全一致，all-root question=`1337/1500`；
4. `[COMPLETE]` 在v6上完成store-first + historical-fallback property closure，2轮物化后收敛；
5. `[COMPLETE]` 输出1500行strict eligibility telemetry，1287 strict complete、四类均≥200；
6. 若root question rate、nonempty、complete或每类strict容量未过冻结门，保留失败并停止。

旧 n300 closure-v2 的 `nonempty=241/300`、`complete=222/300` 虽通过相应门，但 root-question
rate=`232/300=0.773` 未过0.80，因此正确状态仍为 `FAIL_DIAGNOSTIC_STRUCTURE_RETAINED`。审计确认
错误来自跨 resolver 的 baseline+delta 投影，不是 join/cache 消费 bug；n1500 禁止重复该算法。

### Stage C：Proof800 与 final v4 release（完成）

1. 依据 closure 结果预注册 official-raw answer-free retrieval scope，不读取答案或 test 信息；
2. 用 canonical Wiki18 + E5/BM25 + BGE 栈物化每题10 passages，并禁止 fallback；
3. 将 unified-v2 的 reserve-specific `source_release/status` 语义升版为 official-raw 正式接口；
4. 对 closure 合格项物化 canonical unified supply；
5. 因 retrieval/unified 实现 hash 改变，保留 selector protocol v1 并生成 superseding v2；
6. 运行 answer-blind strict selector，四类各200、总800；
7. 合并 H1000 + W(Proof800+ordinary200) + M1000；
8. 生成 `silver_train/question_kg_records/source_gate_records/sampling_weights/prompt_groups/fixed_rollout_schedule`；
9. 验证 m_graph=800、H/2W/M=1000/1000/1000、K4=12,000、replay/ledger overlap=0；
10. 只有全部通过才将数据标为 `COMPLETE_DATA_NOT_TRAINED`。

retrieval scope 已在 closure 最终结果出现前确定为：**全部且仅 strict answer-free structural eligible**；
正式 protocol/manifest 已封版并通过 23/23 数据门和 46/46 preflight。该 scope 要求
schema/root/hops/nonempty/runtime/provenance/edge-trace/duplicate 等机械门全部通过，且每类容量≥200；
不读取答案、EM/F1或候选排序。这样既不为节省计算而事后只挑“看起来好的800”，也不对明确结构失败项
浪费检索资源。

### Stage D：冻结 α 与 reward 单变量实验（当前阶段；执行 §0 的完整方法计划）

1. 冻结 `fixed-step-source-gated-ppo-alpha-v1` successor protocol；
2. 冻结 continuous α target、特征、normalization、temperature 与 constant-α；
3. 老 mixed fast path 保持逐位兼容，新路径显式启用 `m_graph` 与 T/F/A；
4. 固定 ReaRAG/ProofKG 尺度、token credit placement、validity 与 outcome reward；
5. 增加 SFT/PPO/eval system prompt、step schema、chat-template 的逐字节/hash contract test；
6. 生成三份除 α 机制外等价的 resolved config、lock、Experiment ID；
7. 核心 reward/loss 修改和协议冻结须研究者批准。

### Stage E：零更新验证

在相同 candidate bank 上比较 T/F/A reward，至少报告：

- format valid rate；
- greedy 与 oracle@K EM/F1；
- reward-top1 EM/F1；
- eligible-only correct-vs-wrong pairwise、Spearman、tie rate；
- passage-only 的 `α_eff=0` 正确率；
- 按 dataset 与 source eligibility 分层。

阈值必须在看 confirmation 前冻结。若 exploration headroom 不足，停止并归因于候选生成；若 pairwise/
top1 失败，停止并修 reward/α，不能靠增加 PPO 步数碰运气。

### Stage F：96GB GPU runtime probe

每臂最多 8 trajectories，检查：

- policy/reference 均为同一 Strong SFT；
- K=4 grouping 与固定 schedule；
- ReaRAG fail-hard、Proof 分支和 α checkpoint；
- KL、critic、reward placement、10% replay；
- TensorBoard、checkpoint、manifest；
- 无 NaN、OOM 或静默 fallback。

### Stage G：600-trajectory matched smoke

T/F/A 使用同一 schedule、seed 和预算。smoke 只作继续/停止决策，不作论文 headline。重点检查训练健康、
PPO-A 不退化 Strong SFT、A 相对 F 的方向。

### Stage H：正式 v4 PPO

仅在前述门全部通过并再次获得研究者批准后启动。每臂共享同一 3000×K4=12,000 轨迹 schedule；
“是否正好跑一遍 schedule、更新次数及 early-stop”仍须在正式配置中冻结，不能根据 test 结果临时延长。
若 α 是论文主张，T/F/A 必须同预算；checkpoint 只能按冻结 validation 选择，不能按 test 追选。
seed42 出现稳定正向信号后再补 seeds 13/2024。

## 8. 评估与论文报告

### 8.1 表 A：标准同资源 baseline

在冻结 canonical legacy contexts 上 fresh 评估 Strong SFT、PPO-T、PPO-F、PPO-A：

- 同 qid/order；
- 同 passages 与 legacy KG；
- 同 prompt、decode 和 scorer；
- 报三数据集 EM、F1、macro、paired CI、McNemar、gained/lost/tied；
- 外部论文 checkpoint 只放在独立“非严格同资源参考”区，并注明模型、训练数据和资源差异；不参与
  本方法的配对显著性或因果比较。

### 8.2 表 B：source-adaptive matched system

每个 `dataset::qid` 使用同一冻结 record-level quality gate 决定是否存在 eligible ProofKG。所有模型逐题
共享同一 source state，不能只给 PPO-A 更好的证据，也不能在 evaluator 中硬编码数据集路由。

当前分布会表现为 2Wiki eligible 子集获得 Graph、其余样本 passage-only；这反映供给层事实。表 B 用于
比较 checkpoint，不把额外供给效果伪装成训练效果。

### 8.3 供给层独立表

2Wiki matched legacy-vs-ProofKG `+23.7pp EM` 独立作为“额外资源/Graph supply”结果。它说明 Graph
质量有效，但不得计入 `PPO-A − SFT` 或 `PPO-A − PPO-F`。

### 8.4 IHR

IHR judge 的模型、prompt、版本、qid 与输出 manifest 必须冻结。α 在 eval 中最多作为 telemetry；只有
PPO-A checkpoint 在相同输入下的 IHR 差值才能支持“训练降低中间幻觉”的结论。

## 9. α/PRM 历史资产的正确口径

旧 Phase2 checkpoint：

`checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42_no_text_head/alpha_gate.pt`

它是五特征轻量 gate，在 Hotpot held-out 1,886 steps 上得到 Brier `0.0620`、R²-vs-constant `0.5109`。
其 target 是“legacy rule KG 是否给出非中性判决”，不是新 Graph-vs-Text 相对质量。

同时：

- Phase2 三分类 PRM head 没有进入历史 PPO reward path；无需为新方案重训 8B PRM adapter；
- 历史 PPO 的 `R_KG` 来自 rule annotator 的连续值，不是 PRM 三分类输出；
- 历史 hybrid 的 α mean 约 `0.731`，但模型只追平 SFT，且没有 matched fixed-vs-learned α 消融；
- 因此旧 gate 只能作历史基线，新主线要训练轻量 α successor。

## 10. 暂停分支与科研边界

当前不做：

- Query Controller、动态子问题拆解、逐步 search action；
- QPEG-v1/v2/v3/v4、SAEG、passage-derived pseudo-KG；
- Hotpot Controller 银标与 DeepSeek API 生成；
- continued-SFT 或重新 SFT；
- 覆盖/删除失败 checkpoint、日志、manifest；
- 修改 Gold、baseline 或已冻结 evaluation protocol。

暂停不等于否定这些方向，只表示它们不进入当前一周内的因果主线。若未来恢复，必须新建协议和 Experiment
ID，不能改写旧失败门。

## 11. 需要研究者确认的节点

根据 `AGENTS.md`，以下动作必须先确认：

1. 冻结新 α target、mask、normalization 与核心 reward 公式；
2. 修改生产 reward/loss；
3. 启动 600×3 smoke；
4. 启动每臂最多 12,000 scheduled trajectories 的正式 PPO；
5. 修改正式 evaluation protocol；
6. 发布或提交论文。

普通代码测试、数据只读审计、文档更新和小规模 CPU 验证可直接执行。

## 12. 当前正在执行与紧接着的工作

### 12.1 已完成的数据封版

- closure-v3保持权威PASS：strict eligible=`1287/1500`，四类=`361/353/317/256`；
- canonical retrieval Attempt1在完成3批、第4批开始后未知退出且无release；失败记录保留于
  `outputs/audits/2wiki_official_raw_canonical_retrieval_v1_attempt1_failure/report.json`，exit code/root cause
  仍为`UNKNOWN`，无OOM或GPU故障证据；
- Attempt2已完成41/41批并通过正式release validator：1287/1287 identity/hash/order join、每题10 passages、
  BGE cross-encoder load成功且fallback=false。权威目录为
  `outputs/audits/2wiki_official_raw_canonical_retrieval_v1/`，contexts SHA=`a935351a...f9b4`、
  report SHA=`76a68d07...faa4`、manifest SHA=`c94503d8...476f`；
- official-raw unified-v3已物化1287个四路一致候选，13/13 checks通过。目录为
  `data/derived/2wiki_unified_proofkg_official_raw_v3/`；silver/QKG/gate/wrapper SHA分别为
  `53f25006...ebc8`、`3f613418...4503`、`fbcee82a...e89`、`f7228b6d...86d`；
- selector P0REFRESH1因silver schema不存冗余`question_sha256`而0/1287通过，未产生result且未训练；
  append-only失败记录为
  `outputs/audits/2wiki_proof800_strict_selection_v2_seed42_result_p0refresh1_failure/report.json`。
  修复仅改为从silver.question现场重算同一身份hash；不改quota、seed、ranking或科学predicate；
- selector P0REFRESH2正式通过：1287 strict admitted，四类各200，共800；qid/hash各800唯一，current
  family=728。result目录为
  `outputs/audits/2wiki_proof800_strict_selection_v2_seed42_result_p0refresh2/`，proof SHA=
  `8e58d829...1fc3`、report SHA=`dcad7b6f...410c`、manifest SHA=`eba5706f...e625`；
- final v4 protocol、data与preflight均完成：H/W/M各1000，Graph=800、ordinary=2200，K=4共12000，
  scheduled graph trajectories=3200，replay=2000。data的23/23 gates与preflight的46/46 checks均通过；
  权威preflight状态为`PASS_NOT_TRAINED`。

### 12.2 当前边界

- 数据状态仍为 `DATA_READY_PPO_NOT_STARTED`；PPO-A/T/F 配置与版本化修复已经实现并通过 CPU 测试，尚无 GPU 更新；
- 新 α 的 calibration artifact 与真实候选 rankability 尚不存在，不能标为训练就绪；
- 历史 baseline、canonical evaluation protocol 和数据封版资产保持只读；开发代理单独版本化；
- v2.3 scorer 是保守的结构/严格答案匹配修复，不是语义 verifier；完整方法必须报告其实际排序局限。

### 12.3 当前下一阶段

1. 完成独立远端 release 的文件 hash、环境与 CPU 检查；无卡阶段不启动模型训练；
2. GPU 恢复后先生成/评分 train-only 候选，校准新版 α 并运行零更新过程排序检查；
3. 通过后运行完整 A-probe12 与 A-smoke600，验证 EOS/masks、reference KL、两过程分量、α 和 replay；
4. 按冻结 development150 规则比较 SFT/A；有健康且值得继续的信号时执行正式预算；
5. 选定模型后按 canonical Scheme A 做综合 baseline 报告；用 matched T/F/A、多 seed 支撑方法结论。

详细逐项清单见 `docs/todo2.md`。

2026-09-05 远端准备已复验完成：独立目录
`/root/autodl-tmp/kgpaper_releases/source_gated_mixed4_emf1_v1_20260905/`，现有环境 Torch2.11.0+cu128 /
Transformers4.49.0 / TRL0.11.4 / PEFT0.19.1，CUDA=false。308 CPU tests通过，972文件共37,723,852,837B
（含两个模型全部权重分片）SHA一致；ReaRAG代码差异通过独立目录的版本副本解决，原模型和旧项目未改。
证据：`outputs/audits/source_gated_mixed4_emf1_v1_remote_validation/`。主模型仍为完整 PPO-A；
下一步等待 GPU 后生成真实候选、校准 α 并检查过程奖励。此处进度文档晚于部署快照更新，远端代码快照和
原 release manifest 保持冻结，不把文档状态更新冒充源码/模型的新实验。

## 13. 重要文件地图

所有路径均相对于项目根目录 `/home/zjulab/kgpaper`。标为 `PENDING` 的路径不得假装已存在。

### 13.1 项目规则与入口

| 用途 | 路径 |
|---|---|
| 全局科研规则 | `AGENTS.md` |
| 项目说明 | `README.md`（其中旧 5k light-curriculum/通用训练示例属历史说明；当前训练以本文件为准） |
| 当前权威工作流 | `RESEARCH_WORKFLOW.md` |
| 当前逐项执行清单 | `docs/todo2.md` |
| 身份账本人工可读审计 | `docs/mixed_ppo_v4_identity_ledger_audit.md` |
| 历史完整 TODO | `docs/todo.md` |
| 历史重训练记录 | `docs/retraining_plan.md` |
| baseline 汇总 | `docs/baselines_final.md` |
| 当前论文整合稿 | `docs/paper/polished_draft.md` |
| 论文分章节文件 | `docs/paper/00_abstract.md` -- `docs/paper/07_references.md` |
| 本地参考论文 | `docs/paper/references/` |

### 13.2 固定模型与历史日志

| 用途 | 路径 / 状态 |
|---|---|
| Strong SFT 起点 | `checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final/` |
| Strong SFT manifest | `checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/manifest.json` |
| Strong SFT 日志 | `logs/training/sft_quota70_hard_seed42_no_text_head.log` |
| ReaRAG 文本奖励模型 | `models/rearag-9b/` |
| 旧 α checkpoint（仅历史基线） | `checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42_no_text_head/alpha_gate.pt` |
| 历史 combined smoke 日志 | `logs/training/ppo_quota70_hard_seed42_no_text_head_smoke600_combined_stability_v1.log` |
| 历史 hybrid smoke 日志 | `logs/training/ppo_quota70_hard_seed42_no_text_head_smoke600_hybrid_old10_bridge5_v3.log` |

### 13.3 v4 身份、passages 与 replay

| 用途 | 权威路径 |
|---|---|
| complete protected ledger v2 | `outputs/audits/mixed_ppo_v4_protected_identity_ledger_v2/` |
| H/M reconciliation protocol | `outputs/audits/mixed_ppo_v4_hm_full_ledger_reconciliation_v2_seed42_preregistration/` |
| H/M 823 final retrieval | `outputs/audits/mixed3_v4_expansion_retrieval_h417_m406_seed42_v2/` |
| 2Wiki ordinary200 successor | `outputs/audits/2wiki_ordinary200_full_ledger_v2_seed42_preregistration/` |
| clean replay v2 | `data/silver_data/sft_replay_legacy_strong_train_rendered3to5_n2000_seed42_v2/` |
| v4 protocol | `outputs/audits/mixed_ppo_three_dataset_v4_proof800_n3000_k4_seed42_protocol/`（`FROZEN/PASS`；protocol SHA `85640c54...bccc`） |
| final v4 dataset | `data/silver_data/mixed_ppo_three_dataset_v4_proof800_n3000_k4_seed42/`（`COMPLETE_DATA_NOT_TRAINED`；report SHA `375e3b20...eefa`；manifest SHA `c7688f11...c752`） |
| final v4 preflight | `outputs/audits/mixed_ppo_three_dataset_v4_proof800_n3000_k4_seed42_preflight/`（`PASS_NOT_TRAINED`；46/46；report SHA `94876f0e...fbb2`） |

### 13.4 2Wiki n1500 → Proof800 链路

| 阶段 | 权威路径 / 状态 |
|---|---|
| Gold-free n1500 cohort | `outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_seed42_preregistration/` |
| planner execution protocol | `outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_seed42_planner_execution_v1_preregistration/` |
| planner predictions | `outputs/validation/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_seed42_plans_v1/` |
| planner postflight | `outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_seed42_plans_v1_postflight/` |
| v6 evidence store | `indexes/versioned_2wiki_evidence_store_v6_mixed3_v4_complete_ledger_seed42/` |
| 当前 root protocol v2 | `outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_root_resolution_v2_preregistration/`（目录别名v2；内部 Experiment ID 为V2B） |
| 当前 root runtime | `indexes/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_root_resolution_v2/` (`COMPLETE/PASS`) |
| 当前 closure method lock | `outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_clean_closure_v3_preregistration/` |
| closure-v3 execution lock | `outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_clean_closure_v3_execution_lock/` |
| closure-v3 runtime | `data/derived/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_clean_closure_v3/` (`COMPLETE/PASS`) |
| closure-v3 authority result | `outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_clean_closure_v3_result/`（strict eligible=`1287/1500`） |
| retrieval scope policy | `outputs/audits/2wiki_official_raw_canonical_retrieval_v1_scope_policy_preregistration/`（`FROZEN`；SHA `16e41a7...f4879`） |
| exact retrieval scope | `outputs/audits/2wiki_official_raw_canonical_retrieval_v1_scope_preregistration/`（`FROZEN`；1287题） |
| official-raw canonical retrieval | `outputs/audits/2wiki_official_raw_canonical_retrieval_v1/`（`COMPLETE/PASS`；1287题、41/41批；contexts SHA `a935351a...f9b4`） |
| retrieval Attempt1 failure audit | `outputs/audits/2wiki_official_raw_canonical_retrieval_v1_attempt1_failure/report.json`（`FAILED_NO_RELEASE_UNKNOWN_EXIT_NOT_TRAINED`） |
| unified-v3 contract | `outputs/audits/2wiki_unified_proofkg_official_raw_v3_contract/unified_contract.json`（`FROZEN`；SHA `3950ac3...2ba9`） |
| unified-v3 candidate supply | `data/derived/2wiki_unified_proofkg_official_raw_v3/`（`COMPLETE/PASS`；1287四路join；report SHA `f0677560...2468`） |
| strict Proof800 protocol v1 | `outputs/audits/2wiki_proof800_strict_selection_v1_seed42_preregistration/`（只接受closure-v2，已知不兼容权威closure-v3；保留、不执行） |
| strict Proof800 v2 protocol P0REFRESH2 | `outputs/audits/2wiki_proof800_strict_selection_v2_seed42_preregistration_p0refresh2/`（`FROZEN/PASS`；protocol SHA `86803138...e10b`） |
| strict Proof800 v2 result P0REFRESH2 | `outputs/audits/2wiki_proof800_strict_selection_v2_seed42_result_p0refresh2/`（`COMPLETE_STRICT_PROOF800_NOT_TRAINED`；四类各200；proof SHA `8e58d829...1fc3`） |
| selector P0REFRESH1失败记录 | `outputs/audits/2wiki_proof800_strict_selection_v2_seed42_result_p0refresh1_failure/report.json`（schema-adapter失败；无result、未训练；保留） |

不得使用：root-resolution v1、closure-v1、closure-v2；它们均已有 append-only supersession 标记。

Proof800-v2 应保持下列单向接口；脚本名、协议目录和实现 hash 以 v2 正式冻结结果为准：

```text
<authoritative-selector-v2> select
  --closure-dir <完成的closure release>
  --unified-supply-dir <完成的canonical unified supply>
        ↓
outputs/<proof800-v2-result>/proof_candidates.jsonl
        ↓
freeze_mixed_ppo_three_dataset_v4_proof800.py --proof_candidates <上述文件>
        ↓
materialize_mixed_ppo_three_dataset_v4_proof800.py
```

### 13.5 核心实现

| 功能 | 路径 |
|---|---|
| PPO 主训练实现 | `kgproweight/training/phase3_ppo.py` |
| PPO reward 装配 | `kgproweight/training/reward_function.py` |
| trajectory source hard gate | `kgproweight/reward/trajectory_source_gate.py` |
| ProofKG process scorer v2.2 | `kgproweight/reward/proofkg_process_v2_2.py` |
| ReaRAG text reward wrapper | `kgproweight/reward/text_reward_model.py` |
| α gate 实现底座 | `kgproweight/reward/alpha_gate.py` |
| n1500 full-root freezer | `scripts/prepare/freeze_2wiki_official_raw_full_root_resolution_v2.py` |
| n1500 full-root materializer | `scripts/prepare/materialize_2wiki_full_root_resolution_v2.py` |
| n1500 closure runner | `scripts/prepare/run_2wiki_official_raw_n1500_clean_closure_v1_locked.py` |
| 历史 unified ProofKG materializer | `scripts/prepare/materialize_2wiki_proofkg_unified_v2.py`（旧 extension/reserve 行为保持不变） |
| official-raw retrieval scope freezer | `scripts/prepare/freeze_2wiki_official_raw_canonical_retrieval_v1.py` |
| official-raw retrieval materializer | `scripts/prepare/materialize_2wiki_official_raw_canonical_retrieval_v1.py` |
| official-raw unified-v3 materializer | `scripts/prepare/materialize_2wiki_proofkg_unified_v3.py` |
| strict Proof800 selector | `scripts/prepare/select_2wiki_proof800_v1.py` |
| strict Proof800 selector v2 | `scripts/prepare/select_2wiki_proof800_v2.py`（P0REFRESH2已完成；四类各200） |
| v4 protocol freezer | `scripts/prepare/freeze_mixed_ppo_three_dataset_v4_proof800.py` |
| v4 final materializer | `scripts/prepare/materialize_mixed_ppo_three_dataset_v4_proof800.py` |
| v4 strict preflight | `scripts/prepare/preflight_mixed_ppo_three_dataset_v4_proof800.py` |

注意：closure runner 文件名中的 `v1` 是历史脚本名，不表示当前方法版本；权威方法由传入的 closure-v3
protocol 及后续 execution lock 决定。

### 13.6 Evaluation 与已成立证据

| 用途 | 路径 |
|---|---|
| canonical evaluator | `scripts/eval/run_kg_proweight.py` |
| canonical legacy eval config | `configs/eval/kg_proweight.yaml` |
| IHR judge | `scripts/eval/run_ihr_judge.py` |
| 2Wiki matched-control 分数 | `outputs/audits/2wiki_matched_control_score_v2/matched_control_score.json` |
| n300 root projection 错误审计 | `outputs/audits/2wiki_proofkg_extension_v1b_n300_root_projection_gap_audit_v1/report.json` |
| n300 closure-v2 失败记录 | `outputs/audits/2wiki_proofkg_extension_v1b_n300_clean_closure_v2_result/report.json` |
| Proof scorer v2.2 静态审计 | `outputs/audits/proofkg_process_v2_2_static_audit_v3/report.json` |

### 13.7 尚不存在、禁止误用的正式资产

本节按source-credit-v1最新状态更新；历史sourcegate-v1与format-only artifact不能替代当前门。

- v4 PPO-T/F/A YAML已存在：`configs/training/phase3_ppo_mixed4_source_credit_v1_{a,f,t}_{probe,smoke,full}_seed42.yaml`；正式训练launch lock未签发；
- 新α artifact已存在：`outputs/calibration/source_credit_gate_v1_local_seed42/gate.json`，仅通过启发式校准；
- 冻结source credit mask：`outputs/audits/source_credit_mask_v1_local_seed42/manifest.json`；
- v4 train-only过程排序和cached真实分数runtime检查已完成；独立过程效用confirmation仍`PENDING`；
- v4 96GB runtime probe、600 smoke、正式 PPO checkpoints：`NOT_STARTED`；
- v4 三数据集 fresh evaluation 与 IHR：`NOT_STARTED`。

看到旧 `...text7200...`、`...proof400...` 或旧 α checkpoint 时，应先回到本文件判断其历史角色，
不得把它们当作当前 v4 的正式配置。
