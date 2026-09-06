# Proof-KG curriculum SFT/PPO 启动说明

当前状态：代码、数据、配置与CPU预检均完成；本机CUDA driver不可用，训练需在96GB服务器运行。

训练数据为8000条：4000条既有Hotpot train replay和4000条2Wiki train-only、gold-derived
Proof-KG（四题型各1000）。后者只用于教模型使用完整证据链，禁止作为自动planner或评估结果。

关键修复：

- SFT可从现有强LoRA继续训练，不再从裸base重新开始；
- SFT、PPO rollout、RewardSpec和10% replay按`dataset::qid + question hash`使用同一KG；
- 8000/8000覆盖在加载模型前硬检查；
- records模式已明确旁路legacy question-text索引，避免100%假miss及短Proof-KG被二次过滤；
- Proof-KG PPO允许精确的一边KG产生过程奖励（阈值1），历史配置仍为3；
- 全量token预检最长4959<6144，无截断、无空监督。

2026-08-30二次审计：8000个PPO prompt全部构建成功，最大4255 tokens；4000条Proof均非空；
10069个Proof证据步在PPO规则奖励下全部得到`r_kg=1.0`；项目全量393项测试通过。

边界：这表示训练工程链可用，不表示端到端天花板全部解除。自动planner、Hotpot/MuSiQue
Proof覆盖、推理期缺失第二跳和PPO探索/采样上限仍需训练后及独立未见集验证。

96GB服务器按顺序运行：

```bash
bash launch_proofkg_curriculum_sft_remote.sh
# SFT完成并确认manifest=COMPLETE、loss正常后：
bash launch_proofkg_curriculum_ppo_smoke_remote.sh
```

旧SFT在96GB上4751条/148 updates耗时约1小时51分；本轮8000条/250 updates，按旧吞吐粗估
约3小时，实际为`UNKNOWN`直到日志产生。PPO smoke为600 trajectories/150 updates。

PPO成功标准不是只“不退化”：必须相对本轮新SFT checkpoint比较三数据集pipeline、旧replay
与未见Proof-KG集，并同时查看`r_kg`、citation、valid、KL与critic。只有超过新SFT起点才算
达到目标；与旧SFT打平不算成功。
