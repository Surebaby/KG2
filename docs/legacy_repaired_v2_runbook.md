# Legacy-repaired-v2 后续实验运行手册

> 状态（2026-08-26）：Phase 2、SFT及CE=0.10/0.30两轮PPO smoke已完成；两轮均在
> 同qid val上显著低于SFT。已修复TRL PEFT分支绕过显式SFT reference的问题，待运行
> correctness smoke；`kl_coef=0.25`方案已否决。  
> 研究边界：这是“不重新调用 Teacher”的旧轨迹修复线，只能检验下游训练恢复，不能证明银标生成质量提高。
> 可逐项执行与签字验收的完整清单见
> [`quota70_training_checklist.md`](quota70_training_checklist.md)。

所有命令固定使用 `/home/zjulab/anaconda3/envs/kgpaper/bin/python`。系统默认
`python` 指向另一套 Python 3.12/torch 环境，不得用于正式运行。每个训练配置的
`output_dir` basename 即唯一 Experiment ID；目录已存在时代码会拒绝覆盖。

## 固定数据资产

- 推荐主线 Silver（quota-70）：`data/silver_data/silver_legacy_repaired_v2_quota70_hotpot_train.jsonl`
  - MD5：`5974fca49f4d944ad1badcd8e2595aa1`
  - 24,998 records；5,903 accepted；仅 HotpotQA train 来源。
  - split seed=42、val=0.10、test=0.10：train/val/test accepted = 4,751/573/579。
  - train fold 15,653 steps：ONE=2,198、ZERO=12,817、NEG=623、严格小数=15。
- 高KG密度对照 Silver（quota-25）：`data/silver_data/silver_legacy_repaired_v2_hotpot_train.jsonl`
  - MD5：`79326c3f23fc46a0fe5705ae11a2c4c9`
  - 2,361 accepted；train/val/test accepted = 1,909/219/233。
- Candidate sidecar：`data/silver_data/silver_legacy_repaired_v2_hotpot_train.candidates.jsonl`
  - 8,598 intrinsic quality-pass；配额淘汰的记录仍保留，禁止删除。
- 推荐主线 KG index：`indexes/kg_cache/question_kg_index_legacy_repaired_v2_quota70_hotpot_train.json`
  - MD5：`bb47221b263840e64af82250cb56ac87`
  - 24,997 unique questions；重复问题1条且 KG 完全一致。
- quota-70 数据报告：`data/silver_data/silver_legacy_repaired_v2_quota70_hotpot_train.report.json`
- quota-70 索引报告：`data/silver_data/silver_legacy_repaired_v2_quota70_hotpot_train.kg_index_report.json`
- quota-70 fold 对齐报告：同目录的 `index_alignment_{train,val,test}.report.json`。

## 已冻结协议

- 原始 `silver_trajectories.jsonl` 只读，MD5=`d0501ae80359e7038925ccd60e7e681d`。
- current parser + current conservative PRM；不使用 `rule_continuous_v1.1`。
- KG：strict top-12/min-5，不把被过滤的历史引用 union 回学生可见 KG。
- KG 外引用或 citation-contract 错误：整条轨迹 quality reject。
- passages：派生文件保留前15篇，与 PPO 最大输入篇数一致；原始50篇仍在旧文件中。
- quota-70 是对同一不可变candidate的单变量重选：medium<=35%保持不变，仅sparse上限由25%改为70%，seed=42；逐条`{qid,steps,kg_subgraph}`哈希与quota-25完全一致。
- quota-70 train fold索引 absent=0、KG mismatch=0、非空KG=3,600/4,751（75.77%）；quota-25为92.35%，因此两者不得当作同一KG密度条件直接比较。
- 正式Llama tokenizer预检：4,751/4,751 prompt均完整保留KG block，最大4,255 tokens
  （预算6,144）；3,600条为非空KG，1,151条为已覆盖但确实为空，不是索引缺失。
- train/val/test：0.80/0.10/0.10，split seed=42，所有训练阶段必须使用 `split=train`。

## Phase 2 hard-target 恢复基线

这是 GPU 正式训练，运行前仍需研究者单独确认。

```bash
PYTHONPATH=/home/zjulab/kgpaper:/home/zjulab/kgpaper/flashrag_src \
/home/zjulab/anaconda3/envs/kgpaper/bin/python scripts/train/phase2_train_prm.py \
  --config configs/training/phase2_prm_legacy_repaired_v2_quota70.yaml
```

验收：train fold=4,751 accepted / 15,653 steps；输出 `alpha_gate.pt` 的特征数为5；
manifest 必须为 `COMPLETE`，记录 Silver/enriched Silver/模型/entity cache 哈希、split、
seed、Git dirty/diff/source-tree hash；不得覆盖旧 Phase 2 checkpoint。

Phase 2 完成后、进入 SFT 前必须执行：

```bash
PYTHONPATH=/home/zjulab/kgpaper:/home/zjulab/kgpaper/flashrag_src \
/home/zjulab/anaconda3/envs/kgpaper/bin/python scripts/pilot/audit_silver_index_alignment.py \
  --silver checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42/silver_with_logprobs.jsonl \
  --index indexes/kg_cache/question_kg_index_legacy_repaired_v2_quota70_hotpot_train.json \
  --split train \
  --output checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42/kg_alignment_train.report.json

PYTHONPATH=/home/zjulab/kgpaper:/home/zjulab/kgpaper/flashrag_src \
/home/zjulab/anaconda3/envs/kgpaper/bin/python scripts/pilot/score_alpha_gate.py \
  --silver checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42/silver_with_logprobs.jsonl \
  --gate hard=checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42/alpha_gate.pt \
  --split val \
  --output checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42/alpha_gate_val_report.json
```

完整性硬门：KG alignment 必须 `PASS`、absent=0、mismatch=0。候选科学判据是
gate 的 held-out `r2_vs_constant>=0`（至少不劣于常数预测），但这属于进入下一阶段
的 evaluation gate，须由研究者看完 calibration 报告后明确批准，代码不会自行放行。

## Phase 3a SFT

Phase 2 成功后才运行；使用 Phase 2 写回 token logprobs 的数据副本。

```bash
PYTHONPATH=/home/zjulab/kgpaper:/home/zjulab/kgpaper/flashrag_src \
/home/zjulab/anaconda3/envs/kgpaper/bin/python scripts/train/phase3_sft.py \
  --config configs/training/phase3_sft_legacy_repaired_v2_quota70.yaml
```

完成后只用 val（不得提前看 test）验证：

```bash
PYTHONPATH=/home/zjulab/kgpaper:/home/zjulab/kgpaper/flashrag_src \
/home/zjulab/anaconda3/envs/kgpaper/bin/python scripts/eval/validate_sft.py \
  --adapter checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42/final \
  --silver checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42/silver_with_logprobs.jsonl \
  --split val --n 200 --seed 42 \
  --out checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42/val_seed42_n200.jsonl
```

## Phase 3b PPO smoke

Phase 2/SFT 均通过后，再单独请求研究者批准。600 trajectories 是 smoke，不是正式结果。

```bash
PYTHONPATH=/home/zjulab/kgpaper:/home/zjulab/kgpaper/flashrag_src \
/home/zjulab/anaconda3/envs/kgpaper/bin/python scripts/train/phase3_ppo.py \
  --config configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600.yaml
```

该 smoke 使用**10% 同题完整轨迹监督 replay**：PPO rollout 不含 gold；batch=4 下由
fractional-credit scheduler 长期精确实现10%，并记录 `sft_replay_items_seen`、
`sft_replay_actual_ratio` 和 `sft_replay_loss`。旧的裸答案 interval anchor 已关闭。
10%指replay样本数占PPO样本数；每个replay CE另乘当前冻结的
`sft_anchor_weight=0.10`。不能在结果出来前把它描述成已证明能防退化。

停止/验收门：update约50时 `valid_rate` 仍未超过0.5则停止；结束后检查最后10%更新：
`valid_rate>=0.85`、每轨迹平均解析步数 `>=2.8`、`kg_reward_share>=0.10`、
replay实际比例接近0.10、无 NaN/Inf，且 `dR/dalpha` 不应持续为负。600 trajectories
只用于判断稳定性，不能作为正式论文结果。

### 首轮结果与CE=0.30重跑

首轮`...smoke600_replay10`已完成：60/600 replay、末段步数和KG share通过，但末段
valid rate=0.7667；同一val n=200上SFT→PPO的EM `0.790→0.730`、F1
`0.83787→0.78820`，因此不得进入正式PPO。旧输出、日志和失败结论必须保留。

研究者批准只提高完整轨迹CE强度，ratio继续保持10%。新运行命令为：

```bash
PYTHONPATH=/home/zjulab/kgpaper:/home/zjulab/kgpaper/flashrag_src \
/home/zjulab/anaconda3/envs/kgpaper/bin/python scripts/train/phase3_ppo.py \
  --config configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600_replay10_ce030.yaml
```

新ID为`ppo_legacy_repaired_v2_quota70_hard_seed42_no_text_head_smoke600_replay10_ce030`。
resolved config相对首轮除output目录外只能有`sft_anchor_weight: 0.10→0.30`这一项差异。
启动仍属于GPU训练，运行前检查新输出目录和新日志不存在，并单独获得研究者启动确认。

### 显式SFT reference correctness smoke

CE=0.30最终EM/F1=`0.725/0.78137`，未改善。代码复核确认TRL 0.11.4在PEFT模式下
忽略传入的冻结SFT reference，实际通过`disable_adapter()`对裸base计算KL；首批
`objective/kl=65.4414`。因此不得用该错误KL把`kl_coef`提高到0.25。

修复后新run恢复`kl_coef=0.15`、replay=10%、CE=0.10和tokens=256：

```bash
bash launch_ppo_smoke_explicit_sft_ref_remote.sh
```

对应配置为
`configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600_explicit_sft_ref.yaml`。
首批KL必须finite且绝对值<=1.0，否则代码自动终止。该run仍需单独GPU启动批准；不得
同时改到tokens=384或调整Reward/value loss。

开跑前确认新manifest的`source_tree_sha256`非空；远端无Git时
`source_tree_hash_mode`应为`filesystem_fallback`。history中的原始引用诊断应同时查看
`citation_raw_citing_step_frac`、`citation_unknown_surface_count`、
`citation_malformed_content_step_frac`和`citation_known_frac_recognized_surfaces`；
`cite_match_mean_citing_step`只是reward-visible已知引用口径。

### 一次性组合稳定性 smoke（2026-08-27）

研究者因付费GPU成本批准不再逐项smoke，改跑一次多变量组合。新配置同时使用384 tokens、
`kl_coef=0.25`、correct-reference `target_kl=8`、零初始化/无dropout critic和2个PPO
epochs；10% replay与CE=0.10保持不变。该实验不能用于单项消融归因。

```bash
bash launch_ppo_smoke_combined_stability_remote.sh
```

launcher先运行定向回归测试，再创建唯一输出。训练从200 trajectories起按15-update窗口
执行成本止损；若valid<0.50、cap>0.50或KL>20，会保留history/checkpoint并写
`FAILED` manifest。完整跑到600仍不代表通过，必须继续执行冻结n=200验证。

## 当前不运行的分支

- `phase2_prm_soft_alpha.yaml`：quota-70 train的15,653 steps中只有15个严格小数正例（0.096%），当前数据上 hard/soft target几乎无信息差；先不消耗GPU。
- quota-25正式全链训练：保留为高KG密度对照候选；先让quota-70通过Phase2验收和PPO smoke，不同时启动两条GPU线。
- rule-continuous v1.1：尚无人工 heldout 验证，不接核心 Reward/Label。
- V4-Flash/V4-Pro 全量 Teacher 生成：本路线明确选择不重新生成。
- 2Wiki/MuSiQue 训练银标：仓库已有的 `silver_v6_full_20260801_0136.jsonl` 来自 dev split，不能混入训练；本路线对两数据集只能做 OOD evaluation。
