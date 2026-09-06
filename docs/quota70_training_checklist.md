# quota70 后续训练执行清单

> 状态（2026-08-26）：Phase2、SFT和首轮PPO smoke已完成。首轮PPO未通过同qid val
> 保持门；当前待运行CE weight=0.30单变量smoke，不得进入正式PPO。  
> 数据线：`legacy_repaired_v2_quota70`，只使用 HotpotQA train 来源。  
> 原则：每个训练阶段使用唯一 Experiment ID；失败目录、日志和 checkpoint 一律保留；
> Phase2、SFT、PPO 分阶段审批，前一阶段验收前不得启动后一阶段。

## 0. 固定资产与当前边界

- [x] Silver：`data/silver_data/silver_legacy_repaired_v2_quota70_hotpot_train.jsonl`
  - MD5：`5974fca49f4d944ad1badcd8e2595aa1`
  - 24,998 records；5,903 accepted。
  - train/val/test accepted：4,751 / 573 / 579，split seed=42。
- [x] KG index：`indexes/kg_cache/question_kg_index_legacy_repaired_v2_quota70_hotpot_train.json`
  - MD5：`bb47221b263840e64af82250cb56ac87`
  - 三折 absent=0、ordered KG mismatch=0。
  - train 非空 KG：3,600/4,751（75.77%）；其余1,151条是covered-empty，不是索引缺失。
- [x] Wiki18 corpus/dense/BM25：均为21,015,324条。
- [x] 正式Llama tokenizer prompt预检：4,751/4,751保留完整KG block；最大4,255
  tokens，小于6,144；当前无需删除任何15篇passage。
- [x] PPO replay预检：2,000个唯一train qid，完整`[Step] + [Final Answer]`，最大
  4,771 tokens；不向rollout传入gold。
- [x] 全量测试271项通过；Python编译、Shell语法和`git diff --check`通过。
- [ ] 当前CUDA/NVIDIA driver不可用，尚不能开始Phase2。
- [ ] 工作区仍有大量未提交修改；正式训练前固定Git commit或明确记录dirty tree。
- [ ] 文件系统仅约65GB可用、使用率97%；启动前再次确认空间并持续监控。
- [ ] 所有曾在聊天或Git历史中出现的API key已在服务端轮换。当前路线不调用Teacher，
  训练不需要DeepSeek API。

## 1. GPU与训练环境预检

以下命令全部通过后，才能申请启动Phase2：

```bash
set -euo pipefail
KGPW_ROOT=/home/zjulab/kgpaper
KGPW_PY=/home/zjulab/anaconda3/envs/kgpaper/bin/python
cd "$KGPW_ROOT"

nvidia-smi
"$KGPW_PY" -c 'import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported()'
df -h "$KGPW_ROOT"
git rev-parse HEAD
git status --short

PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src" \
  "$KGPW_PY" -m pytest -q

md5sum \
  data/silver_data/silver_legacy_repaired_v2_quota70_hotpot_train.jsonl \
  indexes/kg_cache/question_kg_index_legacy_repaired_v2_quota70_hotpot_train.json

PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src" \
  "$KGPW_PY" scripts/prepare/check_wiki18_assets.py \
    --corpus indexes_wiki18/bm25/corpus.jsonl \
    --dense indexes_wiki18/e5_fp16.dat \
    --bm25 indexes_wiki18/bm25
```

必须同时满足：

- [ ] `torch.cuda.is_available()`和bf16检查通过；GPU型号、显存写入运行记录。
- [ ] Silver与index MD5和第0节一致。
- [ ] Wiki18报告`PASS`且三个计数均为21,015,324。
- [ ] 271项测试通过；若测试数变化，记录新增/删除原因。
- [ ] 下列正式输出目录均不存在：

```bash
test ! -e checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42
test ! -e checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42
test ! -e outputs/ppo_legacy_repaired_v2_quota70_hard_seed42_smoke600_replay10
```

任何一项失败：停止，不创建替代数据、不覆盖旧目录。

## 2. Phase2 启动前人工判定

固定项：

- alpha target：`hard_verdict`。
- Phase2辅助PRM：三分类CE，仅作诊断，不进入PPO。
- alpha gate：5 features，`BCEWithLogitsLoss` logits形式。
- fold：train；seed=42；split seed=42。
- 输入：quota70 Silver；输出使用新的固定Experiment ID。

仍需研究者选择一项：

- [ ] **A：关闭未消费的`text_reward_head`（建议）**。命令增加`--no_text_head`；
  PPO使用ReaRAG，因此不保存一个不会被加载的fallback head，也避免其辅助MSE改变共享
  LoRA表征。相对历史Phase2属于核心loss变化，必须在manifest中记录。
- [ ] **B：保留历史辅助head**。不加`--no_text_head`；恢复历史Phase2组合loss，但正式
  PPO仍不会加载该head。不能把它的存在描述成PPO文本奖励来源。

在A/B未勾选前，不运行Phase2。若选择A，建议把Experiment ID/output目录名增加
`_no_text_head`，避免和B混淆；下面命令以当前冻结的B目录名为例。

## 3. 运行 Phase2

### 3.1 启动审批

- [ ] 研究者明确回复批准启动Phase2。
- [ ] 记录Git commit、dirty状态、GPU、CUDA、配置路径和A/B选择。
- [ ] 日志文件不存在，避免`tee`覆盖失败记录。

### 3.2 命令

```bash
set -euo pipefail
KGPW_ROOT=/home/zjulab/kgpaper
KGPW_PY=/home/zjulab/anaconda3/envs/kgpaper/bin/python
PHASE2_LOG="$KGPW_ROOT/logs/training/phase2_quota70_hard_seed42.log"
cd "$KGPW_ROOT"
mkdir -p logs/training
test ! -e "$PHASE2_LOG"
test ! -e checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42

PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src" \
  "$KGPW_PY" scripts/train/phase2_train_prm.py \
    --config configs/training/phase2_prm_legacy_repaired_v2_quota70.yaml \
    2>&1 | tee "$PHASE2_LOG"
```

若第2节选择A，在`--config ...`后增加`--no_text_head`，并同步修改专用配置的
`output_dir`/日志名/Experiment ID；不得复用上面B分支的目录名。

### 3.3 运行中检查

- [ ] 日志解析值：fold=train、4,751 accepted、15,653 step samples。
- [ ] batch=4、grad accum=4、effective batch=16、epochs=3、max length=1,024。
- [ ] 无OOM、NaN/Inf、数据路径回退或split=None警告。
- [ ] GPU显存和磁盘空间没有逼近上限。
- [ ] 若失败，保留`RUNNING` manifest、输出目录和日志；修复后使用新Experiment ID。

## 4. Phase2 完成后验收

### 4.1 产物完整性

- [ ] `manifest.json`状态为`COMPLETE`，且Experiment ID、seed、resolved config、
  Silver/base model/entity cache哈希齐全。
- [ ] `silver_with_logprobs.jsonl`、`alpha_gate.pt`、`prm_head/`存在。
- [ ] A分支不应产生`text_reward_head.pt`；B分支应记录它存在但不接PPO。
- [ ] enriched Silver仍为24,998 records，accepted仍为5,903；不得只剩train fold。
- [ ] alpha gate真实权重数为5，不能依赖旧3-feature checkpoint的兼容零填充。

```bash
jq '{status,started_at,completed_at,run}' \
  checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42/manifest.json

/home/zjulab/anaconda3/envs/kgpaper/bin/python -c \
  "import torch; p='checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42/alpha_gate.pt'; s=torch.load(p,map_location='cpu'); print({k:tuple(v.shape) for k,v in s.items()}); assert tuple(s['W'].shape)==(5,)"
```

### 4.2 enriched Silver 与 KG index 三折重审计

```bash
set -euo pipefail
KGPW_ROOT=/home/zjulab/kgpaper
KGPW_PY=/home/zjulab/anaconda3/envs/kgpaper/bin/python
AUDIT_DIR="$KGPW_ROOT/outputs/audits/phase2_quota70_hard_seed42"
cd "$KGPW_ROOT"
test ! -e "$AUDIT_DIR"
mkdir -p "$AUDIT_DIR"

for KGPW_FOLD in train val test; do
  PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src" \
    "$KGPW_PY" scripts/pilot/audit_silver_index_alignment.py \
      --silver checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42/silver_with_logprobs.jsonl \
      --index indexes/kg_cache/question_kg_index_legacy_repaired_v2_quota70_hotpot_train.json \
      --split "$KGPW_FOLD" \
      --output "$AUDIT_DIR/kg_alignment_${KGPW_FOLD}.report.json"
done
```

三份报告都必须满足：

- [ ] `status=PASS`。
- [ ] `absent=0`、`kg_mismatch=0`。
- [ ] accepted计数仍为4,751 / 573 / 579。
- [ ] train非空KG仍为3,600；若改变，停止并定位Phase2是否改写了KG。

### 4.3 held-out alpha gate 验收

```bash
PYTHONPATH=/home/zjulab/kgpaper:/home/zjulab/kgpaper/flashrag_src \
/home/zjulab/anaconda3/envs/kgpaper/bin/python scripts/pilot/score_alpha_gate.py \
  --silver checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42/silver_with_logprobs.jsonl \
  --gate hard=checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42/alpha_gate.pt \
  --split val --val_ratio 0.10 --test_ratio 0.10 --split_seed 42 \
  --output outputs/audits/phase2_quota70_hard_seed42/alpha_gate_val_report.json
```

- [ ] 只看val，不看test。
- [ ] 报告BCE、Brier、MAE、`r2_vs_constant`、alpha mean/std、10-bin calibration。
- [ ] 候选门：hard target的`r2_vs_constant>=0`；这是研究者判定门，不由代码自动批准。
- [ ] 检查alpha是否饱和；不得因均值“看起来合理”就宣称学会自适应。
- [ ] 当前评分脚本尚未给出`cite_any=0/1`条件分层；若该分层被定为硬门，需先经研究者
  批准补充evaluation protocol，再决定SFT。
- [ ] 研究者查看完整报告并明确批准/拒绝进入SFT。

## 5. 运行 Phase3a SFT

### 5.1 启动条件

- [ ] Phase2 manifest为`COMPLETE`。
- [ ] 三折KG alignment全部通过。
- [ ] alpha held-out报告已人工审阅。
- [ ] 研究者明确批准启动SFT。
- [ ] SFT输出目录和日志均不存在。

### 5.2 命令

```bash
set -euo pipefail
KGPW_ROOT=/home/zjulab/kgpaper
KGPW_PY=/home/zjulab/anaconda3/envs/kgpaper/bin/python
SFT_LOG="$KGPW_ROOT/logs/training/sft_quota70_hard_seed42.log"
cd "$KGPW_ROOT"
mkdir -p logs/training
test ! -e "$SFT_LOG"
test ! -e checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42

PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src" \
  "$KGPW_PY" scripts/train/phase3_sft.py \
    --config configs/training/phase3_sft_legacy_repaired_v2_quota70.yaml \
    2>&1 | tee "$SFT_LOG"
```

### 5.3 SFT完成检查

- [ ] manifest为`COMPLETE`，Silver和base model哈希正确。
- [ ] `final/adapter_model.safetensors`、`adapter_config.json`和tokenizer文件存在。
- [ ] 日志确认只使用train fold，没有val/test训练泄漏。
- [ ] assistant labels只覆盖完整轨迹，不覆盖prompt；无裸答案target和右截断答案警告。
- [ ] 失败时保留目录和日志，新实验使用新ID。

## 6. SFT held-out val 验收

先运行base-only对照，再运行SFT；两者使用相同fold、n、seed和greedy decode。

```bash
set -euo pipefail
KGPW_ROOT=/home/zjulab/kgpaper
KGPW_PY=/home/zjulab/anaconda3/envs/kgpaper/bin/python
SILVER="$KGPW_ROOT/checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42/silver_with_logprobs.jsonl"
VAL_DIR="$KGPW_ROOT/outputs/validation/sft_quota70_hard_seed42_val_n200"
cd "$KGPW_ROOT"
test ! -e "$VAL_DIR"
mkdir -p "$VAL_DIR"

PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src" \
  "$KGPW_PY" scripts/eval/validate_sft.py \
    --base_only --silver "$SILVER" --split val --n 200 --seed 42 \
    --out "$VAL_DIR/base_seed42_n200.jsonl" \
    2>&1 | tee "$VAL_DIR/base_seed42_n200.log"

PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src" \
  "$KGPW_PY" scripts/eval/validate_sft.py \
    --adapter checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42/final \
    --silver "$SILVER" --split val --n 200 --seed 42 \
    --out "$VAL_DIR/sft_seed42_n200.jsonl" \
    2>&1 | tee "$VAL_DIR/sft_seed42_n200.log"
```

必须报告，不得只挑有利数字：

- [ ] 两臂实际qids完全相同。
- [ ] parse rate、step numbering contiguous、EM、F1、step histogram。
- [ ] gold visible/not-visible passages分层EM。
- [ ] SFT是否相对base提高格式合规且没有明显损害EM/F1。
- [ ] 当前`validate_sft.py`的`well_formed`只要求至少1步+final answer，并不等同PPO的
  `min_valid_steps=3 + min_reasoning_chars=20`。进入PPO前必须人工查看生成，并报告
  `n_steps>=3`比例；若要把完整PPO predicate设为新的硬门，需要研究者批准evaluation
  protocol后再实现，不能偷偷改口径。
- [ ] 不查看test；研究者明确批准/拒绝进入PPO smoke。

## 7. 运行 Phase3b PPO 600-trajectory smoke

### 7.1 冻结配置与审批

- [ ] Phase2/SFT及val全部通过并获人工批准。
- [ ] PPO配置解析为：600 trajectories、batch=4、150 updates、save every 200
  trajectories、ReaRAG、split=train、exact KG alignment、10% replay。
- [ ] replay含义确认：样本比例0.10；CE系数`sft_anchor_weight=0.10`；旧interval=0。
- [ ] 不把CE系数擅自改成1.0；若修改，建立新配置和Experiment ID并再次批准。
- [ ] PPO输出目录与日志不存在。
- [ ] 研究者明确批准启动PPO smoke。

### 7.2 命令

```bash
set -euo pipefail
KGPW_ROOT=/home/zjulab/kgpaper
KGPW_PY=/home/zjulab/anaconda3/envs/kgpaper/bin/python
PPO_LOG="$KGPW_ROOT/logs/training/ppo_quota70_hard_seed42_smoke600_replay10.log"
cd "$KGPW_ROOT"
mkdir -p logs/training
test ! -e "$PPO_LOG"
test ! -e outputs/ppo_legacy_repaired_v2_quota70_hard_seed42_smoke600_replay10

PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src" \
  "$KGPW_PY" scripts/train/phase3_ppo.py \
    --config configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600.yaml \
    2>&1 | tee "$PPO_LOG"
```

### 7.3 update约50的强制停止门

在约200 trajectories / update 50检查：

- [ ] `valid_rate`在updates 40–60未超过0.5：停止运行，保留checkpoint和日志。
- [ ] 出现NaN/Inf、负KL、KG alignment错误、dummy scorer、随机alpha gate：立即停止。
- [ ] 出现OOM：停止；优先只调整batch/rollout chunk，建立新ID，不同时改reward。
- [ ] 不因为单个batch好看就继续；查看窗口均值和真实生成样本。

### 7.4 smoke完成验收

- [ ] 600 trajectories / 150 history rows完成；manifest=`COMPLETE`。
- [ ] checkpoints 200/400/600附近及`final/`存在；samples和history保留。
- [ ] 最后10% updates平均：`valid_rate>=0.85`。
- [ ] 最后10% updates平均解析步数`>=2.8`，同时检查是否只是卡在最小3步门槛。
- [ ] `kg_reward_share>=0.10`；单batch波动不能作为结论。
- [ ] 末段KL `<25`，且全程不得为负。
- [ ] `dR/dalpha`不持续为负；`r_text_used`围绕0而raw `r_text`仍有信号。
- [ ] replay累计60条，最终`sft_replay_actual_ratio=0.10`，replay loss有限。
- [ ] raw advantage/return诊断无NaN/Inf；若重尾，先分解reward来源，不归因于CE标签。
- [ ] 目视检查末checkpoint至少20条真实输出：不少于3步、Reasoning非空、KG引用不越界。
- [ ] 引用诊断同时报告raw citing、unknown surface、malformed和known fraction；不得把
  reward过滤后的`cite_match=1.0`写成原始引用准确率100%。
- [ ] manifest的`source_tree_sha256`非空，并记录`source_tree_hash_mode`；Git-less远端
  应为`filesystem_fallback`。
- [ ] 10% replay是否防退化在这里才可判定；实现正确不等于实验有效。

可用以下命令汇总最后15个updates；输出必须连同日志保存：

```bash
PPO_OUT=outputs/ppo_legacy_repaired_v2_quota70_hard_seed42_smoke600_replay10
test "$(wc -l < "$PPO_OUT/history.jsonl")" -eq 150
! rg -n 'NaN|Infinity' "$PPO_OUT/history.jsonl"
tail -n 15 "$PPO_OUT/history.jsonl" | jq -s '
  {
    updates:length,
    valid_rate_mean:(map(.valid_rate)|add/length),
    parsed_steps_per_traj_mean:(map(.n_steps_sample/4)|add/length),
    kg_reward_share_mean:(map(.kg_reward_share)|add/length),
    kl_mean:(map(.ppo_mean_kl)|add/length),
    dr_dalpha_mean:(map(.dr_dalpha)|add/length),
    replay_actual_ratio_final:.[-1].sft_replay_actual_ratio,
    replay_items_seen_final:.[-1].sft_replay_items_seen
  }'
```

研究者查看完整smoke报告后明确：通过 / 失败 / 单变量重跑。不得自动进入正式PPO。

### 7.5 已批准的CE=0.30单变量重跑

首轮`CE weight=0.10`实际完成60/600 replay，但SFT→PPO在同一val n=200上的EM
`0.790→0.730`、F1 `0.83787→0.78820`；该run判为失败并永久保留。下一轮仅改
`sft_anchor_weight: 0.10→0.30`，replay ratio继续为0.10；不得同时提高
`max_new_tokens`或修改Reward/alpha/value loss/学习率。

启动前：

- [x] 新配置resolved diff已验证只有output ID和CE weight两项；
- [x] 新输出目录当前不存在；
- [x] 诊断代码定向测试通过；
- [ ] 重新运行CUDA/bf16、磁盘和输入artifact预检；
- [ ] 建立不覆盖旧日志的新日志名；
- [ ] 研究者明确批准启动本次GPU smoke。

```bash
set -euo pipefail
KGPW_ROOT=/home/zjulab/kgpaper
KGPW_PY=/home/zjulab/anaconda3/envs/kgpaper/bin/python
PPO_LOG="$KGPW_ROOT/logs/training/ppo_quota70_hard_seed42_no_text_head_smoke600_replay10_ce030.log"
cd "$KGPW_ROOT"
mkdir -p logs/training
test ! -e "$PPO_LOG"
test ! -e outputs/ppo_legacy_repaired_v2_quota70_hard_seed42_no_text_head_smoke600_replay10_ce030

PYTHONPATH="$KGPW_ROOT:$KGPW_ROOT/flashrag_src" \
  "$KGPW_PY" scripts/train/phase3_ppo.py \
    --config configs/training/phase3_ppo_legacy_repaired_v2_quota70_smoke600_replay10_ce030.yaml \
    2>&1 | tee "$PPO_LOG"
```

仍执行7.3/7.4的既有稳定性门，并增加预注册的保持门：同一200 qid、seed42、greedy
验证下，PPO相对SFT的EM下降不得超过1pp，且EM差的paired bootstrap 95% CI必须包含0。
样本文件每个checkpoint应为20条，并核查length-cap、citation-contract、citation-match
分层和adaptive-KL遥测。若正确率恢复但valid仍不足，再请求批准单独建立tok384实验。

### 7.6 显式SFT reference correctness smoke（取代kl025）

CE=0.30已失败。后续复核发现两轮的首批KL均为65.4414：TRL 0.11.4的PEFT分支忽略
传入的显式ref，并在policy上禁用LoRA后对裸base计算reference logprob。因此：

- [x] `kl_coef=0.25`方案已撤回；保留配置，旧launcher主动退出；
- [x] 修复trainer，使PEFT policy仍正常训练/保存，但reference dispatch强制使用冻结SFT副本；
- [x] tiny PEFT真实PPO step验证首批KL=0；
- [x] 新run硬门为第一batch `abs(objective/kl)<=1.0`且finite；
- [x] 新配置固定`kl_coef=0.15`、replay=0.10、CE=0.10、tokens=256；
- [x] 新输出目录和日志按唯一Experiment ID生成，旧实验未覆盖；
- [x] 研究者批准并完成GPU correctness smoke；首批KL=0，reference修复通过；
- [ ] smoke整体仍FAIL：末15 valid=0.583、critic EV=-1.23，禁止自动进入正式PPO；
  详细报告见`outputs/audits/ppo_explicit_sft_ref_smoke600_report.md`。

```bash
bash launch_ppo_smoke_explicit_sft_ref_remote.sh
```

第一条history若KL超过1.0，停止且不得通过提高阈值继续。完成后仍用完全相同的200 qid、
seed42、greedy验证；先判断EM/F1保持，再看rollout valid。tokens=384是后续独立实验。

### 7.7 一次性组合稳定性 smoke（2026-08-27批准）

研究者因GPU成本明确批准覆盖7.6的单变量顺序。本实验同时修复截断、KL校准和critic，
必须标记为多变量组合实验，不能做单项归因。

- [x] tokens=384；`kl_coef=0.25`；correct-reference `target_kl=8`；
- [x] value head零初始化、dropout=0、PPO epochs=2，`vf_coef=0.5`不变；
- [x] replay=10%、CE=0.10保持；Reward/eval/data/KG/seed不变；
- [x] 200 trajectories后启用15-update滚动止损：valid>=0.50、cap<=0.50、KL<=20；
- [x] launcher启动前自动执行22项定向回归；CPU真实TRL step通过；
- [x] 已完成远端运行和同qid验证；combined为当前最稳定PPO smoke，但同条件EM仍比SFT低3pp。

```bash
bash launch_ppo_smoke_combined_stability_remote.sh
```

- [x] manifest=`COMPLETE`；valid=0.983、cap=0.017、KL=1.13、critic修复通过；
- [x] 同条件val n=200：SFT EM/F1=`0.790/0.838`，combined=`0.760/0.806`；训练稳定，
  但PPO仍小幅退化，不能进入正式PPO。

### 7.8 hybrid old10+bridge5 retrieval-input smoke（2026-08-28批准）

本轮跳过IHR，只测试改善rollout passages后PPO相对SFT的EM/F1保持能力。它不是Reward
消融；除passages外，完全继承7.7的combined_stability配置。

- [x] train schedule：600 rollouts / 559 unique qids / seed42 / train fold；
- [x] train override：old first10 + bridge最多5篇，559/559覆盖、每题15篇；
- [x] override SHA256：`7c199a0e272323a8739d232c21ee4f084ba0fc071f1de67fb97a7cb593eb1a1f`；
- [x] schedule SHA256：`6e2b4871d9035ec7c3b494212c382c67aeb4d113ed5ef8367a1b5510af19b771`；
- [x] 本地定向回归28项通过，真实配置preflight为600/559、150 updates、10% replay；
- [ ] 远端同步后再次核对上述两个SHA256，并确认新output/log不存在；
- [ ] 启动：

```bash
bash launch_ppo_smoke_hybrid_old10_bridge5_v3_remote.sh
```

- [ ] 第一batch要求显式SFT reference KL finite且绝对值<=1；否则保留失败run并停止；
- [ ] 完成后必须有150条history、60 replay items、final adapter/value head、COMPLETE manifest；
- [ ] 下载checkpoint后，先在冻结确认集hybrid输入上与SFT和旧combined做paired EM/F1；
  再在开发集复核，不看test，不运行IHR；
- [ ] 通过门：训练稳定性继续满足7.7；新PPO相对hybrid SFT EM下降<=1pp且paired CI包含0。
  未达到时停止扩量，先做日志和paired样本归因。

## 8. 正式 PPO 前必须再次判定

目前尚无quota70正式PPO专用配置，不能直接把历史主配置当正式运行配置。smoke通过后，
研究者需要决定：

- [ ] 正式总trajectories：quota70 train一整遍约需4,752 trajectories；是否跑一遍或更多
  由smoke曲线决定，当前为`UNKNOWN`，不得沿用旧16,000而不说明约3.37遍的数据复用。
- [ ] seed方案：重要结论不得只依赖seed42；明确正式主seed及复现实验seed。
- [ ] checkpoint选择协议和停止协议；不得事后只留最好checkpoint。
- [ ] replay CE系数保持0.10还是单变量比较1.0。
- [ ] 是否运行quota25高KG密度对照；不得与quota70混作同数据条件。
- [ ] 建立新的正式YAML和唯一Experiment ID，记录这是组合修复实验。
- [ ] 研究者明确批准大规模正式PPO。

## 9. 正式 PPO 后评估顺序

- [ ] 先按冻结协议在HotpotQA、2WikiMultiHopQA、MuSiQue执行KG/noKG共6 cells。
- [ ] 所有cells使用相同checkpoint、seed、样本、temperature、retriever、reranker、top-k。
- [ ] 每个+KG cell报告实际注入KG数、非空率；MuSiQue若300/300为空，该cell作废。
- [ ] 不混用历史PPO checkpoint；历史泄漏/失败实验继续保留但不并表冒充同条件。
- [ ] 主结果补多seed、McNemar、paired bootstrap；IHR使用统一judge模型与同一paired样本。
- [ ] 最后才更新论文主表和结论，建立`Claim -> Evidence -> Experiment -> Evaluation -> Data/Model`链。

## 10. 当前最近一步

```text
恢复CUDA/NVIDIA driver
  -> 完成第1节环境预检
  -> 研究者选择Phase2 text_reward_head A/B
  -> 研究者批准Phase2
  -> 只运行Phase2，不并行启动SFT/PPO
```
