# R7 实验进度记录

> **启动日期**: 2026-07-02
> **方案**: Format-as-Constraint (ValidTrajectory + SFT Anchor)
> **状态**: 代码修改完成，待训练

---

## 一、R7 方案概述

### 核心修改

| # | 修改项 | 原值 (R6-A) | 新值 (R7) | 文件 |
|---|--------|------------|----------|------|
| 1 | Outcome Reward | `10×EM` (无条件) | `10×EM×I(ValidTrajectory)` | `composite_reward.py`, `reward_function.py` |
| 2 | Format Bonus | `+0.3 per step` | **删除** | `reward_function.py` |
| 3 | ValidTrajectory | 无 | Step≥3, 有Final Answer, 索引连续, 非空文本 | `reward_function.py` |
| 4 | SFT Anchor | 无 | λ=0.02, 每50步交替SFT | `phase3_ppo.py` |
| 5 | Elite SFT | Elite SFT (2k) | Elite SFT (2k) — **保留** | 无改动 |
| 6 | α-gate | 不变 | 不变 — **论文核心创新不改** | 无改动 |
| 7 | KG/Text Reward | 不变 | 不变 — `α·R_KG + (1-α)·R_Text×0.3` | 无改动 |

### 奖励函数对比

```
R6-A: R_step = α·R_KG + (1-α)·R_Text×0.3 + 0.3(format) + 10×EM(unconditional, last step)
R7:   R_step = α·R_KG + (1-α)·R_Text×0.3                      + 10×EM(conditional, last step only if ValidTrajectory)
```

### 训练目标对比

```
R6-A: L = L_PPO
R7:   L = L_PPO + 0.02×L_SFT  (每50步PPO执行1步SFT anchor)
```

---

## 二、修改的文件清单

| 文件 | 修改内容 | 备份位置 |
|------|---------|---------|
| `kgproweight/reward/composite_reward.py` | `compute_trajectory_rewards` 新增 `trajectory_valid` 参数，gate outcome reward | `backups/R7_rollback/` |
| `kgproweight/training/reward_function.py` | 删除 `step_format_bonus`，新增 `_is_valid_trajectory()`，新增 `min_valid_steps`，gate outcome fallback | `backups/R7_rollback/` |
| `kgproweight/training/phase3_ppo.py` | 删除 `step_format_bonus`，新增 SFT anchor 数据准备与交替训练逻辑，新增 validity rate logging | `backups/R7_rollback/` |
| `configs/training/phase3_ppo.yaml` | 删除 `step_format_bonus`，新增 `min_valid_steps`/`sft_anchor_weight`/`sft_anchor_interval` | `backups/R7_rollback/` |

### 回退方法

```powershell
cd c:\Users\ycc\Desktop\kgpaper\autodl-tmp-backup\kgpaper
powershell -File backups\R7_rollback\rollback.ps1
```

或手动复制：
```bash
cp backups/R7_rollback/composite_reward.py.bak kgproweight/reward/composite_reward.py
cp backups/R7_rollback/reward_function.py.bak kgproweight/training/reward_function.py
cp backups/R7_rollback/phase3_ppo.py.bak kgproweight/training/phase3_ppo.py
cp backups/R7_rollback/phase3_ppo.yaml.bak configs/training/phase3_ppo.yaml
```

---

## 三、训练计划

### R7-A (首选配置)

| 参数 | 值 |
|------|-----|
| SFT 基座 | Elite SFT (2,000 条精品) |
| PPO 数据 | 全量 9,839 条 silver |
| total_steps | 5,000 |
| batch_size | 8 |
| mini_batch_size | 1 |
| kl_coef | 0.1 |
| outcome_weight | 10.0 |
| text_reward_scale | 0.3 |
| min_valid_steps | 3 |
| sft_anchor_weight | 0.02 |
| sft_anchor_interval | 50 |
| temperature | 1.0 |
| max_input_length | 6144 |
| save_every_steps | 500 |

### 预估训练时间

| 进度 | 步数 | 时间 (累计) | 检查内容 |
|------|------|-----------|---------|
| 🟢 首次检查 | ~500 | ~1 小时 | 确认训练启动正常，无 OOM/崩溃 |
| 🟡 格式检查 | ~1000 | ~2 小时 | **重点**: valid_rate 是否上升，是否恢复 [Step N] 格式 |
| 🟡 中期检查 | ~2500 | ~5 小时 | valid_rate 趋势、reward 稳定性、KL 是否正常 |
| 🔴 最终 | ~5000 | ~10 小时 | 训练完成，准备评估 |

> 基于 R6-A 训练数据: 5000 步 ≈ 10 小时 (RTX PRO 6000 Blackwell 96GB)。
> SFT Anchor 每 50 步额外一次 forward+backward，额外开销 ≈ 2%。

### 存档点

`save_every_steps=500`，结合 `batch_size=8`，实际存档在：
`~504, ~1008, ~1512, ~2016, ~2520, ~3024, ~3528, ~4032, ~4536, ~5040`

每个存档点保存完整 adapter 权重 + history.jsonl，可独立恢复。

### TensorBoard 实时监控

```bash
# 启动 TensorBoard（在训练机上新开一个终端）
tensorboard --logdir <output_dir>/tensorboard --port 6006

# 浏览器访问 http://localhost:6006
# 远程服务器需端口转发: ssh -L 6006:localhost:6006 user@server
```

**TensorBoard 指标一览**：

| 面板 | 指标 | 来源 | 说明 |
|------|------|------|------|
| **TRL 内置** | `ppo/loss/total`, `ppo/loss/policy`, `ppo/loss/value` | TRL | PPO 各项 loss |
| | `ppo/policy/clipfrac`, `ppo/policy/approxkl` | TRL | PPO clip 比例和 KL |
| | `objective/kl` | TRL | 策略 KL 散度 |
| **custom/** | `mean_reward` | 我们 | 每 batch 平均轨迹奖励 |
| | `valid_rate` | 我们 | 🔑 **最重要**: 合法轨迹占比 |
| | `n_valid` | 我们 | 合法轨迹数量 |
| | `sft_anchor_loss` | 我们 | SFT anchor CE loss |
| | `advantage_var` | 我们 | Advantage 方差 (>0 = 有学习信号) |

> 训练启动后立即打开 TensorBoard 监控，不错过早期信号。

### 训练监控重点

1. **mean_reward**: 预期初期较低（因为 ValidTrajectory 门槛），逐步上升。不应出现 reward→0 崩塌
2. **valid_rate**: 预期初期 ~10-30%（模型刚开始学习格式），逐步上升至 60-80%
3. **ppo_mean_kl**: 应在 0-8 范围内。KL 爆炸(>20)或崩塌(<0.01)均为异常
4. **advantage_var**: 应 > 0，0 表示无学习信号
5. **sft_anchor_loss**: 应在 1-3 范围内稳定，猛增表示 policy drift 过大
6. **policy_clipfrac**: 应在 0.05-0.3 范围内

### 异常预警阈值

| 指标 | 正常范围 | 预警 | 严重 |
|------|---------|------|------|
| mean_reward | 0.3 ~ 5.0 | < 0.1 | → 0 (崩塌) |
| valid_rate | 20% ~ 80% | < 10% for > 500 steps | → 0% |
| ppo_mean_kl | 0.01 ~ 20 | > 50 | > 100 or < 0.001 |
| advantage_var | > 0.001 | → 0 for > 200 steps | N/A |
| policy_clipfrac | 0.05 ~ 0.30 | > 0.5 | N/A |

### 如果训练崩塌，按以下顺序排查

1. **reward → 0**: 检查是否所有 rollout 都 invalid → 降低 `min_valid_steps` 到 1 允许模型渐进学习
2. **KL 爆炸**: 增大 `kl_coef` 到 0.2，或降低 `sft_anchor_weight`
3. **valid_rate → 0**: 增大 `sft_anchor_interval` 或 `sft_anchor_weight`，增强格式约束
4. **OOM**: 降低 batch_size 到 4，或降低 ppo_max_passages 到 10

---

## 四、评估计划

### 待评估

- [ ] 训练 R7-A PPO (约 10 小时)
- [ ] 评估 R7-A on hotpotqa/2wiki/musique × 3 seeds
- [ ] IHR (LLM-as-Judge) 评估 (如果 R7 恢复了步骤)
- [ ] α 分布分析
- [ ] 更新 Baseline 对比文档

### 消融实验矩阵

| 实验 | SFT 基座 | format 机制 | sft_anchor | 预期 |
|------|---------|-----------|-----------|------|
| R6-A (baseline) | Elite | bonus 0.3 | 无 | EM=0.246, 100% skip steps |
| **R7-A** | Elite | ValidTrajectory | λ=0.02/50步 | 恢复步骤, EM不降 |
| R7-B (备选) | Elite | ValidTrajectory | λ=0.05/20步 | 更强格式约束 |
| R7-C (备选) | Full SFT | ValidTrajectory | λ=0.02/50步 | 对比基座影响 |

---

## 五、论文段落（待插入 Discussion）

> In R6-A, increasing the step-format bonus successfully stabilized PPO optimization but failed to preserve reasoning traces during inference. This observation suggests that positive format rewards merely increase the utility of generating reasoning steps, while they do not make reasoning a prerequisite for receiving the final outcome reward. Consequently, PPO discovers a shortcut policy that directly predicts the final answer while bypassing intermediate reasoning. To address this objective mismatch, we replace the unconditional outcome reward with a trajectory-conditioned outcome reward and introduce a lightweight SFT anchoring loss to preserve the reasoning-format prior throughout reinforcement learning.

---

## 六、训练运行记录

| 运行 | 日期 | 配置 | 结果 | 备注 |
|------|------|------|------|------|
| R7-A | TBD | 见§3 | - | 待训练 |
| | | | | |
| | | | | |

---

## 七、设计决策记录

1. **为什么不用 Full SFT**: Full SFT 已经过拟合到数据分布，PPO 的探索空间很小，难以体现 α-gate 的贡献。Elite SFT 虽然能力稍弱但更适合作为 RL 的初始化。
2. **为什么删 format_bonus 而不保留**: Format 本质是 constraint 不是 reward target。`+0.3 per step` 告诉模型"写步骤多赚一点"而非"不写步骤不能领奖"。R6-A 证明了正向 bonus 不足以在评估时保持格式。
3. **为什么 SFT Anchor 而不是 KL penalty**: KL penalty 是对整个输出分布的无差别约束，会同时限制答案优化。SFT Anchor 只约束输出格式（通过 trajectory token 的 CE loss），对答案内容的影响更小。
4. **为什么 min_valid_steps=3**: 数据集的 min_steps 就是 3，设 3 是格式要求的最低门槛，不额外收紧。

---

*最后更新: 2026-07-02*
*下次更新: 训练完成后*
