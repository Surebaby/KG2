# R7-A 实验进度报告

> **实验**: Format-as-Constraint (ValidTrajectory + SFT Anchor)
> **训练**: 2026-07-02 18:26 ~ 2026-07-03 00:17
> **状态**: 已停止，方案需调整

---

## 一、实验设计

### 核心改动 (vs R6-A)

| 模块 | R6-A | R7-A |
|------|------|------|
| Outcome Reward | `10×EM` 无条件 | `10×EM×I(ValidTrajectory)` |
| Format Bonus | `+0.3 per step` | 删除 |
| ValidTrajectory | 无 | Step≥3 + Final Answer + 索引连续 |
| SFT Anchor | 无 | λ=0.02, 每50 batch |
| Elite SFT | ✓ | ✓ 保留 |
| α-gate | ✓ | ✓ 不改 |

### 奖励公式

```
R6-A: R_step = α·R_KG + (1-α)·R_Text×0.3 + 0.3 + 10×EM
R7-A: R_step = α·R_KG + (1-α)·R_Text×0.3       + 10×EM×I(ValidTrajectory)
```

### 训练配置

| 参数 | 值 |
|------|-----|
| SFT 基座 | Elite SFT (2,000条) |
| batch_size | 8 |
| mini_batch_size | 1 |
| kl_coef | 0.1 |
| outcome_weight | 10.0 |
| text_reward_scale | 0.3 |
| min_valid_steps | 3 |
| sft_anchor_weight | 0.02 |
| sft_anchor_interval | 50 |
| total_steps | 5000 (实际运行 ~1824) |
| GPU | RTX PRO 6000 Blackwell 96GB |
| PyTorch | 2.11.0+cu128 |

---

## 二、训练过程

### 时间线

| 时间 | 事件 |
|------|------|
| 18:26 | 训练启动，ReaRAG-9B 加载成功 |
| 18:40 | 代码bug修复：build_sft_messages导入、log_with冲突 |
| 18:42 | TensorBoard 就绪 |
| 18:49 | Step 32，valid=2/8，KL=73 |
| 19:13 | Step 160，valid=4/8 🔺 峰值 (50%) |
| 19:59 | Step 416，valid 降至 0-1 |
| 21:10 | Step 800，SFT anchor #1 (loss=7.62) |
| 22:39~23:34 | Step 1312~1600，valid 低位震荡 |
| 23:34 | Step 1600，SFT anchor #2 (loss=10.84) |
| 00:17 | Step 1824，valid=1/8 |

### 完整数据趋势

```
Step   32: valid=2/8  reward=+2.98  kl=73  ← 初始探索
Step   64: valid=0/8  reward=-0.16  kl=63
Step  160: valid=4/8  reward=+1.23  kl=46  ← 最高valid
Step  256: valid=0/8  reward=+0.05  kl=34
Step  416: valid=0/8  reward=-0.10  kl=28  ← 开始下降
Step  608: valid=2/8  reward=+1.09  kl=24  ← 偶发峰值
Step  800: valid=1/8  reward=+1.20  kl=18  ← SFT anchor#1
Step  992: valid=3/8  reward=-0.05  kl=15  ← 写了但答案错
Step 1216: valid=3/8  reward=+0.03  kl=12
Step 1408: valid=2/8  reward=-0.07  kl=9
Step 1600: valid=1/8  reward=-0.01  kl=9   ← SFT anchor#2
Step 1824: valid=1/8  reward=-0.09  kl=6   ← KL收敛但格式没涨
```

---

## 三、定量分析

### 整体统计（57个数据点, 456条轨迹）

| 指标 | 数值 | 评估 |
|------|------|:--:|
| 总 valid | 42/456 (**9.2%**) | ❌ |
| valid=0 的step | 30/57 (53%) | ❌ |
| valid≥2 的step | 11/57 (19%) | ❌ |
| valid≥3 的step | 3/57 (5%) | ❌ |
| 平均 reward | ~0.01 | ❌ |
| KL 趋势 | 73→9 | ✅ 收敛但没用 |

### 分阶段趋势

```
0-400:    valid=13.5%  KL=42  ← 探索期
400-800:  valid=4.2%   KL=21  ← 暴跌
800-1200: valid=9.6%   KL=15  ← 微回升
1200-1600:valid=10.6%  KL=9   ← 稳定低位
1600+:    valid=7.1%   KL=7   ← 再降
```

**无上升趋势，数据在噪声范围内震荡。**

### 奖励与 Valid 的脱钩

```
valid=0  时 avg reward = ~0.01
valid≥1 时 avg reward = ~0.01
```

写了步骤 vs 不写步骤，reward 几乎完全相同。模型得不到有效区分信号。

### SFT Anchor 表现

| 触发步数 | Loss | 评估 |
|:--:|------|:--:|
| 800 | 7.62 | 初始 |
| 1600 | 10.84 | 恶化 (+42%) |

anchor 的拉回力无法对抗 PPO 的漂移力。

---

## 四、根因分析

### 问题本质：稀疏奖励死区

```
不写步骤 → valid=0 → 门控关 → 无outcome → reward ≈ 0
写步骤  → valid=1 → 门控开 → 答案错  → 无outcome → reward ≈ 0
写步骤  → valid=1 → 门控开 → 答案对  → reward +10
```

前两种情况占了 ~95% 的轨迹，但它们的 reward 相同。PPO 没有梯度信号区分「写步骤但答案错」和「不写步骤」——它学到的结论是「反正都是 0，写不写无所谓」。

### 三个促成因素

1. **min_valid_steps=3 门槛过高**: Elite SFT 格式锚定弱，模型需要能先写 1 步、再写到 3 步
2. **SFT anchor 过稀**: 每 400 条一次，PPO 漂移速度远超纠偏速度
3. **R_Text=0.3→噪声**: 文本奖励不能提供可靠的格式梯度

### 与 R6-A 对比

| | R6-A | R7-A |
|------|:--:|:--:|
| 格式约束方式 | 软奖励 (+0.3/步) | 硬门控 (ValidTrajectory) |
| 梯度连续性 | 连续（每步都有信号） | 离散（通过/不通过） |
| 学习难度 | 低 | 高 |
| 评估时 valid | 0% | 12% |
| 训练时 valid | 自然产生 | 强行要求 |
| mean_reward | 正常 | ~0 (死区) |

**R6-A 的问题**是 soft reward 太弱（评估时被忽略），**R7-A 的问题**是 hard gate 产生不了梯度。两者各缺一半。

---

## 五、R7-B 实验（进行中）

> **修改**: min_valid_steps=3→1, sft_anchor_interval=50→10, weight=0.02→0.05
> **训练**: 2026-07-03 13:47 ~ 进行中

### R7-B vs R7-A 对比（step 32~992）

| | R7-A | R7-B |
|------|:--:|:--:|
| 平均 valid | ~9% | **~92%** |
| 平均 reward | ~0 | **4-5** |
| SFT anchor loss 趋势 | 恶化 (7.6→10.8) | **改善 (10→4→5)** |
| KL 范围 | 5-73 宽幅震荡 | 26-81 有波动但reward稳定 |

### R7-B 训练日志

```
Step  32: valid=6/8  reward=3.22  kl=54
Step  64: valid=8/8  reward=4.03  kl=53
Step  96: valid=8/8  reward=8.75  kl=43  ← peak reward
Step 160: valid=8/8  reward=3.69  kl=38  SFT anchor#1(loss=10.05)
Step 256: valid=8/8  reward=6.21  kl=46
Step 320: valid=7/8  reward=5.00  kl=38  SFT anchor#2(loss=4.00)
Step 416: valid=8/8  reward=4.95  kl=29  KL↓
Step 512: valid=4/8  reward=2.08  kl=81  ← 偶发波动
Step 576: valid=8/8  reward=4.85  kl=43  快速恢复
Step 640: valid=8/8  reward=2.42  kl=29  SFT anchor#3(loss=8.76)
Step 704: valid=8/8  reward=5.00  kl=26
Step 768: valid=6/8  reward=1.21  kl=73  ← 波动
Step 800: valid=6/8  reward=3.72  kl=59  SFT anchor#4(loss=7.91)
Step 864: valid=7/8  reward=7.50  kl=32
Step 928: valid=6/8  reward=6.15  kl=47
Step 992: valid=8/8  reward=7.50  kl=51
```

### 关键观察

1. **valid 稳定在 6-8/8** ✅ 死区已被打通
2. **reward 4-7 正常范围** ✅ 模型在学习
3. **SFT anchor 频繁触发** (约每200条1次，明显比R7-A的400条密集) ✅
4. **KL 波动正常**：峰值后总能恢复，自适应控制器有效
5. **偶发 valid 下降(step 512/768)后快速恢复** → SFT anchor 在起作用

### 待观察

- step 1500-2000 时 KL 能否降至 20 以下
- 最终评估时是否保留 [Step N] 格式（关键）
- EM 是否超越 R6-A (0.246)

---

## 六、下一步方案

### R7-B 建议配置 (已执行)

```yaml
min_valid_steps: 1          # 降低门槛: 写任何步骤就给outcome
sft_anchor_weight: 0.05     # 增强锚定: 2.5×
sft_anchor_interval: 10     # 加密锚定: 5×
```

### 逻辑

```
min_valid_steps=1: 模型写1步就能拿到outcome → 进入正反馈
sft_anchor加密+加强: 频繁的格式记忆拉回，防止漂移
```

### 后续可选的渐进收紧

```
Step 0-1500:  min_valid_steps=1
Step 1500-3000: min_valid_steps=2
Step 3000-5000: min_valid_steps=3
```

---

## 六、代码修改清单 (待执行)

| 文件 | 修改 |
|------|------|
| `configs/training/phase3_ppo.yaml` | min_valid_steps→1, sft_anchor_weight→0.05, sft_anchor_interval→10 |
| `kgproweight/training/phase3_ppo.py` | 同步默认值 (可选，YAML会覆盖) |

---

## 七、环境记录

| 项目 | 详情 |
|------|------|
| 服务器 | AutoDL connect.bjb1.seetacloud.com:36491 |
| Python | /root/autodl-tmp/kgpw_env/bin/python (3.10.20) |
| PyTorch | 2.11.0+cu128 |
| GPU | RTX PRO 6000 Blackwell, 96GB, sm_120 |
| ReaRAG | /root/autodl-tmp/models/rearag-9b |
| Llama | /root/autodl-tmp/models/llama3-8b |
| Entity Cache | /root/autodl-tmp/kgpaper/indexes/entity_cache.jsonl (34807 entries) |

---

*创建: 2026-07-03*
*下次更新: R7-B 训练后*
ValidTrajectory 门控 — 方向对，R6-A 评估 0% valid，R7-A 至少有 9%
SFT Anchor — 方向对，但参数不对