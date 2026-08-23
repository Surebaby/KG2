# KG-ProWeight 实验数据统计总表
## 生成日期: 2026-08-10

---

## 一、主结果矩阵

| 数据集 | N | 模型 | EM | F1 | Tok | α mean | α range | KG效果 |
|--------|---|------|-----|-----|-----|--------|---------|--------|
| HotpotQA | 300 | SFT noKG | 0.3400 | 0.4558 | 1837 | — | — | baseline |
| HotpotQA | 300 | SFT+KG | 0.3533 | 0.4575 | 1968 | ~~0.292~~ † | ~~[0.07,0.51]~~ † | +0.0133 |
| HotpotQA | 300 | PPO+KG | 0.2133 | 0.2853 | 1947 | — | — | -0.1267 |
| HotpotQA | 300 | PPO noKG | 0.3133 | 0.4244 | 1837 | — | — | -0.0267 |
| HotpotQA | 300 | PPO v1 (旧) | 0.3267 | 0.4241 | 2136 | — | — | — |
| 2Wiki | 300 | SFT noKG | 0.3367 | 0.4265 | 1906 | — | — | baseline |
| 2Wiki | 300 | SFT+KG | 0.3600 | 0.4280 | 2038 | ~~0.804~~ † | ~~[0.39,0.96]~~ † | +0.0233 |
| 2Wiki | 300 | PPO+KG | 0.3233 | 0.3979 | 2040 | — | — | -0.0133 |
| 2Wiki | 300 | PPO noKG | 0.3200 | 0.4001 | 1906 | — | — | -0.0167 |
| 2Wiki | 300 | PPO v1 (旧) | 0.2900 | 0.3625 | 2038 | — | — | — |
| 2Wiki | 1000 | SFT noKG | 0.3170 | 0.3837 | 1902 | — | — | baseline |
| 2Wiki | 1000 | SFT+KG | 0.3440 | 0.3965 | 2037 | ~~0.807~~ † | ~~[0.39,0.96]~~ † | +0.0270 |
| 2Wiki | 1000 | PPO+KG | 0.3010 | 0.3654 | 2037 | — | — | -0.0160 |
| MuSiQue | 300 | SFT noKG | 0.2067 | 0.3234 | 1834 | — | — | baseline |
| MuSiQue | 300 | SFT+KG | 0.2300 | 0.3147 | 1921 | ~~0.854~~ † | ~~[0.59,0.97]~~ † | +0.0233 |
| MuSiQue | 300 | PPO noKG | 0.0367 | 0.1117 | 1805 | — | — | -0.1700 |
| MuSiQue | 300 | PPO+KG* | 0.0367 | 0.1117 | 1805 | — | — | -0.1700 |

> \* MuSiQue PPO+KG 实际注入 0 KG（`question_kg_index` 不覆盖 musique，live Wikidata 兜底在 2026-08-16 起 SSL 超时不可达），故与 noKG 逐位相同，不可作 +KG 消融结论。

> † **α 两列已撤回**（2026-08-22）。这批 α 与 §二 旧表同源，是在 EntityLinker cache 冷
> （`f_confidence ≡ 0`）+ pre-D1 `f_entropy ≡ 1.0` + bias 校正未生效的混合状态下测的，
> 彼此不可比。统一状态重测值见 §二.2（0.918 / 0.914 / 0.908）。**EM/F1 各列不受影响** ——
> α 在评测期只是只读遥测，不参与生成。

## 二、α-Gate 跨数据集行为（2026-08-22 重测，推翻旧表）

> **旧表已撤。** 此前本节报告 HotpotQA 0.292 / 2Wiki 0.804 / MuSiQue 0.855，并作为
> "α-门控涌现式跨数据集自适应" 写进摘要、intro、§5.2、§5.4 和结论。**那三个数不是同一个
> 函数产生的**，因此不能并列比较。见 §二.1 的三条独立证据。

### 二.1 旧表为何不可用

**(a) 0.292 在它自己声称的配置下无解。** 门控为
`α = σ((W₀·density + W₁·conf + W₂·ent + b)/τ)`，`W=[1.2096, 1.7088, −0.5833]`。
`W₁·conf + W₂·ent` 在 conf,ent ∈ [0,1] 上只能取 **[−0.583, +1.709]**。
α=0.292 在实测 density 0.924 下需要残差 **−0.647**，低于下界 —— **任何 (conf, ent) 都解不出来**。
它只在 `bias_correction = 0` 时可达，而 `03_method.md` §3.4 明确写 +0.78 已应用于主结果。
两者不可能同时为真。

**(b) 同数据集同 seed，α 可取四个值。** 重新诊断 08-07 的三个 hotpotqa run（均 seed=42）：

| run | α | 残差 | 诊断 |
|---|---|---|---|
| `kgfix_smoke` 13:09 | 0.1251 | −1.3633 | BIAS MISMATCH + DEGENERATE（bias 未生效且 EntityLinker cache 冷，conf≡0） |
| `val_select_smoke` 03:04 | 0.4405 | −0.4042 | pre-D1，f_entropy≡1.0 强制 |
| `val_select/final` 05:40 | 0.4925 | −0.2572 | pre-D1，f_entropy≡1.0 强制 |
| 本次重测 08-22 | 0.9183 | +1.3564 | 真实 logprobs，conf/ent 均有变化 |

**同数据集内的状态漂移（0.79）远大于旧表声称的跨数据集差异（0.29→0.86）。**

**(c) 两个漂移源已定位。** ① `EntityLinker.link_confidence` 在 cache 为空时返回 0.0
（`entity_linker.py`，`if not cache_items: return 0.0`），而 `indexes/entity_cache.jsonl`
不在 git 里，是运行时产物 —— 08-08 的 run 是冷 cache，08-14 起反解出 conf≈0.28–0.73（已暖）。
② D1（commit `e6b2198`）把推理端 `f_entropy` 从硬编码 1.0 换成真实 per-token logprobs；
`W₂` 为负，故此改动本身把 α-logit 抬高 +0.19…+0.26。

### 二.2 重测结果（统一状态）

协议：同一份 EntityLinker cache 快照（127,967 条，每个数据集独立副本以防互相污染）、
`KGPW_KG_OFFLINE=1`、同一 checkpoint 与门控、`--rerank 10`、`bias_correction=+0.78`、
n=300、seed=42。脚本：`scripts/analysis/remeasure_alpha.sh`；诊断：`alpha_diagnose.py`。

按 question 聚合（与诊断脚本一致）：

| 数据集 | n（非空子图） | α mean | α sd | density mean | r(α, density) | 残差 W₁·conf+W₂·ent |
|---|---|---|---|---|---|---|
| HotpotQA | 285 | **0.9183** | 0.0217 | 0.9242 | +0.902 | +1.3564 (sd 0.0550) |
| 2Wiki | 290 | **0.9143** | 0.0224 | 0.9061 | +0.905 | +1.3454 (sd 0.0577) |
| MuSiQue | 253 | **0.9084** | 0.0301 | 0.8851 | +0.949 | +1.3357 (sd 0.0534) |

按 step 聚合（与旧表 "α range / α>0.5 占比 / KG覆盖 / triples/q" 四列同口径，便于逐列对照）：

| 数据集 | n steps | α mean | α sd | α range | α>0.5 占比 | KG 覆盖 | triples/q |
|---|---|---|---|---|---|---|---|
| HotpotQA | 896 | 0.9015 | 0.0732 | [0.542, 0.991] | **100%** | 95% | 10.7 |
| 2Wiki | 905 | 0.9061 | 0.0533 | [0.575, 0.979] | **100%** | 97% | 10.5 |
| MuSiQue | 853 | 0.8642 | 0.1055 | [0.529, 0.972] | **100%** | 84% | 9.1 |

旧表的对照列是 HotpotQA `[0.07,0.51]` / **13%** vs 2Wiki `[0.39,0.96]` / **97%** /
MuSiQue `[0.57,0.98]` / 100%。重测后**三个数据集都是 100%**，区间也基本重合
（下界 0.529–0.575，上界 0.972–0.991）—— §5.2 里 "只有 13% 超过 0.50 vs 97% 超过 0.50"
这个对比整体消失。

EM/F1 同批记录（**不可与 §一 对比**，protocol 不同：`--rerank 10` + 固定 cache + offline）：
hotpotqa 0.2267/0.2826、2wiki 0.1867/0.2145、musique 0.1367/0.2173。

**诊断脚本判定 `comparable`：跨 run 残差离散度 0.021，小于最大的 run 内 sd 0.058。**
这是三个臂第一次落在同一把尺子上。结论：

| | 论文声称 | 重测 |
|---|---|---|
| α 三数据集极差 | **0.563**（0.292 → 0.855） | **0.0099**（0.9084 → 0.9183） |
| 参照：最大 run 内 sd | — | 0.0301 |

**跨数据集极差（0.0099）只有单个 run 内部离散度（0.0301）的三分之一** —— 即数据集之间的
差异比同一数据集内不同问题之间的差异还小得多。α 在三个数据集上**没有可测的差异**。
论文声称的 ≈2.8–3.0× 自适应，在统一状态下是 1.01×。

### 二.3 现在能说什么、不能说什么

- **不能说**："α 随数据集推理复杂度自适应"。这是旧表的伪影。
- **能说**：α 在本配置下**由子图密度主导**（每个 run 内 r(α,density) ≈ +0.90），
  且三个数据集的密度分布本身相近（0.885–0.924），所以 α 也就相近。这是对门控行为的
  真实描述，而且它本身是个值得报的负面结果：**当前门控几乎不携带 link-confidence
  与 entropy 的信息**。
- **已确认（2026-08-23，`BIAS=0.0` 对照臂实测）**：α 高位有 **+0.162** 直接来自 `+0.78` 校正。
  HotpotQA 臂在**完全相同的协议**下（同一 checkpoint、同一门控、同一实体链接缓存快照、
  `--rerank 10`、n=300、seed=42，唯一差别是 `--alpha_bias_correction 0.0`）测得
  **α = 0.7563 (sd 0.0523)**，对照 `bias=+0.78` 的 **0.9183 (sd 0.0217)**，差 **+0.1620** ——
  与此前按同一残差解析预测的 +0.16 吻合（run：`outputs/alpha_remeasure_bias0/hotpotqa/`）。
  两点旁证同时成立：① 残差 `W₁·conf+W₂·ent = +1.3567 (sd 0.0550)` 与 `+0.78` 臂的
  +1.3564 几乎逐位相同，证明**校正只平移 logit，不改变特征本身**，两臂确实只差这一个变量；
  ② `r(α,density) = +0.922`，即**关掉校正后 α 仍然是单特征密度函数** —— 密度主导
  不是校正造成的假象，这条负面结论在 bias=0 下同样成立。
  **三个数据集全部复现**（2026-08-23 03:16 三臂跑完，脚本自判 `comparable`：残差跨 run 极差
  0.021 < 最大 run 内 sd 0.058）：

  | 数据集 | α (bias=0) | α (bias=+0.78) | 校正贡献 | 残差 bias=0 / +0.78 | r(α,density) bias=0 | 反解 f_entropy |
  |---|---|---|---|---|---|---|
  | HotpotQA | 0.7563 (sd **0.0523**) | 0.9183 (sd 0.0217) | **+0.1620** | +1.3567 / +1.3564 | +0.922 | 0.603 |
  | 2Wiki | 0.7466 (sd **0.0529**) | 0.9143 (sd 0.0224) | **+0.1677** | +1.3458 / +1.3454 | +0.924 | 0.622 |
  | MuSiQue | 0.7341 (sd **0.0676**) | 0.9084 (sd 0.0301) | **+0.1743** | +1.3361 / +1.3357 | +0.960 | 0.639 |

  三条结论，都是三个数据集一致的：

  1. **校正贡献 +0.162 ~ +0.174**，与按固定残差的解析预测 +0.16 吻合。每个数据集的两臂残差
     都到四位小数一致（如 +1.3361 vs +1.3357），确认两次运行**只差这一个变量**。
  2. **α 仍是单特征密度函数**（r = +0.922 / +0.924 / **+0.960**）。「退化」不是校正造成的
     假象 —— 关掉校正后相关性反而更高。§二.1 的负面结论在 bias=0 下同样成立，比只有一个
     状态时更硬。
  3. **关掉校正后 α 的 sd 反而大一倍以上**（0.052/0.053/0.068 vs 0.022/0.022/0.030）。
     `+0.78` 把 α 推进 sigmoid 饱和区，**压掉了门控本就微弱的区分度**。α 的职责是逐步调节
     对 KG 的依赖，饱和的 α 什么都调不了。

  第 3 点是本轮新增的、指向改代码的理由：此前反对沿用 `+0.78` 只有「与 D1 实测熵重复计入」
  这一条（推导层面），现在多了一条实测层面的 —— **该校正主动削弱了门控功能**。
  两条独立理由都指向同一动作：重新推导，而非沿用。

- **✅ 已重新推导（2026-08-23），结论比上面三条更硬：`+0.78` 从来不是推导值。**
  `b_trained = -1.7818`，`-1.7818 + 0.78 = -1.0018` —— 它是为把 `b_eff` 凑成整数 `-1.0`
  反解出来的，`kg_proweight_pipeline.py:141` 的注释本身就写着 "inference override →
  **beff=-1.0**"。按它自己声称的机理定量（补偿 `f_entropy` 的 regime 偏移 `Δe`，经
  `W₂ = -0.5833` 进入 α-logit，故所需加性偏置为 `-W₂·Δe = 0.5833·Δe`）：

  | 情形 | `Δe` | 所需校正 | 已施加 | 倍数 |
  |---|---|---|---|---|
  | 原始前提（`e_inf` 硬编码 1.0，`e_tr ≈ 0.4`） | 0.60 | +0.350 | +0.78 | 2.2× |
  | **理论上界**（`e_tr = 0`，`entropy_from_logprobs` 的下限） | 0.62 | **+0.363** | +0.78 | 2.15× |
  | **D1 之后真实情形**（`e_inf ≈ 0.62` 实测） | 0.22 | +0.129 | +0.78 | 6.0× |

  即 `+0.78` 要求 `Δe = 1.337`，而 `f_entropy` 自身实测只有 ≈0.62，**该偏移在数值上不可达**。
  第三条独立理由：**训练侧从不施加校正**（`phase3_ppo.py:787-791` 直接 load 后 `eval()`），
  所以 `+0.78` 让训练与推理用了两个不同的 α 函数，本身就是一处 train/eval 错位。
  → 已把 `alpha_bias_correction` 默认值改为 **0.0**（`kg_proweight_pipeline.py`、
  `run_kg_proweight.py`、`alpha_diagnose.py` 三处），并在 `tests/test_alpha_gate.py`
  加 2 条回归钉住默认值与上界算术。**不影响任何 EM/F1**（α 是推理期遥测量）。
- **顺带证实**：`f_entropy` 确实来自真实 logprobs 而非硬编码 1.0 —— 诊断脚本报
  「若 ent=1.0 则需 conf=+1.135，超出 [0,1]，不可行」，反解 ent≈0.603 才可行（D1, `e6b2198`）。

## 三、KG 效果汇总

| 数据集 | N | SFT noKG | SFT+KG | KG Effect | McNemar p | 显著性 |
|--------|---|----------|--------|-----------|-----------|--------|
| HotpotQA | 300 | 0.3400 | 0.3533 | +0.0133 (+1.3pp) | 0.618 | NS |
| 2Wiki | 300 | 0.3367 | 0.3600 | +0.0233 (+2.3pp) | 0.450 | NS |
| 2Wiki | 1000 | 0.3170 | 0.3440 | +0.0270 (+2.7pp) | 0.0910 | NS (marginal) |
| MuSiQue | 300 | 0.2067 | 0.2300 | +0.0233 (+2.3pp) | — | — |

## 四、PPO 效果对比（KG 必要性）

| 数据集 | SFT noKG | PPO noKG | PPO+KG | PPO noKG vs SFT | PPO+KG vs SFT | KG恢复量 |
|--------|----------|---------|--------|-----------------|---------------|---------|
| HotpotQA | 0.3400 | 0.3133 | 0.2133 | -2.7pp | -12.7pp | -10.0pp |
| 2Wiki | 0.3367 | 0.3200 | 0.3233 | -1.7pp | -1.3pp | +0.3pp |
| MuSiQue | 0.2067 | 0.0367 | 0.0367* | -17.0pp | -17.0pp | 0.0pp |

**核心发现（2026-08-16 更正）**: 上表 PPO 均为 `kg_proweight_ppo_v2`（"Plan B"奖励权重训成，见 §十）。该 checkpoint 在三个数据集上**全线低于 SFT**，且 KG 不再回升——HotpotQA +KG 反比 noKG 低 10pp（0.3133→0.2133），MuSiQue 则 OOD 崩溃 -17pp。旧 checkpoint `ppo_r10_split/final` 上 KG 曾恢复 +3pp；"KG 是 PPO 必要正则化器"的结论只在旧 checkpoint 成立。

## 五、PPO v1 vs v2 对比（KG 质量修复效果）

| 数据集 | PPO v1 (旧KG) | PPO v2 (新KG) | 改善 |
|--------|-------------|-------------|------|
| HotpotQA | 0.3267 | 0.3433 | +1.7pp |
| 2Wiki | 0.2900 | 0.3233 | +3.3pp |

~~KG 质量修复（§3.4）消除了 PPO 的灾难性退化（2Wiki -4.6pp → -1.3pp）。~~

> ⚠️ **2026-08-23 配对检验后撤回此句。** 两端都不显著：`2wiki_sft_nokg` vs `2wiki_ppo_kg`
> 为 −4.67pp（McNemar 精确 p=**0.120**，CI [−10.3,+1.0]pp）；vs `ppo_v2_2wiki` 为 −1.33pp
> （p=**0.712**，CI [−6.7,+4.0]pp）。「从 −4.6 改善到 −1.3」是**两个零结果之间的变化**，
> 不能称为"消除了退化"。论文原写的 $p=0.005$ 无出处 —— 扫过仓库内全部 2Wiki n=300 运行的
> 两两配对，没有任何一对能在 p≈0.005 处给出 −4.6pp（见 P7）。已同步改 `03_method.md` §3.3/§3.4
> 与 `05_results.md` 发现 3。

> ⚠️ **口径警告**: 上表 "PPO v2 (新KG)" 一列混用了两个 checkpoint——HotpotQA 的 0.3433 来自 `ppo_r10_split/final`（outcome_weight 4.0），2Wiki 的 0.3233 来自 `kg_proweight_ppo_v2`（outcome_weight 8.0 "Plan B"）。同一 checkpoint 下 HotpotQA PPO+KG 实为 **0.2133**（见 §一/§四），故 §五 的 "+1.7pp 改善" 不可信，待按回退后的配置重训再重测。

## 六、PPO 训练指标（TensorBoard）

| 指标 | 值 | 说明 |
|------|-----|------|
| KL divergence (最终) | 29.42 | 策略偏离 SFT 显著 |
| valid_rate | 100% | 全部轨迹通过格式检查 |
| α mean (训练中) | 0.848 | 门控高 KG 权重。**不受 §二 撤回影响**（训练走 `reward_function.py`，不是 pipeline 的 `_compute_alpha_stats`），且与统一状态重测的推理期 0.908–0.918 同量级 —— 反过来印证 §二 旧表的 0.292 是推理端退化态，不是模型行为 |
| mean_reward | 6.26 | 正向且稳定 |
| r_kg_mean | 0.250 | KG 引用质量中等 |
| r_text_mean | 0.764 | 文本流畅度高 |
| kg_reward_share | 2.7% | KG 分量贡献低 (EM 主导) |
| n_steps | 8.0 | 生成步数打满 |
| sft_anchor_loss | 10.46 | 偏离 SFT 输出较多 |
| epoch timing | 18s/update | rollout 9s + reward 3s + ppo 6s |

## 七、KG 噪声审计

| 类别 | 占比 | 示例 |
|------|------|------|
| 可能有用 | 30.2% | occupation, educated_at, country of citizenship |
| 平凡无用 | 21.4% | given_name, family_name, sex_or_gender, instance_of |
| 噪声/错误 | 48.4% | entity linker 错误, Wikimedia metadata, 无关属性 |
| **总计三元组** | **6,852** (300 题) | 修复前 22.8/q → 修复后 10.3/q (−55%) |

## 八、IHR（中间幻觉率）评估

**方法**: deepseek-v4-pro 作为 LLM Judge, 每数据集 50 样本。

| 数据集 | SFT noKG | SFT+KG | SFT Δ | PPO noKG | PPO+KG | PPO Δ |
|--------|---------|--------|-------|----------|--------|-------|
| HotpotQA | 22.1% | 20.6% | −1.5pp | 18.0% | 25.0% | +7.0pp |
| 2WikiMultiHopQA | 28.7% | 20.6% | −8.2pp | 15.0% | 13.0% | −2.0pp |
| MuSiQue | 31.9% | 26.3% | −5.6pp | 52.0% | 52.0% | 0.0pp* |

> **数据来源（2026-08-23 补记）**：`outputs/ihr_eval/*.json`，judge = `deepseek-chat`，n=50/臂。逐一核对：hotpotqa sft 0.2212/0.2062、ppo 0.18/0.25；2wiki sft 0.2873/0.2057、ppo 0.15/0.13；musique sft 0.3194/0.2633、ppo 0.52/0.52 —— 与本表一致。

> \* MuSiQue PPO 两臂均注入 0 KG（wikidata 断网），故 Δ=0；PPO 各臂均为 `kg_proweight_ppo_v2`（"Plan B"，见 §十）。

**核心发现**: SFT 阶段 KG 在三个数据集上均一致降低中间幻觉率（幅度 **2wiki −8.2pp > MuSiQue −5.6pp > HotpotQA −1.5pp**）。但 PPO（v2）阶段结论反转：**MuSiQue IHR 从 31.9% 飙到 52.0%**（坐实 -17pp 的 OOD 崩溃是幻觉驱动），HotpotQA +KG 反使幻觉升高（18%→25%，KG 在过拟合模型上起反作用），仅 2Wiki 方向正确但幅度微弱（-2pp）。进一步印证 v2 checkpoint 被 "Plan B" 训坏（§十）。

> ⚠️ **"IHR 降幅与 α / 推理复杂度正相关" 这个说法不成立**（2026-08-22 更正）。两点：
> ① **排序本身是错的**：本表 SFT Δ 的排序是 2wiki(−8.2) > MuSiQue(−5.6) > HotpotQA(−1.5)，
> 而"推理复杂度"和旧 α 的排序都是 MuSiQue > 2wiki > HotpotQA —— **后两名反了**，
> 三个点里只有 HotpotQA 最小这一点对。原正文写成 "MuSiQue > 2wiki > HotpotQA"，与本表自相矛盾，已改。
> ② **α 侧的前提也没了**：统一状态重测后 α 三个数据集无可测差异（0.918/0.914/0.908，见 §二），
> 所以不存在"与 α 一致"这回事。`05_results.md` §5.4 里同样的句子须一并删除。
>
> ⚠️ **judge 口径（2026-08-23 已溯源，本节写错了）**：本表 12 个数字全部来自 `outputs/ihr_eval/{hotpotqa,2wiki,musique}_{sft,ppo}_{kg,nokg}.json`，这 12 个文件的 `judge_model` 字段**一律是 `deepseek-chat`**，不是本节原先写的 `deepseek-v4-pro`。
> 因此 `05_results.md` §5.5 写的 DeepSeek-Chat 才是对的，本节的 v4-pro 表述已改。
> **但真正的问题没解决，反而更明确了**：所有基线 IHR（`outputs/baselines*/**/ihr_result_*.json`，共 10 个文件）用的是 `deepseek-v4-pro`，而主方法这 12 臂用的是 `deepseek-chat` —— **主方法与基线不是同一个判官，现在同表并列的 IHR 全部不可比**。这正是 P5 要修的东西：用 `deepseek-v4-pro` 重判主方法 12 臂（`run_ihr_judge.py` 现已把 `--judge_model` 改成必填，漏传会直接报错而不是静默换判官）。在 P5 跑完之前，**任何主方法 vs 基线的 IHR 对比都不能写进论文**。

## 九、实验环境

| 项目 | 配置 |
|------|------|
| 基座模型 | Llama-3-8B-Instruct (LoRA: r=32, α=64) |
| GPU | NVIDIA RTX PRO 6000 Blackwell (96 GB) |
| 检索 | E5-base + BM25, RRF k=60, bge-reranker-v2-m3 → top-10 |
| 推理 | max_input=6144, max_new_tokens=512, greedy decoding |
| KG 源 | Wikidata, 2-hop SPARQL, 离线实体缓存 35K+ |
| Phase 2 训练 | 26,583 samples, 3 epochs, ~50 min, 20 GB VRAM |
| Phase 3 PPO 训练 | 750 updates, ~4 hours, ~95 GB VRAM |

---

*数据截止: 2026-08-16. 所有评估使用 wiki18 全量 Wikipedia 检索 (21M 文档).*

---

## 十、数据口径与已知限制（2026-08-16 更新）

> 以下为 eval 质量排查结论，供组会汇报参考。**主表已按 canonical `kg_proweight_ppo_v2` 更正**（HotpotQA PPO+KG 0.3433→0.2133、补 MuSiQue PPO 两行、§八 新增 PPO IHR）。

| # | 问题 | 说明 |
|---|------|------|
| 1 | **"Plan B"奖励权重训坏了 PPO（已回退）** | `phase3_ppo.yaml` 曾未提交地把 `outcome_weight 4→8`、`step_reward_scale 1.5→0.5`（EM 独大、砍掉 KG 过程奖励），训出的 `kg_proweight_ppo_v2` 全线低于 SFT（hotpotqa+KG -14pp、musique -17pp），musique IHR 飙到 52%。已回退到 4.0/1.5，**待重训**。 |
| 2 | checkpoint 混用（§五 不可信） | 历史表把 `ppo_r10_split/final`（outcome 4.0）与 `kg_proweight_ppo_v2`（outcome 8.0）数值混在同一"PPO v2"名下，两者质量差 13pp。§一/§四 已统一到 v2；§五 待重训后重测。 |
| 3 | wikidata 断网 → musique 无 KG | `question_kg_index` 只覆盖 hotpotqa-dev；musique 全靠 live Wikidata 兜底，而 2026-08-16 起 wikidata.org SSL 超时不可达，PPO+KG 实测注入 0 KG（与 noKG 逐位相同）。修网络/补离线缓存前，musique +KG 无有效结论。 |
| 4 | **银标数据质量是上游瓶颈** | `silver_v1_reannotated.jsonl`（24,998 条）：teacher=`deepseek-v4-flash`；仅 31% 推理步引用 KG（`cited_triples` 非空）；accepted 率仅 39.4%；PRM 标签 80% 为 0.0（中性类、判别信号弱）；`kg_reward_share` 仅 2.7%。这决定 phase1/2/3 上限，比缓存重要得多。 |
| 5 | 单 seed、噪声地板 ~4pp | 主表为 seed_42 单次 n=300。KG 效果 +1.3~2.7pp 均在噪声内，§三 McNemar 已标 NS——汇报应表述为"方向一致、未达显著"。 |
| 6 | KG 键归一化 bug（已修） | 构建 index 时键 strip、查找未 strip，~10% 题静默丢 KG。修复后覆盖 90%→96%，EM +1pp。 |
| 7 | α 在 eval 阶段是 telemetry | eval 时 α 不参与 KG 门控，只用于训练期 reward 重加权。 |
| 8 | 实体链接覆盖缺口 | 4% 题在 index 构建时 n_linked=0（Vermont Catamounts、Euromarché、D1NZ 等专名漏链），KG 为空。 |
