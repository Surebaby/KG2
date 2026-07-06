# PPO 阶段格式退化 — 整体解决方案

> **问题**: Elite SFT 学会 100% 格式 + 丰富推理内容 → R7-B PPO 后 69% 有 `[Step N]` 但推理内容 100% 空白（`"Reasoning: \nFinal Answer: X"`）
> **根因**: ValidTrajectory gate 只检查标记存在性，不检查内容质量。PPO 发现「有标记 + 无推理 + 给答案」拿到的 reward 和「有标记 + 有推理 + 给答案」一样。

---

## 方案总览

| # | 方案 | 等级 | 改动量 | 预期效果 |
|---|------|:--:|:--:|---|
| A | **内容感知 ValidTrajectory** | 🔴 必做 | 小 (~30行) | 堵死「空推理」捷径 |
| B | **动态 KL β** | 🟡 建议 | 中 (~50行) | 自动稳定，减少调参 |
| C | **增强 SFT Anchor** | 🟡 建议 | 小 (改参数) | 强化格式记忆 |
| D | **SFT 数据混入 PPO Prompt 池** | 🟢 选做 | 中 (~80行) | 提升 rollout 多样性 |

---

## A. 内容感知 ValidTrajectory（必做）

### 当前问题

`_is_valid_trajectory()` 检查：
```python
if not s.raw_text or not s.raw_text.strip():  # ← 只检查是否为空
    return False
```

R7-B 输出 `[Step 1]\nReasoning: \nFinal Answer: X`，`raw_text` = `"Reasoning: \nFinal Answer: X"` → strip 后非空 → gate 通过。

### 修改方案

在 `reward_function.py` 的 `_is_valid_trajectory()` 中增加内容质量检查：

```python
@staticmethod
def _is_valid_trajectory(
    steps: list,
    response: str,
    min_steps: int = 3,
    min_reasoning_chars: int = 20,      # ← 新增
    min_knowledge_items: int = 1,       # ← 新增（当 KG 非空时）
    kg_is_empty: bool = True,           # ← 新增
) -> bool:
    # ... 原有检查 ...
    
    # --- 新增：内容质量检查 ---
    for s in steps:
        body = s.raw_text.strip()
        
        # 1) 必须有 Reasoning 段且内容 > min_reasoning_chars
        if "Reasoning:" in body:
            reasoning = body.split("Reasoning:", 1)[1]
            # 截止到 Knowledge/Conclusion/Final Answer
            reasoning = re.split(r'Knowledge Used:|Conclusion:|Final Answer:', reasoning)[0]
            reasoning = reasoning.strip()
            if len(reasoning) < min_reasoning_chars:
                return False  # ← 空推理，gate 关闭
        
        # 2) 非空 KG 时必须有 Knowledge Used
        if not kg_is_empty:
            if "Knowledge Used:" not in body:
                return False
            ku = body.split("Knowledge Used:", 1)[1].split("Conclusion:", 1)[0]
            if "(" not in ku or ")" not in ku:
                return False  # ← 无有效三元组
    
    return True
```

### 效果

| | 当前 R7-B | 加内容 Gate |
|---|---|---|
| `[Step 1]\nReasoning: \nFinal Answer: X` | ✅ valid | ❌ invalid（无奖励） |
| `[Step 1]\nReasoning: Beethoven was born in Bonn...` | ✅ valid | ✅ valid |

PPO 会发现「不写推理 = 拿不到 outcome reward」，被迫恢复推理内容。

---

## B. 动态 KL 惩罚（建议）

### 思路

对不同 token 施加不同 KL 系数：**格式关键区放大，内容区正常**。

```
[Step 1]  Reasoning:  <推理内容>  Knowledge Used:  <三元组>  Conclusion:  <结论>
  ↑↑↑        ↑↑↑         ↑           ↑↑↑               ↑          ↑↑↑          ↑
  β×3       β×2        β×1          β×2              β×1         β×2         β×1
```

### 实现方案

在 `step_reward_ppo_trainer.py` 或 `phase3_ppo.py` 中：

```python
# 1. 检测格式 token 区间
FORMAT_PATTERNS = [
    r"\[Step \d+\]",         # 步骤标记
    r"Reasoning:",            # 推理头
    r"Knowledge Used:",       # 知识头
    r"Conclusion:",           # 结论头
    r"\[Final Answer\]",      # 最终答案标记
]

def get_format_token_mask(response_ids, tokenizer, format_weight=3.0):
    """返回每个 token 的 KL 权重向量。格式区 = format_weight，内容区 = 1.0"""
    response_text = tokenizer.decode(response_ids)
    weights = torch.ones(len(response_ids))
    for pattern in FORMAT_PATTERNS:
        for m in re.finditer(pattern, response_text):
            start_char, end_char = m.span()
            # char → token 映射
            start_tok = _char_to_token(response_ids, tokenizer, start_char)
            end_tok = _char_to_token(response_ids, tokenizer, end_char)
            weights[start_tok:end_tok] = format_weight
    return weights

# 2. 在 PPO loss 中使用
# kl_div = (logprob - ref_logprob) * format_weights  # 加权 KL
```

### 自适应 β 动态调节

```python
# 每个 batch 后更新
def update_kl_coef(current_kl, format_violation_rate, base_kl=0.1):
    if format_violation_rate > 0.5:     # 超过半数丢格式
        return base_kl * 3.0            # 紧急拉回
    elif format_violation_rate > 0.2:   # 部分丢格式
        return base_kl * 1.5            # 适度加强
    elif current_kl < 0.01:             # KL 过低（策略漂移）
        return base_kl * 2.0
    else:
        return base_kl                  # 正常
```

---

## C. 增强 SFT Anchor（建议 — 已有框架）

### 当前参数

```yaml
sft_anchor_weight: 0.05    # λ
sft_anchor_interval: 10    # 每 80 条 trajectory 做 1 次
```

### 建议调整

R7-B 的 anchor 把格式拉回来了（69% 标记），但内容没拉住。两个方向：

**方向 1: 增大权重**

```yaml
sft_anchor_weight: 0.15    # 0.05 → 0.15 (3×)
sft_anchor_interval: 8     # 10 → 8 (1.25×)
```

**方向 2: 内容感知 Anchor 采样**

当前 anchor 随机采样 SFT 数据。改为优先采样**多步推理样本**（≥3 步），强化模型对完整格式的记忆：

```python
def _prepare_sft_anchor_data(silver_reader, tokenizer, cfg):
    # 按 step_count 加权采样
    weights = [d.get("n_steps", 1) for d in silver_reader]  # 步数越多权重越大
    # 采样时使用 weights
```

---

## D. SFT 数据混入 PPO Prompt 池（选做）

### 思路

PPO rollout 时，90% 用真实检索结果（探索），10-20% 混入带完整 SFT answer 的 prompt（格式记忆）。

```python
def _mixed_rollout(policy, prompts, sft_prompts, sft_ratio=0.15):
    batch_size = len(prompts)
    n_sft = int(batch_size * sft_ratio)
    n_explore = batch_size - n_sft

    # 探索 prompt: 正常检索，模型自由生成
    explore_outputs = policy.generate(prompts[:n_explore])

    # 格式记忆 prompt: 给完整 SFT answer，让模型见过正确格式
    sft_outputs = policy.generate(sft_prompts[:n_sft])

    return explore_outputs + sft_outputs
```

每 batch 15% 的 rollout 看到完整 SFT 格式轨迹，其余 85% 正常 PPO 探索。SFT 样本的 outcome reward 直接给满分（因为答案已知正确）。

---

## 推荐实施路线

### Phase 1: 最低可行（1-2 小时）

```
A（内容感知 Gate） + C（SFT Anchor 参数调整）
```

- `_is_valid_trajectory` 加 `min_reasoning_chars=20`
- `sft_anchor_weight: 0.05 → 0.15`
- 重新训练，预期恢复推理内容 + 70-80% 格式保持率

### Phase 2: 稳定性增强（半天）

```
Phase 1 + B（动态 KL β）
```

- 添加 token-level KL 权重
- 自适应 β 调节
- 格式保持率目标: 85-95%

### Phase 3: 最终优化（可选）

```
Phase 2 + D（SFT 数据混入）
```

- PPO prompt 池混入 15% SFT 数据
- 格式保持率目标: 90-100%

---

## 关键指标监控

| 指标 | 当前 R7-B final | 目标 | 监控方式 |
|---|---|---|---|
| `[Step N]` 完成率 | 69% | ≥ 90% | TensorBoard |
| Reasoning 内容率 | **0%** | ≥ 80% | 每 500 step 采样 50 条抽查 |
| Knowledge Used 率 | **0%** | ≥ 60% | 同上 |
| EM (hotpotqa) | 0.323 | ≥ 0.35 | metric_score.txt |
| valid_rate (训练中) | ~92% | ~60-80%（因为 gate 更严） | TensorBoard |

---

*2026-07-06*
