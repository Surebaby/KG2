# Plan: Finding 2 修复 — Phase 2 的 link_confidence 改用步级 entity-linker(与 PPO 对齐)

## 目标
让 α-gate 的 `link_confidence` 特征在 **Phase 2 训练**和 **PPO 推理**两端用**同一种量**:
步级 `compute_link_confidence(step_entities, EntityLinker())`。消除当前"训练用轨迹级
coverage、推理用步级 entity-linker 模糊匹配"的分布不一致(静默 miscalibration)。

## 关键发现(已核实)
1. **无需改 silver schema**:步实体由 parser 从 step 文本派生(`ParsedStep.from_text` →
   `ENTITY_RE.findall`),PPO 也是这么拿的(`parse_steps(response)`)。Phase 2 用它已有的
   step 文本重算即可。
2. `EntityLinker.link_confidence` **纯本地**(cache 查 + rapidfuzz),无联网,Phase-2 规模安全。
3. PPO 用 `ImprovedPRMAnnotator()` + 默认 `EntityLinker()`;Phase 2 必须用同一个以对齐。
4. `_StepSample.coverage` 字段在 fixed loop 里**只**作为 link_confidence 来源(282→297),
   校准目标已不再用它 → 改动面很小。

## 改动(全部在 try 文件,包内不动)

### 文件:`scripts/train/try/phase2_prm/phase2_prm_try.py`

**(1) 新增 import**
- `from kgproweight.data.parsers import parsed_step_from_silver_dict`
- `from kgproweight.reward.alpha_gate import compute_link_confidence`
- `from kgproweight.kg.entity_linker import EntityLinker`

**(2) `_build_samples_accepted_only` 增加 entity_linker 参数,计算步级 link_confidence**
- 函数签名加 `entity_linker: EntityLinker`。
- 每个 step:用 `parsed_step_from_silver_dict(step.to_dict())` 拿 `mentioned_entities`
  (与 PPO 完全同源),再 `compute_link_confidence(entities, entity_linker)`。
- 把结果写进 `_StepSample.coverage` 字段(该字段现在语义=步级 link_confidence;
  加注释说明)。这样 `_StepDataset.__getitem__` / `_collate` / 训练循环**零改动**地
  把它当 link_confidence 用。

**(3) `run_phase2_fixed` 构造 EntityLinker 并传入**
- 在调用 `_build_samples_accepted_only` 前:`entity_linker = EntityLinker()`。
- 传给该函数。

**(4) 训练循环 line 282/297 改名 + 注释(语义已变,逻辑不变)**
- `coverage = batch["coverage"]` → 变量名保留(batch key 不变),但注释改为
  "step-level link_confidence(与 PPO 对齐,Finding 2)"。
- line 297 `link_confidence = coverage.clamp(0,1)`:coverage 现已是 [0,1] 的 link_conf,
  clamp 保留作防御。注释更新。

> 注:校准 target(`kg_has_verdict = labels_class != 1`)**不变**——它本就独立于三特征,
> #2 的非退化性质不受影响。

## 验证(CPU,无需 GPU)
1. **单测**:构造若干 silver step dict,断言 Phase 2 算出的 link_confidence ==
   PPO 路径 `compute_link_confidence(parse_steps(text)[i].mentioned_entities, linker)`
   对同一文本的结果(逐位相等)→ 证明两端真正对齐。
2. **import + 离线测试**:`test_ppo_offline.py` / `test_distill_offline.py` 全绿。
3. **(可选 GPU)**:80 条 silver 重跑 fixed Phase 2,确认 EXIT=0、cal loss 仍不坍缩、
   link_confidence 现在每步不同(不再是轨迹常数)。

## 风险 / 权衡
- **EntityLinker cache 为空时** `link_confidence` 恒为 0 → 特征退化为常数。需在验证里
  确认 cache 路径有内容(否则训练端和推理端虽"对齐"但都是 0,α 只由 density+entropy 驱动)。
  这是 PPO 推理端**本来就有**的行为,对齐后两端一致,不引入新问题。
- 不动 silver schema → Phase 1 无需重跑,改动局限 Phase 2 一个文件。
