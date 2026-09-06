# 剩余格式失败与生成约束只读审计

截至2026-09-06，冻结1660条候选中format-v2有效1471、无效189（88.61%通过）。
本审计没有读取Gold、重生成候选、修改prompt、放宽校验或改动采样runtime。
证据位于 `outputs/audits/sourcegate_remaining_format_failures_20260905_v1/report.json`；
逐条记录在同目录 `candidate_diagnostics.jsonl`，可由
`python -m scripts.pilot.audit_sourcegate_remaining_format_failures_v1` 重现到新的输出目录。

## 剩余问题分布

189条无效中81条触及384-token上限，108条由模型生成真正的EOS终止。
全1660条token/decode/身份及EOS记录一致，1573条EOS终止均为128009；不存在误把padding当EOS、
丢弃真实结束token或无EOS却提前截断的证据。另有6条触及上限但仍满足格式；不能一概判所有cap无效。

下列原因重叠，不可相加：

| 原因 | 候选数 |
|---|---:|
| 缺Final marker | 109 |
| 重复Final marker | 2 |
| Conclusion字段不是恰好1个 | 79 |
| Knowledge Used字段不是恰好1个 | 74 |
| 步数不足 | 43 |
| 超过5步 | 18 |
| Reasoning字段不是恰好1个 | 6 |
| 找不到精确Reasoning字段 | 5 |
| 空Final真实内容 | 1 |
| Reasoning不足20字符 | 1 |

43条步数不足包括36条2<3、2条1<3、5条1<2；其中32条只有步数不足，其余格式已经成立。
830题冻结prompt和当前PPO renderer逐题完全相同。SFT系统prompt没有显式动态最小步数，
仍含“Stop generating after [Final Answer].”措辞；这属于冻结提示与训练格式意图之间的表达局限，
本次没有修改。已知空Final样本是到达长度上限，而非在marker后采样EOS，不能单凭该样本归因于提示措辞。

## 不建议增加统一EOS最小长度

实际已安装Transformers的 `min_length` 比较完整prompt+response长度，不能直接当输出长度。
`min_new_tokens` 会把阈值前EOS的logit改为负无穷，因而改变当前承诺的无logit裁剪采样分布。
现PPO注释明确要求采样分布与TRL原logits重算一致，不能把此项当无研究影响的普通开关。

| min_new_tokens | 会抑制的原有效EOS | 会抑制的原无效EOS |
|---|---:|---:|
| 32 | 0 | 0 |
| 64 | 0 | 3 |
| 96 | 5 | 7 |
| 128 | 74 | 16 |
| 192 | 486 | 33 |
| 256 | 1064 | 70 |

这些只是已观察到EOS会被改变的数量，不能解释为修复数。延长生成仍可能漏字段、重复Final或超步数。
同理，遇第5步就停止会丢掉答案；状态机强制字段/Final则是新的受约束策略，须连同old/new policy
概率归一化、KL与终止语义验证，不能直接塞进现PPO generate。

## 可实施的最小下一步

1. 保留现有format-v2和-4非法奖励，按cap/EOS、缺Final/缺字段/少步/多步分开监控小PPO阶段。
   这些异常主要体现冻结SFT的输出格式行为和有限生成预算，不是已证实的token裁剪实现错误。
2. 若要测试输出预算，另立小规模384→512长度probe，保持原输入、模型、seed、采样温度和A/F/T预算一致；
   不重生成整个校准银行，不把新长轨迹混入旧冻结银行。
   81条cap失败中至少10条已有追加token不能修复的前缀错误，因此长度提升收益应实测，不能预报81条全恢复。
3. 对32条只有步数不足的样本，统一最小长度没有直接针对性；当前合法的2步图轨迹也会受到影响。
   暂不增加EOS遮罩、重采样过滤、后处理补空步骤或放宽min_steps。

## 下一版cached-real runtime审计接口

v1审计 `check_source_credit_runtime_cached_v1.py` 及其结果保持为历史证据。新版本应：

- 沿用原32个候选ID和同样的真实raw_text、prompt/step、tokenizer预算与response_token_ids检查。
- 使用新版artifact/schema/loader与显式runtime credit version，从artifact.feature_names动态读取维度，
  独立按weights、means、scales计算logistic alpha，不能调用待验证的gate.predict充当oracle。
- 若启用softsign，独立oracle改为每步 `z/(1+abs(z))` 后平均，再分配 `.3*(1-alpha)/n`；Graph原clip保持。
- 将hard_clip=0、soft_saturation和raw_z_outside_unit分开；hardclip为0只是映射属性。
- 新gate无fresh confirmation时，只能以显式诊断flag注入CPU reward，正常PPO CLI必须拒绝。

这仍然是零更新数值/路由检查，不构成GPU PPO probe、独立过程效用或EM/F1提升证据。
