# Source-credit v2：代表性 Text 统计重绑定复核

最终产物：`outputs/calibration/source_credit_gate_v2_representative_local_seed42_20260906_v1/manifest.json`。Experiment ID 为 `SOURCE-CREDIT-V2-REPRESENTATIVE-TEXT-20260906-V1`，状态为 `V2_REPRESENTATIVE_TEXT_REPAIRED_NOT_INDEPENDENT_CONFIRMATION`。

父校准为 `outputs/calibration/source_credit_gate_v2_stepfeatures_local_seed42_20260906_v2/`；补充候选银行为 `outputs/audits/normalization_representative_bank_v1_seed42_20260906_r2/`。旧版本与失败的 r1 均保留。本次只替换 Text 归一化的拟合人口及相应统计/来源字段，没有重新训练 α 或 PPO。

## 完成的来源与不变变量校验

`scripts/train/rebind_source_credit_v2_text_population.py` 在读取前验证 parent 两个 variant 各自精确的 gate/report/candidates/assignments 输出集合，绑定路径必须等于随后实际读取的文件。新银行的相对文件路径只在其所属银行目录解析，并校验 `origin_path`；消除了校验一个文件却读取另一个固定路径的缺口。

240 条 compact normalization row 均从绑定的输入、问题选择清单、完整 generation record、完整 scored record 重放。核对 dataset/qid、question/family hash、input hash、K2 index、冻结 SFT policy/base/generation contract、raw Text scores、format validity 与 origin。`generation_sha256` 按 producer 合同指完整 prediction record 的 canonical digest，不误解为生成文本字符串的 hash。94 条旧候选精确复用，146 条新候选来自原 SFT 和原 bf16 ReaRAG；失败候选未删除或补抽。

从绑定的 3000 题 question-only groups、旧 family split 和保护账本重新执行固定 seed 42 的 identity-only、reuse-first hash 选择，得到相同 H40、M40、W32 图 + W8 普通题人口。新人口与旧已消费 calibration/confirmation family 的重叠为 0，保护身份/问题/family 重叠为 0。选择与统计均未使用 Gold、EM 或 F1。

执行开始冻结实际使用的 18 个代码依赖以及所有读取的来源文件，发布前再次检查字节一致，并保存源码快照。两组 variant 的 candidates 与 assignments 均为原字节复制；各自 1660 条 α 预测逐条完全相同。额外字段比较确保 α 权重、标准化、Graph 统计、固定 α、来源 mask 和其他研究变量均未改变。

## 拟合人口与新数值

输入人口为 120 题 / 240 个 K2 候选，其中 192 个 format-valid 候选提供 654 步 Text 观测，118 题至少有一个有效候选。两个全部失败的 HotpotQA 问题保留在输入记录中，不参与有效观测均值，也没有替换。

| 数据集 | 输入题数 | 有有效观测题数 | 有效候选数 | 步数 |
|---|---:|---:|---:|---:|
| HotpotQA | 40 | 38 | 62 | 209 |
| 2Wiki | 40 | 40 | 71 | 226 |
| MuSiQue | 40 | 40 | 59 | 219 |

统计仍严格遵循 question → valid candidate → step 的层级等权合同，得到 `text_center = 0.6359942726986936`，`raw_step_std = text_scale = 0.20923710334704368`，scale floor 为 0.1。方差分解为 within-trajectory `0.026167980372965943`，between-trajectory-means `0.01761218504409549`。应用仍为每步中心化/缩放、softsign、最后取均值。

输入的 H/W/M 题数相同，但有效题数为 38/40/40，因此实际有效问题权重并非强行每数据集 1/3。这个差异按原“保留失败、不补抽”规则公开记录；没有为了得到更整齐的统计再次选题。

## 测试与科研边界

最终相关 CPU 回归共 97 项通过：feature-v2 20 项、gate-v2 30 项、calibrator 20 项、representative rebind 27 项。除来源/路径/人口/冻结模型回归外，包含六步 invalid 轨迹与 production 共用前五步 feature view 的回归；修复以新的 stepfeatures v2 校准目录发布，初版两条 invalid telemetry 差异记录保留。

训练/独立 confirmation/PPO 放行标志均为 False，PPO optimizer update 为 0。代表性 train 补充银行用于 Text 统计修复，不能当新独立 confirmation。Source PASS 图信用人口仍为 671 / 800。softsign 的无 hard clipping 是映射性质，表示容量增加与归一化修复也都不是过程奖励有效、语义可靠或 EM/F1 提升的证据。
