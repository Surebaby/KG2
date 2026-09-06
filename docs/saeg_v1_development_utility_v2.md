# SAEG-v1 development 零训练效用结果（2026-09-03）

状态：`FAIL_STOP_BEFORE_SFT`。这是development n=150结果；confirmation仍未打开，SFT/PPO均未启动。

## 严格配对结果

| 数据集 | A 无图 EM/F1 | D Passage-QPEG EM/F1 | ΔEM | gained/lost/tied |
|---|---:|---:|---:|---:|
| HotpotQA | .4000/.5988 | .3200/.5147 | -.0800 | 0/4/46 |
| 2WikiMultiHopQA | .3800/.4607 | .3000/.3933 | -.0800 | 0/4/46 |
| MuSiQue | .1000/.1689 | .0600/.1214 | -.0400 | 1/3/46 |
| macro | — | — | -.0667 | — |

三数据集macro F1差为`-.0663`。A/B/D各150条，C/W因fresh结构门失败而明确为`NOT_EVALUABLE`。
14个Passage-QPEG为空的题全部精确回退A，prompt/generation/prediction一致。

## 只读分解

- P非空136题：EM差`-.0735`，净增1、净损11；P为空14题差为0。
- 只有31/136题的P句子包含Gold答案词面；该分组仍净损2、净增0。其余105题净增1、净损9。
- 强SFT在P臂的已知passage citation使用率为`.7533`，而A臂为0；说明模型确实消费了P块，
  但被不完整或歧义的高分句子带偏，不是“模型完全忽略图”。
- P块平均增加约163 prompt tokens；负效应集中在2–4条边，不是只有超长/超密图才失败。

代表性失败包括：比较题只提供一侧日期/国籍、桥接题给出多个可能的电视剧/实体却没有消歧第二跳，
以及选出的高相关句子没有包含问题所需终点。selector score很高仍可能语义不完整，因此不能通过简单
下调/上调现有score阈值宣称修复。

## 根因边界

当前训练P轨迹由raw-train Gold supporting facts/decomposition构建，是完整正例；development P则由
answer-free sentence selector从检索Top-10自动选择。也就是说，当前SFT候选只教“信任完整P”，
没有教模型在自动P不完整/歧义时回退full passages。这是明确的train/eval evidence-quality错配。

本结果只否定“当前P构造+当前prompt可直接进入SAEG-SFT”，不否定passage-derived evidence的所有
可能实现，也不改变2Wiki canonical ProofKG额外资源结果。

## 建议的单变量修复

若继续三数据集主线，优先做`P hard-negative alignment v2`，而不是越过失败门直接训练：

1. 在新的raw-train题上运行与evaluation完全相同的answer-free P selector；
2. 用仅训练可见的Gold supporting facts/decomposition离线判定`complete / partial / misleading`；
3. complete正例保留P citation目标；partial/misleading例保留模型可见的P块，但监督轨迹必须回退full
   passages、不得引用错误P；
4. 冻结正/负比例及family隔离，再在新的development协议上只检验这一数据对齐变量；
5. 若仍不能把D−A拉回非负，则停止P输入增强，不能启动SAEG-SFT/PPO-K。

该修复会改变训练数据构造，需研究者批准后执行。不得事后覆盖本次负结果或修改v2门。

机器结果：`outputs/validation/saeg_v1_development_strong_sft_npdf_v2_attempt2/report.json`。
