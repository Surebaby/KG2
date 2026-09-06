# TRACE-Gate / KG-ProWeight

面向多跳 RAG 的图文过程奖励与 PPO 研究代码。代码仓库：[Surebaby/KG2](https://github.com/Surebaby/KG2)。Python 包名保持为 `kgproweight`。

## 当前版本（2026-09-06）

当前主线使用原 Strong SFT，结合答案奖励、Text / Graph 过程奖励、六维来源特征的冻结 α 门控和 SFT replay。

- 正式 PPO 数据已封版：HotpotQA / 2Wiki / MuSiQue 各 1,000 题，K=4；800 个 strict ProofKG 问题，2,000 条 replay，每题 10 passages。
- A-smoke600 实际运行到 300 条 rollout、75 个不同问题后，触发原有效率保护规则停止。训练 `FAILED` 与 `step_200`、`aborted_step_300` 均保留。
- 固定 150 题开发评估已完成：三模型 × legacy / no_graph 两视图，共 900 条预测。逐题评分独立复验通过，原 EM 优先规则仍选中 **Strong SFT**。
- 本轮尚未证明 PPO 综合提升或 α 独立贡献；这些数字不是正式 canonical900 主表。

开发集三域宏平均，单位为百分比：

| 模型 | legacy EM | legacy F1 | no_graph EM | no_graph F1 |
| --- | ---: | ---: | ---: | ---: |
| Strong SFT | 24.00 | 34.70 | 25.33 | 36.66 |
| PPO step_200 | 22.67 | 35.34 | 24.67 | 35.25 |
| PPO aborted_step_300 | 22.00 | 32.79 | 24.67 | 36.50 |

当前进度以 [RESEARCH_WORKFLOW.md](RESEARCH_WORKFLOW.md) 和 [docs/todo2.md](docs/todo2.md) 顶部为准。历史实验与失败记录保留在研究文档中，不能与当前条件直接混比。

## 安装与本地检查

使用 Python 3.10 或更新版本。GPU 训练环境需另行核对 CUDA、模型路径和冻结实验配置。

```bash
git clone https://github.com/Surebaby/KG2.git
cd KG2
python -m pip install -e ".[dev]"
export PYTHONPATH="$PWD:$PWD/flashrag_src${PYTHONPATH:+:$PYTHONPATH}"
export KGPW_PROJECT_ROOT="$PWD"
python -m pytest -q tests/test_ppo_emf1_development_v1.py
```

`flashrag_src/flashrag` 是随仓库提供的源码，通过上述 `PYTHONPATH` 使用。
依赖声明见 [pyproject.toml](pyproject.toml)；已有部署环境的锁定清单见 [scripts/deploy/requirements-lock.txt](scripts/deploy/requirements-lock.txt)。锁定清单记录特定部署环境，新机器仍需兼容性检查。

密钥和远端登录凭据通过环境变量或本机凭据管理器配置，变量示例见 [.env.example](.env.example)。

## 代码入口

| 目录 | 内容 |
| --- | --- |
| `kgproweight/data/` | 数据契约、提示与解析 |
| `kgproweight/reward/` | 答案目标、ProofKG、Text 校准与 α 来源门控 |
| `kgproweight/training/` | SFT、PPO、逐 token 奖励和 replay |
| `scripts/prepare/` | 数据构建、身份隔离和协议封存 |
| `scripts/train/` | 校准与训练入口 |
| `scripts/eval/` | 冻结开发评估、baseline 与选模 |
| `scripts/diagnose/` | 工程和科研诊断 |
| `scripts/deploy/` | 环境、远端执行与 TensorBoard |
| `configs/` | 版本化配置 |
| `tests/` | 单元与契约测试 |

具体实验使用对应文档给出的配置、数据版本和唯一 Experiment ID。通用脚本和历史 launcher 的存在不表示应直接启动这些任务。

## 研究记录

- [项目规则](AGENTS.md)
- [完整奖励与 α 设计复核](docs/ppo_alpha_review_20260905_v1.md)
- [source-credit-v2 修复](docs/source_credit_v2_repair_20260906.md)
- [A-smoke600 训练与停止](docs/ppo_a_smoke600_supervision_20260906.md)
- [三模型开发评估](docs/ppo_a_stopped300_development_eval_20260906.md)
- [PPO 丢分诊断](docs/ppo_a_stopped300_regression_diagnosis_20260906.md)
- [AutoDL TensorBoard 配置](docs/ppo_tensorboard_autodl_20260905.md)

## 代码与实验资产

本仓库同步代码、配置、测试和研究进度文档。模型权重、tokenizer 资产、原始数据、索引、checkpoint、完整预测与审计输出、论文草稿和参考 PDF 保留在研究工作区，不包含在此次代码同步中。

文档和冻结配置中的 `data/`、`models/`、`checkpoints/`、`indexes/`、`outputs/` 路径指向这些外部实验资产。复现实验前需将对应版本放入原相对路径并核对 SHA；仅克隆代码不能直接复现已封存训练。

新版工作树移除已淘汰的 `archive/`、`_archive/`、`scripts/train/try/`、旧 R7 部署包及含硬编码凭据的旧辅助脚本。被运行路径、测试或实验复现使用的版本化模块继续保留。旧提交历史保持可追溯。

许可证见 [LICENSE](LICENSE)。
