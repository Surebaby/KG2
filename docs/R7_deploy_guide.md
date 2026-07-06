# R7 部署与训练指南

> 适用: AutoDL RTX PRO 6000 Blackwell 96GB

---

## 一、代码同步清单

以下文件需要同步到服务器。相对路径从 `kgpaper/` 根目录起。

### 核心 R7 修改（必须同步）

| 文件 | 状态 |
|------|:---:|
| `kgproweight/reward/composite_reward.py` | ✏️ 已修改 |
| `kgproweight/training/reward_function.py` | ✏️ 已修改 |
| `kgproweight/training/phase3_ppo.py` | ✏️ 已修改 |
| `configs/training/phase3_ppo.yaml` | ✏️ 已修改 |

### 依赖修复（必须同步，否则 CLI 报错）

| 文件 | 状态 |
|------|:---:|
| `kgproweight/config/schemas.py` | ✏️ 已修改 (删除 step_format_bonus, 新增 R7 字段) |
| `scripts/train/phase3_ppo.py` | ✏️ 已修改 (移除 step_format_bonus 引用) |
| `schemas.py` | ✏️ 已修改 (根目录副本) |
| `phase3_ppo.py` | ✏️ 已修改 (根目录副本) |

### 可选同步

| 文件 | 说明 |
|------|------|
| `docs/R7_experiment_log.md` | 实验记录 |
| `docs/R7_deploy_guide.md` | 本文档 |
| `docs/problem_and_solutions.md` | 方案讨论 |

### 快速同步命令

```bash
# 在本地执行 (AutoDL 的 JupyterLab 内置终端或 SSH):
# 假设 kgpaper 目录已通过 AutoDL 文件上传同步

# 验证所有修改已到位:
cd ~/kgpaper
grep -r "step_format_bonus" kgproweight/training/reward_function.py kgproweight/training/phase3_ppo.py configs/training/phase3_ppo.yaml kgproweight/config/schemas.py scripts/train/phase3_ppo.py
# 预期: 只有注释中出现，无可执行代码
```

---

## 二、环境检查

```bash
# 1. 确认 GPU
nvidia-smi
# 预期: RTX PRO 6000 Blackwell, 96GB

# 2. 确认 Python 包
pip list 2>/dev/null | grep -iE "torch|transformers|trl|peft|tensorboard"
# 必须: torch>=2.0, transformers>=4.40, trl>=0.11, peft, tensorboard

# 3. 如果缺包:
pip install tensorboard trl peft -U
```

---

## 三、训练启动

### 3.1 确认数据路径

```bash
cd ~/kgpaper

# 检查必要文件是否存在:
ls -lh checkpoints/sft_student_elite/final/adapter_config.json   # Elite SFT 基座
ls -lh checkpoints/prm_alpha_gate/alpha_gate.pt                  # α-gate 权重
ls -lh checkpoints/prm_alpha_gate/silver_with_logprobs.jsonl      # PPO 训练数据
ls -lh data/silver_data/silver_trajectories.jsonl                 # 银标数据（SFT anchor 用）
```

### 3.2 启动命令

```bash
cd ~/kgpaper

python scripts/train/phase3_ppo.py \
  --config configs/training/phase3_ppo.yaml \
  --sft_checkpoint checkpoints/sft_student_elite/final \
  --alpha_gate_path checkpoints/prm_alpha_gate/alpha_gate.pt \
  --silver_data checkpoints/prm_alpha_gate/silver_with_logprobs.jsonl \
  --output_dir checkpoints/kg_proweight_R7A \
  --seed 42
```

### 3.3 后台运行 (推荐)

```bash
nohup python scripts/train/phase3_ppo.py \
  --config configs/training/phase3_ppo.yaml \
  --sft_checkpoint checkpoints/sft_student_elite/final \
  --alpha_gate_path checkpoints/prm_alpha_gate/alpha_gate.pt \
  --silver_data checkpoints/prm_alpha_gate/silver_with_logprobs.jsonl \
  --output_dir checkpoints/kg_proweight_R7A \
  --seed 42 \
  > logs/R7A_train.log 2>&1 &

# 记录 PID
echo $! > logs/R7A_train.pid
```

### 3.4 TensorBoard 监控

```bash
# 另开终端:
tensorboard --logdir checkpoints/kg_proweight_R7A/tensorboard --port 6006 --bind_all

# 浏览器访问: http://<服务器IP>:6006
```

---

## 四、进度检查

### 4.1 查看训练日志

```bash
tail -f logs/R7A_train.log
```

### 4.2 查看 history.jsonl (最近 5 条)

```bash
tail -5 checkpoints/kg_proweight_R7A/history.jsonl | python -m json.tool
```

### 4.3 检查关键指标 (1000步/2h)

```bash
python -c "
import json
with open('checkpoints/kg_proweight_R7A/history.jsonl') as f:
    lines = f.readlines()
    last = json.loads(lines[-1])
    print(f'step={last[\"step\"]}')
    print(f'mean_reward={last[\"mean_reward\"]:.4f}')
    print(f'valid_rate={last[\"valid_rate\"]:.3f}')
    print(f'sft_anchor_loss={last[\"sft_anchor_loss\"]:.4f}')
    print(f'ppo_mean_kl={last[\"ppo_mean_kl\"]:.4f}')
"
```

### 4.4 检查存档点

```bash
ls -d checkpoints/kg_proweight_R7A/step_*
# 预期: step_504, step_1008, step_1512, step_2016, step_2520, ...
```

---

## 五、如果训练崩溃

### 回退到 R6-A 代码

```bash
cd ~/kgpaper
cp backups/R7_rollback/composite_reward.py.bak kgproweight/reward/composite_reward.py
cp backups/R7_rollback/reward_function.py.bak kgproweight/training/reward_function.py
cp backups/R7_rollback/phase3_ppo.py.bak kgproweight/training/phase3_ppo.py
cp backups/R7_rollback/phase3_ppo.yaml.bak configs/training/phase3_ppo.yaml
# schemas.py 和 scripts 也需回退
```

### 从存档点恢复训练

```bash
# 如果训练在 step 2000 崩溃，从 step_1512 恢复:
python scripts/train/phase3_ppo.py \
  --config configs/training/phase3_ppo.yaml \
  --sft_checkpoint checkpoints/kg_proweight_R7A/step_1512 \
  ...
```

---

*最后更新: 2026-07-02*
