# 完整 PPO-A probe12 执行与归档

2026-09-06：`A_PROBE12_ENGINEERING_PASS_A_SMOKE600_AUTHORIZED_PREPARING_NOT_STARTED`。

完整 A-probe12 已在远端正常结束（exit 0），独立工程验收通过。研究者随后明确回复“进入”，授权下一版完整 A-smoke600；截至本记录，600正在准备独立配置、scope与发布，尚未启动。原probe产物和fresh132健康FAIL保留，未据此改写评估规则或自动启动full/F/T。

## 实验与模型

训练 Experiment ID：`ppo_mixed4_answer_format_v2_a_probe12_scoped_seed42_20260906_v1`。
监督 Experiment ID：`SOURCE-CREDIT-V2-A-PROBE12-GPU-SUPERVISED-20260906-V1`。

配置为 [A-probe12 scoped seed42](../configs/training/phase3_ppo_mixed4_answer_format_v2_a_probe12_scoped_seed42.yaml)，保留原 Strong SFT `sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final`；policy LoRA可训练，reference为其显式冻结副本，passage过程分由真实冻结ReaRAG计算。答案/格式v2、来源信用v2、ProofKG v2.3、六维 learned α和冻结归一化共同参与。

本轮使用RTX PRO 6000 Blackwell Server Edition 96GB、BF16、seed42、batch4、K4、mini-batch1、PPO epochs2、lr1e-6、最大生成384 tokens。固定顺序为HotpotQA四条→2Wiki四条→MuSiQue四条，共3题/12条轨迹，TensorBoard横轴为4、8、12。原训练800图题/671 PASS信用mask不变。

## 实测与独立验收

| 项目 | 实测结果与含义 |
|---|---|
| 训练完成 | 3批/12条，COMPLETE且exit 0；history与固定schedule一致 |
| 真实过程评分 | Text步骤9+9+16=34；中间图批4条eligible，其中3条合法并实际执行Graph过程分 |
| 格式 | 3/4、3/4、4/4，共10/12；全部输出与失败保留，不重采样 |
| 显式reference KL | 0、0.0209981、0.0607076；三批loss/KL/奖励均有限 |
| α覆盖 | 有效图轨迹预测α均值0.379423；含1条无效零信用后的图批有效α均值0.284567；六维scalar与histogram实际出现 |
| Replay | 逐批0、0、1条，最终1/12；末批CE为0.350869 |
| TensorBoard | 606 scalar、133 histogram、3文本标签；314项event/history/冻结契约数值对照通过 |
| LoRA更新 | 256/256张量发生变化，27,262,334/27,262,976元素改变；key/shape/dtype一致且全部有限，L2变化0.03942885 |
| Value head更新 | 零初始化的4096个weight元素及1个bias元素全部改变，全部有限 |
| 显存 | PyTorch峰值allocated69.55585GiB；监督期间整卡最高观测约78.743GiB。整卡采样包括allocator预留/CUDA等开销，不能与张量分配量混称 |
| 本地归档 | 12份训练文件共126,619,933字节，含完整final、history、manifest、12条samples与TB指针；逐项SHA与远端一致，真实events及日志另存 |

第二批KL在第一次PPO更新之后、任何replay之前已非零，结合显式冻结reference与实际加载契约，支持actor在PPO阶段确有变化。最终LoRA差异包含PPO与replay共同影响，不能按现有日志分摊两者贡献；当前没有逐层梯度范数。Replay不更新value head，其从零变化独立支持PPO critic已更新。

这里只验收完整工程链路。12条在线rollout的EM/F1不构成综合baseline性能结果，3个有效图样本不证明learned α优于固定混合。原独立确认的health FAIL不改成PASS；600用户授权与probe工程通过分别记录，新的600执行仍使用独立版本。

## 权威产物

- [监督归档报告](../outputs/audits/ppo_a_probe12_gpu_supervision_20260906_v1/report.json)
- [独立事件/参数验收](../outputs/audits/ppo_a_probe12_independent_acceptance_20260906_v1/report.json)，SHA `5d6a3d46bc1fd501e5643dce2d225fa802d066d16f0e46f955bbfc77da4b7ef4`
- [完整训练产物](../outputs/audits/ppo_a_probe12_gpu_supervision_20260906_v1/ppo_mixed4_answer_format_v2_a_probe12_scoped_seed42_20260906_v1)
- [远端12文件SHA清单](../outputs/audits/ppo_a_probe12_gpu_supervision_20260906_v1/ppo_a_probe12_scoped_20260906_v1_supervision/output_sha256.json)
- [原始训练日志](../outputs/audits/ppo_a_probe12_gpu_supervision_20260906_v1/ppo_a_probe12_scoped_20260906_v1.log)、[真实TensorBoard events](../outputs/audits/ppo_a_probe12_gpu_supervision_20260906_v1/tensorboard)

原始启动监督status保留`PROCESS_EXITED_ZERO_AWAITING_AUDIT`，这是进程退出时的历史快照；独立验收及本归档报告补充后验结果，不覆盖该记录。

## 查看日志与 TensorBoard

实例开启时，通过AutoPanel→TensorBoard访问6007服务，选择此真实训练run：

```text
kgpaper/ppo_mixed4_answer_format_v2_a_probe12_scoped_seed42_20260906_v1/20260906T081100Z_d7ca78b0
```

该run在`/root/tf-logs`下；`_diagnostics`中的旧缓存或连通性事件分别保留，不能代替本次训练曲线。

远端查看已结束的日志与记录：

```bash
ssh -p 30481 root@connect.bjb1.seetacloud.com
cd /root/autodl-tmp/kgpaper_releases/source_credit_v2_probe12_20260906_v1
tail -n 100 outputs/launches/ppo_a_probe12_scoped_20260906_v1.log
cat outputs/ppo_mixed4_answer_format_v2_a_probe12_scoped_seed42_20260906_v1/tensorboard_run.json
cat outputs/ppo_mixed4_answer_format_v2_a_probe12_scoped_seed42_20260906_v1/history.jsonl
```

本轮已经结束，无需再次启动probe。下一版600的日志与run名称由其独立发布记录给出，不能续写或覆盖这里的产物。
