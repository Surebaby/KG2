#!/usr/bin/env python
"""G5 — does the PRM head GENERALISE, or did it memorise the silver labels?

The 0.991 accuracy reported after training was measured on data the head was
trained on, so it establishes only that optimisation succeeded. Phase 2's
actual objective is a PRM that scores UNSEEN steps, because Phase 3b calls it
on freshly generated PPO rollouts.

A real train/val/test split now exists (kgproweight.data.silver_split), but the
checkpoints evaluated here were trained BEFORE it, with split=None — i.e. on the
whole file — so no fold was held back from them and a same-distribution held-out
number is not available for them retroactively. Once a checkpoint is trained with
``--split train``, evaluate it on the ``test`` fold instead of the proxy below
and report THAT as held-out; this script prints the checkpoint's recorded fold
(from manifest.json) so the two cannot be confused.

For a split=None checkpoint, what we CAN measure is the next best thing, and we
label it honestly:

  A. Trajectory-disjoint eval on REJECTED trajectories. Phase 2 trained on
     accepted-only (_build_samples_accepted_only), so every step of a rejected
     trajectory is genuinely unseen. Labels come from the same PRMAnnotator, so
     the target semantics match.

  B. Majority-class and random baselines on that same set, so the accuracy
     number is interpretable rather than impressive-sounding.

  C. Per-class recall + confusion matrix. NEG is only ~4% of data; a head that
     silently drops NEG still scores ~70% overall but is useless to PPO, whose
     entire purpose is penalising hallucinated steps.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kgproweight.data.parsers import parsed_step_from_silver_dict
from kgproweight.training.phase2_prm import (
    PRMHead,
    _StepSample,
    _last_nonpad_hidden,
    build_prm_input,
)
from kgproweight.utils.paths import model_path

CKPT = Path(sys.argv[1] if len(sys.argv) > 1 else
            "checkpoints/prm_alpha_gate_v1reann")
SILVER = sys.argv[2] if len(sys.argv) > 2 else \
    "data/silver_data/silver_v1_reannotated.jsonl"
MAX_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
# Checkpoints trained BEFORE the NEG fix never saw a "[Prior Conclusions]"
# prefix. Scoring them with one measures a train/eval format mismatch on top of
# the model, so a pre-fix checkpoint must be evaluated the way it was trained.
# PRM_EVAL_PREFIX=0 reproduces the old bare-step input.
USE_PREFIX = os.environ.get("PRM_EVAL_PREFIX", "1") != "0"
BASE = "llama3-8B-instruct"
# MUST match the cap the checkpoint was TRAINED with (Phase2Config.max_length /
# prm_max_length), because truncation_side="left" drops the OLDEST prior
# conclusions. Evaluating a 1024-trained head at 512 silently strips the
# contradiction evidence from exactly the long chains that carry the most NEG
# signal — depressing the metric under test rather than measuring the model.
MAXLEN = int(os.environ.get("PRM_EVAL_MAXLEN", "1024"))
# Inference only, but the 4090 has 24 GB vs the remote box's 96 GB. bf16 8B
# weights are ~16 GB, leaving ~8 GB for activations; 8 x 512 is comfortable.
BS = int(os.environ.get("PRM_EVAL_BS", "8"))

# What fold, if any, this checkpoint was trained on. A checkpoint trained with
# --split train has a genuine test fold available, and the rejected-trajectory
# proxy below should NOT be the number reported for it. Read from the manifest
# rather than assumed, so an old (split=None) and a new checkpoint can never be
# compared as though both were held-out.
CKPT_SPLIT = None
_manifest = CKPT / "manifest.json"
if _manifest.exists():
    try:
        with open(_manifest, encoding="utf-8") as fh:
            CKPT_SPLIT = (json.load(fh).get("split_info") or {}).get("split")
    except (json.JSONDecodeError, OSError):
        CKPT_SPLIT = None

print("=" * 72)
print("G5  PRM 头泛化能力 (未参与训练的 REJECTED 轨迹)")
print("=" * 72)
print("输入格式: %s" % ("带前序结论前缀 (NEG 修复后)" if USE_PREFIX
                     else "裸步骤文本 (NEG 修复前, PRM_EVAL_PREFIX=0)"))
if CKPT_SPLIT:
    print("检查点训练 fold: %s" % CKPT_SPLIT)
    print("  注意: 该检查点有真正的 held-out fold。本脚本测的是 rejected 代理指标,")
    print("  不要把它当作 held-out 结果上报 —— 请改用 test fold 评测。")
else:
    print("检查点训练 fold: 无 (split=None, 全量文件训练)")
    print("  因此不存在同分布 held-out 集, 下面是分布偏移代理指标。")

# ---------------------------------------------------------- collect unseen
rej_texts, rej_cls = [], []
acc_texts, acc_cls = [], []
n_rej = n_acc = 0
with open(SILVER, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            t = json.loads(line)
        except json.JSONDecodeError:
            continue
        is_acc = bool(t.get("accepted"))
        if is_acc:
            n_acc += 1
        else:
            n_rej += 1
        # Build the input exactly as training does, via the shared
        # build_prm_input(): step text prefixed with prior conclusions. Scoring
        # bare step text here would measure a train/eval mismatch rather than
        # the model.
        prev = []
        for i, s in enumerate(t.get("steps") or []):
            txt = str(s.get("text") or "")
            parsed = parsed_step_from_silver_dict(s, fallback_index=i)
            if txt.strip():
                lab = float(s.get("label", 0.0))
                cls = 2 if lab >= 0.5 else (0 if lab <= -0.5 else 1)
                smp = _StepSample(
                    text=txt, label=lab, label_class=cls, kg_subgraph=[],
                    coverage=0.0, binary_quality=1 if is_acc else -1,
                    semantic_entropy=0.0, prev_conclusions=list(prev),
                )
                rendered = build_prm_input(smp) if USE_PREFIX else txt
                if is_acc:
                    if len(acc_texts) < MAX_STEPS:
                        acc_texts.append(rendered)
                        acc_cls.append(cls)
                else:
                    if len(rej_texts) < MAX_STEPS:
                        rej_texts.append(rendered)
                        rej_cls.append(cls)
            if parsed.intermediate_conclusion:
                prev.append(parsed.intermediate_conclusion)
        if len(rej_texts) >= MAX_STEPS and len(acc_texts) >= MAX_STEPS:
            break

print("轨迹: accepted=%d rejected=%d" % (n_acc, n_rej))
print("评测步骤: 未见(rejected)=%d | 对照(accepted, 训练过)=%d"
      % (len(rej_texts), len(acc_texts)))
if not rej_texts:
    print("没有 rejected 步骤可评测 -> 无法测泛化")
    sys.exit(1)

# ------------------------------------------------------------- load model
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

prm_dir = CKPT / "prm_head"
base_id = model_path(BASE)
print("\n加载 base=%s + LoRA=%s" % (base_id, prm_dir))
tok = AutoTokenizer.from_pretrained(str(prm_dir))
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
# Match _StepDataset: drop the oldest prior conclusions, never the current step.
tok.truncation_side = "left"
base = AutoModel.from_pretrained(base_id, torch_dtype=torch.bfloat16,
                                 device_map="cuda")
# adapter_config.json records the REMOTE base path (/root/autodl-tmp/...), which
# does not exist locally. We already loaded the base explicitly above, so tell
# PEFT to attach the adapter to it rather than re-resolving that stale path.
base = PeftModel.from_pretrained(base, str(prm_dir), is_trainable=False)
base.eval()
hidden = getattr(base.config, "hidden_size", 4096)
head = PRMHead(hidden_size=hidden, n_classes=3).to("cuda", torch.float32)
head.load_state_dict(torch.load(prm_dir / "prm_head.pt", map_location="cpu"))
head.eval()


def predict(texts):
    preds = []
    for i in range(0, len(texts), BS):
        chunk = texts[i:i + BS]
        enc = tok(chunk, return_tensors="pt", truncation=True,
                  max_length=MAXLEN, padding=True).to("cuda")
        with torch.no_grad():
            out = base(input_ids=enc["input_ids"],
                       attention_mask=enc["attention_mask"])
            h = _last_nonpad_hidden(out.last_hidden_state,
                                    enc["attention_mask"])
            preds.extend(head(h).argmax(dim=-1).cpu().tolist())
        if (i // BS) % 20 == 0:
            print("  %d/%d" % (i, len(texts)), flush=True)
    return preds


def report(name, y_true, y_pred):
    names = ["NEG", "NEU", "POS"]
    cm = [[0, 0, 0] for _ in range(3)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    n = len(y_true)
    acc = sum(cm[i][i] for i in range(3)) / n
    counts = [sum(cm[i]) for i in range(3)]
    major = max(counts) / n
    print("\n" + "-" * 72)
    print("%s  (n=%d)" % (name, n))
    print("-" * 72)
    print("混淆矩阵 (行=真实, 列=预测)")
    print("        %8s %8s %8s  |  召回   支持" % tuple(names))
    for i in range(3):
        rec = cm[i][i] / counts[i] if counts[i] else float("nan")
        print("  %-5s %8d %8d %8d  | %.3f  %6d"
              % (names[i], cm[i][0], cm[i][1], cm[i][2], rec, counts[i]))
    f1s = []
    for j in range(3):
        pc = sum(cm[i][j] for i in range(3))
        prec = cm[j][j] / pc if pc else 0.0
        rec = cm[j][j] / counts[j] if counts[j] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        print("  预测为 %-4s %6d 次 (%5.1f%%)  精确率 %.3f  F1 %.3f"
              % (names[j], pc, 100 * pc / n, prec, f1))
    macro = sum(f1s) / 3
    print("  准确率 %.4f | 多数类基线 %.4f | 提升 %+.4f | macro-F1 %.4f"
          % (acc, major, acc - major, macro))
    return {"acc": acc, "majority": major, "macro_f1": macro,
            "n": n, "cm": cm,
            "recall": [cm[i][i] / counts[i] if counts[i] else None
                       for i in range(3)]}


print("\n预测未见步骤 (rejected 轨迹)...")
rej_pred = predict(rej_texts)
r_unseen = report("未见步骤 — 泛化 (REJECTED 轨迹, 训练时被排除)",
                  rej_cls, rej_pred)

print("\n预测训练过的步骤 (accepted 轨迹, 对照)...")
acc_pred = predict(acc_texts)
r_seen = report("训练过的步骤 — 记忆 (ACCEPTED 轨迹)", acc_cls, acc_pred)

gap = r_seen["acc"] - r_unseen["acc"]
neg_rec = r_unseen["recall"][0]

print("\n" + "=" * 72)
print("G5 判定")
print("=" * 72)
print("未见准确率 %.4f (多数类 %.4f, 提升 %+.4f)"
      % (r_unseen["acc"], r_unseen["majority"],
         r_unseen["acc"] - r_unseen["majority"]))
print("训练准确率 %.4f" % r_seen["acc"])
print("泛化 gap   %+.4f" % gap)
print("未见 NEG 召回 %s" % ("%.3f" % neg_rec if neg_rec is not None else "n/a"))
beats = r_unseen["acc"] > r_unseen["majority"] + 0.05
uses_neg = (neg_rec or 0.0) > 0.30
small_gap = gap < 0.20
for label, ok in (("显著优于多数类基线 (+0.05)", beats),
                  ("未见数据上仍能识别 NEG (召回>0.30)", uses_neg),
                  ("过拟合 gap < 0.20", small_gap)):
    print("  %-40s %s" % (label, "PASS" if ok else "FAIL"))
print("\nG5 => %s" % ("PASS" if (beats and uses_neg and small_gap) else "FAIL"))
print("\n注意: rejected 轨迹被 StratifiedSilverFilter 拒绝, 其标签分布与")
print("accepted 不同, 因此这是分布偏移下的泛化测试, 比同分布 held-out 更严格。")

with open(CKPT / "prm_holdout.json", "w") as fh:
    json.dump({"unseen": r_unseen, "seen": r_seen, "gap": gap,
               "pass": bool(beats and uses_neg and small_gap),
               # Recorded so a later comparison cannot silently mix the two
               # input formats and attribute a format effect to the model.
               "input_format": "prefixed" if USE_PREFIX else "bare_step",
               # Same reason: with left truncation, the cap decides how many
               # prior conclusions survive, so a cap mismatch is a train/eval
               # mismatch. Must equal the checkpoint's training max_length.
               "max_length": MAXLEN,
               # The fold the checkpoint was TRAINED on, and what this file
               # actually measures. "rejected_proxy" is distribution-shifted and
               # must not be written up as a held-out result; only an eval run on
               # the test fold of a --split train checkpoint may be.
               "trained_split": CKPT_SPLIT,
               "eval_set": "rejected_proxy",
               "is_held_out": False},
              fh, indent=2)
print("写入 %s" % (CKPT / "prm_holdout.json"))
