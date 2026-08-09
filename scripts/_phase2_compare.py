#!/usr/bin/env python
"""Side-by-side comparison of two Phase 2 checkpoints on the SAME unseen steps.

Run after retraining so the NEG fix is judged against the old checkpoint under
identical conditions, rather than against a remembered number.

Usage:
  python scripts/_phase2_compare.py OLD_CKPT NEW_CKPT [SILVER] [MAX_STEPS]

Reads prm_holdout.json from each checkpoint dir if present; otherwise tells you
which one still needs _phase2_prm_holdout.py run against it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OLD = Path(sys.argv[1] if len(sys.argv) > 1 else
           "checkpoints/prm_alpha_gate_v1reann")
NEW = Path(sys.argv[2] if len(sys.argv) > 2 else
           "checkpoints/prm_alpha_gate_v1reann_negfix")

NAMES = ["NEG", "NEU", "POS"]


def load(p: Path):
    f = p / "prm_holdout.json"
    if not f.exists():
        print("缺少 %s" % f)
        print("  先运行: python scripts/_phase2_prm_holdout.py %s" % p)
        return None
    with open(f) as fh:
        return json.load(fh)


def macro_f1(cm, counts):
    f1s = []
    for j in range(3):
        pc = sum(cm[i][j] for i in range(3))
        prec = cm[j][j] / pc if pc else 0.0
        rec = cm[j][j] / counts[j] if counts[j] else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / 3, f1s


def prec_rec(cm, counts, j):
    pc = sum(cm[i][j] for i in range(3))
    prec = cm[j][j] / pc if pc else 0.0
    rec = cm[j][j] / counts[j] if counts[j] else 0.0
    return prec, rec, pc


a, b = load(OLD), load(NEW)
if a is None or b is None:
    sys.exit(1)

print("=" * 74)
print("Phase 2 前后对比 — 未见步骤 (rejected 轨迹)")
print("=" * 74)
print("旧: %s" % OLD)
print("新: %s" % NEW)

ua, ub = a["unseen"], b["unseen"]
sa, sb = a["seen"], b["seen"]

if ua["n"] != ub["n"]:
    print("\n注意: 评测样本数不同 (%d vs %d), 对比仅供参考"
          % (ua["n"], ub["n"]))

# Each checkpoint must be scored in the format it was TRAINED on, otherwise the
# delta conflates the model change with an input-format change.
fa_, fb_ = a.get("input_format", "?"), b.get("input_format", "?")
print("\n输入格式  旧=%s  新=%s" % (fa_, fb_))
# Left truncation means the token cap decides how many prior conclusions survive,
# so evaluating at a cap other than the training one is itself a train/eval
# mismatch. Each side must be scored at ITS OWN training cap; differing caps
# across the two sides is expected here (512-trained vs 1024-trained).
la_, lb_ = a.get("max_length", "?"), b.get("max_length", "?")
print("token 上限  旧=%s  新=%s" % (la_, lb_))
if la_ == "?" or lb_ == "?":
    print("  警告: 某一侧未记录 max_length, 无法确认它是在训练时的上限下评测的。")
if fa_ == "prefixed":
    print("  警告: 旧 checkpoint 在带前缀的输入上评测, 但它训练时没见过前缀。")
    print("        应以 PRM_EVAL_PREFIX=0 重跑, 否则分数被格式错配拉低。")
elif fa_ == "bare_step" and fb_ == "prefixed":
    print("  各自使用其训练时的格式 — 这是干净的「两次训练产出」对比。")

# A fold difference invalidates the comparison in a way no metric reveals: a
# split=None side trained on the whole file, so its "unseen" proxy set overlaps
# nothing while a --split train side has a real test fold. Comparing the two as
# if they measured the same thing overstates whichever side had more data.
sa_, sb_ = a.get("trained_split", "?"), b.get("trained_split", "?")
print("训练 fold   旧=%s  新=%s" % (sa_, sb_))
if sa_ != sb_:
    print("  警告: 两侧训练 fold 不同, 训练数据量不同, 差值混入了数据量效应。")
if a.get("is_held_out") or b.get("is_held_out"):
    print("  注意: 某一侧声明为 held-out, 与 rejected 代理指标不可直接比较。")
else:
    print("  两侧均为 rejected 代理指标 (非 held-out) — 可比, 但不可写成 held-out。")

ca = [sum(ua["cm"][i]) for i in range(3)]
cb = [sum(ub["cm"][i]) for i in range(3)]
ma, fa = macro_f1(ua["cm"], ca)
mb, fb = macro_f1(ub["cm"], cb)


def row(label, x, y, fmt="%.4f", better="up"):
    d = y - x
    if abs(d) < 1e-9:
        mark = "  ="
    elif (d > 0) == (better == "up"):
        mark = " 改善"
    else:
        mark = " 退化"
    print("  %-26s %10s %10s  %+8.4f%s"
          % (label, fmt % x, fmt % y, d, mark))


print("\n" + "-" * 74)
print("核心指标 (未见数据)")
print("-" * 74)
print("  %-26s %10s %10s  %8s" % ("", "旧", "新", "变化"))
row("NEG 召回", ua["recall"][0] or 0.0, ub["recall"][0] or 0.0)
pa_, ra_, _ = prec_rec(ua["cm"], ca, 0)
pb_, rb_, _ = prec_rec(ub["cm"], cb, 0)
row("NEG 精确率", pa_, pb_)
row("NEG F1", fa[0], fb[0])
print()
row("macro-F1", ma, mb)
row("准确率", ua["acc"], ub["acc"])
row("多数类基线", ua["majority"], ub["majority"], better="down")
print()
row("NEU 召回", ua["recall"][1] or 0.0, ub["recall"][1] or 0.0)
row("POS 召回", ua["recall"][2] or 0.0, ub["recall"][2] or 0.0)
print()
row("训练集准确率", sa["acc"], sb["acc"])
row("过拟合 gap", a["gap"], b["gap"], better="down")

for tag, d, cm, counts in (("旧", a, ua["cm"], ca), ("新", b, ub["cm"], cb)):
    print("\n" + "-" * 74)
    print("%s — 混淆矩阵 (未见, n=%d)" % (tag, sum(counts)))
    print("-" * 74)
    print("        %8s %8s %8s  |  召回   支持" % tuple(NAMES))
    for i in range(3):
        rec = cm[i][i] / counts[i] if counts[i] else 0.0
        print("  %-5s %8d %8d %8d  | %.3f  %6d"
              % (NAMES[i], cm[i][0], cm[i][1], cm[i][2], rec, counts[i]))

print("\n" + "=" * 74)
print("判定")
print("=" * 74)
neg_r = ub["recall"][0] or 0.0
checks = [
    ("NEG 召回 > 0.30 (G5 判据)", neg_r > 0.30),
    ("NEG 召回相比旧版提升", neg_r > (ua["recall"][0] or 0.0)),
    ("NEG 精确率未崩 (>0.40)", pb_ > 0.40),
    ("macro-F1 相比旧版提升", mb > ma),
    ("过拟合 gap < 0.20", b["gap"] < 0.20),
]
for label, ok in checks:
    print("  %-38s %s" % (label, "PASS" if ok else "FAIL"))
n_ok = sum(1 for _, ok in checks if ok)
print("\n%d/%d 通过" % (n_ok, len(checks)))
print("\n注: 类别加权会主动牺牲总准确率换取 NEG 召回, 所以准确率下降是")
print("    预期的。真正该看 macro-F1 和 NEG F1。")
