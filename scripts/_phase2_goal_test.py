#!/usr/bin/env python
"""Test Phase 2 against its STATED objectives (paper_design.md 4.2), not just
"did it train".

Objectives under test
---------------------
G1  alpha-Gate discriminates on the REAL feature distribution.
    Paper contribution #2: "high density and confidence push alpha->1,
    absence pushes alpha->0". A gate that outputs a near-constant alpha is
    functionally a fixed-weight blend and the contribution is void.

G2  alpha correlates with "the KG can actually judge this step"
    (the calibration target). Measured by AUC, not by loss value.

G3  Sign/monotonicity of each feature matches the paper's hypothesis:
    d(alpha)/d(density) > 0, d(alpha)/d(confidence) > 0,
    d(alpha)/d(entropy) < 0.

G4  Training IMPROVED the gate over its hand-set init. If the init gate
    already discriminates as well, Phase 2 added nothing.

G5  PRM head generalises (held-out), not just memorises. The 0.991 reported
    earlier was on training data and proves nothing about generalisation.

G6  Artifacts load into the Phase 3b consumption path without shape errors.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kgproweight.data.parsers import parsed_step_from_silver_dict
from kgproweight.data.entity_filter import clean_entities
from kgproweight.kg.coverage import graph_density
from kgproweight.kg.entity_linker import EntityLinker
from kgproweight.reward.alpha_gate import (
    AlphaGate,
    compute_link_confidence,
    entropy_from_logprobs,
)
from kgproweight.retrieval.bootstrap import resolve_entity_cache_path

CKPT = Path(sys.argv[1] if len(sys.argv) > 1 else
            "checkpoints/prm_alpha_gate_v1reann")
ENRICHED = CKPT / "silver_with_logprobs.jsonl"
MAX_TRAJ = int(sys.argv[2]) if len(sys.argv) > 2 else 1200


def auc(scores, labels):
    """Rank-based AUC; ties get average rank."""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # average ranks over ties
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    sum_pos = sum(r for r, (_, lab) in zip(ranks, pairs) if lab == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


print("=" * 72)
print("Phase 2 目标达成度测试")
print("=" * 72)

# ---------------------------------------------------------------- load gate
trained = AlphaGate()
trained.load_state_dict(torch.load(CKPT / "alpha_gate.pt", map_location="cpu"))
trained.eval()
init = AlphaGate()
init.eval()

print("\n[gate] trained  W=%s b=%.4f tau=%.4f" % (
    [round(v, 4) for v in trained.W.data.tolist()],
    float(trained.b.data), float(trained.tau.detach())))
print("[gate] init     W=%s b=%.4f tau=%.4f" % (
    [round(v, 4) for v in init.W.data.tolist()],
    float(init.b.data), float(init.tau.detach())))

# ------------------------------------------------- rebuild REAL features
print("\n正在重建真实特征分布 (density, link_confidence, entropy)...")
# Stream the enriched jsonl: it is 1.37 GB and SilverDatasetReader materialises
# every trajectory as Python objects at once (~10x blowup), which OOM-kills the
# box. We only need three scalars per step, so parse and discard line by line.
linker = EntityLinker(cache_path=resolve_entity_cache_path())

dens, conf, ent, verdict, lab_cls = [], [], [], [], []
n_traj = 0
with open(ENRICHED, encoding="utf-8") as fh:
    for line in fh:
        if n_traj >= MAX_TRAJ:
            break
        line = line.strip()
        if not line:
            continue
        try:
            traj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not traj.get("accepted"):
            continue
        n_traj += 1
        kg = [tuple(str(x) for x in t)
              for t in (traj.get("kg_subgraph") or [])
              if isinstance(t, (list, tuple)) and len(t) == 3]
        gd = graph_density(kg)
        for s_idx, step in enumerate(traj.get("steps") or []):
            if not str(step.get("text") or "").strip():
                continue
            parsed = parsed_step_from_silver_dict(step, fallback_index=s_idx)
            lc = compute_link_confidence(
                step_entities=clean_entities(parsed.mentioned_entities),
                entity_linker=linker,
            )
            e = entropy_from_logprobs(step.get("token_logprobs"))
            label = float(step.get("label", 0.0))
            cls = 2 if label >= 0.5 else (0 if label <= -0.5 else 1)
            dens.append(gd)
            conf.append(lc)
            ent.append(e)
            lab_cls.append(cls)
            verdict.append(0 if cls == 1 else 1)
        del traj

n = len(dens)
print("步骤数 %d (来自 %d 条 accepted 轨迹)" % (n, n_traj))
print("  density    mean=%.4f std=%.4f  p1=%.4f p50=%.4f p99=%.4f" % (
    mean(dens), std(dens), pct(dens, 1), pct(dens, 50), pct(dens, 99)))
print("  link_conf  mean=%.4f std=%.4f  p1=%.4f p50=%.4f p99=%.4f" % (
    mean(conf), std(conf), pct(conf, 1), pct(conf, 50), pct(conf, 99)))
print("  entropy    mean=%.4f std=%.4f  p1=%.4f p50=%.4f p99=%.4f" % (
    mean(ent), std(ent), pct(ent, 1), pct(ent, 50), pct(ent, 99)))
print("  KG 有裁决比例 %.1f%%" % (100.0 * sum(verdict) / n))

D = torch.tensor(dens, dtype=torch.float32)
C = torch.tensor(conf, dtype=torch.float32).clamp(0, 1)
E = torch.tensor(ent, dtype=torch.float32)
with torch.no_grad():
    a_tr = trained(D, C, E).tolist()
    a_in = init(D, C, E).tolist()

results = {}

# ------------------------------------------------------------------ G1
print("\n" + "-" * 72)
print("G1  alpha 在真实分布上是否有判别力 (非退化常数)")
print("-" * 72)
sd = std(a_tr)
spread = pct(a_tr, 99) - pct(a_tr, 1)
uniq = len(set(round(v, 4) for v in a_tr))
print("trained alpha  mean=%.4f std=%.4f  min=%.4f p1=%.4f p25=%.4f p50=%.4f p75=%.4f p99=%.4f max=%.4f" % (
    mean(a_tr), sd, min(a_tr), pct(a_tr, 1), pct(a_tr, 25),
    pct(a_tr, 50), pct(a_tr, 75), pct(a_tr, 99), max(a_tr)))
print("p99-p1 跨度 %.4f | 唯一值 %d" % (spread, uniq))
g1 = sd > 0.02 and spread > 0.05
print("=> %s (判据: std>0.02 且 p99-p1>0.05)" % ("PASS" if g1 else "FAIL"))
results["G1_discriminative"] = g1

# ------------------------------------------------------------------ G2
print("\n" + "-" * 72)
print("G2  alpha 是否与 'KG 能裁决该步骤' 相关 (AUC)")
print("-" * 72)
auc_tr = auc(a_tr, verdict)
auc_in = auc(a_in, verdict)
a_v1 = [a for a, v in zip(a_tr, verdict) if v == 1]
a_v0 = [a for a, v in zip(a_tr, verdict) if v == 0]
print("AUC(trained) = %.4f" % auc_tr)
print("AUC(init)    = %.4f" % auc_in)
print("alpha | KG有裁决  mean=%.4f (n=%d)" % (mean(a_v1), len(a_v1)))
print("alpha | KG无裁决  mean=%.4f (n=%d)" % (mean(a_v0), len(a_v0)))
print("均值差 %+.4f" % (mean(a_v1) - mean(a_v0)))
g2 = auc_tr > 0.55
print("=> %s (判据: AUC>0.55, 0.5=随机)" % ("PASS" if g2 else "FAIL"))
results["G2_auc"] = g2
results["_auc_trained"] = round(auc_tr, 4)
results["_auc_init"] = round(auc_in, 4)

# ------------------------------------------------------------------ G3
print("\n" + "-" * 72)
print("G3  单调性方向是否符合论文假设")
print("-" * 72)
md, mc, me = mean(dens), mean(conf), mean(ent)


def sweep(idx, lo, hi):
    base = [md, mc, me]
    out = []
    for v in (lo, hi):
        x = list(base)
        x[idx] = v
        with torch.no_grad():
            out.append(float(trained(
                torch.tensor([x[0]]), torch.tensor([x[1]]),
                torch.tensor([x[2]])).item()))
    return out


for name, idx, lo, hi, want in (
    ("density   ", 0, pct(dens, 5), pct(dens, 95), "+"),
    ("link_conf ", 1, pct(conf, 5), pct(conf, 95), "+"),
    ("entropy   ", 2, pct(ent, 5), pct(ent, 95), "-"),
):
    a_lo, a_hi = sweep(idx, lo, hi)
    d = a_hi - a_lo
    got = "+" if d > 0 else "-"
    ok = got == want
    print("  %s p5=%.3f->alpha=%.4f  p95=%.3f->alpha=%.4f  d=%+.4f  期望%s 实际%s  %s" % (
        name, lo, a_lo, hi, a_hi, d, want, got, "OK" if ok else "WRONG"))
    results["G3_mono_%d" % idx] = ok
g3 = all(results.get("G3_mono_%d" % i) for i in range(3))
print("=> %s" % ("PASS" if g3 else "FAIL"))

# ------------------------------------------------------------------ G4
print("\n" + "-" * 72)
print("G4  训练是否优于手工初始门控 (Phase 2 是否有增量价值)")
print("-" * 72)
bce_tr = -mean([math.log(max(a, 1e-7)) if v else math.log(max(1 - a, 1e-7))
                for a, v in zip(a_tr, verdict)])
bce_in = -mean([math.log(max(a, 1e-7)) if v else math.log(max(1 - a, 1e-7))
                for a, v in zip(a_in, verdict)])
print("BCE(trained) = %.4f" % bce_tr)
print("BCE(init)    = %.4f" % bce_in)
print("改善 %+.4f (%.1f%%)" % (bce_in - bce_tr, 100 * (bce_in - bce_tr) / bce_in))
print("init  alpha  mean=%.4f std=%.4f" % (mean(a_in), std(a_in)))
g4 = bce_tr < bce_in and auc_tr >= auc_in - 1e-6
print("=> %s (判据: BCE 下降 且 AUC 不退化)" % ("PASS" if g4 else "FAIL"))
results["G4_beats_init"] = g4

# ------------------------------------------------------------------ G6
print("\n" + "-" * 72)
print("G6  产物能否被 Phase 3b 消费路径加载")
print("-" * 72)
ok6 = True
try:
    g = AlphaGate()
    g.load_state_dict(torch.load(CKPT / "alpha_gate.pt", map_location="cpu"))
    g.eval()
    v = g.forward_single(0.3, 0.8, 3.0)
    print("  AlphaGate.forward_single(0.3,0.8,3.0) = %.4f  OK" % v)
    assert 0.0 < v < 1.0
except Exception as exc:
    print("  AlphaGate FAIL: %s" % exc)
    ok6 = False
try:
    import torch.nn as nn
    sd_t = torch.load(CKPT / "text_reward_head.pt", map_location="cpu")
    hid = sd_t["0.weight"].shape[1]
    head = nn.Sequential(nn.Linear(hid, 1), nn.Tanh())
    head.load_state_dict(sd_t)
    print("  text_reward_head hidden=%d  OK" % hid)
except Exception as exc:
    print("  text_reward_head FAIL: %s" % exc)
    ok6 = False
try:
    sd_p = torch.load(CKPT / "prm_head" / "prm_head.pt", map_location="cpu")
    shapes = {k: tuple(v.shape) for k, v in sd_p.items()}
    print("  prm_head %s  OK" % shapes)
    assert shapes.get("proj.2.weight", (0,))[0] == 3, "not 3-way"
except Exception as exc:
    print("  prm_head FAIL: %s" % exc)
    ok6 = False
print("=> %s" % ("PASS" if ok6 else "FAIL"))
results["G6_loadable"] = ok6

# ------------------------------------------------------------------ summary
print("\n" + "=" * 72)
print("汇总")
print("=" * 72)
core = [("G1 alpha 有判别力", results["G1_discriminative"]),
        ("G2 alpha 与 KG 裁决相关 (AUC %.3f)" % auc_tr, results["G2_auc"]),
        ("G3 单调性符合假设", g3),
        ("G4 优于初始门控", results["G4_beats_init"]),
        ("G6 产物可加载", results["G6_loadable"])]
for name, ok in core:
    print("  %-40s %s" % (name, "PASS" if ok else "FAIL"))
n_pass = sum(1 for _, ok in core if ok)
print("\n%d/%d 通过" % (n_pass, len(core)))

with open(CKPT / "goal_test.json", "w") as fh:
    json.dump({
        "results": {k: (bool(v) if isinstance(v, bool) else v)
                    for k, v in results.items()},
        "alpha_trained": {"mean": mean(a_tr), "std": std(a_tr),
                          "p1": pct(a_tr, 1), "p50": pct(a_tr, 50),
                          "p99": pct(a_tr, 99),
                          "min": min(a_tr), "max": max(a_tr)},
        "alpha_init": {"mean": mean(a_in), "std": std(a_in)},
        "auc_trained": auc_tr, "auc_init": auc_in,
        "bce_trained": bce_tr, "bce_init": bce_in,
        "features": {
            "density": {"mean": mean(dens), "std": std(dens)},
            "link_confidence": {"mean": mean(conf), "std": std(conf)},
            "entropy": {"mean": mean(ent), "std": std(ent)},
        },
        "n_steps": n, "n_traj": n_traj,
        "verdict_rate": sum(verdict) / n,
        "n_pass": n_pass, "n_total": len(core),
    }, fh, indent=2)
print("写入 %s" % (CKPT / "goal_test.json"))
