#!/usr/bin/env python
"""Real-time PPO training monitor — run this ON the AutoDL box's terminal.

Reads the local ``ppo_rerun.log`` + checkpoint dir directly (no SSH, stdlib
only) and renders a live dashboard: training phase, step/total, mean_reward,
KL (with a health verdict), loss and a short trend. The thing to watch after
the init_kl_coef fix is that KL stays anchored (~2–8) instead of running away
to three digits, and that reward does not collapse to 0.

Usage (from /root/autodl-tmp/kgpaper):
    python scripts/monitor_ppo.py                 # live, refresh every 10s
    python scripts/monitor_ppo.py -n 5            # refresh every 5s
    python scripts/monitor_ppo.py --once          # one snapshot, then exit
    python scripts/monitor_ppo.py --raw 30        # also show last 30 raw log lines
    python scripts/monitor_ppo.py --log other.log # point at a different log file

Paths default to this run; override with --log / --ckpt or env KGPPO_LOG/KGPPO_CKPT.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DEF_LOG = os.environ.get("KGPPO_LOG", os.path.join(HERE, "ppo_rerun.log"))
DEF_CKPT = os.environ.get("KGPPO_CKPT", os.path.join(HERE, "checkpoints", "kg_proweight_final"))
TOTAL_STEPS = int(os.environ.get("KGPPO_TOTAL", "1500"))

# step=NN mean_reward=X kl=Y loss=Z clipfrac=W  (emitted every batch_size*4 steps)
STEP_RE = re.compile(
    r"step=(\d+)\s+mean_reward=([-\d.eE+]+)\s+kl=([-\d.eE+]+)\s+"
    r"loss=([-\d.eE+]+|nan)\s+clipfrac=([-\d.eE+]+|nan)"
)
SHARD_RE = re.compile(r"Downloading shards:\s*\d+%[^\d]*(\d+)/(\d+)")

C = {"g": "\033[32m", "y": "\033[33m", "r": "\033[31m", "c": "\033[36m",
     "b": "\033[1m", "d": "\033[2m", "x": "\033[0m"}


def read_tail(path: str, nbytes: int = 80000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            data = f.read()
    except FileNotFoundError:
        return ""
    # progress bars use \r; flatten so we see real lines
    return data.decode("utf-8", "replace").replace("\r", "\n")


def proc_alive() -> bool:
    try:
        out = subprocess.run(
            ["pgrep", "-f", "scripts/train/phase3_ppo.py"],
            capture_output=True, text=True, timeout=5,
        )
        return out.returncode == 0 and out.stdout.strip() != ""
    except Exception:
        return False


def gpu_line() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    except Exception:
        return ""


def parse(log: str, ckpt_dir: str) -> dict:
    rows = [m.groups() for m in STEP_RE.finditer(log)]
    steps = [
        {"step": int(s), "reward": float(mr), "kl": float(kl),
         "loss": (float(ls) if ls != "nan" else None),
         "clip": (float(cf) if cf != "nan" else None)}
        for (s, mr, kl, ls, cf) in rows
    ]
    alive = proc_alive()
    shard = None
    for m in SHARD_RE.finditer(log):
        shard = (int(m.group(1)), int(m.group(2)))
    low = log.lower()
    if steps:
        phase = "TRAINING"
    elif shard and shard[0] < shard[1]:
        phase = "DOWNLOADING"
    elif "loading checkpoint shards" in low or "policy trainable params" in low:
        phase = "LOADING"
    elif not alive:
        phase = "STOPPED"
    else:
        phase = "STARTING"
    if any(k in log for k in ("Traceback (most recent call last)", "CUDA out of memory")):
        phase = "ERROR" if not steps else phase
    if "Phase 3b PPO done" in log:
        phase = "DONE"
    try:
        entries = os.listdir(ckpt_dir)
    except FileNotFoundError:
        entries = []
    ckpts = sorted([c for c in entries if c.startswith("step_")],
                   key=lambda s: int(s.split("_")[1]))
    if "final" in entries:
        ckpts.append("final")
    return {"phase": phase, "steps": steps, "alive": alive, "shard": shard, "ckpts": ckpts}


def kl_verdict(kl: float) -> str:
    if kl != kl:
        return f"{C['r']}nan{C['x']}"
    if kl < 0 or kl > 50:
        return f"{C['r']}DIVERGING (the collapse signature){C['x']}"
    if 1.0 <= kl <= 12.0:
        return f"{C['g']}healthy{C['x']}"
    if kl < 1.0:
        return f"{C['y']}very low (policy may be stuck){C['x']}"
    return f"{C['y']}high{C['x']}"


def spark(vals):
    bars = "▁▂▃▄▅▆▇█"
    nums = [v for v in vals if v is not None]
    if not nums:
        return ""
    lo, hi = min(nums), max(nums)
    rng = (hi - lo) or 1.0
    return "".join(bars[min(7, int((v - lo) / rng * 7))] if v is not None else " " for v in vals)


def render(st: dict, log: str, raw_lines: int) -> str:
    out, ph = [], st["phase"]
    pc = {"TRAINING": "g", "DONE": "g", "DOWNLOADING": "c", "LOADING": "c",
          "STARTING": "y", "STOPPED": "r", "ERROR": "r"}.get(ph, "y")
    out.append(f"{C['b']}── PPO monitor ──{C['x']}   phase: {C[pc]}{C['b']}{ph}{C['x']}   "
               f"proc: {'alive' if st['alive'] else C['r']+'DEAD'+C['x']}")
    g = gpu_line()
    if g:
        out.append(f"  GPU: {g}")
    if ph == "DOWNLOADING" and st["shard"]:
        a, b = st["shard"]
        out.append(f"  reward-model shards: {C['c']}{a}/{b}{C['x']} ({a*100//b}%) — "
                   f"training starts after this")
    steps = st["steps"]
    if steps:
        last = steps[-1]
        pct = last["step"] * 100 // max(1, TOTAL_STEPS)
        loss_s = f"{last['loss']:.4f}" if last["loss"] is not None else "n/a"
        clip_s = f"{last['clip']:.3f}" if last["clip"] is not None else "n/a"
        out += ["",
                f"  {C['b']}step {last['step']}/{TOTAL_STEPS}{C['x']} ({pct}%)",
                f"    mean_reward = {C['b']}{last['reward']:+.4f}{C['x']}",
                f"    KL          = {C['b']}{last['kl']:.3f}{C['x']}   [{kl_verdict(last['kl'])}]",
                f"    loss        = {loss_s}",
                f"    clipfrac    = {clip_s}"]
        tail = steps[-24:]
        out += ["",
                f"  {C['d']}trend (last {len(tail)} logged points){C['x']}",
                f"    reward {spark([s['reward'] for s in tail])}  "
                f"[{min(s['reward'] for s in tail):+.3f} … {max(s['reward'] for s in tail):+.3f}]",
                f"    KL     {spark([s['kl'] for s in tail])}  "
                f"[{min(s['kl'] for s in tail):.2f} … {max(s['kl'] for s in tail):.2f}]",
                f"    loss   {spark([s['loss'] for s in tail])}"]
        rew = [s["reward"] for s in steps]
        if len(rew) >= 8 and max(rew[:-4]) > 0.05 and all(r <= 1e-6 for r in rew[-4:]):
            out.append(f"  {C['r']}{C['b']}⚠ reward collapsed to 0 — roll back to the last "
                       f"step_* checkpoint.{C['x']}")
    else:
        out.append(f"  {C['d']}(no training steps logged yet){C['x']}")
    out.append("")
    out.append(f"  checkpoints: {C['g']}{', '.join(st['ckpts']) if st['ckpts'] else '(none yet)'}{C['x']}")
    if raw_lines:
        out.append(f"  {C['d']}── last {raw_lines} log lines ──{C['x']}")
        for ln in [l for l in log.splitlines() if l.strip()][-raw_lines:]:
            out.append(f"  {C['d']}{ln[:160]}{C['x']}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--interval", type=int, default=10, help="refresh seconds (default 10)")
    ap.add_argument("--once", action="store_true", help="one snapshot then exit")
    ap.add_argument("--raw", type=int, default=0, help="also show last N raw log lines")
    ap.add_argument("--log", default=DEF_LOG, help=f"log path (default {DEF_LOG})")
    ap.add_argument("--ckpt", default=DEF_CKPT, help="checkpoint dir")
    args = ap.parse_args()

    try:
        while True:
            log = read_tail(args.log)
            st = parse(log, args.ckpt)
            frame = render(st, log, args.raw)
            if args.once:
                print(frame)
                return
            sys.stdout.write("\033[2J\033[H")
            print(f"{C['d']}{time.strftime('%F %T')}  refresh {args.interval}s  Ctrl-C to quit{C['x']}\n")
            print(frame)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
