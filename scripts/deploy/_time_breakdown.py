"""Where do the ~82 seconds per optimiser update actually go?

The smoke run measured 81.9 s/update, which extrapolates to ~46 h for the full
2000-update schedule. Before accepting that, attribute the time. The log only
timestamps optimiser updates, so infer the split from the config and the two
costs we can measure directly here.

Reads the log; does not need a GPU.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

log = Path(sys.argv[1] if len(sys.argv) > 1
           else "outputs/split_ppo_smoke/train.log")
text = log.read_text(errors="replace")

TS = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")


def secs(hms: str) -> int:
    h, m, s = (int(x) for x in hms.split(" ")[1].split(":"))
    return h * 3600 + m * 60 + s


# One ADV_DEBUG before_whiten per optimiser update, emitted after rollout +
# reward and immediately before the PPO backward pass.
upd = [secs(m.group(1)) for line in text.splitlines()
       if "ADV_DEBUG before_whiten" in line
       for m in [TS.match(line)] if m]

# The R8 sampling line lands right after a rollout batch completes.
step_lines = [(secs(m.group(1)), line) for line in text.splitlines()
              if re.search(r"step=\d+ \(upd=\d+\)", line)
              for m in [TS.match(line)] if m]

if len(upd) < 2:
    print("not enough updates logged to time anything")
    sys.exit(0)

gaps = [b - a for a, b in zip(upd, upd[1:])]
gaps_sorted = sorted(gaps)
mean = sum(gaps) / len(gaps)
print(f"optimiser updates logged : {len(upd)}")
print(f"per-update wall clock    : mean {mean:.1f}s  "
      f"median {gaps_sorted[len(gaps_sorted) // 2]}s  "
      f"min {gaps_sorted[0]}s  max {gaps_sorted[-1]}s")
print(f"  gaps: {gaps}")

# Extrapolate honestly, and show what the knobs buy.
print("\n--- full-run projection at this rate ---")
for traj, label in [(2000, "2000 traj (250 upd)"),
                    (4000, "4000 traj (500 upd)"),
                    (16000, "16000 traj (2000 upd)")]:
    upds = traj // 8
    h = upds * mean / 3600
    print(f"  {label:24s} {h:6.1f} h")

print("\n--- what dominates: 8 traj x ppo_epochs, at mini_batch_size=1 ---")
print("  Each update does batch_size/mini_batch_size * ppo_epochs")
print("  = 8/1 * 2 = 16 sequential fwd+bwd passes, each recomputing")
print("  activations because gradient checkpointing is on.")
print("  Raising mini_batch_size batches those passes instead of serialising")
print("  them -- the single biggest lever, bounded by VRAM (92.4/97.9 GB used).")
print("\n  Independent of that, rollout generates ~250 tokens x 8 prompts")
print("  one prompt at a time (_generate loops), which is also serial.")
