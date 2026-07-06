#!/usr/bin/env python
"""Build R5 mixed training set: 70% hard examples + 30% random full silver.

Usage:
  python scripts/prepare/build_mixed_data.py \
    --hard checkpoints/prm_alpha_gate/silver_hard_examples.jsonl \
    --full data/silver_data/silver_trajectories.jsonl \
    --output checkpoints/prm_alpha_gate/silver_mixed_r5.jsonl \
    --hard_ratio 0.7 --seed 42
"""
import argparse, json, random

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hard", required=True)
    ap.add_argument("--full", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--hard_ratio", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    print("Loading hard examples...")
    hard = [json.loads(l) for l in open(args.hard) if l.strip()]
    print(f"  Hard: {len(hard)}")

    print("Loading full silver...")
    accepted = []
    with open(args.full) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if d.get("accepted"):
                    accepted.append(d)
    print(f"  Full accepted: {len(accepted)}")

    # Sample from full silver to match the desired ratio
    n_total = len(hard)
    n_full_needed = int(n_total / args.hard_ratio * (1 - args.hard_ratio))
    n_full_needed = min(n_full_needed, len(accepted))
    full_sample = random.sample(accepted, n_full_needed)

    mixed = hard + full_sample
    random.shuffle(mixed)

    print(f"  Mixed total: {len(mixed)} ({len(hard)} hard + {len(full_sample)} full)")
    print(f"  Hard ratio: {len(hard)/len(mixed):.1%}")
    print(f"  Full ratio: {len(full_sample)/len(mixed):.1%}")

    with open(args.output, "w") as f:
        for t in mixed:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    import os
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"\nSaved → {args.output} ({size_mb:.0f} MB)")

if __name__ == "__main__":
    main()
