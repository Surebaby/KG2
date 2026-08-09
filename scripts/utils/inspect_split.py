#!/usr/bin/env python
"""Print the train/val/test split of a silver file without running a phase.

Two uses. Before a run: confirm the folds are the size and composition you
expect. After a run: confirm a checkpoint's fold matches the one you are about
to evaluate on — the split is a pure function of the spec, so recomputing it
here gives exactly the folds any phase would use.

    python scripts/utils/inspect_split.py data/silver_data/silver_v1_reannotated.jsonl
    python scripts/utils/inspect_split.py <file> --val_ratio 0.1 --test_ratio 0.1
    python scripts/utils/inspect_split.py <file> --dump-qids test > test_qids.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.data.silver_split import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_VAL_RATIO,
    SPLIT_NAMES,
    SplitSpec,
    check_no_group_leak,
    group_key,
    summarize_split,
)


def _label_class(v: float) -> int:
    # Same bucketing as phase2_prm._label_to_class.
    return 0 if v <= -0.5 else (2 if v >= 0.5 else 1)


@dataclass(frozen=True)
class _Keyed:
    """The only three fields ``assign_split`` / ``group_key`` actually read.

    Structurally compatible with ``SilverTrajectory`` for those two functions, so
    the streaming path computes *identical* folds — it just never builds the
    steps / passages / kg_subgraph that make a full trajectory large.
    """

    qid: str
    question: str
    accepted: bool


def _stream_tally(path, spec):
    """Fold tallies read one JSON line at a time, discarding each row after use.

    Why this exists: the AutoDL container in no-GPU mode has a 2 GiB cgroup
    memory limit (``/sys/fs/cgroup/memory.max``) while the silver file is 1.37 GB
    of JSON, so ``SilverDatasetReader`` — which materialises all 24,998
    trajectories — is SIGKILLed by the OOM killer (exit 137) before printing
    anything. Peak RSS here is one parsed line plus ~25k group-key strings.

    Returns the same quantities the eager path derives from ``folds``:
    ``(counts_n, counts_acc, group_keys_per_fold, label_hist_per_fold, total,
    n_accepted, qids_per_fold_or_None)``.
    """
    from kgproweight.data.silver_split import assign_split

    n = {k: 0 for k in SPLIT_NAMES}
    n_acc = {k: 0 for k in SPLIT_NAMES}
    keys = {k: set() for k in SPLIT_NAMES}
    hist = {k: Counter() for k in SPLIT_NAMES}
    total = 0
    total_acc = 0

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            traj = _Keyed(
                qid=str(d.get("qid") or d.get("id") or ""),
                question=str(d.get("question", "")),
                accepted=bool(d.get("accepted", True)),
            )
            fold = assign_split(traj, spec)
            total += 1
            n[fold] += 1
            keys[fold].add(group_key(traj, spec))
            if traj.accepted:
                total_acc += 1
                n_acc[fold] += 1
                for s in d.get("steps", []) or []:
                    if not str(s.get("text", "") or "").strip():
                        continue
                    hist[fold][_label_class(float(s.get("label", 0)))] += 1
            del d

    return n, n_acc, keys, hist, total, total_acc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("silver_path")
    ap.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    ap.add_argument("--test_ratio", type=float, default=DEFAULT_TEST_RATIO)
    ap.add_argument("--split_seed", type=int, default=DEFAULT_SPLIT_SEED)
    ap.add_argument("--no_stratify", action="store_true")
    ap.add_argument("--group_by_qid", action="store_true")
    ap.add_argument("--dump-qids", choices=list(SPLIT_NAMES), default=None,
                    help="Print one qid per line for this fold and exit.")
    ap.add_argument("--stream", action="store_true",
                    help="Tally line-by-line instead of loading the file. Same "
                         "folds, ~1 GB less RAM. Required on the AutoDL box in "
                         "no-GPU mode, where a 2 GiB cgroup cap OOM-kills the "
                         "eager reader on a 1.37 GB silver file.")
    args = ap.parse_args()

    spec = SplitSpec(
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.split_seed,
        stratify_accepted=not args.no_stratify,
        group_by_question=not args.group_by_qid,
    )
    if args.stream:
        if args.dump_qids:
            # Streamed too: the qid list is what an evaluator filters on, and it
            # must be obtainable on the memory-capped box as well.
            from kgproweight.data.silver_split import assign_split
            with open(args.silver_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    t = _Keyed(str(d.get("qid") or d.get("id") or ""),
                               str(d.get("question", "")),
                               bool(d.get("accepted", True)))
                    if assign_split(t, spec) == args.dump_qids:
                        print(t.qid)
            return 0
        n_map, acc_map, key_map, hist_map, total, n_acc = _stream_tally(
            args.silver_path, spec)
        from kgproweight.data.silver_split import SplitCounts
        counts = SplitCounts(
            n=n_map, n_accepted=acc_map,
            n_groups={k: len(v) for k, v in key_map.items()})
        label_hist = hist_map
        leaks = {}
        for i, a in enumerate(SPLIT_NAMES):
            for b in SPLIT_NAMES[i + 1:]:
                shared = key_map[a] & key_map[b]
                if shared:
                    leaks["%s|%s" % (a, b)] = len(shared)
    else:
        reader = SilverDatasetReader(args.silver_path)
        folds = reader.splits(spec)

        if args.dump_qids:
            for t in folds[args.dump_qids]:
                print(t.qid)
            return 0

        counts = summarize_split(folds, spec)
        total = len(reader.trajectories)
        n_acc = len(reader.accepted())
        label_hist = {}
        for name in SPLIT_NAMES:
            h = Counter()
            for t in folds[name]:
                if not t.accepted:
                    continue
                for s in t.steps:
                    if not (s.text or "").strip():
                        continue
                    h[_label_class(float(s.label))] += 1
            label_hist[name] = h
        leaks = check_no_group_leak(folds, spec)

    print("=" * 74)
    print("Silver split — %s" % args.silver_path)
    print("=" * 74)
    print("spec: val=%.3f test=%.3f seed=%d stratify=%s group_by=%s"
          % (spec.val_ratio, spec.test_ratio, spec.seed,
             spec.stratify_accepted, "question" if spec.group_by_question else "qid"))
    print("file: %d trajectories, %d accepted (%.2f%%)"
          % (total, n_acc, 100.0 * n_acc / max(total, 1)))
    print()
    print("%-6s %8s %8s %10s %9s %8s" %
          ("fold", "traj", "%file", "accepted", "%of fold", "groups"))
    for name in SPLIT_NAMES:
        n, a = counts.n[name], counts.n_accepted[name]
        print("%-6s %8d %7.2f%% %10d %8.2f%% %8d"
              % (name, n, 100.0 * n / max(total, 1), a,
                 100.0 * a / max(n, 1), counts.n_groups[name]))

    # Step-level label histogram per fold. The class weights are computed on the
    # train fold, so they only transfer if the held-out folds have a comparable
    # label mix — and NEG is the rare class the whole PRM claim rests on.
    print("\nstep labels over ACCEPTED trajectories (the Phase 2 training pool)")
    print("%-6s %8s %9s %9s %9s" % ("fold", "steps", "NEG", "NEU", "POS"))
    neg_counts = {}
    for name in SPLIT_NAMES:
        hist = label_hist[name]
        n = sum(hist.values())
        neg_counts[name] = hist[0]
        if not n:
            print("%-6s %8d %9s %9s %9s" % (name, 0, "-", "-", "-"))
            continue
        print("%-6s %8d %8.2f%% %8.2f%% %8.2f%%"
              % (name, n, 100.0 * hist[0] / n, 100.0 * hist[1] / n,
                 100.0 * hist[2] / n))

    print("\ngroup leakage across folds: %s"
          % ("NONE" if not leaks else "LEAK -> %s" % leaks))

    # A held-out NEG recall is only as precise as the NEG count behind it, and
    # that count is what usually makes a small fold useless for this metric.
    import math
    print("\nprecision of a held-out NEG-recall estimate (Wilson, p=0.68):")
    for name in ("val", "test"):
        n = neg_counts.get(name, 0)
        if n < 5:
            print("  %-5s NEG n=%-5d too few to estimate" % (name, n))
            continue
        half = 1.96 * math.sqrt(0.68 * 0.32 / n)
        verdict = "usable" if half <= 0.10 else "TOO WIDE to report"
        print("  %-5s NEG n=%-5d 95%% CI +/-%.3f  (%s)" % (name, n, half, verdict))

    if leaks:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
