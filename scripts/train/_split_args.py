"""Shared ``--split`` CLI plumbing for the phase training scripts.

Every phase that reads silver data needs the same four flags, and they have to
mean the same thing in each — a Phase 3 run on a different fold than Phase 2 is
worse than no split at all, because the resulting number looks held-out and
isn't. Defining them once removes the chance of the defaults drifting apart.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

from kgproweight.data.silver_split import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_VAL_RATIO,
)


def add_split_args(p: argparse.ArgumentParser) -> None:
    """Add ``--split`` / ``--val_ratio`` / ``--test_ratio`` / ``--split_seed``.

    All use ``argparse.SUPPRESS`` so ``getattr(args, name, yaml_value)`` can tell
    "user passed it" from "argparse filled in a default" — the same convention
    the memory-critical Phase 2 flags already use.
    """
    g = p.add_argument_group("train/val/test split")
    g.add_argument(
        "--split", choices=["train", "val", "test"], default=argparse.SUPPRESS,
        help="Fold to use. Omit for the whole file (pre-split behaviour). Use the "
             "SAME fold in every phase, or a later phase trains on what an "
             "earlier phase held out.",
    )
    g.add_argument(
        "--val_ratio", type=float, default=argparse.SUPPRESS,
        help="Fraction of question groups held out for validation (default 0.10).",
    )
    g.add_argument(
        "--test_ratio", type=float, default=argparse.SUPPRESS,
        help="Fraction of question groups held out for test (default 0.10).",
    )
    g.add_argument(
        "--split_seed", type=int, default=argparse.SUPPRESS,
        help="Seed for fold assignment. Separate from --seed so a seed sweep over "
             "training randomness does not redraw the held-out set (default 42).",
    )


def split_kwargs(args: argparse.Namespace, tcfg: Any = None) -> Dict[str, Any]:
    """Build the split kwargs for a phase config.

    CLI wins over YAML; YAML wins over the built-in default. ``tcfg=None`` is the
    no-config branch.
    """
    def pick(name: str, fallback: Any) -> Any:
        if hasattr(args, name):
            return getattr(args, name)
        if tcfg is not None:
            return getattr(tcfg, name, fallback)
        return fallback

    return {
        "split": pick("split", None),
        "val_ratio": pick("val_ratio", DEFAULT_VAL_RATIO),
        "test_ratio": pick("test_ratio", DEFAULT_TEST_RATIO),
        "split_seed": pick("split_seed", DEFAULT_SPLIT_SEED),
    }


def log_split(logger: Any, phase: str, cfg: Any) -> None:
    """Warn loudly when no fold is set.

    A silent whole-file run is exactly how an in-sample number ends up in a paper
    labelled as held-out, so this is a warning rather than an info line.
    """
    if getattr(cfg, "split", None) is None:
        logger.warning(
            "%s split: NONE — using the whole silver file. val/test are NOT held "
            "back, so nothing measured on this data is out-of-sample. Pass "
            "--split train to hold them back.", phase,
        )
    else:
        logger.info(
            "%s split: fold=%s val=%.3f test=%.3f split_seed=%s",
            phase, cfg.split, cfg.val_ratio, cfg.test_ratio,
            cfg.seed if cfg.split_seed is None else cfg.split_seed,
        )
