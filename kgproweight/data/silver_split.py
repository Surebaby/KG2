"""Deterministic train/val/test split for silver trajectories.

Every phase currently calls ``reader.accepted()`` on the whole file, so there is
no held-out set anywhere in the pipeline: Phase 2's generalisation number is
measured on ``StratifiedSilverFilter``-rejected trajectories, which is a
distribution-shifted proxy, not a same-distribution held-out set. This module
supplies the missing split.

Three properties drive the design.

**Hash bucketing, not shuffle-and-slice.** ``random.shuffle`` on a list assigns
folds by *position*, so appending trajectories (Phase 1 is re-run to add 2Wiki /
MuSique) or reordering the file reshuffles every previous assignment and leaks
old training items into the new test fold. Hashing the group key to a fixed
number of buckets makes each trajectory's fold a function of its key alone:
stable under append, reorder, filtering and subsetting.

**Grouping by normalised question, not qid.** qids are unique in the current
file but one question text appears twice (``train_3398`` / ``train_22486``).
Splitting on qid would put paraphrase-identical items on both sides of the
boundary. Grouping guarantees all trajectories for a question land in one fold.

**Stratifying on ``accepted``.** Accepted / rejected is 39.4% / 60.6% and the
two carry different label distributions — the whole reason the rejected set is a
shifted proxy. Independent bucketing per stratum keeps that ratio inside each
fold, so a val fold is comparable to a train fold.

Fold assignment is a pure function of ``(group_key, stratum, seed, ratios)``.
Nothing here reads or writes the silver file, so the split can be recomputed
identically in any phase without materialising three copies of a 1.28 GB file.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from kgproweight.data.silver_dataset import SilverTrajectory

__all__ = [
    "DEFAULT_VAL_RATIO",
    "DEFAULT_TEST_RATIO",
    "DEFAULT_SPLIT_SEED",
    "SplitSpec",
    "SplitCounts",
    "TRAIN",
    "VAL",
    "TEST",
    "SPLIT_NAMES",
    "group_key",
    "assign_split",
    "split_trajectories",
    "summarize_split",
]

TRAIN = "train"
VAL = "val"
TEST = "test"
SPLIT_NAMES: Tuple[str, str, str] = (TRAIN, VAL, TEST)

# Single source of truth for the defaults. Every phase config and CLI reads these
# rather than repeating the literals: nine copies of "0.10" is nine chances for
# one phase to hold back a different fold than another, which produces a
# held-out number that quietly isn't one.
#
# 10% rather than 5% because the metric these folds exist to measure is NEG
# recall and NEG is only ~3.4% of held-out accepted steps — see SplitSpec.
DEFAULT_VAL_RATIO = 0.10
DEFAULT_TEST_RATIO = 0.10
# Independent of the training seed on purpose, so a seed sweep does not redraw
# the held-out set.
DEFAULT_SPLIT_SEED = 42

# Resolution of the bucket grid. 10_000 buckets means a ratio is honoured to
# within 0.01% in the limit, and with ~25k groups the realised fold sizes land
# within a few tenths of a percent of the requested ratio.
_N_BUCKETS = 10_000

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True)
class SplitSpec:
    """Ratios and seed defining a split.

    ``val`` and ``test`` are given explicitly and train takes the remainder, so
    the three always sum to exactly 1.0 with no floating-point residue to
    allocate.
    """

    # 10% rather than the more usual 5%, because the metric these folds exist to
    # measure is NEG recall and NEG is only ~3.4% of held-out accepted steps. At
    # 5% the test fold holds ~57 NEG steps, so a recall of 0.68 carries a 95% CI
    # of +/-0.12 — too wide to state as a result. At 10% it is ~114 steps and
    # +/-0.086. The cost is small: train still keeps ~20,078 trajectories, far
    # more than a LoRA head needs.
    val_ratio: float = DEFAULT_VAL_RATIO
    test_ratio: float = DEFAULT_TEST_RATIO
    seed: int = DEFAULT_SPLIT_SEED
    # Set False to key on qid instead of question text. Only useful when the
    # corpus is known to have no repeated questions AND qids must be the unit of
    # analysis; the default is safer.
    group_by_question: bool = True
    # Set False to bucket all trajectories together rather than per accepted
    # flag. Keeps the split valid but lets the accepted ratio drift between
    # folds, which makes val/train comparisons noisier.
    stratify_accepted: bool = True

    def __post_init__(self) -> None:
        for name, v in (("val_ratio", self.val_ratio), ("test_ratio", self.test_ratio)):
            if not (0.0 <= v < 1.0):
                raise ValueError(f"{name} must be in [0, 1), got {v}")
        if self.val_ratio + self.test_ratio >= 1.0:
            raise ValueError(
                "val_ratio + test_ratio must leave a non-empty train fold, got "
                f"{self.val_ratio} + {self.test_ratio}"
            )

    @property
    def train_ratio(self) -> float:
        return 1.0 - self.val_ratio - self.test_ratio

    # Bucket boundaries, computed once. Test occupies the lowest buckets and val
    # the next band, so *lowering* test_ratio never moves an item out of val and
    # vice versa is contained — shrinking a holdout keeps the remaining holdout
    # items in place instead of re-drawing them.
    @property
    def _test_cut(self) -> int:
        return int(round(self.test_ratio * _N_BUCKETS))

    @property
    def _val_cut(self) -> int:
        return self._test_cut + int(round(self.val_ratio * _N_BUCKETS))


@dataclass(frozen=True)
class SplitCounts:
    """Per-fold trajectory / accepted counts, for logging and manifests."""

    n: Dict[str, int]
    n_accepted: Dict[str, int]
    n_groups: Dict[str, int]

    def as_dict(self) -> Dict[str, Dict[str, int]]:
        return {"n": dict(self.n), "n_accepted": dict(self.n_accepted),
                "n_groups": dict(self.n_groups)}


def _normalize_question(q: str) -> str:
    """Casefold, strip punctuation, collapse whitespace.

    Catches the near-duplicates that exact string equality misses (trailing
    ``?``, double spaces, capitalisation) without going as far as stemming,
    which would start merging genuinely different questions.
    """
    s = _PUNCT_RE.sub(" ", str(q).lower())
    return _WS_RE.sub(" ", s).strip()


def group_key(traj: SilverTrajectory, spec: SplitSpec = SplitSpec()) -> str:
    """Key whose fold assignment all co-keyed trajectories share.

    Falls back to qid when the question is empty, so a malformed record gets its
    own fold rather than colliding with every other empty-question record into
    one giant group (which would then swallow a whole fold).
    """
    if spec.group_by_question:
        norm = _normalize_question(traj.question)
        if norm:
            return "q:" + norm
    return "id:" + str(traj.qid)


def _bucket(key: str, stratum: str, seed: int) -> int:
    """Map a key to ``[0, _N_BUCKETS)``.

    blake2b rather than ``hash()``: Python's string hash is salted per process
    (PYTHONHASHSEED), so ``hash()`` would hand back a *different* split on every
    run — silently training on last run's test fold.
    """
    payload = f"{seed}\x00{stratum}\x00{key}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % _N_BUCKETS


def assign_split(traj: SilverTrajectory, spec: SplitSpec = SplitSpec()) -> str:
    """Return ``"train"``, ``"val"`` or ``"test"`` for one trajectory.

    Pure function of the trajectory's key and the spec — no dataset context, no
    ordering, no mutable state.
    """
    stratum = ("acc" if traj.accepted else "rej") if spec.stratify_accepted else "all"
    b = _bucket(group_key(traj, spec), stratum, spec.seed)
    if b < spec._test_cut:
        return TEST
    if b < spec._val_cut:
        return VAL
    return TRAIN


def split_trajectories(
    trajectories: Iterable[SilverTrajectory],
    spec: SplitSpec = SplitSpec(),
) -> Dict[str, List[SilverTrajectory]]:
    """Partition into the three folds, preserving input order within each."""
    out: Dict[str, List[SilverTrajectory]] = {name: [] for name in SPLIT_NAMES}
    for t in trajectories:
        out[assign_split(t, spec)].append(t)
    return out


def summarize_split(
    folds: Dict[str, Sequence[SilverTrajectory]],
    spec: SplitSpec = SplitSpec(),
) -> SplitCounts:
    n: Dict[str, int] = {}
    n_acc: Dict[str, int] = {}
    n_grp: Dict[str, int] = {}
    for name in SPLIT_NAMES:
        items = folds.get(name, [])
        n[name] = len(items)
        n_acc[name] = sum(1 for t in items if t.accepted)
        n_grp[name] = len({group_key(t, spec) for t in items})
    return SplitCounts(n=n, n_accepted=n_acc, n_groups=n_grp)


def check_no_group_leak(
    folds: Dict[str, Sequence[SilverTrajectory]],
    spec: SplitSpec = SplitSpec(),
) -> Dict[str, int]:
    """Return group keys shared between folds, keyed by ``"a|b"``.

    Should always be empty by construction — this is a cheap assertion for
    callers and tests, not a repair step. A non-empty result means the group key
    is not deterministic (e.g. mutated question text) rather than that the
    bucketing is wrong.
    """
    keys = {
        name: {group_key(t, spec) for t in folds.get(name, [])} for name in SPLIT_NAMES
    }
    leaks: Dict[str, int] = {}
    for i, a in enumerate(SPLIT_NAMES):
        for b in SPLIT_NAMES[i + 1:]:
            shared = keys[a] & keys[b]
            if shared:
                leaks[f"{a}|{b}"] = len(shared)
    return leaks
