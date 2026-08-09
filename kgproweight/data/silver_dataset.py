"""Silver trajectory dataset reader.

Phase 1 writes one trajectory per line to
``data/silver_data/silver_trajectories.jsonl``. Phase 2 then augments
the file with per-step LLM logprobs (``silver_with_logprobs.jsonl``)
which fixes bug #3 (hardcoded ``semantic_entropy=0.5``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

if TYPE_CHECKING:  # avoids a circular import at runtime (silver_split imports us)
    from kgproweight.data.silver_split import SplitSpec


@dataclass
class SilverStepRecord:
    index: int
    text: str
    label: float  # continuous R_KG = precision × relevance in [-1, 1]
    cited_triples: List[Tuple[str, str, str]] = field(default_factory=list)
    token_logprobs: Optional[List[float]] = None  # filled by Phase 2 logprob pass

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SilverStepRecord":
        triples: List[Tuple[str, str, str]] = []
        for t in d.get("cited_triples", []) or []:
            if isinstance(t, (list, tuple)) and len(t) == 3:
                triples.append(tuple(str(x) for x in t))
        return cls(
            index=int(d.get("index", 0)),
            text=str(d.get("text", "")),
            label=float(d.get("label", 0)),
            cited_triples=triples,
            token_logprobs=d.get("token_logprobs"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "label": self.label,
            "cited_triples": [list(t) for t in self.cited_triples],
            **({"token_logprobs": self.token_logprobs} if self.token_logprobs is not None else {}),
        }


@dataclass
class SilverTrajectory:
    qid: str
    question: str
    answer: str
    dataset: str
    steps: List[SilverStepRecord]
    kg_subgraph: List[Tuple[str, str, str]] = field(default_factory=list)
    retrieved_passages: List[Dict[str, Any]] = field(default_factory=list)
    teacher_output: Optional[str] = None
    teacher_model: Optional[str] = None
    accepted: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SilverTrajectory":
        steps = [SilverStepRecord.from_dict(s) for s in d.get("steps", [])]
        kg: List[Tuple[str, str, str]] = []
        for t in d.get("kg_subgraph", []) or []:
            if isinstance(t, (list, tuple)) and len(t) == 3:
                kg.append(tuple(str(x) for x in t))
        return cls(
            qid=str(d.get("qid") or d.get("id") or ""),
            question=str(d.get("question", "")),
            answer=str(d.get("answer", "")),
            dataset=str(d.get("dataset", "")),
            steps=steps,
            kg_subgraph=kg,
            retrieved_passages=d.get("retrieved_passages", []) or [],
            teacher_output=d.get("teacher_output"),
            teacher_model=d.get("teacher_model"),
            accepted=bool(d.get("accepted", True)),
            metadata=d.get("metadata", {}) or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "qid": self.qid,
            "question": self.question,
            "answer": self.answer,
            "dataset": self.dataset,
            "steps": [s.to_dict() for s in self.steps],
            "kg_subgraph": [list(t) for t in self.kg_subgraph],
            "retrieved_passages": self.retrieved_passages,
            "accepted": self.accepted,
            "metadata": self.metadata,
        }
        if self.teacher_output is not None:
            out["teacher_output"] = self.teacher_output
        if self.teacher_model is not None:
            out["teacher_model"] = self.teacher_model
        return out


def iter_silver_trajectories(path: str | Path) -> Iterator[SilverTrajectory]:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield SilverTrajectory.from_dict(json.loads(line))
            except json.JSONDecodeError:
                continue


class SilverDatasetReader:
    """Load and slice silver data in memory.

    Memory footprint of 15k trajectories with ~7 steps each is ~50 MB JSON,
    fine for a single GPU node. For larger datasets, switch to an iterable
    loader.
    """

    def __init__(
        self,
        path: str | Path,
        split: Optional[str] = None,
        split_spec: Optional["SplitSpec"] = None,
    ) -> None:
        """Load the file, optionally keeping only one fold.

        ``split`` filters at load time so every downstream call — ``accepted()``,
        ``subset()``, ``__iter__`` — sees only that fold. This is deliberate:
        12+ call sites across Phase 2/3a/3b treat ``accepted()`` as "the data",
        and threading a fold argument through each one invites the failure where
        one site is missed and quietly trains on test. Filtering at the source
        makes the reader itself the single point of enforcement.

        ``split=None`` keeps the historical whole-file behaviour, so existing
        callers and the rejected-trajectory G5 proxy are unaffected until they
        opt in.
        """
        self.path = Path(path)
        self.split = split
        self.split_spec = split_spec
        all_trajectories = list(iter_silver_trajectories(self.path))
        if split is None:
            self.trajectories: List[SilverTrajectory] = all_trajectories
        else:
            from kgproweight.data.silver_split import (
                SPLIT_NAMES,
                SplitSpec,
                assign_split,
            )

            if split not in SPLIT_NAMES:
                raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")
            spec = split_spec or SplitSpec()
            self.split_spec = spec
            self.trajectories = [t for t in all_trajectories if assign_split(t, spec) == split]
        # Kept so callers can report what fraction of the file a fold covers
        # without re-reading 1.28 GB.
        self.n_total_in_file = len(all_trajectories)

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> SilverTrajectory:
        return self.trajectories[idx]

    def __iter__(self) -> Iterator[SilverTrajectory]:
        return iter(self.trajectories)

    def accepted(self) -> List[SilverTrajectory]:
        return [t for t in self.trajectories if t.accepted]

    def subset(self, n: int, seed: int = 42) -> List[SilverTrajectory]:
        """Reproducible random subset; used by the data-efficiency rigour scan.

        Shuffles a copy: the previous version shuffled the list returned by
        ``accepted()`` and returned a prefix of it, which is fine, but taking a
        copy makes it explicit that repeated calls are independent.
        """
        import random

        rng = random.Random(seed)
        accepted = list(self.accepted())
        rng.shuffle(accepted)
        return accepted[:n]

    def splits(
        self, spec: Optional["SplitSpec"] = None
    ) -> Dict[str, List[SilverTrajectory]]:
        """Partition the loaded trajectories into all three folds at once."""
        from kgproweight.data.silver_split import SplitSpec, split_trajectories

        return split_trajectories(self.trajectories, spec or SplitSpec())

    @staticmethod
    def write_jsonl(path: str | Path, trajectories: Iterable[SilverTrajectory]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            for t in trajectories:
                fh.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")


def attach_logprobs(
    trajectory: SilverTrajectory,
    step_logprobs: Sequence[Sequence[float]],
) -> SilverTrajectory:
    """Mutate ``trajectory`` in place with one logprob list per step."""
    for step, lp in zip(trajectory.steps, step_logprobs):
        step.token_logprobs = [float(x) for x in lp]
    return trajectory
