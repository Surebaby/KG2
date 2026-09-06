"""Fail-fast per-question KG overrides for SFT/PPO training.

The identity contract is ``dataset::qid`` plus an exact question hash.  This
module deliberately does not accept question-text-only indexes: those remain a
separate legacy path for reproducing historical runs.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from kgproweight.kg.question_kg import load_question_kg_index, question_key, question_sha256


@dataclass
class TrainingQuestionKGStats:
    trajectories: int = 0
    covered: int = 0
    absent: int = 0
    covered_empty: int = 0
    changed: int = 0

    @property
    def coverage_rate(self) -> float:
        return self.covered / max(1, self.trajectories)

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "coverage_rate": self.coverage_rate}


def read_question_kg_records(path: str | Path) -> Dict[str, Dict[str, Any]]:
    import json

    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid question-KG JSONL at line {line_number}") from exc
    return load_question_kg_index(rows)


def apply_training_question_kg(
    trajectories: Iterable[Any],
    records: Mapping[str, Mapping[str, Any]],
    *,
    min_coverage: float = 1.0,
    require_nonempty: bool = False,
) -> TrainingQuestionKGStats:
    if not 0.0 <= float(min_coverage) <= 1.0:
        raise ValueError("min_coverage must be in [0, 1]")
    stats = TrainingQuestionKGStats()
    for traj in trajectories:
        stats.trajectories += 1
        key = question_key(str(traj.dataset), str(traj.qid))
        record = records.get(key)
        if record is None:
            stats.absent += 1
            continue
        expected_hash = question_sha256(str(traj.question))
        if str(record.get("question_sha256") or "") != expected_hash:
            raise ValueError(f"question hash mismatch for training KG key={key}")
        if str(record.get("question") or "").strip() != str(traj.question).strip():
            raise ValueError(f"question text mismatch for training KG key={key}")
        triples = [tuple(str(part) for part in value) for value in record.get("kg_subgraph") or []]
        stats.covered += 1
        stats.covered_empty += int(not triples)
        stats.changed += int(list(traj.kg_subgraph) != triples)
        traj.kg_subgraph = triples
        # Preserve the Gold-free execution contract for reward routing.  Earlier
        # code copied only the triples, so PPO could not distinguish an
        # automatically planned two-hop proof from a legacy neighbourhood and
        # had no principled way to choose its validity target.
        metadata = getattr(traj, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            traj.metadata = metadata
        runtime = {
            "question_key": key,
            "query_plan": dict(record.get("query_plan") or {}),
            "provenance": dict(record.get("provenance") or {}),
        }
        # Reward-v2.1 needs the Gold-free executor trace. Historical records do
        # not contain it, so preserve their metadata byte-for-byte in meaning
        # and propagate it only for explicitly enriched versioned records.
        if "execution" in record:
            runtime["execution"] = dict(record.get("execution") or {})
        metadata["question_kg_runtime"] = runtime
        # Preserve original schema, identity and provenance for the opt-in
        # source gate. Never synthesize missing identity from the lookup key.
        metadata["source_quality_record"] = deepcopy(record)
    if stats.coverage_rate < float(min_coverage):
        raise ValueError(
            "question-KG record coverage "
            f"{stats.covered}/{stats.trajectories}={stats.coverage_rate:.1%} is below "
            f"min_coverage={float(min_coverage):.1%}"
        )
    if require_nonempty and stats.covered_empty:
        raise ValueError(
            f"question-KG records contain {stats.covered_empty} covered-but-empty trajectories"
        )
    return stats
